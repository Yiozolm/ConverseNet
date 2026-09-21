"""Isolate the production s=1 kernel benefit with unchanged current preparation/models.

The reference is a namespaced copy of current source with exactly one selector
changed to false. The current production extension is the candidate. Both use
current Python forwards, including the same per-forward spectrum reuse policy.
"""
import argparse
import hashlib
import json
import os
import sys
import types

import torch
from torch.utils import cpp_extension

import benchmark_fp32_training as common
from benchmark_training_refinements import report_metadata, training_case
from extension_loader import load_extension
from training_refinement_baseline import ROOT, SOURCE, SOURCE_NAMES

SELECTOR = "bool use_scale1(I H,I W,I s) { return s==1 && H*W>=65536; }"
DISABLED_SELECTOR = "bool use_scale1(I H,I W,I s) { return false; }"
_loaded = {}


def load_scale1_disabled(verbose=False):
    """Verify and patch one known selector in an isolated current-source copy."""
    originals = {name: (ROOT / SOURCE / name).read_bytes() for name in SOURCE_NAMES}
    original_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in originals.items()}
    content = {name: value.decode("utf-8").replace("\r\n", "\n")
               for name, value in originals.items()}
    target = "converse2d_training.cu"
    occurrences = content[target].count(SELECTOR)
    if occurrences != 1:
        raise RuntimeError(f"Expected one exact current use_scale1 selector, found {occurrences}; inspect the source before updating this ablation")
    content[target] = content[target].replace(SELECTOR, DISABLED_SELECTOR, 1)
    patched_hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in content.items()}
    fingerprint = hashlib.sha256(json.dumps(dict(original=original_hashes, patched=patched_hashes),
                                           sort_keys=True).encode()).hexdigest()[:16]
    if fingerprint in _loaded:
        return _loaded[fingerprint]
    prefix = "s1_disabled_" + fingerprint + "_converse"
    namespace = prefix + "2d"
    build = ROOT / ".build/training_s1_ablation" / fingerprint
    build.mkdir(parents=True, exist_ok=True)
    sources, transformed_hashes = [], {}
    for name, value in content.items():
        transformed = value.replace("converse", prefix)
        path = build / name.replace("converse", prefix)
        if not path.exists() or path.read_text(encoding="utf-8") != transformed:
            path.write_text(transformed, encoding="utf-8")
        transformed_hashes[path.name] = hashlib.sha256(transformed.encode()).hexdigest()
        if path.suffix != ".h":
            sources.append(str(path))
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG", "1033")
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    flags += ["-DCONVERSE2D_WITH_CUDA=1", "-DS1_ABLATION_SOURCE_REV=0x" + fingerprint[:12]]
    name = "training_s1_disabled_" + fingerprint
    cpp_extension.load(name=name, sources=sources, extra_include_paths=[str(build)],
                       extra_cflags=flags, extra_cuda_cflags=["-O3", "-lineinfo"],
                       with_cuda=True, is_python_module=False,
                       build_directory=str(build), verbose=verbose)
    manifest = dict(ref="current-with-s1-disabled:" + fingerprint,
                    source_sha256={f"{SOURCE}/{name}": value for name, value in original_hashes.items()},
                    patched_source_sha256=patched_hashes, namespaced_source_sha256=transformed_hashes,
                    patch=dict(file=f"{SOURCE}/{target}", occurrence_count=1,
                               before=SELECTOR, after=DISABLED_SELECTOR),
                    namespace=namespace, extension_name=name, build_directory=str(build),
                    cflags=flags, cuda_flags=["-O3", "-lineinfo"])
    _loaded[fingerprint] = (getattr(torch.ops, namespace), manifest)
    return _loaded[fingerprint]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=("all", "operators", "training"), default="all")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", default="artifacts/training_refinements/s1_ablation.json")
    args = parser.parse_args()
    if min(args.iters, args.rounds, args.warmup) < 1:
        parser.error("iters, rounds and warmup must be positive")
    if os.environ.get("CONVERSE2D_CPU_ONLY") == "1" or not torch.cuda.is_available():
        parser.error("A CUDA-enabled production build is required")
    if os.environ.get("CONVERSE2D_BACKEND", "").lower() not in ("", "auto", "cuda"):
        parser.error("CONVERSE2D_BACKEND must not select a reference backend")
    # Reject a source change between loading the production and derived builds.
    source_hashes = {name: hashlib.sha256((ROOT / SOURCE / name).read_bytes()).hexdigest()
                     for name in SOURCE_NAMES}
    skipped = os.environ.pop("CONVERSE2D_SKIP_BUILD", None)
    try:
        load_extension(verbose=args.verbose_build)
    finally:
        if skipped is not None:
            os.environ["CONVERSE2D_SKIP_BUILD"] = skipped
    current = torch.ops.converse2d
    disabled, baseline = load_scale1_disabled(args.verbose_build)
    after_hashes = {name: hashlib.sha256((ROOT / SOURCE / name).read_bytes()).hexdigest()
                    for name in SOURCE_NAMES}
    if source_hashes != after_hashes or any(
            source_hashes[name] != baseline["source_sha256"][f"{SOURCE}/{name}"]
            for name in SOURCE_NAMES):
        raise RuntimeError("Source changed during ablation setup; restart in a fresh process")
    variants = dict(dev=disabled, current=current)
    dispatch = {name: common.verify_fused_dispatch(ops) for name, ops in variants.items()}
    common.release_cuda()
    sys.path.insert(0, str(ROOT))
    from models import util_converse, converse_usrnet
    # The reused model factory receives current classes for BOTH variants.
    current_models = types.SimpleNamespace(util_converse=util_converse, converse_usrnet=converse_usrnet)
    torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = report_metadata(args, baseline, dispatch)
    report["variant_labels"] = dict(dev="current with s1 selector disabled", current="current production s1 selector enabled")
    report["model_scope"] = "Identical current model forwards, kernel preparation and within-forward reuse; only CUDA use_scale1 selector differs"
    report["benchmark_sha256"]["training_s1_ablation.py"] = hashlib.sha256(
        (ROOT / "test/training_s1_ablation.py").read_bytes()).hexdigest()
    configs = [("operator", (1, 32, 256, 256), 1, 1),
               ("operator", (4, 32, 256, 256), 1, 1),
               ("operator", (1, 32, 256, 256), 2, 1),
               ("operator", (1, 32, 256, 256), 3, 1)]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for config in configs[:1] if args.quick else configs:
        _, shape, scale, _ = config
        if args.mode in ("all", "operators"):
            report["operators"].append(common.operator_case(shape, scale, variants, args))
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.mode in ("all", "training"):
            report["training"].append(training_case(config, variants, current_models, args))
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
