"""Isolated Converse2D s1 dispatch experiment; production files are unchanged.

    python test/probe_training_s1_shapes.py

Copy the exact current C++/CUDA/header sources into a distinct namespace and
replace exactly one known use_scale1 selector with `return s==1`. Preparation,
kernel bodies, precision, lambda parameterization and parameters stay identical.
Direct solver input sizes include 100x100 padded prior features and 96x96
DataNet features; wrapper pad/crop and the surrounding network are not timed.
Synthetic fixed activations/parameters are shape-matched, not claimed captured
real activations. Observation and prior are the SAME requires_grad input x.

First validate output/dx/dw/db against independent full-FFT FP64:
normal output atol=rtol=3e-5, gradients atol=rtol=5e-5;
weak output atol=1e-6/rtol=1e-5, gradients unchanged. Any failure prevents ALL
timing. Weak cases reuse the existing scale-one regression's parameter fixture.
Then verify actual CUDA kernel names in an untimed one-call profiler capture.
Formal timing has no profiler: warm5, alternating round4/iters20, one GPU
fixture at a time, no leaf .grad accumulation. This is forward plus complete
x/kernel/bias VJP, not loss/optimizer time, full-network speed or a numerical
equivalence claim about ConvTranspose2d.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

from probe_pointwise_training import capture, clear_cuda, fixture, tensor_hash, timed_fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Converse2D/torch_converse2d"
SOURCE_NAMES = ("converse2d.cpp", "converse2d_kernels.cu", "converse2d_training.cu", "converse2d_training.h")
SELECTOR = "bool use_scale1(I H,I W,I s) { return s==1 && H*W>=65536; }"
FORCED_SELECTOR = "bool use_scale1(I H,I W,I s) { return s==1; }"
NORMAL_CASES = (
    ("prior_b4_c128_100_shared3", 4, 128, 100, 100, 1, 128, 3),
    ("datanet_b4_c64_96_batched7", 4, 64, 96, 96, 4, 64, 7),
    ("prior_b1_c128_100_shared3", 1, 128, 100, 100, 1, 128, 3),
    ("prior_b8_c128_100_shared3", 8, 128, 100, 100, 1, 128, 3),
    ("old_small_b1_c32_64x80", 1, 32, 64, 80, 1, 32, 3),
    ("old_small_b8_c32_64", 8, 32, 64, 64, 1, 32, 3),
    ("already_specialized_control_b1_c32_256", 1, 32, 256, 256, 1, 32, 3),
)
CORE_KERNELS = ("forward_scale1", "backward_scale1", "filter_scale1", "solve_alias",
                "solve_output", "adjoint_q", "adjoint_inputs", "adjoint_filter")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_hashes():
    return {name: digest((SOURCE / name).read_bytes()) for name in SOURCE_NAMES}


def load_forced(verbose=False):
    import torch
    from torch.utils import cpp_extension

    original = {name: (SOURCE / name).read_bytes() for name in SOURCE_NAMES}
    hashes = {name: digest(value) for name, value in original.items()}
    texts = {name: value.decode("utf-8").replace("\r\n", "\n") for name, value in original.items()}
    count = texts["converse2d_training.cu"].count(SELECTOR)
    if count != 1:
        raise RuntimeError(f"Expected one exact s1 gate, found {count}; do not patch an unknown source version")
    texts["converse2d_training.cu"] = texts["converse2d_training.cu"].replace(SELECTOR, FORCED_SELECTOR, 1)
    patched = {name: digest(value.encode()) for name, value in texts.items()}
    fingerprint = digest(json.dumps(dict(original=hashes, patched=patched), sort_keys=True).encode())[:16]
    prefix = "s1_forced_" + fingerprint + "_converse"
    build = ROOT / ".build/training_s1_shapes" / fingerprint
    build.mkdir(parents=True, exist_ok=True)
    sources, namespaced = [], {}
    for name, text in texts.items():
        text = text.replace("converse", prefix)
        target = build / name.replace("converse", prefix)
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        namespaced[target.name] = digest(text.encode())
        if target.suffix != ".h":
            sources.append(str(target))
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG", "1033")
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    flags = (["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"])
    flags += ["-DCONVERSE2D_WITH_CUDA=1", "-DS1_SHAPES_SOURCE_REV=0x" + fingerprint[:12]]
    wrapper_cl = "/Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX"
    normalize_cl = os.name == "nt" and os.environ.get("CL") == wrapper_cl
    if normalize_cl:
        # Preserve the exact compiler options, but let PyTorch's no-argument
        # cl.exe version probe run without an implicit source-less command.
        flags += wrapper_cl.split()
    extension = "training_s1_forced_" + fingerprint
    try:
        if normalize_cl:
            os.environ.pop("CL")
        cpp_extension.load(name=extension, sources=sources, extra_include_paths=[str(build)],
                           extra_cflags=flags, extra_cuda_cflags=["-O3", "-lineinfo"],
                           with_cuda=True, is_python_module=False, build_directory=str(build), verbose=verbose)
    finally:
        if normalize_cl:
            os.environ["CL"] = wrapper_cl
    namespace = prefix + "2d"
    return getattr(torch.ops, namespace), dict(source_sha256=hashes, patched_source_sha256=patched,
        namespaced_source_sha256=namespaced, namespace=namespace, extension=extension,
        build_directory=str(build), patch=dict(file="converse2d_training.cu", count=1,
        before=SELECTOR, after=FORCED_SELECTOR), cflags=flags, cuda_flags=["-O3", "-lineinfo"],
        known_wrapper_cl_moved_to_cflags=normalize_cl,
        nvcc_prepend_flags=os.environ.get("NVCC_PREPEND_FLAGS"))


def cases():
    import torch
    generator = torch.Generator(device="cpu").manual_seed(9214)
    result = []
    for name, batch, channels, height, width, kb, kc, kernel in NORMAL_CASES:
        x = torch.randn(batch, channels, height, width, generator=generator)
        weight = torch.rand(kb, kc, kernel, kernel, generator=generator)/(kernel*kernel)
        bias = torch.randn(1, channels, 1, 1, generator=generator)
        upstream = torch.randn(x.shape, generator=generator)/x.numel()**.5
        result.append(dict(name=name, eps=1e-3, weak=False, timed=True,
                           tensors=(x, weight, bias, upstream)))
    # Use the established fixture generation, including its strided x and
    # unused independent-prior RNG draw; this probe then shares x as the prior.
    from test_fp32_training import FP32Training
    torch.manual_seed(9214)
    for amplitude in (0., 1e-6, 1e-3):
        x, prior, weight, bias = FP32Training.data(None, 3, 4, 1, kb=1, kc=1)
        with torch.no_grad():
            x.mul_(1e-5)
            prior.mul_(1e-5)
            weight.mul_(amplitude)
            bias.fill_(-40.)
        upstream = torch.randn(x.shape, device="cuda", dtype=torch.float32)*1e-5
        tensors = tuple(t.detach().cpu().clone() for t in (x, weight, bias, upstream))
        result.append(dict(name=f"weak_s1_amplitude_{amplitude:g}", eps=1e-8,
                           weak=True, timed=False, tensors=tensors))
        del x, prior, weight, bias, upstream
    clear_cuda()
    return result


def method(ops, eps):
    return lambda x, weight, bias: ops.forward(x, x, weight, bias, 1, eps, "v7")


def metrics(actual, expected, *, output, weak):
    import torch
    atol, rtol = ((1e-6, 1e-5) if weak else (3e-5, 3e-5)) if output else (5e-5, 5e-5)
    if actual.shape != expected.shape:
        raise RuntimeError("Numerical reference shape mismatch")
    a, e = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return dict(passed=False, finite=False, atol=atol, rtol=rtol)
    delta, budget = (a-e).abs(), atol+rtol*e.abs()
    failed = int((delta > budget).sum())
    return dict(passed=failed == 0, finite=True, atol=atol, rtol=rtol,
                max_abs=delta.max().item(), relative_l2=(delta.norm()/e.norm().clamp_min(1e-30)).item(),
                max_pointwise_budget_ratio=(delta/budget).max().item(), failed_elements=failed,
                shape=list(a.shape))


def validate(case, variants):
    import torch
    from models.converse_core import converse2d_reference
    eps = case["eps"]
    reference = capture(case["tensors"], torch.float64,
                        lambda x, w, b: converse2d_reference(x, x, w, b, 1, eps))
    result = {}
    for name, ops in variants.items():
        actual = capture(case["tensors"], torch.float32, method(ops, eps))
        if any(value.dtype != torch.float32 for value in actual.values()):
            raise RuntimeError(f"{name}: expected FP32 outputs and gradients")
        checked = {key: metrics(actual[key], reference[key], output=key == "output", weak=case["weak"])
                   for key in reference}
        result[name] = dict(passed=all(value["passed"] for value in checked.values()), tensors=checked)
    return result


def verify_kernel_hit(case, ops, forced):
    import torch
    clear_cuda()
    inputs, run = fixture(case["tensors"], torch.float32, method(ops, case["eps"]))
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        run()
        torch.cuda.synchronize()
    names = [event.name for event in profile.events() if str(event.device_type).endswith("CUDA")]
    hits = {key: sum(key in name for name in names) for key in CORE_KERNELS}
    height, width = inputs[0].shape[-2:]
    specialized = forced or height*width >= 65536
    broadcast = inputs[1].shape[:2] != inputs[0].shape[:2]
    expected = (["forward_scale1", "backward_scale1"] if specialized else
                ["solve_alias", "solve_output", "adjoint_q", "adjoint_inputs"])
    if broadcast:
        expected.append("filter_scale1" if specialized else "adjoint_filter")
    valid = all(hits[key] == (1 if key in expected else 0) for key in CORE_KERNELS)
    if any(tensor.grad is not None for tensor in inputs):
        raise RuntimeError("Dispatch profiling accumulated leaf gradients")
    del profile, run, inputs
    clear_cuda()
    return dict(passed=valid, expected=expected, observed_kernel_counts=hits,
                scope="One untimed profiler call; not used in benchmark timings")


def time_case(case, variants, args):
    rounds = []
    for index in range(args.rounds):
        order = ("current", "forced_s1") if index % 2 == 0 else ("forced_s1", "current")
        values = {name: timed_fixture(case["tensors"], method(variants[name], case["eps"]), args)
                  for name in order}
        rounds.append(dict(round=index+1, order=list(order), variants=values))
    medians = {name: {key: statistics.median(row["variants"][name][key] for row in rounds)
                      for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in variants}
    ratios = {key: [row["variants"]["current"][key]/row["variants"]["forced_s1"][key] for row in rounds]
              for key in ("wall_ms", "cuda_event_ms")}
    return dict(rounds=rounds, medians=medians, paired_current_over_forced=ratios,
                median_paired_ratio={key: statistics.median(value) for key, value in ratios.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/training_research/s1_shapes.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("warmup, rounds and iters must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    import torch
    from extension_loader import load_extension
    if not torch.cuda.is_available() or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        parser.error("A CUDA-enabled production build is required")
    sys.path.insert(0, str(ROOT))
    before = source_hashes()
    load_extension(verbose=args.verbose_build)
    current = torch.ops.converse2d
    forced, build = load_forced(args.verbose_build)
    if before != source_hashes() or before != build["source_sha256"]:
        raise RuntimeError("Source changed during setup; restart the experiment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    variants = dict(current=current, forced_s1=forced)
    inputs = cases()
    report = dict(status="validating", scope=__doc__, forced_build=build,
                  current_build=json.loads((ROOT/".build/cuda/source_manifest.json").read_text()),
                  source_sha256=before,
                  script_sha256={name: digest((ROOT/"test"/name).read_bytes()) for name in
                                 ("probe_training_s1_shapes.py", "probe_pointwise_training.py",
                                  "test_fp32_training.py", "extension_loader.py")},
                  reference_sha256=digest((ROOT/"models/converse_core.py").read_bytes()),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(), tf32=False, cudnn_deterministic=True,
                                   CL=os.environ.get("CL"), NVCC_PREPEND_FLAGS=os.environ.get("NVCC_PREPEND_FLAGS")),
                  settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        for case in inputs:
            x, weight, bias, _ = case["tensors"]
            row = dict(name=case["name"], shape=list(x.shape), kernel_shape=list(weight.shape),
                       bias_shape=list(bias.shape), scale=1, eps=case["eps"], weak=case["weak"],
                       input_prior_identity="same requires_grad x", tensors_sha256=tensor_hash(case["tensors"]),
                       validation=validate(case, variants), timing=None)
            report["cases"].append(row)
            save()
            print(json.dumps(dict(case=case["name"], validation=row["validation"])), flush=True)
        if not all(check["passed"] for row in report["cases"] for check in row["validation"].values()):
            report["status"] = "numerical_gate_failed_no_timing"
            save()
            raise SystemExit(1)
        for case, row in zip(inputs, report["cases"]):
            row["dispatch"] = {name: verify_kernel_hit(case, ops, name == "forced_s1")
                               for name, ops in variants.items()}
            save()
            if not all(value["passed"] for value in row["dispatch"].values()):
                report["status"] = "kernel_dispatch_unverified_no_timing"
                save()
                raise SystemExit(1)
        report["status"] = "timing"
        for case, row in zip(inputs, report["cases"]):
            if case["timed"]:
                row["timing"] = time_case(case, variants, args)
                save()
                print(json.dumps(dict(case=case["name"], timing=row["timing"]["medians"])), flush=True)
        report["status"] = "complete_local_dispatch_experiment"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save()
        raise
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
