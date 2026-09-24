"""Isolated checked-build benchmark for differentiable PSF pad/roll fusion.

Run the baseline and candidate in separate Python processes with identical
arguments. The baseline for this study is d37e963. Example:
  python tools/benchmark_fp32_psf.py --root artifacts/fp32_psf/before --output artifacts/fp32_psf/before.json --include-model --profile
  python tools/benchmark_fp32_psf.py --root . --output artifacts/fp32_psf/after.json --include-model --profile
  python tools/benchmark_fp32_psf.py --compare artifacts/fp32_psf/before.json artifacts/fp32_psf/after.json --output artifacts/fp32_psf/comparison.json

Times include the entire operator: activation padding/cropping, nearest prior,
PSF preparation, every FFT and the requested VJP. FP64 is only a reference.
Optional CPU profiler event counts come from a separate untimed call, with
activation padding excluded from the internal-operator forward scope.
Use --deterministic-algorithms as a separate explicit comparison lane; its
results do not replace failures from the default nondeterministic lane.
"""
import argparse
from collections import Counter
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import benchmark_fp32_p0 as common


def specifications():
    result = [
        dict(name="usrnet_prior", shape=(1, 128, 32, 40), scale=1, kernel=(3, 3),
             kb=1, kc=128, padding=2, padding_mode="circular"),
    ]
    for scale in (1, 2, 3):
        spec = dict(name=f"dynamic_k7_s{scale}", shape=(2, 64, 32, 40),
                    scale=scale, kernel=(7, 7), kb=2, kc=64, padding=0)
        if scale == 2:
            spec["masks"] = ("all", "input_only", "kernel_only", "kernel_bias")
        result.append(spec)
    result.append(dict(name="s2_b1_no_broadcast_k7", shape=(1, 64, 32, 40),
                       scale=2, kernel=(7, 7), kb=1, kc=64, padding=0,
                       masks=("all", "input_only", "kernel_only", "kernel_bias")))
    for index, shape in enumerate(((1, 64, 24, 28), (1, 64, 48, 56)), 1):
        result.append(dict(name=f"srresnet_up{index}", shape=shape, scale=2,
                           kernel=(2, 2), kb=1, kc=64, padding=2,
                           padding_mode="replicate", eps=1e-3))
    for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
        result.append(dict(name=f"s3_broadcast_kb{kb}_kc{kc}", shape=(2, 3, 7, 8),
                           scale=3, kernel=(3, 2), kb=kb, kc=kc, padding=0))
    for scale in (1, 2):
        result.append(dict(name=f"s{scale}_transpose", shape=(2, 3, 9, 7), scale=scale,
                           kernel=(2, 3), kb=1, kc=3, padding=0, transpose=True))
    for spec in result:
        spec.setdefault("padding_mode", "circular")
        spec.setdefault("eps", 1e-5)
        spec.setdefault("transpose", False)
    return result


def fixture(torch, spec, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def random(shape):
        if spec["transpose"]:
            return torch.randn((*shape[:-2], shape[-1], shape[-2]), generator=generator).transpose(-1, -2)
        return torch.randn(shape, generator=generator)
    shape = spec["shape"]
    kh, kw = spec["kernel"]
    x = random(shape)
    weight = random((spec["kb"], spec["kc"], kh, kw)) / (kh * kw)
    bias = torch.randn((1, shape[1], 1, 1), generator=generator) * 0.1
    upstream = random((*shape[:2], shape[-2] * spec["scale"], shape[-1] * spec["scale"]))
    upstream /= upstream.numel() ** 0.5
    return x, weight, bias, upstream


def forward_vjp(torch, solver, data, upstream, spec, *, annotate=False):
    def scope(name):
        return torch.profiler.record_function(name) if annotate else contextlib.nullcontext()
    x, weight, bias = data
    scale, padding = spec["scale"], spec["padding"]
    with scope("psf_benchmark/activation_padding"):
        padded = torch.nn.functional.pad(x, (padding,) * 4, mode=spec["padding_mode"]) if padding else x
    with scope("psf_benchmark/prior"):
        prior = padded if scale == 1 else torch.nn.functional.interpolate(padded, scale_factor=scale, mode="nearest")
    with scope("psf_benchmark/operator_forward"):
        output = solver(padded, prior, weight, bias, scale, spec["eps"])
    if padding:
        crop = padding * scale
        output = output[..., crop:-crop, crop:-crop]
    required = [(name, value) for name, value in zip(("dx", "dweight", "dbias"), data) if value.requires_grad]
    with scope("psf_benchmark/vjp"):
        gradients = torch.autograd.grad(output, [value for _, value in required], upstream)
    return {"output": output.detach(), **{name: value.detach() for (name, _), value in zip(required, gradients)}}


def profile_counts(torch, function, spec):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        function()
        torch.cuda.synchronize()
    counts = {scope: Counter() for scope in ("activation_padding", "prior", "operator_forward", "vjp", "other")}
    fft_inputs = []
    for event in profile.events():
        parent = event
        category = "other"
        while parent is not None:
            if parent.name.startswith("psf_benchmark/"):
                category = parent.name.split("/", 1)[1]
                break
            parent = parent.cpu_parent
        lowered = event.name.lower()
        if any(token in lowered for token in ("pad", "roll", "fft", "psf", "copy", "slice", "zero", "converse2d")):
            counts[category][event.name] += 1
        if category == "operator_forward" and event.name == "aten::fft_fft2":
            fft_inputs.append({"input_shapes": event.input_shapes})
    return {
        "scope": "One independent complete forward+VJP; CPU API events only, no timing conclusion or CUDA launch-count claim",
        "counts": {name: dict(sorted(value.items())) for name, value in counts.items()},
        "operator_forward_fft2_events_in_order": fft_inputs,
        "expected_forward_fft2_count": 2 if spec["scale"] == 1 else 3,
        "attribution_note": "Production prepares the kernel FFT before input/prior FFTs; inspect the first fft2 input and source fingerprint. Recursive ATen roll events are API counts, not distinct GPU kernels. VJP scope includes activation-pad/crop backward.",
    }


def run_operators(torch, reference, args):
    masks = {"all": (True, True, True), "input_only": (True, False, False),
             "kernel_only": (False, True, False), "kernel_bias": (False, True, True)}
    cases = {}
    for index, spec in enumerate(specifications()):
        raw = fixture(torch, spec, args.seed + index)
        for mask in spec.get("masks", ("all", "input_only", "kernel_only")):
            needs = masks[mask]
            name = f"operator/{spec['name']}/{mask}"
            data = common.prepare(torch, raw, needs, torch.float32)
            ref_data = common.prepare(torch, raw, needs, torch.float64)
            upstream = raw[3].cuda()
            expected = forward_vjp(torch, reference, ref_data, upstream.double(), spec)
            python32 = forward_vjp(torch, reference, data, upstream, spec)
            def call():
                return forward_vjp(torch, torch.ops.converse2d.forward, data, upstream, spec)
            actual = call()
            snapshot = common.records(actual, expected)
            control = common.records(python32, expected)
            del expected, actual, python32, ref_data
            item = {
                "spec": spec, "needs_grad": dict(zip(("x", "weight", "bias"), needs)),
                "fft_shape": [(extent + 2 * spec["padding"]) * spec["scale"] for extent in spec["shape"][-2:]],
                "strides": {name: list(value.stride()) for name, value in zip(("x", "weight", "bias", "upstream"), (*data, upstream))},
                "fixture": common.records(dict(zip(("x", "weight", "bias", "upstream"), (*data, upstream)))),
                "snapshot": snapshot, "python_fp32": control,
                "scope": "Complete operator forward+selected VJPs; shared s1 or differentiable nearest prior; pad/crop and per-call kernel FFT included",
                "timing": common.measure(torch, call, args),
            }
            item["python_fp32_noninferiority"] = common.noninferiority(item)
            if args.profile:
                item["cpu_event_profile"] = profile_counts(
                    torch, lambda: forward_vjp(torch, torch.ops.converse2d.forward, data, upstream, spec, annotate=True), spec)
            cases[name] = item
            print(name, item["timing"]["median"], flush=True)
            del data, upstream
    torch.ops.converse2d.clear_cache()
    return cases


def compare(paths):
    result = common.compare(paths)
    result["kind"] = "psf_comparison"
    result["note"] = "PSF pad/roll fusion must retain each output/VJP hash; every FP64 error gate is reported independently. Profiler counts are untimed CPU API events. Timing medians do not establish statistical significance."
    before, after = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result["harness_match"] = before["harness_sha256"] == after["harness_sha256"]
    result["helpers_match"] = before["helper_sha256"] == after["helper_sha256"]
    result["cpu_event_profiles"] = {
        name: {label: report["cases"][name]["cpu_event_profile"] for label, report in (("before", before), ("after", after))}
        for name in sorted(before["cases"].keys() & after["cases"].keys())
        if all("cpu_event_profile" in report["cases"][name] for report in (before, after))
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Separate untimed CPU event-count capture per operator case")
    parser.add_argument("--deterministic-algorithms", action="store_true",
                        help="Separate deterministic-algorithm lane; enable PyTorch deterministic operators, including replicate-pad decomposition")
    parser.add_argument("--build", action="store_true", help="Explicitly build; otherwise require a matching checked CUDA binary")
    args = parser.parse_args()
    if args.compare:
        result = compare(args.compare)
    else:
        if args.warmup < 0 or args.rounds < 1 or args.iters < 1:
            parser.error("warmup must be nonnegative; rounds and iters must be positive")
        if os.environ.get("CONVERSE2D_BACKEND") or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
            raise RuntimeError("Unset CONVERSE2D_BACKEND and CONVERSE2D_CPU_ONLY=1; checked CUDA routing is required")
        args.root = args.root.resolve()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["CONVERSE2D_SKIP_BUILD"] = "0" if args.build else "1"
        sys.path.insert(0, str(args.root))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This benchmark requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(args.deterministic_algorithms)
        torch.manual_seed(args.seed)
        loader_path = args.root / "test/extension_loader.py"
        loader_spec = importlib.util.spec_from_file_location("psf_checked_loader", loader_path)
        loader = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(loader)
        loader.load_extension()
        from models.converse_core import converse2d_reference
        manifest_path = args.root / ".build/cuda/source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes = loader.production_source_hashes()
        python_paths = ("models/converse_core.py", "models/util_converse.py", "models/converse_usrnet.py", "test/extension_loader.py")
        python_hashes = {path: common.file_sha(args.root / path) for path in python_paths}
        harness_sha = common.file_sha(__file__)
        helper_sha = common.file_sha(common.__file__)
        result = {
            "kind": "psf_benchmark", "root": str(args.root), "study_baseline_commit": "d37e963",
            "harness_sha256": harness_sha, "helper_sha256": helper_sha,
            "settings": {key: getattr(args, key) for key in ("seed", "warmup", "rounds", "iters", "include_model", "profile", "deterministic_algorithms")},
            "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(), "tf32": False, "amp": False,
                            "cudnn_deterministic": True, "cudnn_benchmark": False,
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "source_sha256": source_hashes, "python_sha256": python_hashes,
            "checked_build_manifest": manifest,
            "memory_scope": "Peak additional PyTorch allocated bytes over live fixture tensors before a timed round; excludes non-PyTorch CUDA allocations and whole-process VRAM.",
            "cases": run_operators(torch, converse2d_reference, args),
        }
        if args.include_model:
            result["cases"].update(common.run_model(torch, args))
        if source_hashes != loader.production_source_hashes() or python_hashes != {p: common.file_sha(args.root / p) for p in python_paths}:
            raise RuntimeError("Source changed during benchmark; result is ineligible")
        if common.file_sha(args.root / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
            raise RuntimeError("Binary changed during benchmark; result is ineligible")
        if harness_sha != common.file_sha(__file__) or helper_sha != common.file_sha(common.__file__):
            raise RuntimeError("Benchmark harness changed during run; result is ineligible")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
