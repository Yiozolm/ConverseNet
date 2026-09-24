"""Checked-build P0 comparison; run each checkout in a separate process.

Examples (build both checkouts first with their test/extension_loader.py):
  python tools/benchmark_fp32_p0.py --root .build/p0-before --output before.json
  python tools/benchmark_fp32_p0.py --root . --output after.json --include-model
  python tools/benchmark_fp32_p0.py --compare before.json after.json --output diff.json

No profiler, Graph capture, data loading, or quality/convergence claim. Cold
means an empty Converse kernel cache, not a cold CUDA/cuFFT process. Model
timing includes eager checks and Adam; operator timing includes padding,
nearest prior, all preparation/FFTs and the requested complete VJP.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_record(tensor, reference=None):
    value = tensor.detach().cpu().contiguous()
    result = {"shape": list(value.shape), "dtype": str(value.dtype),
              "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
              "finite": bool(value.isfinite().all())}
    if reference is not None:
        ref = reference.detach().cpu().double()
        error = value.double() - ref
        norm = ref.norm().item()
        result["fp64_error"] = {
            "max_abs": error.abs().max().item(),
            "relative_l2": error.norm().item() / norm if norm else (0.0 if not error.any() else None),
            "reference_l2": norm,
        }
    return result


def records(values, reference=None):
    return {name: tensor_record(value, None if reference is None else reference[name])
            for name, value in values.items()}


def measure(torch, function, args, *, before_call=None, before_round=None):
    """Wall time includes synchronization; events measure the submitted stream span."""
    for _ in range(args.warmup):
        if before_call:
            before_call()
        function()
    torch.cuda.synchronize()
    rows = []
    for _ in range(args.rounds):
        if before_round:
            before_round()
        if before_call:
            # Cold preparation remains inside the operator, cache clearing outside.
            wall_ms = cuda_ms = 0.0
            peaks = []
            for _ in range(args.iters):
                before_call()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                tick = time.perf_counter()
                start.record()
                function()
                end.record()
                end.synchronize()
                wall_ms += (time.perf_counter() - tick) * 1000
                cuda_ms += start.elapsed_time(end)
                peaks.append(max(0, torch.cuda.max_memory_allocated() - base))
            rows.append({"wall_ms": wall_ms / args.iters, "cuda_event_ms": cuda_ms / args.iters,
                         "peak_extra_allocated_bytes": max(peaks)})
        else:
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter()
            start.record()
            for _ in range(args.iters):
                function()
            end.record()
            end.synchronize()
            rows.append({"wall_ms": (time.perf_counter() - tick) * 1000 / args.iters,
                         "cuda_event_ms": start.elapsed_time(end) / args.iters,
                         "peak_extra_allocated_bytes": max(0, torch.cuda.max_memory_allocated() - base)})
    return {"rounds": rows,
            "median": {key: statistics.median(row[key] for row in rows) for key in rows[0]},
            "timing_policy": "one synchronized call at a time" if before_call else "synchronized batch"}


def make_fixture(torch, shape, scale, kernel, seed, *, kernel_channels=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=generator)
    kc = shape[1] if kernel_channels is None else kernel_channels
    weight = torch.randn((1, kc, kernel, kernel), generator=generator) / (kernel * kernel)
    bias = torch.randn((1, shape[1], 1, 1), generator=generator) * 0.1
    upstream = torch.randn((shape[0], shape[1], shape[2] * scale, shape[3] * scale), generator=generator)
    upstream /= upstream.numel() ** 0.5
    return x, weight, bias, upstream


def operator(torch, solver, data, scale, padding, eps):
    x, weight, bias = data
    if padding:
        x = torch.nn.functional.pad(x, (padding,) * 4, mode="circular")
    prior = x if scale == 1 else torch.nn.functional.interpolate(x, scale_factor=scale, mode="nearest")
    output = solver(x, prior, weight, bias, scale, eps)
    if padding:
        crop = padding * scale
        output = output[..., crop:-crop, crop:-crop]
    return output


def prepare(torch, raw, needs, dtype):
    return tuple(value.to(device="cuda", dtype=dtype).detach().requires_grad_(need)
                 for value, need in zip(raw[:3], needs))


def forward_vjp(torch, solver, data, upstream, scale, padding, eps):
    output = operator(torch, solver, data, scale, padding, eps)
    required = [(name, value) for name, value in zip(("dx", "dweight", "dbias"), data) if value.requires_grad]
    gradients = torch.autograd.grad(output, [value for _, value in required], upstream) if required else ()
    return {"output": output.detach(), **{name: value.detach() for (name, _), value in zip(required, gradients)}}


def run_operators(torch, reference, args):
    cases = {}
    # (NCHW, scale, padding, kernel channels): retain the original two s1
    # fixtures and add tiny odd/even dimensions and B/C-broadcast kernels.
    inference_specs = (
        ((1, 64, 128, 128), 1, 0, 64),
        ((1, 128, 24, 28), 1, 2, 128),
        ((2, 3, 7, 8), 2, 0, 1),
        ((1, 3, 8, 7), 3, 0, 3),
        ((2, 3, 7, 9), 4, 0, 1),
    )
    for shape_index, (shape, scale, padding, kernel_channels) in enumerate(inference_specs):
        raw = make_fixture(torch, shape, scale, 3, args.seed + shape_index, kernel_channels=kernel_channels)
        data = prepare(torch, raw, (False,) * 3, torch.float32)
        ref_data = prepare(torch, raw, (False,) * 3, torch.float64)
        with torch.no_grad():
            ref = operator(torch, reference, ref_data, scale, padding, 1e-5)
            python_output = operator(torch, reference, data, scale, padding, 1e-5)
            python_snapshot = {"output": tensor_record(python_output, ref)}
            del python_output
        for mode in ("no_grad", "frozen_gradmode"):
            for cache in ("cold", "warm"):
                name = f"inference/s{scale}_{'x'.join(map(str, shape))}_pad{padding}/{mode}/{cache}"
                context = torch.no_grad if mode == "no_grad" else torch.enable_grad
                def call():
                    with context():
                        return operator(torch, torch.ops.converse2d.forward, data, scale, padding, 1e-5)
                torch.ops.converse2d.clear_cache()
                output = call()
                if cache == "warm":
                    output = call()
                snapshot = {"output": tensor_record(output, ref)}
                del output
                cases[name] = {
                    "scope": "shared s1/nearest s>1 prior; complete padded operator; frozen FP32 inputs",
                    "input_shape": shape, "fft_shape": [(shape[-2] + 2 * padding) * scale, (shape[-1] + 2 * padding) * scale],
                    "fixture": records(dict(zip(("x", "weight", "bias"), data))),
                    "snapshot": snapshot, "python_fp32": python_snapshot,
                    "timing": measure(torch, call, args, before_call=torch.ops.converse2d.clear_cache if cache == "cold" else None),
                }
                print(name, cases[name]["timing"]["median"], flush=True)
        del ref, ref_data, data
        torch.ops.converse2d.clear_cache()

    masks = {"all": (True, True, True), "input_only": (True, False, False),
             "kernel_only": (False, True, False), "bias_only": (False, False, True)}
    for scale in (1, 2, 3):
        shape = (2, 32, 32, 40)
        padding = 2 if scale == 1 else 0
        kernel = 3 if scale == 1 else 7
        raw = make_fixture(torch, shape, scale, kernel, args.seed + 10 + scale)
        for mask_name, needs in masks.items():
            name = f"training/s{scale}_{'x'.join(map(str, shape))}_k{kernel}_pad{padding}/{mask_name}"
            data = prepare(torch, raw, needs, torch.float32)
            ref_data = prepare(torch, raw, needs, torch.float64)
            upstream = raw[3].cuda()
            expected = forward_vjp(torch, reference, ref_data, upstream.double(), scale, padding, 1e-5)
            python32 = forward_vjp(torch, reference, data, upstream, scale, padding, 1e-5)
            def call():
                return forward_vjp(torch, torch.ops.converse2d.forward, data, upstream, scale, padding, 1e-5)
            actual = call()
            snapshot = records(actual, expected)
            python_snapshot = records(python32, expected)
            del expected, actual, python32, ref_data
            cases[name] = {
                "scope": "full forward and selected VJPs; nearest prior chain included; per-call differentiable kernel FFT",
                "input_shape": shape, "fft_shape": [(shape[-2] + 2 * padding) * scale, (shape[-1] + 2 * padding) * scale],
                "needs_grad": dict(zip(("x", "weight", "bias"), needs)),
                "fixture": records(dict(zip(("x", "weight", "bias", "upstream"), (*data, upstream)))),
                "snapshot": snapshot, "python_fp32": python_snapshot,
                "timing": measure(torch, call, args),
            }
            print(name, cases[name]["timing"]["median"], flush=True)
            del data, upstream
    torch.ops.converse2d.clear_cache()
    return cases


def run_model(torch, args):
    from models.converse_usrnet import ConverseUSRNet
    checkpoint = args.root / "model_zoo/converse_usrnet.pth"
    model = ConverseUSRNet(backend="cuda").cuda()
    initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(initial, strict=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    x = torch.rand((1, 3, 8, 10), generator=generator).cuda()
    kernel = torch.rand((1, 1, 7, 7), generator=generator)
    kernel = (kernel / kernel.sum()).cuda()
    target = torch.rand((1, 3, 16, 20), generator=generator).cuda()
    model.eval()
    def inference():
        with torch.no_grad():
            return model(x, kernel, 2)
    infer_snapshot = records({"output": inference()})
    inference_timing = measure(torch, inference, args)
    torch.ops.converse2d.clear_cache()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)
    def step(snapshot=False):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, kernel, 2)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        result = {"output": tensor_record(output), "loss": tensor_record(loss)} if snapshot else None
        if snapshot:
            result.update({f"grad/{name}": tensor_record(parameter.grad)
                           for name, parameter in model.named_parameters() if parameter.grad is not None})
        optimizer.step()
        if snapshot:
            result.update({f"parameter/{name}": tensor_record(parameter) for name, parameter in model.named_parameters()})
        return result
    train_snapshot = step(snapshot=True)
    # Restore the same checkpoint/empty Adam state for every formal round.
    # This keeps parameter trajectories comparable without timed CPU transfers.
    def restore():
        model.load_state_dict(initial, strict=True)
        optimizer.state.clear()
        # Allocate and initialize Adam state through one identical complete step.
        step()
        torch.cuda.synchronize()
    training_timing = measure(torch, step, args, before_round=restore)
    return {
        "usrnet/inference": {"checkpoint_sha256": file_sha(checkpoint), "snapshot": infer_snapshot,
                             "timing": inference_timing, "scope": "complete 5 iterations/7 blocks; eager no_grad; warm fixed-kernel cache"},
        "usrnet/adam_training": {"checkpoint_sha256": file_sha(checkpoint), "snapshot": train_snapshot,
                                 "timing": training_timing, "scope": "zero_grad+forward+MSE+all backward+Adam; GPU resident synthetic input, no data/quality claim",
                                 "fixture": records({"x": x, "kernel": kernel, "target": target}),
                                 "fft_shapes": {"DataNet": [16, 20], "35_prior_calls": [20, 24]}},
    }


def noninferiority(case):
    """Every tensor must independently meet BOTH zero-margin FP64 error gates."""
    result = {}
    for name, actual in case.get("snapshot", {}).items():
        control = case.get("python_fp32", {}).get(name)
        if control is None or "fp64_error" not in actual or "fp64_error" not in control:
            continue  # Full-model records deliberately have no FP64 reference.
        errors, baseline = actual["fp64_error"], control["fp64_error"]
        metrics = {metric: (errors[metric] is not None and baseline[metric] is not None
                            and errors[metric] <= baseline[metric])
                   for metric in ("max_abs", "relative_l2")}
        finite = bool(actual["finite"] and control["finite"])
        result[name] = {"passed": finite and all(metrics.values()), "finite": finite,
                        "metric_passed": metrics, "candidate_error": errors,
                        "python_fp32_error": baseline}
    return result


def compare(paths):
    before, after = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    differences = []
    def walk(left, right, path):
        if left is None and right is None:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if "sha256" in left or "sha256" in right:
                if left.get("sha256") != right.get("sha256"):
                    differences.append({"tensor": path, "before": left, "after": right})
            else:
                for key in sorted(left.keys() | right.keys()):
                    walk(left.get(key), right.get(key), f"{path}/{key}")
        elif left is None or right is None:
            differences.append({"missing": path})
    common = before["cases"].keys() & after["cases"].keys()
    timings, numerical = {}, {}
    for name in sorted(common):
        for field in ("fixture", "snapshot", "python_fp32"):
            walk(before["cases"][name].get(field), after["cases"][name].get(field), f"{name}/{field}")
        b, a = [value["cases"][name]["timing"]["median"] for value in (before, after)]
        timings[name] = {"before": b, "after": a,
                         "wall_speedup": b["wall_ms"] / a["wall_ms"],
                         "cuda_speedup": b["cuda_event_ms"] / a["cuda_event_ms"]}
        checks = {label: noninferiority(value["cases"][name])
                  for label, value in (("before", before), ("after", after))}
        if any(checks.values()):
            numerical[name] = checks
    return {"kind": "p0_comparison", "files": [str(path) for path in paths],
            "settings_match": before["settings"] == after["settings"],
            "missing_cases": sorted(before["cases"].keys() ^ after["cases"].keys()),
            "tensor_hash_differences": differences, "timings": timings,
            "python_fp32_noninferiority": numerical,
            "noninferiority_failures": {
                label: [f"{name}/{tensor}" for name, checks in numerical.items()
                        for tensor, check in checks[label].items() if not check["passed"]]
                for label in ("before", "after")},
            "note": "Frozen GradMode inference may intentionally change to match no_grad; inspect each difference and FP64 error. Performance medians are not significance tests."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--build", action="store_true", help="Explicitly allow the checked loader to build; default verifies an existing binary")
    args = parser.parse_args()
    if args.compare:
        result = compare(args.compare)
    else:
        if args.warmup < 0 or args.rounds < 1 or args.iters < 1:
            parser.error("warmup must be nonnegative; rounds and iters must be positive")
        if os.environ.get("CONVERSE2D_BACKEND"):
            raise RuntimeError("Unset CONVERSE2D_BACKEND; this benchmark requires explicit checked CUDA routing")
        if os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
            raise RuntimeError("Unset CONVERSE2D_CPU_ONLY=1; this benchmark requires the checked CUDA build")
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
        torch.manual_seed(args.seed)
        loader_path = args.root / "test/extension_loader.py"
        spec = importlib.util.spec_from_file_location("p0_checked_loader", loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.load_extension()
        from models.converse_core import converse2d_reference
        manifest_path = args.root / ".build/cuda/source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes = loader.production_source_hashes()
        python_paths = ("models/converse_core.py", "models/util_converse.py", "models/converse_usrnet.py", "test/extension_loader.py")
        python_hashes = {path: file_sha(args.root / path) for path in python_paths}
        result = {
            "kind": "p0_benchmark", "root": str(args.root), "harness_sha256": file_sha(__file__),
            "settings": {key: getattr(args, key) for key in ("seed", "warmup", "rounds", "iters", "include_model")},
            "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(), "tf32": False, "amp": False,
                            "cudnn_deterministic": True, "cudnn_benchmark": False},
            "source_sha256": source_hashes, "python_sha256": python_hashes,
            "checked_build_manifest": manifest,
            "memory_scope": "Peak additional PyTorch allocated bytes relative to live tensors before each timed round/call; excludes non-PyTorch allocations and is not whole-process VRAM.",
            "cases": run_operators(torch, converse2d_reference, args),
        }
        if args.include_model:
            result["cases"].update(run_model(torch, args))
        if source_hashes != loader.production_source_hashes() or python_hashes != {p: file_sha(args.root / p) for p in python_paths}:
            raise RuntimeError("Source changed during benchmark; result is not eligible")
        if file_sha(args.root / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
            raise RuntimeError("Binary changed during benchmark; result is not eligible")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
