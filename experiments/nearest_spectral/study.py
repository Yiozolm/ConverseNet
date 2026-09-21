"""Isolated nearest-prior spectral fusion experiment (CUDA inference only).

Builds a renamed production baseline and experimental kernels in a private
dispatcher namespace. It does not install an extension or modify model code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils import cpp_extension


ROOT = Path(__file__).resolve().parents[2]
import sys as _layout_sys
_layout_sys.path.insert(0,str(ROOT/"test"))
from extension_loader import legacy_source_texts, production_source_hashes

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "Converse2D" / "torch_converse2d"
BUILD = ROOT / ".build" / "nearest_spectral"
NAMESPACE = "converse2d_nearest_spectral_experiment"
OP = getattr(torch.ops, NAMESPACE)
DTYPES = (torch.float64, torch.float32, torch.float16, torch.bfloat16)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference


def load(*, verbose=True):
    """Compile without changing production registration or installed modules."""
    BUILD.mkdir(parents=True, exist_ok=True)
    private = legacy_source_texts()["converse2d.cpp"]
    for before, after in {
        "TORCH_LIBRARY(converse2d, m)": f"TORCH_LIBRARY({NAMESPACE}, m)",
        "TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m)":
            f"TORCH_LIBRARY_IMPL({NAMESPACE}, CompositeImplicitAutograd, m)",
    }.items():
        if private.count(before) != 1:
            raise RuntimeError(f"Expected one production registration: {before}")
        private = private.replace(before, after)
    generated = BUILD / "baseline.cpp"
    if not generated.exists() or generated.read_text(encoding="utf-8") != private:
        generated.write_text(private, encoding="utf-8")
    if not torch.cuda.is_available() or cpp_extension.CUDA_HOME is None:
        raise RuntimeError("This experiment requires CUDA and a CUDA toolkit")
    flags = ["/O2", "/std:c++17", "/Zc:preprocessor"] if os.name == "nt" else ["-O3", "-std=c++17"]
    if os.name == "nt":
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
        os.environ.setdefault("VSLANG", "1033")
    # MSVC's localized include output can hide dependencies from Ninja.
    digest = hashlib.sha256(private.encode("utf-8") +
                            json.dumps(production_source_hashes(),sort_keys=True).encode()).hexdigest()
    dependency_flag = f"-DNEAREST_SPECTRAL_BASELINE_SHA256_{digest}=1"
    flags.extend(["-DCONVERSE2D_WITH_CUDA=1", dependency_flag])
    cuda_flags = ["-O3", "-lineinfo", dependency_flag]
    if os.name == "nt":
        cuda_flags.extend(["-Xcompiler", "/Zc:preprocessor"])
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    cpp_extension.load(
        name="converse2d_nearest_spectral_experiment_ext",
        sources=[str(HERE / "bindings.cpp"), str(HERE / "kernels.cu"), str(SOURCE / "converse2d_training.cu")],
        extra_include_paths=[str(BUILD)], extra_cflags=flags,
        extra_cuda_cflags=cuda_flags, with_cuda=True, is_python_module=False,
        build_directory=str(BUILD), verbose=verbose,
    )


def baseline(x, weight, bias, scale, eps=1e-5):
    prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
    return OP.forward(x, prior, weight, bias, scale, eps, "v7")


def fused(x, weight, bias, scale, eps=1e-5):
    return OP.forward_nearest(x, weight, bias, scale, eps)


def reference(x, weight, bias, scale, eps=1e-5):
    x, weight, bias = (t.double() for t in (x, weight, bias))
    prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
    return converse2d_reference(x, prior, weight, bias, scale, eps)


def data(dtype, *, batch=2, channels=3, height=5, width=7,
         kernel_batches=1, kernel_channels=3, scale=2, noncontiguous=False):
    x = torch.randn(batch, channels, height, width, device="cuda", dtype=dtype)
    kh, kw = min(3, height * scale), min(3, width * scale)
    weight = torch.randn(kernel_batches, kernel_channels, kh, kw,
                         device="cuda", dtype=dtype) / (kh * kw)
    bias = torch.randn(1, channels, 1, 1, device="cuda", dtype=dtype)
    if noncontiguous:
        # Strided views preserve the requested shapes, including rectangles.
        x = torch.stack((x, x), dim=-1)[..., 0]
        weight = torch.stack((weight, weight), dim=-1)[..., 0]
        bias = torch.stack((bias, bias), dim=-1)[..., 0]
    return x, weight, bias


def tolerance(dtype):
    if dtype == torch.float64:
        return 1e-10, 1e-10
    if dtype == torch.float32:
        return 1e-4, 1e-4
    # Output rounding is measured independently below. Two representable steps
    # allow either side of a rounding boundary; arithmetic still uses float32.
    return 2 * torch.finfo(dtype).eps, 1e-4


def errors(actual, expected, *, dtype=None):
    a, e = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    delta = a - e
    if finite:
        max_abs = delta.abs().max().item()
        # Scaling first is essential: norm(x) squares subnormal double values
        # and can incorrectly report both numerator and denominator as zero.
        magnitude = e.abs().max().item()
        if magnitude:
            # CUDA scalar division may use a reciprocal. 1 / 1e-310 overflows
            # even though each quotient is ordinary, so rescale in two stages.
            factor = 1.0 / max(magnitude, 1e-300)
            scaled_magnitude = magnitude * factor
            scaled_a = (a * factor) / scaled_magnitude
            scaled_e = (e * factor) / scaled_magnitude
            numerator = torch.linalg.vector_norm(scaled_a - scaled_e).item()
            denominator = torch.linalg.vector_norm(scaled_e).item()
            relative_l2 = numerator / denominator
            if not math.isfinite(relative_l2):
                relative_l2 = None
        else:
            relative_l2 = 0.0 if max_abs == 0 else None
    else:
        max_abs, relative_l2 = None, None
    result = {"finite": finite, "max_abs": max_abs, "relative_l2": relative_l2}
    if dtype is not None:
        rtol, atol = tolerance(dtype)
        close = torch.isclose(a, e, rtol=rtol, atol=atol)
        result.update(passed=finite and bool(close.all()), rtol=rtol, atol=atol,
                      mismatched_elements=int((~close).sum().item()))
    return result


def validate_case(tensors, scale, *, name="matrix", eps=1e-5):
    x, weight, bias = tensors
    expected = reference(*tensors, scale, eps)
    old, new = baseline(*tensors, scale, eps), fused(*tensors, scale, eps)
    old_error = errors(old, expected, dtype=x.dtype)
    new_error = errors(new, expected, dtype=x.dtype)
    row = {
        "name": name, "shape": list(x.shape), "kernel_shape": list(weight.shape),
        "dtype": str(x.dtype), "scale": scale, "eps": eps,
        "noncontiguous": not all(t.is_contiguous() for t in tensors),
        "baseline_vs_double_reference": old_error,
        "fused_vs_double_reference": new_error,
        "fused_vs_baseline": errors(new, old, dtype=x.dtype),
        "reference_output_quantization": errors(expected.to(x.dtype), expected),
    }
    row["passed"] = (old_error["passed"] and new_error["passed"]
                     and new.shape == expected.shape and new.dtype == x.dtype)
    return row


@torch.no_grad()
def validate_underflow():
    """Characterize subnormal regularizers separately, without relaxed tests."""
    rows = []
    for scale in (2, 3):
        for eps, amplitude in ((1e-38, 1e-18), (1e-42, 1e-20), (1e-45, 1e-23)):
            x, weight, bias = data(torch.float32, batch=1, channels=1,
                                   kernel_channels=1, scale=scale)
            x.mul_(1e-20)
            weight.mul_(amplitude)
            bias.fill_(-100)
            for cache_mode in ("warm", "inference_tensor_bypass"):
                if cache_mode != "warm":
                    with torch.inference_mode():
                        weight = weight.clone()
                row = validate_case((x, weight, bias), scale, name="underflow", eps=eps)
                row["weight_amplitude"] = amplitude
                row["weight_cache"] = cache_mode
                rows.append(row)
    # Identity PSFs isolate subnormal input behavior from ill-conditioning.
    # Tiny input errors are judged relatively so the ordinary absolute floor
    # cannot make a flush-to-zero result appear correct.
    for dtype, amplitudes in ((torch.float32, (1e-38, 1e-40)),
                              (torch.float64, (1e-310,))):
        for amplitude in amplitudes:
            for scale in (2, 3, 5):
                x, weight, bias = data(dtype, batch=1, channels=1, kernel_channels=1,
                                       height=5, width=7, scale=scale)
                x.mul_(amplitude)
                weight.zero_()
                weight[..., 1, 1] = 1
                row = validate_case((x, weight, bias), scale, name="subnormal_input")
                row["input_amplitude"] = amplitude
                rows.append(row)
    for row in rows:
        limit = 1e-10 if row["dtype"] == str(torch.float64) else 1e-4
        for key in ("baseline_vs_double_reference", "fused_vs_double_reference"):
            metric = row[key]
            metric["strict_relative_l2_limit"] = limit
            metric["strict_relative_passed"] = (metric["finite"] and
                metric["relative_l2"] is not None and metric["relative_l2"] <= limit)
        row["strict_relative_passed"] = all(row[key]["strict_relative_passed"]
            for key in ("baseline_vs_double_reference", "fused_vs_double_reference"))
    return {"cases": rows, "strict_tolerance_all_passed": all(r["passed"] for r in rows),
            "passed": all(r["strict_relative_passed"] for r in rows),
            "note": "Stress is reported independently and contributes to overall_accuracy_passed. Relative L2 limits remain float32=1e-4, float64=1e-10 with no absolute-error exemption; nonfinite values are explicit."}


@torch.no_grad()
def validate_graph():
    # Inference tensors deliberately bypass the host weight cache. Warming
    # this exact path also creates cuFFT plans before capture.
    x, weight, bias = data(torch.float32, batch=1, channels=2, kernel_channels=2)
    with torch.inference_mode():
        weight = weight.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused(x, weight, bias, 3)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fused(x, weight, bias, 3)
    rows = []
    for index in range(3):
        x.add_(0.125)
        graph.replay()
        torch.cuda.synchronize()
        rows.append({"replay": index, **errors(output, reference(x, weight, bias, 3), dtype=x.dtype)})
    return {"passed": all(row["passed"] for row in rows), "replays": rows,
            "weight_cache": "inference_tensor_bypass"}


@torch.no_grad()
def validate():
    torch.manual_seed(1934)
    rows = []
    shapes = ((1, 1), (1, 5), (5, 1), (5, 7), (6, 8), (5, 8))
    for dtype in DTYPES:
        for scale in (1, 2, 3, 4, 5):
            for shape_index, (height, width) in enumerate(shapes):
                for broadcast_index, (kb, kc) in enumerate(((1, 1), (1, 3), (2, 1), (2, 3))):
                    tensors = data(dtype, height=height, width=width, scale=scale,
                                   kernel_batches=kb, kernel_channels=kc,
                                   noncontiguous=(shape_index + broadcast_index) % 2 == 1)
                    rows.append(validate_case(tensors, scale))
    # Both implementations use the same production cache helper. Verify that
    # source tensor versions invalidate entries, including promoted low types.
    cache_rows = []
    for dtype in DTYPES:
        tensors = data(dtype, scale=3)
        cache_rows.append(validate_case(tensors, 3, name="cache_initial"))
        cache_rows.append(validate_case(tensors, 3, name="cache_reuse"))
        tensors[1].mul_(0.375)
        cache_rows.append(validate_case(tensors, 3, name="cache_weight_mutation"))
        tensors[2].add_(0.5)
        cache_rows.append(validate_case(tensors, 3, name="bias_mutation"))
        with torch.inference_mode():
            dynamic = tensors[1].clone()
            for index in range(3):
                dynamic.add_(0.025)
                cache_rows.append(validate_case((tensors[0], dynamic, tensors[2]), 3,
                                                name=f"inference_weight_mutation_{index}"))
    x, weight, bias = data(torch.float32)
    invalid = {
        "zero_scale": lambda: fused(x, weight, bias, 0),
        "negative_scale": lambda: fused(x, weight, bias, -1),
        "output_size_overflow": lambda: fused(x, weight, bias, 2**62),
        "zero_eps": lambda: fused(x, weight, bias, 2, 0),
        "nan_eps": lambda: fused(x, weight, bias, 2, float("nan")),
        "infinite_eps": lambda: fused(x, weight, bias, 2, float("inf")),
        "rank": lambda: fused(x[0], weight, bias, 2),
        "empty_input": lambda: fused(x[:0], weight, bias, 2),
        "integer_input": lambda: fused(x.int(), weight.int(), bias.int(), 2),
        "mixed_dtype": lambda: fused(x, weight.double(), bias, 2),
        "bias_shape": lambda: fused(x, weight, bias[:, :2], 2),
        "kernel_shape": lambda: fused(x, weight.expand(4, -1, -1, -1), bias, 2),
        "cpu": lambda: fused(x.cpu(), weight.cpu(), bias.cpu(), 2),
    }
    rejected = {}
    for name, invalid_call in invalid.items():
        try:
            invalid_call()
        except (RuntimeError, ValueError) as error:
            rejected[name] = str(error).splitlines()[0]
        else:
            raise AssertionError(f"Invalid input was accepted: {name}")
    for requires_grad in (False, True):
        with torch.enable_grad():
            try:
                fused(x.detach().requires_grad_(requires_grad), weight, bias, 2)
            except (RuntimeError, ValueError) as error:
                rejected[f"grad_enabled_requires_grad_{requires_grad}"] = str(error).splitlines()[0]
            else:
                raise AssertionError("Inference-only API accepted grad-enabled execution")
    graph = validate_graph()
    stress = validate_underflow()
    OP.clear_cache()
    return {
        "passed": all(row["passed"] for row in rows + cache_rows) and graph["passed"],
        "case_count": len(rows), "cases": rows, "cache_cases": cache_rows,
        "cuda_graph": graph, "underflow_stress": stress, "rejected_inputs": rejected,
        "reference": "models.converse_core.converse2d_reference, float64 full spectrum FFT",
        "low_precision_tolerance": "rtol=2*dtype.eps, atol=1e-4; reference rounding error recorded separately",
        "gradcheck": "not applicable: the isolated interface explicitly rejects enabled autograd",
    }


def timed(fn, iterations):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return {"cuda_event_ms": start.elapsed_time(end) / iterations,
            "synchronized_wall_ms": (time.perf_counter() - wall_start) * 1000 / iterations}


def peak_extra_bytes(fn):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    del output
    return extra


@torch.no_grad()
def profile_operations():
    x, weight, bias = data(torch.float32, batch=1, channels=32, height=32,
                           width=40, kernel_channels=32)
    fns = {"baseline": lambda: baseline(x, weight, bias, 2),
           "fused": lambda: fused(x, weight, bias, 2)}
    for fn in fns.values():
        for _ in range(3):
            fn()
    result = {}
    for name, fn in fns.items():
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            fn()
            torch.cuda.synchronize()
        counts = {event.key: event.count for event in prof.key_averages()}
        result[name] = {key: counts.get(key, 0)
                        for key in ("aten::upsample_nearest2d", "aten::fft_rfft2", "aten::fft_irfft2")}
    expected = {"baseline": {"aten::upsample_nearest2d": 1, "aten::fft_rfft2": 2},
                "fused": {"aten::upsample_nearest2d": 0, "aten::fft_rfft2": 1}}
    result["passed"] = all(result[name][key] == count for name, row in expected.items()
                           for key, count in row.items())
    result["scope"] = "one warm-cache forward; CPU operator events, scale=2"
    OP.clear_cache()
    return result


@torch.no_grad()
def benchmark(iterations, rounds, warmup):
    cases = [
        (1, 32, 32, 40, 2, "warm"),
        (1, 64, 128, 128, 2, "warm"),
        (1, 32, 128, 128, 3, "warm"),
        (8, 32, 128, 128, 2, "warm"),
        (1, 64, 128, 128, 1, "warm"),
        (1, 32, 127, 129, 3, "warm"),
        (1, 32, 96, 128, 4, "warm"),
        (1, 64, 128, 128, 2, "inference_tensor_bypass"),
    ]
    torch.manual_seed(1935)
    rows = []
    for batch, channels, height, width, scale, cache_mode in cases:
        OP.clear_cache()
        x, weight, bias = data(torch.float32, batch=batch, channels=channels,
                               height=height, width=width, kernel_channels=channels, scale=scale)
        weight = weight.flatten(2).softmax(-1).reshape_as(weight)
        if cache_mode != "warm":
            with torch.inference_mode():
                weight = weight.clone()
        fns = {"baseline": lambda: baseline(x, weight, bias, scale),
               "fused": lambda: fused(x, weight, bias, scale)}
        for _ in range(warmup):
            for fn in fns.values():
                fn()
        old, new = (fn() for fn in fns.values())
        parity = errors(new, old, dtype=x.dtype)
        if not parity["passed"]:
            raise AssertionError(f"Benchmark parity failed: shape={list(x.shape)}, scale={scale}, errors={parity}")
        del old, new
        row = {"shape": list(x.shape), "scale": scale, "dtype": str(x.dtype),
               "weight_cache": cache_mode, "fused_vs_baseline": parity,
               "peak_extra_bytes": {name: peak_extra_bytes(fn) for name, fn in fns.items()},
               "rounds": {name: [] for name in fns}}
        for repetition in range(rounds):
            order = list(fns) if repetition % 2 == 0 else list(reversed(fns))
            for name in order:
                row["rounds"][name].append(timed(fns[name], iterations))
        metrics = ("cuda_event_ms", "synchronized_wall_ms")
        row["median_ms"] = {name: {metric: statistics.median(sample[metric] for sample in samples)
                                    for metric in metrics}
                            for name, samples in row["rounds"].items()}
        row["latency_reduction_pct"] = {
            metric: 100 * (1 - row["median_ms"]["fused"][metric] / row["median_ms"]["baseline"][metric])
            for metric in metrics}
        row["peak_extra_reduction_bytes"] = row["peak_extra_bytes"]["baseline"] - row["peak_extra_bytes"]["fused"]
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "rounds"}), flush=True)
    OP.clear_cache()
    return rows


def command_output(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15,
                                   encoding="utf-8", errors="replace")
        return (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def environment():
    props = torch.cuda.get_device_properties(0)
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_home": cpp_extension.CUDA_HOME, "device": "cuda", "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()), "total_memory": props.total_memory,
        "multiprocessors": props.multi_processor_count,
        "git_revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "gpu_driver": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "nvcc": command_output([str(Path(cpp_extension.CUDA_HOME) / "bin" / "nvcc"), "--version"])
        if cpp_extension.CUDA_HOME else None,
    }


def source_hashes():
    files = [SOURCE / "converse2d.cpp", SOURCE / "converse2d_kernels.cu",
             ROOT / "models" / "converse_core.py", HERE / "bindings.cpp", HERE / "kernels.cu",
             Path(__file__), BUILD / "baseline.cpp"]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def save(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validation-only", action="store_true")
    mode.add_argument("--benchmark-only", action="store_true")
    mode.add_argument("--profile-only", action="store_true")
    parser.add_argument("--quiet-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "nearest_spectral" / "results.json")
    args = parser.parse_args()
    if min(args.iters, args.rounds, args.warmup) < 1:
        parser.error("--iters, --rounds, and --warmup must be positive")
    result = {
        "experiment": "nearest-prior frequency-domain fusion", "namespace": NAMESPACE,
        "scope": "isolated CUDA inference; removes the enlarged prior and its FFT for scale > 1",
        "settings": {"iterations": args.iters, "rounds": args.rounds, "warmup": args.warmup,
                     "ordering": "alternate baseline-first and fused-first rounds",
                     "benchmark_grad_mode": "no_grad", "scale_one": "unchanged production path",
                     "validation_only": args.validation_only, "benchmark_only": args.benchmark_only,
                     "profile_only": args.profile_only},
    }
    try:
        load(verbose=not args.quiet_build)
        result["environment"] = environment()
        result["source_sha256"] = {**production_source_hashes(), **source_hashes()}
        if not args.benchmark_only and not args.profile_only:
            print("Validating against independent float64 full-spectrum reference...", flush=True)
            result["validation"] = validate()
            save(args.output, result)
            if not result["validation"]["passed"]:
                failed = [row for row in result["validation"]["cases"] + result["validation"]["cache_cases"] if not row["passed"]]
                raise AssertionError(f"Validation failed; first case: {failed[:1]}")
            print(f"Validation passed: {result['validation']['case_count']} matrix cases, cache and graph checks", flush=True)
        result["operation_profile"] = profile_operations()
        save(args.output, result)
        if not result["operation_profile"]["passed"]:
            raise AssertionError(f"Unexpected operation counts: {result['operation_profile']}")
        if not args.validation_only and not args.profile_only:
            result["benchmarks"] = benchmark(args.iters, args.rounds, args.warmup)
        if "validation" in result:
            result["stress_passed"] = result["validation"]["underflow_stress"]["passed"]
            result["overall_accuracy_passed"] = result["validation"]["passed"] and result["stress_passed"]
        # passed describes successful execution of the ordinary suite and
        # benchmark; stress limitations remain explicit in the fields above.
        result["passed"] = True
    except Exception as error:
        result["passed"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        save(args.output, result)
        raise
    save(args.output, result)
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
