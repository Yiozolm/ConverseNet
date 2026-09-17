"""Isolated experiment: move nearest interpolation inside the C++ operator.

The original operator is copied into a private dispatcher namespace at build
time. No production source, installed extension, or model wrapper is changed.
Run in a CUDA/MSVC developer shell; see this directory's README.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.utils import cpp_extension


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "Converse2D" / "torch_converse2d"
BUILD = ROOT / ".build" / "nearest_cpp"
NAMESPACE = "converse2d_nearest_experiment"
OP = getattr(torch.ops, NAMESPACE)
TOLERANCES = {
    torch.float64: (1e-11, 1e-11),
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def load(*, cpu_only=False, verbose=True):
    """Build a separate library containing an unchanged, renamed baseline."""
    BUILD.mkdir(parents=True, exist_ok=True)
    original = (SOURCE / "converse2d.cpp").read_text(encoding="utf-8")
    replacements = {
        "TORCH_LIBRARY(converse2d, m)": f"TORCH_LIBRARY({NAMESPACE}, m)",
        "TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m)":
            f"TORCH_LIBRARY_IMPL({NAMESPACE}, CompositeImplicitAutograd, m)",
    }
    private = original
    for before, after in replacements.items():
        if private.count(before) != 1:
            raise RuntimeError(f"Expected one production registration: {before}")
        private = private.replace(before, after)
    generated = BUILD / "baseline.cpp"
    # Avoid an unnecessary rebuild when this generated include is unchanged.
    if not generated.exists() or generated.read_text(encoding="utf-8") != private:
        generated.write_text(private, encoding="utf-8")
    if os.name == "nt":
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
        os.environ.setdefault("VSLANG", "1033")
    cuda = not cpu_only
    if cuda and (not torch.cuda.is_available() or cpp_extension.CUDA_HOME is None):
        raise RuntimeError("CUDA and its toolkit are required; use --cpu for validation only")
    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    # A command-line dependency is reliable even when localized MSVC
    # /showIncludes output prevents Ninja from discovering baseline.cpp.
    baseline_hash = hashlib.sha256(private.encode("utf-8")).hexdigest()
    flags.append(f"-DNEAREST_CPP_BASELINE_SHA256_{baseline_hash}=1")
    cuda_flags = ["-O3", "-lineinfo"]
    if os.name == "nt":
        cuda_flags.extend(["-Xcompiler", "/Zc:preprocessor"])
    sources = [str(HERE / "bindings.cpp")]
    if cuda:
        flags.append("-DCONVERSE2D_WITH_CUDA=1")
        sources.append(str(HERE / "kernels.cu"))
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    cpp_extension.load(
        name="converse2d_nearest_experiment_ext",
        sources=sources,
        extra_include_paths=[str(BUILD)],
        extra_cflags=flags,
        extra_cuda_cflags=cuda_flags,
        with_cuda=cuda,
        is_python_module=False,
        build_directory=str(BUILD),
        verbose=verbose,
    )


def baseline(x, weight, bias, scale, eps=1e-5, variant="v7"):
    prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
    return OP.forward(x, prior, weight, bias, scale, eps, variant)


def inside_cpp(x, weight, bias, scale, eps=1e-5, variant="v7"):
    return OP.forward_nearest(x, weight, bias, scale, eps, variant)


def data(device, dtype, *, batch=2, channels=3, height=5, width=7,
         kernel_batches=1, kernel_channels=3, noncontiguous=False):
    x = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    weight = torch.randn(kernel_batches, kernel_channels, 3, 3,
                         device=device, dtype=dtype) / 9
    bias = torch.zeros(1, channels, 1, 1, device=device, dtype=dtype)
    if noncontiguous:
        x = x.transpose(-2, -1)
        weight = weight.transpose(-2, -1)
    return x, weight, bias


def compare(actual, expected, dtype):
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return (actual.double() - expected.double()).abs().max().item()


def validate_case(args, scale, *, variant="v7"):
    dtype = args[0].dtype
    args = tuple(t.detach().requires_grad_() for t in args)
    old = baseline(*args, scale, variant=variant)
    new = inside_cpp(*args, scale, variant=variant)
    assert new.dtype == dtype
    forward_error = compare(new, old, dtype)
    upstream = torch.randn_like(old) / old.numel()
    old_grads = torch.autograd.grad(old, args, upstream)
    new_grads = torch.autograd.grad(new, args, upstream)
    gradient_errors = [compare(a, b, dtype) for a, b in zip(new_grads, old_grads)]
    with torch.no_grad():
        inference_error = compare(inside_cpp(*args, scale, variant=variant),
                                  baseline(*args, scale, variant=variant), dtype)
    return {
        "shape": list(args[0].shape),
        "kernel_broadcast": list(args[1].shape[:2]),
        "dtype": str(dtype), "scale": scale, "variant": variant,
        "noncontiguous": not args[0].is_contiguous(),
        "max_abs_forward_error": forward_error,
        "max_abs_gradient_errors": dict(zip(("x", "weight", "bias"), gradient_errors)),
        "max_abs_inference_error": inference_error,
    }


def validate(device):
    """Check routing parity, gradients, higher derivatives, and rejected input."""
    rows = []
    torch.manual_seed(934)
    for dtype in TOLERANCES:
        for scale in (1, 2, 3):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                for noncontiguous in (False, True):
                    args = data(device, dtype, kernel_batches=kb, kernel_channels=kc,
                                noncontiguous=noncontiguous)
                    rows.append(validate_case(args, scale))
    # v7 has the full matrix above; the five legacy labels retain their dispatch.
    for variant in ("v2", "v3", "v4", "v5", "v6"):
        for scale in (1, 2, 3):
            rows.append(validate_case(data(device, torch.float64), scale, variant=variant))
    tiny = tuple(t.requires_grad_() for t in data(
        device, torch.float64, batch=1, channels=1, height=3, width=4,
        kernel_channels=1))
    fn = lambda *tensors: inside_cpp(*tensors, 2, eps=1e-3)
    gradcheck = torch.autograd.gradcheck(fn, tiny, fast_mode=True)
    gradgradcheck = torch.autograd.gradgradcheck(fn, tiny, fast_mode=True)
    x, weight, bias = data(device, torch.float32)
    invalid = {
        "zero_scale": lambda: inside_cpp(x, weight, bias, 0),
        "negative_scale": lambda: inside_cpp(x, weight, bias, -1),
        "output_size_overflow": lambda: inside_cpp(x, weight, bias, 2**62),
        "zero_eps": lambda: inside_cpp(x, weight, bias, 2, eps=0),
        "nan_eps": lambda: inside_cpp(x, weight, bias, 2, eps=float("nan")),
        "infinite_eps": lambda: inside_cpp(x, weight, bias, 2, eps=float("inf")),
        "rank": lambda: inside_cpp(x[0], weight, bias, 2),
        "empty_input": lambda: inside_cpp(x[:0], weight, bias, 2),
        "integer_input": lambda: inside_cpp(x.int(), weight.int(), bias.int(), 2),
        "mixed_dtype": lambda: inside_cpp(x, weight.double(), bias, 2),
        "bias_shape": lambda: inside_cpp(x, weight, bias[:, :2], 2),
        "kernel_shape": lambda: inside_cpp(x, weight.expand(4, -1, -1, -1), bias, 2),
        "variant": lambda: inside_cpp(x, weight, bias, 2, variant="invalid"),
    }
    rejected = {}
    for name, invalid_call in invalid.items():
        try:
            invalid_call()
        except (RuntimeError, ValueError) as error:
            rejected[name] = str(error).splitlines()[0]
        else:
            raise AssertionError(f"Invalid input was accepted: {name}")
    OP.clear_cache()
    return {
        "passed": True, "case_count": len(rows), "cases": rows,
        "gradcheck": bool(gradcheck), "gradgradcheck": bool(gradgradcheck),
        "rejected_inputs": rejected,
        "note": "Parity against the unchanged operator with explicit Python nearest interpolation",
    }


def timed(fn, iterations):
    """Measure the same loop using GPU events and a synchronized wall clock."""
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000 / iterations
    return {"cuda_event_ms": start.elapsed_time(end) / iterations,
            "synchronized_wall_ms": wall_ms}


def peak_extra_bytes(fn):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    del result
    return extra


@torch.no_grad()
def benchmark(iterations, rounds, warmup):
    rows = []
    cases = [
        (1, 32, 32, 40, 2),
        (1, 64, 128, 128, 2),
        (1, 32, 128, 128, 3),
        (8, 32, 128, 128, 2),
        (1, 64, 128, 128, 1),
    ]
    torch.manual_seed(935)
    for batch, channels, height, width, scale in cases:
        OP.clear_cache()
        x, weight, bias = data("cuda", torch.float32, batch=batch, channels=channels,
                               height=height, width=width, kernel_channels=channels)
        weight = weight.flatten(2).softmax(-1).reshape_as(weight)
        fns = {
            "python_interpolate": lambda: baseline(x, weight, bias, scale),
            "cpp_interpolate": lambda: inside_cpp(x, weight, bias, scale),
        }
        for _ in range(warmup):
            for fn in fns.values():
                fn()
        old, new = (fn() for fn in fns.values())
        error = compare(new, old, x.dtype)
        del old, new
        row = {"shape": [batch, channels, height, width], "scale": scale,
               "dtype": str(x.dtype), "variant": "v7", "max_abs_error": error,
               "peak_extra_bytes": {name: peak_extra_bytes(fn) for name, fn in fns.items()},
               "rounds": {name: [] for name in fns}}
        for repetition in range(rounds):
            order = list(fns) if repetition % 2 == 0 else list(reversed(fns))
            for name in order:
                row["rounds"][name].append(timed(fns[name], iterations))
        row["median_ms"] = {
            name: {metric: statistics.median(sample[metric] for sample in samples)
                   for metric in ("cuda_event_ms", "synchronized_wall_ms")}
            for name, samples in row["rounds"].items()
        }
        row["latency_reduction_pct"] = {
            metric: 100 * (1 - row["median_ms"]["cpp_interpolate"][metric]
                          / row["median_ms"]["python_interpolate"][metric])
            for metric in ("cuda_event_ms", "synchronized_wall_ms")
        }
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


def environment(device):
    info = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_home": cpp_extension.CUDA_HOME, "device": device,
        "git_revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, capability=list(torch.cuda.get_device_capability()),
                    total_memory=props.total_memory, multiprocessors=props.multi_processor_count,
                    gpu_driver=command_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                               "--format=csv,noheader"]))
        if cpp_extension.CUDA_HOME:
            info["nvcc"] = command_output([str(Path(cpp_extension.CUDA_HOME) / "bin" / "nvcc"), "--version"])
    return info


def source_hashes():
    files = [SOURCE / "converse2d.cpp", SOURCE / "converse2d_kernels.cu",
             HERE / "bindings.cpp", HERE / "kernels.cu", Path(__file__), BUILD / "baseline.cpp"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def save(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="CPU validation only; no CUDA benchmark")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--quiet-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "nearest_cpp" / "results.json")
    args = parser.parse_args()
    if min(args.iters, args.rounds, args.warmup) < 1:
        parser.error("--iters, --rounds, and --warmup must be positive")
    device = "cpu" if args.cpu else "cuda"
    result = {
        "experiment": "nearest interpolation inside C++ operator boundary",
        "namespace": NAMESPACE,
        "scope": "isolated experiment; x0 allocation and its FFT remain unchanged",
        "environment": environment(device),
        "settings": {"iterations": args.iters, "rounds": args.rounds,
                     "warmup": args.warmup, "device": device,
                     "ordering": "alternate baseline-first and C++-first rounds",
                     "benchmark_grad_mode": "no_grad", "weight_cache": "warm"},
    }
    try:
        load(cpu_only=args.cpu, verbose=not args.quiet_build)
        result["source_sha256"] = source_hashes()
        print("Validating forward, inference, gradients, and higher derivatives...", flush=True)
        result["validation"] = validate(device)
        save(args.output, result)
        print(f"Validation passed: {result['validation']['case_count']} parity cases, "
              "gradcheck, gradgradcheck, and invalid input checks", flush=True)
        if not args.cpu and not args.validation_only:
            result["benchmarks"] = benchmark(args.iters, args.rounds, args.warmup)
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
