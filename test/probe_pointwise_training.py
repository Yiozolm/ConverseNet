"""Isolated FP32 1x1-convolution FWD + dx/dw/db candidate, not an end-to-end result.

    python test/probe_pointwise_training.py
    python test/probe_pointwise_training.py --output artifacts/training_research/pointwise_repeat.json

Real full-USRNet feature shapes: B4, H=W=96, C64->128, C128->64 and C64->3.
Weights/biases come from the supplied pretrained checkpoint; activations are
fixed CPU-generated normal samples with std=input_scale (default 1), not claimed
to be captured real activations. The fixed normal upstream has std equal to
upstream_scale/sqrt(output.numel()). Both methods and the FP64 F.conv2d reference
receive exactly the same FP32-rounded values before reference promotion.

Numerical gates are predeclared: atol=3e-5, rtol=3e-4 for output and all VJPs.
Report maximum absolute error, relative L2 and pointwise budget violations.
Cases failing either method's gate are not timed or called validated.

No optimizer, data loading, profiler, precision reduction or production edits.
Warmup=5; alternate four rounds of twenty iterations. One GPU fixture is resident
at a time, autograd.grad does not accumulate leaf .grad, and returned tensors are
discarded between iterations. Timings include eager forward and all three VJPs;
CUDA event spans and synchronized wall time are separate. Peak memory is total
PyTorch allocated/reserved for that fixture, excluding driver/library memory.
"""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 3e-5, 3e-4
CASES = (
    ("c64_to_128", 64, 128, "p.m_body.0.conv1.1"),
    ("c128_to_64", 128, 64, "p.m_body.0.conv1.5"),
    ("c64_to_3", 64, 3, "conv2"),
)


def conv2d(x, weight, bias):
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)


def matmul(x, weight, bias):
    # Keep NCHW storage; no NHWC materialization is hidden outside timing.
    output = weight[:, :, 0, 0].unsqueeze(0) @ x.flatten(2)
    output = output + bias.reshape(1, -1, 1)
    return output.reshape(x.shape[0], weight.shape[0], x.shape[2], x.shape[3])


METHODS = {"conv2d": conv2d, "matmul": matmul}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((list(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def fixture(cpu_tensors, dtype, method):
    import torch
    inputs = tuple(tensor.to(device="cuda", dtype=dtype).requires_grad_() for tensor in cpu_tensors[:3])
    upstream = cpu_tensors[3].to(device="cuda", dtype=dtype)

    def run():
        output = method(*inputs)
        gradients = torch.autograd.grad(output, inputs, upstream)
        return dict(output=output, dx=gradients[0], dw=gradients[1], db=gradients[2])

    return inputs, run


def capture(cpu_tensors, dtype, method):
    clear_cuda()
    inputs, run = fixture(cpu_tensors, dtype, method)
    values = run()
    if any(tensor.grad is not None for tensor in inputs):
        raise RuntimeError("autograd.grad unexpectedly accumulated leaf gradients")
    result = {name: value.detach().cpu().clone() for name, value in values.items()}
    del values, run, inputs
    clear_cuda()
    return result


def numerical_metrics(actual, reference):
    import torch
    if actual.shape != reference.shape:
        raise RuntimeError(f"Shape mismatch: {actual.shape} versus {reference.shape}")
    actual, reference = actual.double(), reference.double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return dict(passed=False, finite=False, shape=list(actual.shape), atol=ATOL, rtol=RTOL)
    error = (actual-reference).abs()
    budget = ATOL+RTOL*reference.abs()
    failed = int((error > budget).sum())
    return dict(passed=failed == 0, finite=True, shape=list(actual.shape), atol=ATOL, rtol=RTOL,
                max_abs=error.max().item(), relative_l2=(error.norm()/reference.norm().clamp_min(1e-30)).item(),
                max_pointwise_budget_ratio=(error/budget).max().item(), failed_elements=failed,
                elements=actual.numel())


def validate(cpu_tensors):
    import torch
    # Full double-precision convolution is the independent numerical reference.
    reference = capture(cpu_tensors, torch.float64, conv2d)
    report = {}
    for name, method in METHODS.items():
        actual = capture(cpu_tensors, torch.float32, method)
        if any(tensor.dtype != torch.float32 for tensor in actual.values()):
            raise RuntimeError(f"{name}: expected FP32 output and VJPs")
        metrics = {key: numerical_metrics(actual[key], reference[key]) for key in reference}
        report[name] = dict(passed=all(value["passed"] for value in metrics.values()), tensors=metrics)
    return report


def timed_fixture(cpu_tensors, method, args):
    import torch
    clear_cuda()
    inputs, run = fixture(cpu_tensors, torch.float32, method)
    for _ in range(args.warmup):
        run()  # Never retain a previous output or VJP during the next iteration.
    if any(tensor.grad is not None for tensor in inputs):
        raise RuntimeError("Warmup accumulated leaf gradients")
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    began = time.perf_counter()
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    torch.cuda.synchronize()
    result = dict(wall_ms=(time.perf_counter()-began)*1000/args.iters,
                  cuda_event_ms=start.elapsed_time(end)/args.iters,
                  initial_allocated_bytes=initial_allocated, initial_reserved_bytes=initial_reserved,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if any(tensor.grad is not None for tensor in inputs):
        raise RuntimeError("Timed VJPs accumulated leaf gradients")
    check = run()
    if not all(torch.isfinite(value).all().item() for value in check.values()):
        raise RuntimeError("Nonfinite post-timing output/VJP")
    result["leaf_grad_accumulated"] = False
    del check, run, inputs
    clear_cuda()
    return result


def benchmark(cpu_tensors, args):
    rounds = []
    for index in range(args.rounds):
        order = ("conv2d", "matmul") if index % 2 == 0 else ("matmul", "conv2d")
        values = {name: timed_fixture(cpu_tensors, METHODS[name], args) for name in order}
        rounds.append(dict(round=index+1, order=list(order), methods=values))
    medians = {name: {key: statistics.median(row["methods"][name][key] for row in rounds)
                      for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in METHODS}
    ratios = {key: [row["methods"]["conv2d"][key]/row["methods"]["matmul"][key] for row in rounds]
              for key in ("wall_ms", "cuda_event_ms")}
    return dict(rounds=rounds, medians=medians, paired_conv2d_over_matmul=ratios,
                median_paired_ratio={key: statistics.median(values) for key, values in ratios.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"model_zoo/converse_usrnet.pth")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/training_research/pointwise_probe.json")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--upstream-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("warmup, rounds and iters must be positive")
    if not all(math.isfinite(value) and value > 0 for value in (args.input_scale, args.upstream_scale)):
        parser.error("Input/upstream scales must be finite and positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite an existing probe report: {args.output}")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 conflicts with this FP32 protocol")
    # Configure deterministic cuBLAS before the first CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        parser.error("Use deterministic CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU performance fallback is reported")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cases = list(CASES)
    report = dict(status="running", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope=__doc__, settings={key: str(value.resolve()) if isinstance(value, Path) else value
                                          for key, value in vars(args).items()},
                  checkpoint_sha256=file_hash(args.checkpoint), script_sha256=file_hash(Path(__file__).resolve()),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(), tf32=False, cudnn_benchmark=False,
                                   cudnn_deterministic=True, deterministic_algorithms=True,
                                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
                  numerical_budget=dict(atol=ATOL, rtol=RTOL, reference="FP64 torch.nn.functional.conv2d"),
                  results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    save()
    try:
        for index, (name, cin, cout, prefix) in enumerate(cases):
            weight, bias = state[prefix+".weight"].clone(), state[prefix+".bias"].clone()
            if weight.dtype != torch.float32 or bias.dtype != torch.float32 or weight.shape != (cout, cin, 1, 1) or bias.shape != (cout,):
                raise RuntimeError(f"Unexpected pretrained FP32 pointwise tensors: {prefix}")
            generator = torch.Generator(device="cpu").manual_seed(args.seed+index)
            x = torch.randn(4, cin, 96, 96, generator=generator)*args.input_scale
            upstream_std = args.upstream_scale/math.sqrt(4*cout*96*96)
            upstream = torch.randn(4, cout, 96, 96, generator=generator)*upstream_std
            tensors = (x, weight, bias, upstream)
            row = dict(case=name, input_shape=list(x.shape), output_shape=list(upstream.shape),
                       checkpoint_prefix=prefix, input_std=args.input_scale, upstream_std=upstream_std,
                       input_weight_upstream_sha256=tensor_hash(tensors), validation=validate(tensors))
            if all(result["passed"] for result in row["validation"].values()):
                row["status"] = "numerically_validated_local_candidate"
                row["timing"] = benchmark(tensors, args)
            else:
                row["status"], row["timing"] = "numerical_gate_failed_no_timing", None
            report["results"].append(row)
            save()
            print(json.dumps(dict(case=name, status=row["status"], validation=row["validation"],
                                  medians=row["timing"]["medians"] if row["timing"] else None)), flush=True)
        report["status"] = ("complete" if all(row["timing"] is not None for row in report["results"])
                            else "numerical_gate_failed")
        report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save()
        raise
    print(f"Saved {args.output}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
