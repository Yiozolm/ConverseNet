"""Benchmark the current kernels with identical inputs and a float64 reference."""
import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension_loader import load_extension
from models.converse_core import converse2d_reference


def measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    batches = []
    wall = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        batches.append(start.elapsed_time(end)/iters)
        wall.append((time.perf_counter()-t0)*1000/iters)
    return {"gpu_ms_median": statistics.median(batches), "wall_ms_median": statistics.median(wall),
            "gpu_ms_batches": batches,
            "peak_extra_bytes": torch.cuda.max_memory_allocated() - baseline_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--single", action="store_true", help="Benchmark only the requested shape")
    parser.add_argument("--training", action="store_true", help="Time forward plus gradients for all four inputs")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--C", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", help="Result JSON path (default: artifacts/benchmark[_training].json)")
    args = parser.parse_args()
    if min(args.B,args.C,args.H,args.W,args.scale,args.iters) < 1:
        parser.error("shape, scale and iteration counts must be positive")
    if not torch.cuda.is_available():
        parser.error("GPU benchmarking requires CUDA-enabled PyTorch and a CUDA device")
    load_extension()
    torch.manual_seed(14)
    rows = []
    cases = [(args.B,args.C,args.H,args.W,args.scale)] if args.profile or args.single else [
        (1,32,128,128,s) for s in (1,2,3)] + [(1,64,256,256,s) for s in (1,2)]
    for b,c,h,w,s in cases:
        x = torch.randn(b,c,h,w,device="cuda")
        x0 = x if s == 1 else torch.nn.functional.interpolate(x,scale_factor=s,mode="nearest")
        weight = torch.randn(1,c,3,3,device="cuda").flatten(2).softmax(-1).reshape(1,c,3,3)
        bias = torch.zeros(1,c,1,1,device="cuda")
        if args.training:
            x0 = x0.detach().clone()
            for tensor in (x,x0,weight,bias):
                tensor.requires_grad_(True)
        data = (x,x0,weight,bias,s,1e-5)
        with torch.no_grad():
            reference = converse2d_reference(*(t.double() if isinstance(t,torch.Tensor) else t for t in data)).float()
        with (torch.enable_grad() if args.training else torch.no_grad()):
            forward = lambda: torch.ops.converse2d.forward(*data)
            output = forward().detach()
            max_error = (output-reference).abs().max().item()
            torch.testing.assert_close(output, reference, atol=1e-4, rtol=5e-5)
            fn = (lambda: torch.autograd.grad(forward().square().mean(), (x,x0,weight,bias))) if args.training else forward
            if args.profile:
                for _ in range(5): fn()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range(f"Converse2D_s{s}"):
                    for _ in range(args.iters): fn()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                print(json.dumps({"profile":"fused","max_abs_error":max_error}))
            else:
                row = dict(B=b,C=c,H=h,W=w,scale=s,max_abs_error=max_error,
                           **measure(fn,5,args.iters))
                rows.append(row)
                print(json.dumps(row),flush=True)
        torch.ops.converse2d.clear_cache()
    if not args.profile:
        default = "artifacts/benchmark_training.json" if args.training else "artifacts/benchmark.json"
        output = ROOT / (args.output or default)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
                                      "cuda":torch.version.cuda,"training":args.training,
                                      "iters_per_batch":args.iters,"results":rows},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
