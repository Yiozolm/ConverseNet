"""One warmed full-spectrum s1 training call, as in the RTX 5060 Ti NCU capture.

Workload: x [B,128,100,100] used as both input and shared prior, weight
[1,128,3,3], bias zeros, scale 1, forward plus autograd.grad. The extension
must already be built by run.py (CONVERSE2D_SKIP_BUILD=1 enforces that).

--mode ncu     5 warmups, then one call between cudaProfilerStart/Stop.
--mode timing  warmups, then a torch.profiler trace over --iters calls.
"""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--mode', choices=('ncu', 'timing'), required=True)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--iters', type=int, default=20)
parser.add_argument('--trace', type=Path)
args = parser.parse_args()
sys.path[:0] = [str(args.root.resolve() / 'test'), str(args.root.resolve())]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import torch
from extension_loader import load_extension

load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(41001)
x = torch.randn(args.batch, 128, 100, 100, device='cuda', requires_grad=True)
weight = torch.rand(1, 128, 3, 3, device='cuda', requires_grad=True)
bias = torch.zeros(1, 128, 1, 1, device='cuda', requires_grad=True)
upstream = torch.randn_like(x)


def run():
    out = torch.ops.converse2d.forward(x, x, weight, bias, 1, 1e-5)
    return torch.autograd.grad(out, (x, weight, bias), upstream)


for _ in range(5):
    run()
torch.cuda.synchronize()
if args.mode == 'ncu':
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.trace))
    print(json.dumps(dict(trace=str(args.trace), iters=args.iters)))
