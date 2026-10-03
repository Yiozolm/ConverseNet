"""One timing round for a checked build at --root; alternate roots across processes.

Records the s1 forward kernel median (torch.profiler) and the complete
forward+VJP call median (CUDA events) for the production B4/C128/100x100 shape.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--iters', type=int, default=50)
args = parser.parse_args()
sys.path[:0] = [str(args.root.resolve() / 'test'), str(args.root.resolve())]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import torch
from torch.profiler import ProfilerActivity, profile
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


for _ in range(10):
    run()
torch.cuda.synchronize()
start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
calls = []
for _ in range(args.iters):
    start.record()
    run()
    stop.record()
    stop.synchronize()
    calls.append(start.elapsed_time(stop))
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()
kernel = [e.device_time for e in prof.events() if 'scale1_forward' in e.name]
record = dict(label=args.label, root=str(args.root), batch=args.batch, iters=args.iters, gpu=torch.cuda.get_device_name(0),
              binary_sha256=json.loads((args.root / '.build/cuda/source_manifest.json').read_text())['binary_sha256'],
              call_median_ms=statistics.median(calls), forward_kernel_median_us=statistics.median(kernel),
              forward_kernel_name=next(e.name for e in prof.events() if 'scale1_forward' in e.name))
with args.output.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record))
