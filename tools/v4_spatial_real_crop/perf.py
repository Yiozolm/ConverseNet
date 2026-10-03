"""One timing round of the public forward+VJP for a checked build at --root.

Alternate baseline/candidate roots in fresh processes. Records the complete
call median (CUDA events), and from one profiled window the per-call kernel
count, summed kernel time and the real-part VJP kernels by name.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--output', type=Path, required=True)
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
CONFIGS = {'s1_b4_c128_100': (4, 128, 100, 100, 1), 's2_b4_c64_64': (4, 64, 64, 64, 2), 's3_b2_c32_48': (2, 32, 48, 48, 3)}
results = {}
for name, (batch, channels, h, w, scale) in CONFIGS.items():
    torch.manual_seed(41001)
    x = torch.randn(batch, channels, h, w, device='cuda', requires_grad=True)
    x0 = x if scale == 1 else torch.randn(batch, channels, h * scale, w * scale, device='cuda', requires_grad=True)
    weight = torch.rand(1, channels, 3, 3, device='cuda', requires_grad=True)
    bias = torch.zeros(1, channels, 1, 1, device='cuda', requires_grad=True)
    inputs = (x, weight, bias) if scale == 1 else (x, x0, weight, bias)
    upstream = torch.randn(batch, channels, h * scale, w * scale, device='cuda')

    def run():
        out = torch.ops.converse2d.forward(x, x0, weight, bias, scale, 1e-5)
        return torch.autograd.grad(out, inputs, upstream)

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
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time > 0]
    names = Counter(('fill' if 'FillFunctor' in e.name else 'real_crop_backward' if 'real_crop_backward' in e.name
                     else 'other') for e in kernels)
    results[name] = dict(call_median_ms=statistics.median(calls), kernels_per_call=len(kernels) / args.iters,
                         kernel_time_per_call_us=sum(e.device_time for e in kernels) / args.iters,
                         per_call_counts={k: v / args.iters for k, v in names.items()})
record = dict(label=args.label, root=str(args.root), iters=args.iters, gpu=torch.cuda.get_device_name(0),
              binary_sha256=json.loads((args.root / '.build/cuda/source_manifest.json').read_text())['binary_sha256'],
              results=results)
with args.output.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record))
