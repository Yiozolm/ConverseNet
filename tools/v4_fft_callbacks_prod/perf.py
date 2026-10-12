"""One timing round of training forward+VJP with the checked build.

Run alternately with CONVERSE2D_FFT_CALLBACKS=0 (the unchanged ATen sequence in
the same binary) and unset or a site list, each in a fresh process. Records the
complete call median (CUDA events), and from one profiled window kernels per
call, summed kernel time, the number of callback-linked FFT kernels and the
time per call of every kernel name.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import torch
from torch.profiler import ProfilerActivity, profile
from extension_loader import load_extension

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--iters', type=int, default=40)
args = parser.parse_args()
load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# name: (entry, batch, channels, height, width, scale, padding)
CONFIGS = {'circular_s1_b4_c64_96_pad2': ('circular', 4, 64, 96, 96, 1, 2),
           'circular_s1_b4_c128_96_pad2': ('circular', 4, 128, 96, 96, 1, 2),
           'forward_s1_b4_c128_100': ('forward', 4, 128, 100, 100, 1, 0),
           'forward_s2_b4_c64_64': ('forward', 4, 64, 64, 64, 2, 0),
           'forward_s3_b2_c32_48': ('forward', 2, 32, 48, 48, 3, 0)}
results = {}
for name, (entry, batch, channels, h, w, scale, padding) in CONFIGS.items():
    torch.manual_seed(41001)
    x = torch.randn(batch, channels, h, w, device='cuda', requires_grad=True)
    x0 = x if scale == 1 else torch.randn(batch, channels, h * scale, w * scale, device='cuda', requires_grad=True)
    weight = torch.rand(1, channels, 3, 3, device='cuda', requires_grad=True)
    bias = torch.zeros(1, channels, 1, 1, device='cuda', requires_grad=True)
    inputs = (x, weight, bias) if x0 is x else (x, x0, weight, bias)
    upstream = torch.randn(batch, channels, h * scale, w * scale, device='cuda')

    def run():
        if entry == 'circular':
            out = torch.ops.converse2d._training_circular_s1(x, weight, bias, padding, 1e-5)
        else:
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
    by_name = collections.Counter()
    for e in kernels:
        by_name[e.name[:160]] += e.device_time / args.iters
    results[name] = dict(call_median_ms=statistics.median(calls), kernels_per_call=len(kernels) / args.iters,
                         kernel_time_per_call_us=sum(e.device_time for e in kernels) / args.iters,
                         lto_fft_per_call=sum('lto_fft' in e.name for e in kernels) / args.iters,
                         kernel_us_by_name=dict(by_name))
record = dict(label=args.label, callbacks_env=os.environ.get('CONVERSE2D_FFT_CALLBACKS'), iters=args.iters,
              gpu=torch.cuda.get_device_name(0),
              binary_sha256=json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())['binary_sha256'],
              results=results)
with args.output.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record))
