"""Report which training transforms use callbacks on this GPU (profiler kernel names).

Bit-identity admission is decided per plan at run time, so another GPU may
admit a different set of shapes. Per training call this counts:
lto_fft (callback-linked FFT kernels), scale (ATen 1/N passes, AUnaryFunctor),
real_crop (ATen-path real/crop VJP kernel) and pad (circular-pad kernels).
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import torch
from extension_loader import load_extension

# name: (entry, batch, channels, height, width, scale, padding)
CONFIGS = {'circular_s1_96_pad2 (100x100)': ('circular', 4, 64, 96, 96, 1, 2),
           'circular_s1_48_pad1 (50x50)': ('circular', 4, 64, 48, 48, 1, 1),
           'forward_s1_100': ('forward', 4, 64, 100, 100, 1, 0),
           'forward_s1_96': ('forward', 4, 64, 96, 96, 1, 0),
           'forward_s1_64': ('forward', 4, 64, 64, 64, 1, 0),
           'forward_s1_33x17': ('forward', 2, 8, 33, 17, 1, 0),
           'forward_s2_48 (96x96)': ('forward', 4, 32, 48, 48, 2, 0),
           'forward_s2_64 (128x128)': ('forward', 4, 32, 64, 64, 2, 0),
           'forward_s3_32 (96x96)': ('forward', 2, 32, 32, 32, 3, 0),
           'forward_s3_48 (144x144)': ('forward', 2, 32, 48, 48, 3, 0)}


def classify(name):
    for key, token in (('lto_fft', 'lto_fft'), ('scale', 'AUnaryFunctor'), ('real_crop', 'real_crop'),
                       ('pad', 'circular_pad')):
        if token in name:
            return key
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load_extension()
    rows = {}
    for name, (entry, batch, channels, h, w, scale, padding) in CONFIGS.items():
        torch.manual_seed(5)
        x = torch.randn(batch, channels, h, w, device='cuda', requires_grad=True)
        x0 = x if scale == 1 else torch.randn(batch, channels, h * scale, w * scale, device='cuda', requires_grad=True)
        weight = torch.rand(1, channels, 3, 3, device='cuda', requires_grad=True)
        bias = torch.zeros(1, channels, 1, 1, device='cuda', requires_grad=True)
        upstream = torch.randn(batch, channels, h * scale, w * scale, device='cuda')

        def run():
            out = (torch.ops.converse2d._training_circular_s1(x, weight, bias, padding, 1e-5) if entry == 'circular'
                   else torch.ops.converse2d.forward(x, x0, weight, bias, scale, 1e-5))
            return torch.autograd.grad(out, (x, weight, bias), upstream)

        run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
            run()
            torch.cuda.synchronize()
        counts = collections.Counter(classify(e.name) for e in trace.events() if e.device_time > 0)
        counts.pop(None, None)
        rows[name] = {key: counts.get(key, 0) for key in ('lto_fft', 'scale', 'real_crop', 'pad')}
    report = dict(gpu=torch.cuda.get_device_name(0), callbacks_env=os.environ.get('CONVERSE2D_FFT_CALLBACKS'),
                  rows=rows)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, row in rows.items():
        print(f'{name:32s} {row}')


if __name__ == '__main__':
    main()
