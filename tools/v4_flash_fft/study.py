"""SRAM-resident fused s1 forward vs the production training forward.

Accuracy: fused output against the production FP32 forward (baseline) and the
FP64 Python reference, through numerical_policy.comparison.
Timing: paired CUDA events. The fused side includes the per-call kernel
preparation (PSF pad/roll, FP32 fft2, sigmoid) that it consumes. The
production side is the differentiable forward (grad enabled), with and
without its VJP for context. Research only; forward only.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
import torch
import torch.nn.functional as F
from torch.utils import cpp_extension
from extension_loader import load_extension
from models.converse_core import converse2d_reference
from numerical_policy import comparison

EPS = 1e-5
VARIANTS = {0: 'generic', 1: 'static', 2: 'pair'}
CASES = {'circular_s1_b4_c64_96_pad2': (4, 64, 96, 96, 2), 'circular_s1_b4_c128_96_pad2': (4, 128, 96, 96, 2),
         'forward_s1_b4_c128_100': (4, 128, 100, 100, 0), 'forward_s1_b4_c64_96': (4, 64, 96, 96, 0),
         'forward_s1_b4_c64_64': (4, 64, 64, 64, 0)}


def build(verbose=False):
    if os.name == 'nt':
        cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
    (ROOT / '.build' / 'flash_fft').mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(name='flash_fft_research', sources=[str(HERE / 'fused.cu'), str(HERE / 'bind.cpp')],
                              extra_cflags=['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],
                              extra_cuda_cflags=['-O3', '-lineinfo', '-std=c++17'] + (['-Xptxas=-v'] if verbose else []),
                              build_directory=str(ROOT / '.build' / 'flash_fft'), verbose=verbose)


def kernel_prep(weight, bias, H, W):
    kh, kw = weight.shape[-2:]
    psf = torch.roll(F.pad(weight, (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1))
    return torch.fft.fft2(psf).contiguous(), (torch.sigmoid(bias - 9.0) + EPS).reshape(-1).contiguous()


def production(x, weight, bias, pad):
    if pad:
        return torch.ops.converse2d._training_circular_s1(x, weight, bias, pad, EPS)
    return torch.ops.converse2d.forward(x, x, weight, bias, 1, EPS)


def reference64(x, weight, bias, pad):
    x, weight, bias = (t.detach().double() for t in (x, weight, bias))
    xp = F.pad(x, (pad,) * 4, mode='circular') if pad else x
    y = converse2d_reference(xp, xp, weight, bias, 1, EPS)
    return y[..., pad:y.shape[-2] - pad, pad:y.shape[-1] - pad] if pad else y


def fused(ext, x, weight, bias, pad, variant=0):
    H, W = x.shape[-2] + 2 * pad, x.shape[-1] + 2 * pad
    k, l = kernel_prep(weight, bias, H, W)
    return ext.forward(x, k, l, pad, variant)


def paired(fns, rounds, iters):
    times = {name: [] for name in fns}
    for fn in fns.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for name, fn in fns.items():
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            stop.record()
            stop.synchronize()
            times[name].append(start.elapsed_time(stop) * 1000 / iters)
    return {name: dict(median_us=statistics.median(v), rounds=v) for name, v in times.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; keep earlier evidence')
    load_extension()
    ext = build()
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cases={})
    for name, (b, c, h, w, pad) in CASES.items():
        torch.manual_seed(71001)
        x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
        weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
        bias = torch.zeros(1, c, 1, 1, device='cuda', requires_grad=True)
        upstream = torch.randn(b, c, h, w, device='cuda')
        baseline = production(x, weight, bias, pad).detach()
        ref = reference64(x, weight, bias, pad)
        row = dict(accuracy={})
        for variant, label in VARIANTS.items():
            with torch.no_grad():
                candidate = fused(ext, x.detach(), weight.detach(), bias.detach(), pad, variant)
            acc = row['accuracy'][label] = comparison(candidate, baseline, ref)
            print(f"{name:28s} {label:7s} {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 "
                  f"{acc['candidate']['rel_l2']:.3e} (prod {acc['baseline']['rel_l2']:.3e}, "
                  f"{acc['ratio']['rel_l2']:.2f}x) max_abs {acc['ratio']['max_abs']:.2f}x", flush=True)

        def fused_call(variant):
            with torch.no_grad():
                fused(ext, x, weight, bias, pad, variant)

        def production_forward():
            production(x, weight, bias, pad)

        def production_train():
            torch.autograd.grad(production(x, weight, bias, pad), (x, weight, bias), upstream)

        fns = {'production_forward': production_forward}
        fns.update({label: (lambda v=variant: fused_call(v)) for variant, label in VARIANTS.items()})
        fns['production_forward_vjp'] = production_train
        row['timing'] = paired(fns, args.rounds, args.iters)
        t = {k: v['median_us'] for k, v in row['timing'].items()}
        base = t['production_forward']
        print(f"{'':28s} prod fwd {base:7.1f}us | " + ' | '.join(
            f"{label} {t[label]:7.1f}us ({base / t[label]:.2f}x)" for label in VARIANTS.values()) +
            f" | prod fwd+vjp {t['production_forward_vjp']:7.1f}us", flush=True)
        report['cases'][name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
