"""VkFFT vs cuFFT (ATen) feasibility for Converse2D's batched 2-D FP32 transforms.

1. Accuracy: each transform against an FP64 FFT of the same FP32 input, gated by
   test/numerical_policy.comparison with ATen/cuFFT as the FP32 baseline.
   Component-level only; it is not the operator release gate.
2. Paired CUDA-event timing at production shapes, ATen and VkFFT alternating.

VkFFT's FP32 CUDA default (useLUT=-1) evaluates twiddles with __sincosf, a
fast-math intrinsic the FP32 policy excludes; it is measured as `vk_sincosf`
for reference only. `vk_lut` (useLUT=1) uses its host-precomputed twiddle table.
Research only.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'test'), str(HERE)]
import torch
from numerical_policy import comparison
import loader

LUT, SINCOSF = 1, -1


def inputs(shape, kind, generator, real=False):
    def one():
        x = torch.randn(shape, generator=generator)
        if kind == 'scaled':
            x = x * 1e18
        if kind == 'specials':
            atoms = torch.tensor([0., -0., 2. ** -149, 2. ** -126, 1e30, -1e30, 3.])
            x = atoms[torch.randint(0, len(atoms), shape, generator=generator)]
        return x
    x = one() if real else torch.complex(one(), one())
    return x.cuda()


def transforms(ext):
    """name -> (make input, ATen, VkFFT(lut), FP64 reference)."""
    def inverse_scaled(z, lut):
        out = z.clone()
        ext.fft2_(out, 1, True, lut)
        return out
    return {
        'fft2': (False, lambda z: torch.fft.fft2(z), lambda z, lut: ext.fft2(z, lut), lambda z: torch.fft.fft2(z)),
        'ifft2': (False, lambda z: torch.fft.ifft2(z), inverse_scaled, lambda z: torch.fft.ifft2(z)),
        'rfft2': (True, lambda x: torch.fft.rfft2(x), lambda x, lut: ext.rfft2(x, lut), lambda x: torch.fft.rfft2(x)),
        'irfft2': (True, lambda x: torch.fft.irfft2(torch.fft.rfft2(x), s=x.shape[-2:]),
                   lambda x, lut: ext.irfft2(torch.fft.rfft2(x), x.shape[-1], lut),
                   lambda x: torch.fft.irfft2(torch.fft.rfft2(x.double()), s=x.shape[-2:])),
    }


ACCURACY_SHAPES = [(4, 128, 100, 100), (4, 128, 96, 96), (4, 64, 128, 128), (2, 32, 144, 144),
                   (4, 64, 64, 64), (4, 64, 48, 48), (2, 3, 33, 17), (1, 1, 7, 9), (2, 4, 97, 101)]


def accuracy(ext):
    generator = torch.Generator().manual_seed(61001)
    rows = []
    for kind in ('randn', 'scaled', 'specials'):
        for shape in ACCURACY_SHAPES:
            for name, (real, aten, vk, exact) in transforms(ext).items():
                if real and shape[-1] % 2:
                    continue  # irfft2 of an odd width needs s; keep the study to even widths
                x = inputs(shape, kind, generator, real)
                row = dict(kind=kind, shape=list(shape), transform=name)
                try:
                    baseline = aten(x)
                    ref = exact(x.double() if real else x.to(torch.complex128))
                    for label, lut in (('vk_lut', LUT), ('vk_sincosf', SINCOSF)):
                        row[label] = comparison(vk(x, lut), baseline, ref)
                except Exception as error:  # keep every failure in the report
                    row['error'] = repr(error)
                rows.append(row)
                status = 'ERR' if 'error' in row else ' '.join(
                    f"{label} {'ok ' if row[label]['passed'] else 'FAIL'} l2 {row[label]['ratio']['rel_l2'] or 0:5.2f}x "
                    f"max {row[label]['ratio']['max_abs'] or 0:5.2f}x" for label in ('vk_lut', 'vk_sincosf'))
                print(f'{kind:8s} {name:6s} {str(shape):22s} {status}', flush=True)
    return rows


TIMING_SHAPES = {'s1_pad_b4_c128_100': (4, 128, 100, 100), 's1_b4_c128_96': (4, 128, 96, 96),
                 's1_b4_c64_100': (4, 64, 100, 100), 's2_b4_c64_128': (4, 64, 128, 128),
                 's3_b2_c32_144': (2, 32, 144, 144), 'b4_c64_64': (4, 64, 64, 64), 'b16_c64_256': (16, 64, 256, 256)}


def paired(pairs, rounds, iters):
    """Alternate candidates within each round; median per-call microseconds."""
    times = {name: [] for name in pairs}
    for fn in pairs.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for name, fn in pairs.items():
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            stop.record()
            stop.synchronize()
            times[name].append(start.elapsed_time(stop) * 1000 / iters)
    return {name: dict(median_us=statistics.median(v), rounds=v) for name, v in times.items()}


def timing(ext, rounds, iters):
    results = {}
    for label, shape in TIMING_SHAPES.items():
        torch.manual_seed(61002)
        z = torch.randn(shape, dtype=torch.complex64, device='cuda')
        x = torch.randn(shape, device='cuda')
        work = z.clone()
        out = torch.empty_like(z)
        half = torch.fft.rfft2(x)
        groups = {
            'fft2': {'aten': lambda: torch.fft.fft2(z), 'vk_lut': lambda: ext.fft2(z, LUT),
                     'vk_sincosf': lambda: ext.fft2(z, SINCOSF)},
            # Drop-in inverse clones z; `inplace` is the transform alone on a scratch buffer.
            'ifft2': {'aten': lambda: torch.fft.ifft2(z), 'vk_lut': lambda: ext.fft2_(z.clone(), 1, True, LUT),
                      'vk_lut_inplace': lambda: ext.fft2_(work, 1, True, LUT),
                      'vk_sincosf_inplace': lambda: ext.fft2_(work, 1, True, SINCOSF)},
            'rfft2': {'aten': lambda: torch.fft.rfft2(x), 'vk_lut': lambda: ext.rfft2(x, LUT),
                      'vk_sincosf': lambda: ext.rfft2(x, SINCOSF)},
            # Both clone the half spectrum (ATen does so for C2R as well).
            'irfft2': {'aten': lambda: torch.fft.irfft2(half, s=shape[-2:]), 'vk_lut': lambda: ext.irfft2(half, shape[-1], LUT),
                       'vk_sincosf': lambda: ext.irfft2(half, shape[-1], SINCOSF)},
        }
        results[label] = {}
        for group, pairs in groups.items():
            r = paired(pairs, rounds, iters)
            results[label][group] = r
            base = r['aten']['median_us']
            print(f'{label:20s} {group:7s} ' + ' | '.join(
                f"{k} {v['median_us']:8.1f}us ({base / v['median_us']:4.2f}x)" for k, v in r.items()), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--skip-accuracy', action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    ext, identity = loader.build()
    report = dict(identity=identity, device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    if not args.skip_accuracy:
        report['accuracy'] = accuracy(ext)
        lut = [r for r in report['accuracy'] if 'vk_lut' in r]
        report['accuracy_summary'] = dict(
            vk_lut_failed=sum(not r['vk_lut']['passed'] for r in lut),
            vk_sincosf_failed=sum(not r['vk_sincosf']['passed'] for r in lut),
            errors=sum('error' in r for r in report['accuracy']), rows=len(report['accuracy']))
        print(report['accuracy_summary'], flush=True)
    report['timing'] = timing(ext, args.rounds, args.iters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; keep earlier evidence')
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
