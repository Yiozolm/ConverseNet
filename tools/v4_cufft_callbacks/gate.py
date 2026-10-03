"""P1 numerical gate for every cuFFT callback site, plus backward-site timing.

Operations (candidate vs the ATen path production uses today):
  F1 load_real            fft2 of real x                     vs torch.fft.fft2(x)
  F2 load_circular        fft2 of circularly padded x        vs fft2(_training_pad_complex)
  F3 store_scaled         ifft2 with 1/N at the store        vs torch.fft.ifft2
  B1 load_crop_embed+store_scaled  VJP of real_crop(ifft2(Z))  vs autograd through ATen
  B2 store_real           VJP of fft2(real x)                vs autograd through ATen
Each case reports byte identity and test/numerical_policy.comparison (normal
regime) against an independent NumPy FP64 FFT of the same FP32 inputs.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT), str(HERE)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import numpy as np
import torch
from extension_loader import load_extension
import loader
import numerical_policy

SHAPES = [((4, 128, 96, 96), 2), ((4, 128, 96, 96), 0), ((4, 128, 100, 100), 0), ((4, 64, 48, 48), 1), ((4, 64, 48, 48), 0),
          ((2, 3, 33, 17), 1), ((2, 3, 33, 17), 0), ((1, 4, 24, 28), 2), ((3, 5, 64, 64), 0),
          ((2, 3, 8, 6), 3), ((1, 1, 7, 9), 0), ((2, 2, 1, 5), 0)]
ATOMS = torch.tensor([0., -0., 2. ** -149, -(2. ** -149), 2. ** -126, 1e30, -1e30, 3.])


def sample(shape, kind, generator):
    if kind == 'specials':
        return ATOMS[torch.randint(0, len(ATOMS), shape, generator=generator)].cuda()
    x = torch.randn(shape, generator=generator)
    return (x * 1e18 if kind == 'scaled' else x).cuda()


def identical(a, b):
    a, b = a.contiguous(), b.contiguous()
    a = torch.view_as_real(a) if a.is_complex() else a
    b = torch.view_as_real(b) if b.is_complex() else b
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def gate(candidate, baseline, ref64):
    if callable(candidate):
        try:
            candidate = candidate()
        except RuntimeError as error:  # recorded as a failed case, never skipped
            return dict(error=str(error).splitlines()[0], passed=False, identical=False,
                        ratio=dict(rel_l2=None, max_abs=None))
    result = numerical_policy.comparison(candidate, baseline, torch.from_numpy(ref64).to(candidate.device))
    result['identical'] = identical(candidate, baseline)
    return result


def np64(t):
    return t.detach().cpu().numpy().astype(np.complex128 if t.is_complex() else np.float64)


def cases(ext):
    generator = torch.Generator().manual_seed(61001)
    pad_op = torch.ops.converse2d._training_pad_complex
    crop_op = torch.ops.converse2d._training_real_crop
    for kind in ('randn', 'specials', 'scaled'):
        for (b, c, h, w), pad in SHAPES:
            H, W, N = h + 2 * pad, w + 2 * pad, (h + 2 * pad) * (w + 2 * pad)
            label = dict(kind=kind, shape=[b, c, h, w], pad=pad)
            x = sample((b, c, h, w), kind, generator)
            if pad == 0:
                yield dict(op='F1_load_real', **label), gate(lambda: ext.fft2_real(x, 0), torch.fft.fft2(x), np.fft.fft2(np64(x)))
            else:
                padded = np.pad(np64(x), ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='wrap')
                yield dict(op='F2_load_circular', **label), gate(lambda: ext.fft2_real(x, pad), torch.fft.fft2(pad_op(x, pad)),
                                                             np.fft.fft2(padded))
            z = torch.complex(sample((b, c, H, W), kind, generator), sample((b, c, H, W), kind, generator))
            yield dict(op='F3_store_scaled', **label), gate(lambda: ext.ifft2_scaled(z), torch.fft.ifft2(z), np.fft.ifft2(np64(z)))
            for layout in ('contiguous', 'transposed'):
                g = sample((b, c, h, w), kind, generator)
                if layout == 'transposed':
                    g = g.transpose(2, 3).contiguous().transpose(2, 3)
                leaf = torch.zeros(b, c, H, W, device='cuda', dtype=torch.complex64, requires_grad=True)
                baseline, = torch.autograd.grad(crop_op(torch.fft.ifft2(leaf), pad), leaf, g)
                embedded = np.zeros((b, c, H, W), np.complex128)
                embedded[..., pad:pad + h, pad:pad + w] = np64(g)
                yield (dict(op='B1_crop_embed_fft2_scaled', layout=layout, **label),
                       gate(lambda: ext.crop_embed_fft2_scaled(g, pad), baseline, np.fft.fft2(embedded) / N))
            gy = torch.complex(sample((b, c, H, W), kind, generator), sample((b, c, H, W), kind, generator))
            leaf = torch.zeros(b, c, H, W, device='cuda', requires_grad=True)
            baseline, = torch.autograd.grad(torch.fft.fft2(leaf), leaf, gy)
            yield (dict(op='B2_ifft2_real', **label),
                   gate(lambda: ext.ifft2_real(gy), baseline, np.real(np.fft.ifft2(np64(gy)) * N)))


def timing(ext, rounds, iters):
    torch.manual_seed(61002)
    g = torch.randn(4, 128, 96, 96, device='cuda')
    leaf = torch.zeros(4, 128, 100, 100, device='cuda', dtype=torch.complex64, requires_grad=True)
    out = torch.ops.converse2d._training_real_crop(torch.fft.ifft2(leaf), 2)
    gy = torch.randn(4, 128, 100, 100, device='cuda', dtype=torch.complex64)
    real_leaf = torch.zeros(4, 128, 100, 100, device='cuda', requires_grad=True)
    spectrum = torch.fft.fft2(real_leaf)
    pairs = {
        'B1_crop2_ifft2_vjp_B4C128_96to100': (lambda: torch.autograd.grad(out, leaf, g, retain_graph=True),
                                              lambda: ext.crop_embed_fft2_scaled(g, 2)),
        'B2_fft2_real_vjp_B4C128_100': (lambda: torch.autograd.grad(spectrum, real_leaf, gy, retain_graph=True),
                                        lambda: ext.ifft2_real(gy)),
    }
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def measure(fn):
        values = []
        for _ in range(iters):
            start.record(); fn(); stop.record(); stop.synchronize()
            values.append(start.elapsed_time(stop) * 1e3)
        return statistics.median(values)

    result = {}
    for name, (aten, ours) in pairs.items():
        for fn in (aten, ours):
            for _ in range(5):
                fn()
        a, o = [], []
        for r in range(rounds):
            for fn in ((aten, ours) if r % 2 == 0 else (ours, aten)):
                (a if fn is aten else o).append(measure(fn))
        result[name] = dict(aten_us=a, callback_us=o, speedup=statistics.median(a) / statistics.median(o))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    load_extension()
    ext, identity = loader.build()
    rows = [dict(**label, **result) for label, result in cases(ext)]
    summary = {}
    for row in rows:
        s = summary.setdefault(row['op'], dict(cases=0, passed=0, identical=0, errors=[], worst_rel_l2_ratio=0., worst_max_abs_ratio=0.))
        s['cases'] += 1; s['passed'] += row['passed']; s['identical'] += row['identical']
        if 'error' in row:
            s['errors'].append(dict(kind=row['kind'], shape=row['shape'], pad=row['pad'], error=row['error']))
        for metric in ('rel_l2', 'max_abs'):
            ratio = row['ratio'][metric]
            s[f'worst_{metric}_ratio'] = max(s[f'worst_{metric}_ratio'], ratio if ratio is not None else float('inf'))
    report = dict(identity=identity, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  policy=numerical_policy.BUDGETS['normal'], summary=summary, passed=all(r['passed'] for r in rows),
                  cases=rows, timing=timing(ext, args.rounds, args.iters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(passed=report['passed'], summary=summary,
                          speedups={k: round(v['speedup'], 3) for k, v in report['timing'].items()}), indent=2))


if __name__ == '__main__':
    main()
