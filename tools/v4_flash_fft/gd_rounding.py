"""Which FP32 step of the per-frequency regularizer gradient gd dominates grad_l error?

gd = Re(-t * conj(q / d)),  t = sum_alias(G k),  q = (y - mean_alias(k p)) / d,
d = mean_alias(|k|^2) + l.  Starting from the fused kernels' own FP32 spectra (so
the spectrum contribution is held fixed), gl = sum gd is evaluated in FP64 except
for one step at a time, which is rounded to FP32 (complex64 / float32 in torch,
without FMA contraction; a proxy for the kernel's rounding):
  t        the alias sum of G k
  q        y - pm and the division by d
  qd       the second division q / d
  re       the final real part Re(-t conj(q/d)) (a two-term dot product)
  terms    every step above at once (gd rounded per frequency, summed exactly)
  sum      FP64 gd rounded to FP32, then summed in FP32 (sequential and pairwise)
Errors are rel-L2 over channels of gl against the all-FP64 value. Research only.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import torch
import torch.nn.functional as F
import study
import flash_study as fs
from error_anatomy import aliases, spectra

C64, F32 = torch.complex64, torch.float32


def gl(Y, P, G, k, l, s, rounded):
    r = (lambda z: z.to(C64 if z.is_complex() else F32).to(z.dtype))
    pm = aliases(k * P, s, 'mean')
    d = aliases(k.real ** 2 + k.imag ** 2, s, 'mean') + l
    t = aliases(G * k, s, 'sum')
    if 't' in rounded:
        t = r(t)
    num = Y - pm
    q = num / d
    if 'q' in rounded:
        q = r(r(num) / d)
    qd = q / d
    if 'qd' in rounded:
        qd = r(qd)
    if 're' in rounded:
        gd = r(r(-t.real * qd.real) + r(-t.imag * qd.imag))  # -(tx qx + ty qy), each product rounded
    else:
        gd = (-t * qd.conj()).real
    if 'terms' in rounded:
        gd = r(gd)
    if 'sum_seq' in rounded:
        return torch.cumsum(gd.to(F32).permute(1, 0, 2, 3).reshape(gd.shape[1], -1), 1)[:, -1].double()
    if 'sum_pair' in rounded:
        return gd.to(F32).sum((0, 2, 3)).double()
    return gd.sum((0, 2, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--cases', nargs='*', default=['forward_s2_b4_c64_32', 'forward_s2_b4_c64_48',
                                                       'forward_s3_b2_c32_32', 'forward_s3_b2_c32_24'])
    args = parser.parse_args()
    study.load_extension()
    ext = study.build()
    variants = {'t': {'t'}, 'q': {'q'}, 'qd': {'qd'}, 're': {'re'}, 'terms': {'t', 'q', 'qd', 're', 'terms'},
                'sum_seq': {'sum_seq'}, 'sum_pair': {'sum_pair'}}
    rows = []
    for name in args.cases:
        b, c, h, w, s, pad = fs.CASES[name]
        for seed in range(args.seeds):
            torch.manual_seed(75000 + seed)
            x = torch.randn(b, c, h, w, device='cuda')
            x0 = torch.randn(b, c, h * s, w * s, device='cuda')
            weight = torch.rand(1, c, 3, 3, device='cuda')
            bias = torch.randn(1, c, 1, 1, device='cuda')
            g = torch.randn(b, c, h * s, w * s, device='cuda')
            H, W = h * s, w * s
            psf = torch.roll(F.pad(weight, (0, W - 3, 0, H - 3)), (-1, -1), (-2, -1))
            k = torch.fft.fft2(psf).to(torch.complex128)
            l = (torch.sigmoid(bias - 9.0) + fs.EPS).double()
            Y, P, G = spectra('fused', ext, x, x0, g, s)
            exact = gl(Y, P, G, k, l, s, set())
            row = dict(case=name, seed=seed, errors={v: float((gl(Y, P, G, k, l, s, steps) - exact).norm() / exact.norm())
                                                     for v, steps in variants.items()})
            rows.append(row)
            print(f"{name:22s} seed {seed}: " + ' '.join(f'{v} {e:.1e}' for v, e in row['errors'].items()), flush=True)
    print('\nmedian rel-L2 of gl per rounded step:')
    for v in variants:
        print(f"  {v:9s} {statistics.median(r['errors'][v] for r in rows):.2e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
