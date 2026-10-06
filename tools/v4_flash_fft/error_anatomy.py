"""Where do the fused path's grad_bias / grad_weight budget failures come from?

For each case and seed (the seed_sweep inputs), the kernel- and regularizer-gradient
formulas are re-evaluated in FP64 with spectra from different sources:
  exact   FP64 FFTs of the inputs, FP64 k and l
  cufft   production's FP32 spectra: fft2 of (x, 0) and fft2(g)/N, as the admitted
          cuFFT callbacks compute them
  cufftT  the same cuFFT applied to transposed planes: mathematically identical,
          equally accurate, different rounding (a control "second FFT")
  fused   the fused kernels' own FFT (ext.debug_fft2)
Each path's actual error then splits as
  shared       FP32 k = fft2(psf) and l = sigmoid(bias - 9) + eps, common to both paths
  spectrum     the path's FP32 spectra, everything downstream in FP64
  arithmetic   the path's FP32 pointwise math and reductions (actual - FP64 model)
Errors are relative L2 over channels against the FP64 reference gradient, as in the
release gate. Also reported: how concentrated each error is (effective number of
channels and of frequency terms) and the spectrum-only ratios of fused and of the
control to production. Research only.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import torch
import torch.nn.functional as F
import study
import flash_study as fs

CASES = ['circular_s1_b4_c64_96_pad2', 'circular_s1_b4_c128_96_pad2', 'forward_s1_b4_c128_100',
         'forward_s1_b4_c64_96', 'forward_s2_b4_c64_32', 'forward_s2_b4_c64_48', 'forward_s3_b2_c32_32',
         'forward_s3_b2_c32_24', 'forward_s2_b4_c64_64', 'forward_s3_b2_c32_48']


def aliases(t, s, op):
    if s == 1:
        return t
    v = t.reshape(t.shape[0], t.shape[1], s, t.shape[2] // s, s, t.shape[3] // s)
    return v.sum((2, 4)) if op == 'sum' else v.mean((2, 4))


def model(Y, P, G, k, l, s):
    """FP64 kernel/regularizer gradients from given spectra (production's adjoint)."""
    pm = aliases(k * P, s, 'mean')
    d = aliases(k.real ** 2 + k.imag ** 2, s, 'mean') + l
    q = (Y - pm) / d
    t = aliases(G * k, s, 'sum')
    gy = t / d
    gd = (-t * (q / d).conj()).real
    gm = -gy / (s * s)
    rep = (lambda z: z.repeat(1, 1, s, s)) if s > 1 else (lambda z: z)
    gk = ((G * rep(q).conj()).conj() + rep(gm) * P.conj() + 2 * k * rep(gd) / (s * s)).sum(0, keepdim=True)
    return gk, gd


def to_weight_grad(gk, weight, H, W):
    w = weight.detach().double().requires_grad_()
    kh, kw = w.shape[-2:]
    k = torch.fft.fft2(torch.roll(F.pad(w, (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1)))
    return torch.autograd.grad(k, w, gk)[0]


def spectra(source, ext, x, x0, g, s):
    """(Y, P, G) as complex128, from one FFT source; G includes the 1/N factor."""
    n = g.shape[-1] * g.shape[-2]
    if source == 'exact':
        f = lambda t: torch.fft.fft2(t.double().to(torch.complex128))
        Gs = f(g) / n
    elif source == 'cufft':
        f = lambda t: torch.fft.fft2(t.to(torch.complex64))
        Gs = torch.fft.fft2(g.to(torch.complex64), norm='forward')
    elif source == 'cufftT':
        f = lambda t: torch.fft.fft2(t.transpose(-1, -2).contiguous().to(torch.complex64)).transpose(-1, -2)
        Gs = torch.fft.fft2(g.transpose(-1, -2).contiguous().to(torch.complex64), norm='forward').transpose(-1, -2)
    else:
        f = lambda t: ext.debug_fft2(t.contiguous())
        raw = ext.debug_fft2(g.contiguous())
        scale = torch.tensor(1.0 / n, dtype=torch.float64).float().item()
        Gs = torch.complex(raw.real * scale, raw.imag * scale)
    Y = f(x)
    P = Y if s == 1 else f(x0)
    return [t.to(torch.complex128) for t in (Y, P, Gs)]


def rel(e, ref):
    return float(e.norm() / ref.norm())


def n_eff(v):
    v2 = v.reshape(-1).double() ** 2
    return float(v2.sum() ** 2 / (v2 ** 2).sum()) if v2.sum() > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--cases', nargs='*', default=CASES)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists')
    study.load_extension()
    ext = study.build()
    device = torch.cuda.current_device()
    rows = []
    for name in args.cases:
        b, c, h, w, s, pad = fs.CASES[name]
        if not ext.supported(h, w, s, pad, device):
            print(f'{name}: skipped (not eligible here)', flush=True)
            continue
        for seed in range(args.seeds):
            torch.manual_seed(75000 + seed)  # same inputs as seed_sweep.py
            x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
            x0 = torch.randn(b, c, h * s, w * s, device='cuda', requires_grad=s > 1)
            weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
            bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
            g = torch.randn(b, c, h * s, w * s, device='cuda')
            inputs = (x, weight, bias) if s == 1 else (x, x0, weight, bias)
            actual = {}
            for path, fn in (('cufft', fs.production), ('fused', lambda *a: fs.flash(ext, *a))):
                grads = torch.autograd.grad(fn(x, x0, weight, bias, s, pad), inputs, g)
                actual[path] = dict(weight=grads[-2].double(), bias=grads[-1].double())
            ref = fs.reference64(x, x0, weight, bias, s, pad, g)
            ref = dict(weight=ref[-2], bias=ref[-1])
            H, W = h * s + 2 * pad, w * s + 2 * pad
            kh, kw = 3, 3
            psf32 = torch.roll(F.pad(weight.detach(), (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1))
            k32 = torch.fft.fft2(psf32).to(torch.complex128)
            k64 = torch.fft.fft2(psf32.double())
            l32 = (torch.sigmoid(bias.detach() - 9.0) + fs.EPS).double()
            l64 = torch.sigmoid(bias.detach().double() - 9.0) + fs.EPS
            dsig = torch.sigmoid(bias.detach().double() - 9.0) * (1 - torch.sigmoid(bias.detach().double() - 9.0))
            # Circular s1: spectra of the circularly padded x and the zero-embedded gradient,
            # as the production callbacks and the fused kernels form them.
            xs = F.pad(x.detach(), (pad,) * 4, mode='circular') if pad else x.detach()
            x0s, gs = x0.detach(), F.pad(g, (pad,) * 4) if pad else g

            def evaluate(Y, P, G, k, l):
                gk, gd = model(Y, P, G, k, l, s)
                return dict(weight=to_weight_grad(gk, weight, H, W),
                            bias=(gd.sum((0, 2, 3)).reshape(1, c, 1, 1) * dsig)), gd

            spec = {src: spectra(src, ext, xs, x0s, gs, s) for src in ('exact', 'cufft', 'cufftT', 'fused')}
            exact, gd_exact = evaluate(*spec['exact'], k64, l64)
            shared_base, _ = evaluate(*spec['exact'], k32, l32)
            row = dict(case=name, seed=seed, model_vs_reference={
                q: rel(exact[q] - ref[q], ref[q]) for q in ('weight', 'bias')}, grads={})
            per_src = {src: evaluate(*spec[src], k32, l32) for src in ('cufft', 'cufftT', 'fused')}
            for q in ('weight', 'bias'):
                r = ref[q]
                entry = dict(shared=rel(shared_base[q] - exact[q], r))
                for src in ('cufft', 'cufftT', 'fused'):
                    entry[f'spectrum_{src}'] = rel(per_src[src][0][q] - shared_base[q], r)
                for path in ('cufft', 'fused'):
                    entry[f'arithmetic_{path}'] = rel(actual[path][q] - per_src[path][0][q], r)
                    entry[f'total_{path}'] = rel(actual[path][q] - r, r)
                    entry[f'n_eff_channels_{path}'] = n_eff(actual[path][q] - r)
                entry['gate_ratio'] = entry['total_fused'] / entry['total_cufft']
                # Upper bound on what better fused arithmetic could achieve: fused spectra,
                # exact (FP64) pointwise math, sums and sigmoid chain.
                entry['ideal_fused_gate_ratio'] = rel(per_src['fused'][0][q] - r, r) / entry['total_cufft']
                # The same for the transposed-cuFFT control: an equally accurate FFT.
                entry['ideal_control_gate_ratio'] = rel(per_src['cufftT'][0][q] - r, r) / entry['total_cufft']
                entry['spectrum_ratio_fused'] = entry['spectrum_fused'] / entry['spectrum_cufft']
                entry['spectrum_ratio_control'] = entry['spectrum_cufftT'] / entry['spectrum_cufft']
                row['grads'][q] = entry
            # Concentration of the grad_l sum: how many frequency terms carry |gd| and its spectrum error.
            dg = {src: (per_src[src][1] - model(*spec['exact'], k32, l32, s)[1]) for src in ('cufft', 'fused')}
            row['terms'] = dict(
                n_eff_gd=sum(n_eff(gd_exact[:, ch]) for ch in range(c)) / c,
                n_eff_spectrum_error_cufft=sum(n_eff(dg['cufft'][:, ch]) for ch in range(c)) / c,
                n_eff_spectrum_error_fused=sum(n_eff(dg['fused'][:, ch]) for ch in range(c)) / c,
                terms_per_channel=gd_exact[:, 0].numel(),
                cancellation=float((gd_exact.abs().sum((0, 2, 3)) / gd_exact.sum((0, 2, 3)).abs()).median()))
            rows.append(row)
            gb = row['grads']['bias']
            print(f"{name:22s} seed {seed}: bias gate {gb['gate_ratio']:.2f} | spectrum fused/prod "
                  f"{gb['spectrum_ratio_fused']:.2f} control/prod {gb['spectrum_ratio_control']:.2f} | shares "
                  f"shared {gb['shared']:.1e} spec {gb['spectrum_cufft']:.1e}/{gb['spectrum_fused']:.1e} arith "
                  f"{gb['arithmetic_cufft']:.1e}/{gb['arithmetic_fused']:.1e} | n_eff ch "
                  f"{gb['n_eff_channels_cufft']:.1f}/{gb['n_eff_channels_fused']:.1f} | terms "
                  f"{row['terms']['n_eff_spectrum_error_cufft']:.0f}/{row['terms']['terms_per_channel']} "
                  f"cancel {row['terms']['cancellation']:.0f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))
    summarize(rows)


def summarize(rows):
    def geo(v):
        v = [x for x in v if x > 0]
        return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float('nan')

    print('\nper case, geometric means over seeds (fail = gate ratio > 1.25):')
    for name in dict.fromkeys(r['case'] for r in rows):
        rs = [r for r in rows if r['case'] == name]
        for q in ('bias', 'weight'):
            e = [r['grads'][q] for r in rs]
            print(f"  {name:22s} {q:6s} ideal-arithmetic gate fused {geo([x['ideal_fused_gate_ratio'] for x in e]):.2f} "
                  f"(>1.25: {sum(x['ideal_fused_gate_ratio'] > 1.25 for x in e)}) control "
                  f"{geo([x['ideal_control_gate_ratio'] for x in e]):.2f} (>1.25: {sum(x['ideal_control_gate_ratio'] > 1.25 for x in e)})")
            print(f"  {name:22s} {q:6s} gate {geo([x['gate_ratio'] for x in e]):.2f} "
                  f"(fail {sum(x['gate_ratio'] > 1.25 for x in e)}/{len(e)}) | spectrum fused/prod "
                  f"{geo([x['spectrum_ratio_fused'] for x in e]):.2f} (>1.25: {sum(x['spectrum_ratio_fused'] > 1.25 for x in e)}) "
                  f"control/prod {geo([x['spectrum_ratio_control'] for x in e]):.2f} "
                  f"(>1.25: {sum(x['spectrum_ratio_control'] > 1.25 for x in e)})")


if __name__ == '__main__':
    main()
