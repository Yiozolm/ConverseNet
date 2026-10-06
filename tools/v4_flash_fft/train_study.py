"""Fused s1 forward+VJP vs the production training operator.

FlashS1 wraps the SRAM-resident forward and backward kernels. The PSF pad/roll,
the per-call kernel fft2 and sigmoid(bias - 9) + eps stay differentiable ATen
operations, so grad_k and grad_l flow back to weight and bias through autograd.
Every gradient (x, weight, bias) and the output are gated with
numerical_policy.comparison: production FP32 is the baseline and FP64 autograd
through models.converse_core.converse2d_reference the reference. Research only.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import torch.nn.functional as F
import study
from study import EPS, comparison, converse2d_reference, paired, production


class FlashS1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, l, pad, ext):
        ctx.pad, ctx.ext = pad, ext
        ctx.save_for_backward(x, k, l)
        return ext.forward(x, k, l.reshape(-1), pad, 1)

    @staticmethod
    def backward(ctx, g):
        x, k, l = ctx.saved_tensors
        gx, gk, gl = ctx.ext.backward(x, g.contiguous(), k, l.reshape(-1), ctx.pad)
        return gx, gk, gl.reshape(l.shape), None, None


def flash(ext, x, weight, bias, pad):
    H, W = x.shape[-2] + 2 * pad, x.shape[-1] + 2 * pad
    kh, kw = weight.shape[-2:]
    psf = torch.roll(F.pad(weight, (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1))
    k = torch.fft.fft2(psf)
    l = torch.sigmoid(bias - 9.0) + EPS
    return FlashS1.apply(x, k, l, pad, ext)


def reference64(x, weight, bias, pad, upstream):
    x, weight, bias = (t.detach().double().requires_grad_() for t in (x, weight, bias))
    xp = F.pad(x, (pad,) * 4, mode='circular') if pad else x
    y = converse2d_reference(xp, xp, weight, bias, 1, EPS)
    y = y[..., pad:y.shape[-2] - pad, pad:y.shape[-1] - pad] if pad else y
    return [y.detach(), *torch.autograd.grad(y, (x, weight, bias), upstream.double())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; keep earlier evidence')
    study.load_extension()
    ext = study.build()
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cases={})
    for name, (b, c, h, w, pad) in study.CASES.items():
        torch.manual_seed(72001)
        x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
        weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
        bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
        upstream = torch.randn(b, c, h, w, device='cuda')
        inputs = (x, weight, bias)

        def run(fn):
            out = fn(x, weight, bias, pad)
            return [out.detach(), *torch.autograd.grad(out, inputs, upstream)]

        prod_fn = production
        flash_fn = lambda *a: flash(ext, *a)
        baseline, candidate = run(prod_fn), run(flash_fn)
        ref = reference64(x, weight, bias, pad, upstream)
        row = dict(accuracy={})
        for label, cand, base, r in zip(('output', 'grad_x', 'grad_weight', 'grad_bias'), candidate, baseline, ref):
            acc = row['accuracy'][label] = comparison(cand, base, r)
            print(f"{name:28s} {label:11s} {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 {acc['candidate']['rel_l2']:.3e} "
                  f"(prod {acc['baseline']['rel_l2']:.3e}, {acc['ratio']['rel_l2'] or 0:.2f}x) "
                  f"max_abs {acc['ratio']['max_abs'] or 0:.2f}x", flush=True)
        row['timing'] = paired({'production': lambda: run(prod_fn), 'flash': lambda: run(flash_fn)},
                               args.rounds, args.iters)
        t = {k: v['median_us'] for k, v in row['timing'].items()}
        print(f"{'':28s} fwd+vjp production {t['production']:7.1f}us | flash {t['flash']:7.1f}us "
              f"({t['production'] / t['flash']:.2f}x)", flush=True)
        report['cases'][name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
