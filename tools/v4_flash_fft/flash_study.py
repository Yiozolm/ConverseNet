"""SRAM-resident fused training (s1, s2, s3) vs production, with a capacity switch.

Each case runs fused only if ext.supported(): the shape is compiled and its
complex plane (padded s1 plane, or the S*H x S*W s2/s3 plane) fits the device's
opt-in shared memory per block. Otherwise the case records `production` as its
path and its fused columns stay empty. That is the fallback, not a failure.

For every fused case: output and all input gradients are gated with
numerical_policy.comparison (production FP32 baseline, FP64 autograd through
models.converse_core.converse2d_reference), two calls are compared bit for bit,
and forward+VJP is timed against production with paired CUDA events.
Writes <output>/flash.json and <output>/summary.md. Research only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import torch.nn.functional as F
import study
from study import EPS, comparison, converse2d_reference, paired
from train_study import FlashS1

# name: (batch, channels, height, width, scale, circular padding)
CASES = {
    'circular_s1_b4_c64_96_pad2': (4, 64, 96, 96, 1, 2),
    'circular_s1_b4_c128_96_pad2': (4, 128, 96, 96, 1, 2),
    'forward_s1_b4_c128_100': (4, 128, 100, 100, 1, 0),
    'forward_s1_b4_c64_96': (4, 64, 96, 96, 1, 0),
    'forward_s2_b4_c64_64': (4, 64, 64, 64, 2, 0),
    'forward_s2_b4_c64_48': (4, 64, 48, 48, 2, 0),
    'forward_s2_b4_c64_32': (4, 64, 32, 32, 2, 0),
    'forward_s3_b2_c32_48': (2, 32, 48, 48, 3, 0),
    'forward_s3_b2_c32_32': (2, 32, 32, 32, 3, 0),
    'forward_s3_b2_c32_24': (2, 32, 24, 24, 3, 0),
}


class FlashScaled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x0, k, l, scale, ext):
        ctx.scale, ctx.ext = scale, ext
        ctx.save_for_backward(x, x0, k, l)
        return ext.scaled_forward(x, x0, k, l.reshape(-1), scale)

    @staticmethod
    def backward(ctx, g):
        x, x0, k, l = ctx.saved_tensors
        gx, gx0, gk, gl = ctx.ext.scaled_backward(x, x0, g.contiguous(), k, l.reshape(-1), ctx.scale)
        return gx, gx0, gk, gl.reshape(l.shape), None, None


def kernel_spectrum(weight, bias, H, W):
    kh, kw = weight.shape[-2:]
    psf = torch.roll(F.pad(weight, (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1))
    return torch.fft.fft2(psf), torch.sigmoid(bias - 9.0) + EPS


def flash(ext, x, x0, weight, bias, scale, pad):
    if scale == 1:
        k, l = kernel_spectrum(weight, bias, x.shape[-2] + 2 * pad, x.shape[-1] + 2 * pad)
        return FlashS1.apply(x, k, l, pad, ext)
    k, l = kernel_spectrum(weight, bias, x0.shape[-2], x0.shape[-1])
    return FlashScaled.apply(x, x0, k, l, scale, ext)


def production(x, x0, weight, bias, scale, pad):
    if pad:
        return torch.ops.converse2d._training_circular_s1(x, weight, bias, pad, EPS)
    return torch.ops.converse2d.forward(x, x if scale == 1 else x0, weight, bias, scale, EPS)


def reference64(x, x0, weight, bias, scale, pad, upstream):
    leaves = [t.detach().double().requires_grad_() for t in (x, x0, weight, bias)]
    xd, x0d, wd, bd = leaves
    if scale == 1:
        xp = F.pad(xd, (pad,) * 4, mode='circular') if pad else xd
        y = converse2d_reference(xp, xp, wd, bd, 1, EPS)
        y = y[..., pad:y.shape[-2] - pad, pad:y.shape[-1] - pad] if pad else y
        wrt = (xd, wd, bd)
    else:
        y = converse2d_reference(xd, x0d, wd, bd, scale, EPS)
        wrt = (xd, x0d, wd, bd)
    return [y.detach(), *torch.autograd.grad(y, wrt, upstream.double())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--cases', nargs='*', default=list(CASES))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; keep earlier evidence')
    args.output.mkdir(parents=True)
    study.load_extension()
    ext = study.build()
    device = torch.cuda.current_device()
    report = dict(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  torch=torch.__version__, cuda=torch.version.cuda, smem_optin_bytes=ext.smem_capacity(device),
                  started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cases={})
    print(f"{report['device']}: opt-in shared memory per block {report['smem_optin_bytes']} bytes", flush=True)
    for name in args.cases:
        b, c, h, w, scale, pad = CASES[name]
        torch.manual_seed(74001)
        x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
        x0 = torch.randn(b, c, h * scale, w * scale, device='cuda', requires_grad=scale > 1)
        weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
        bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
        upstream = torch.randn(b, c, h * scale, w * scale, device='cuda')
        inputs = (x, weight, bias) if scale == 1 else (x, x0, weight, bias)
        labels = ['output', 'grad_x'] + ([] if scale == 1 else ['grad_x0']) + ['grad_weight', 'grad_bias']
        supported = ext.supported(h, w, scale, pad, device)
        row = dict(shape=[b, c, h, w], scale=scale, pad=pad, path='fused' if supported else 'production',
                   plane_bytes=(scale * (h + 2 * pad)) ** 2 * 8)
        report['cases'][name] = row
        print(f"{name:28s} plane {row['plane_bytes']:7d} B -> {row['path']}", flush=True)

        def run(fn):
            out = fn(x, x0, weight, bias, scale, pad)
            return [out.detach(), *torch.autograd.grad(out, inputs, upstream)]

        flash_fn = lambda *a: flash(ext, *a)
        if supported:
            try:
                candidate, baseline = run(flash_fn), run(production)
                ref = reference64(x, x0, weight, bias, scale, pad, upstream)
                row['accuracy'] = {}
                for label, cand, base, r in zip(labels, candidate, baseline, ref):
                    acc = row['accuracy'][label] = comparison(cand, base, r)
                    print(f"{'':28s} {label:11s} {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 "
                          f"{acc['ratio']['rel_l2'] or 0:.2f}x max_abs {acc['ratio']['max_abs'] or 0:.2f}x", flush=True)
                again = run(flash_fn)
                row['bitwise_repeatable'] = all(torch.equal(a.view(torch.int32), z.view(torch.int32))
                                                for a, z in zip(candidate, again))
            except Exception as error:  # keep failures in the report
                row['error'] = repr(error)
                print(f"{'':28s} ERROR {error!r}", flush=True)
                continue
            row['timing'] = paired({'production': lambda: run(production), 'flash': lambda: run(flash_fn)},
                                   args.rounds, args.iters)
        else:
            row['timing'] = paired({'production': lambda: run(production)}, args.rounds, args.iters)
        t = {k: v['median_us'] for k, v in row['timing'].items()}
        if 'flash' in t:
            row['speedup'] = t['production'] / t['flash']
            print(f"{'':28s} fwd+vjp production {t['production']:8.1f}us | flash {t['flash']:8.1f}us "
                  f"({row['speedup']:.2f}x) repeatable {row['bitwise_repeatable']}", flush=True)
        else:
            print(f"{'':28s} fwd+vjp production {t['production']:8.1f}us (fused not eligible)", flush=True)
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / 'flash.json').write_text(json.dumps(report, indent=1))
    lines = [f"# Fused SRAM training: {report['device']} (sm_{''.join(map(str, report['capability']))})", '',
             f"- torch {report['torch']} / CUDA {report['cuda']}; opt-in shared memory per block "
             f"{report['smem_optin_bytes']} B", '',
             '| case | plane | path | production us | flash us | speedup | worst rel-L2 ratio | budgets | repeatable |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, row in report['cases'].items():
        t = {k: v['median_us'] for k, v in row.get('timing', {}).items()}
        acc = row.get('accuracy', {})
        worst = max((a['ratio']['rel_l2'] or 0 for a in acc.values()), default=None)
        budgets = ('pass' if all(a['passed'] for a in acc.values()) else 'FAIL') if acc else row.get('error', '-')
        lines.append(f"| {name} | {row['plane_bytes'] // 1024} KB | {row['path']} | {t.get('production', 0):.0f} | "
                     f"{t['flash']:.0f} | {row['speedup']:.2f}x | {worst:.2f} | {budgets} | {row.get('bitwise_repeatable')} |"
                     if 'flash' in t else
                     f"| {name} | {row['plane_bytes'] // 1024} KB | {row['path']} | {t.get('production', 0):.0f} | - | - | - | "
                     f"{budgets} | - |")
    lines += ['', 'Operator-level forward+VJP; not a whole-model or convergence result.']
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
