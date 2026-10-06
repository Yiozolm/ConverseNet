"""Fused SRAM training on the other operations the models run.

flash_study.py covers the shared-kernel circular s1 solve and the s2/s3 solve
with an independent prior. The models also run:
- the USRNet data term (ConvReverseDataNet): one 7x7 kernel per (b, c) plane
  (weight batch B), eps 1e-3, no padding, with the nearest-upsampled prior
  built from x. The recipe (--scale 3, patch 96) runs it once at s3 (32 -> 96)
  and four times at s1 (96x96) per forward;
- Converse2D's other padding modes at s1: replicate (ConverseMSRResNet, kernel
  5, pad 4), reflect and zeros, next to the circular control.

Training mode: forward+VJP for (x, weight, bias), each gated by
numerical_policy.comparison (production FP32 baseline, FP64 autograd through
models.converse_core.converse2d_reference), a bitwise repeat check and paired
CUDA-event timing. --seeds N runs the seeded gate over N seeds instead.
--inference compares the fused forward under no_grad with production's
half-spectrum inference forward (output only). Research only.
"""
import argparse
from collections import namedtuple
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import torch.nn.functional as F
import study
from study import comparison, converse2d_reference, paired
from flash_study import FlashScaled, FlashScaledReg

Case = namedtuple('Case', 'b c h w scale pad mode kb k eps prior')
MODES = {'circular': 0, 'replicate': 1, 'reflect': 2, 'constant': 3}
CASES = {
    'data_s1_b4_c64_96_k7': Case(4, 64, 96, 96, 1, 0, 'circular', 4, 7, 1e-3, 'shared'),
    'data_s3_b4_c64_32_k7': Case(4, 64, 32, 32, 3, 0, 'circular', 4, 7, 1e-3, 'nearest'),
    'data_s2_b4_c64_48_k7': Case(4, 64, 48, 48, 2, 0, 'circular', 4, 7, 1e-3, 'nearest'),
    'replicate_s1_b4_c64_92_pad4_k5': Case(4, 64, 92, 92, 1, 4, 'replicate', 1, 5, 1e-5, 'shared'),
    'reflect_s1_b4_c64_96_pad2': Case(4, 64, 96, 96, 1, 2, 'reflect', 1, 3, 1e-5, 'shared'),
    'zeros_s1_b4_c64_96_pad2': Case(4, 64, 96, 96, 1, 2, 'constant', 1, 3, 1e-5, 'shared'),
    'circular_s1_b4_c64_96_pad2': Case(4, 64, 96, 96, 1, 2, 'circular', 1, 3, 1e-5, 'shared'),
    'circular_s1_b4_c128_96_pad2': Case(4, 128, 96, 96, 1, 2, 'circular', 1, 3, 1e-5, 'shared'),
}
LABELS = ['output', 'grad_x', 'grad_weight', 'grad_bias']


class FlashS1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, l, pad, mode, ext):
        ctx.pad, ctx.mode, ctx.ext = pad, mode, ext
        ctx.save_for_backward(x, k, l)
        return ext.forward(x, k, l.reshape(-1), pad, 1, mode)

    @staticmethod
    def backward(ctx, g):
        x, k, l = ctx.saved_tensors
        gx, gk, gl = ctx.ext.backward(x, g.contiguous(), k, l.reshape(-1), ctx.pad, ctx.mode)
        return gx, gk, gl.reshape(l.shape), None, None, None


def make_inputs(case, seed):
    torch.manual_seed(seed)
    x = torch.randn(case.b, case.c, case.h, case.w, device='cuda', requires_grad=True)
    weight = torch.rand(case.kb, case.c, case.k, case.k, device='cuda', requires_grad=True)
    bias = torch.randn(1, case.c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(case.b, case.c, case.h * case.scale, case.w * case.scale, device='cuda')
    return x, weight, bias, upstream


def padded(case, x):
    return F.pad(x, (case.pad,) * 4, mode=case.mode) if case.pad else x


def prior(case, xp):
    return xp if case.scale == 1 else F.interpolate(xp, scale_factor=case.scale, mode='nearest')


def crop(case, out):
    m = case.pad * case.scale
    return out[..., m:out.shape[-2] - m, m:out.shape[-1] - m] if m else out


def kernel_spectrum(weight, bias, H, W, eps):
    kh, kw = weight.shape[-2:]
    psf = torch.roll(F.pad(weight, (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1))
    return torch.fft.fft2(psf).contiguous(), torch.sigmoid(bias - 9.0) + eps


def flash(case, ext, x, weight, bias):
    if case.scale == 1:
        k, l = kernel_spectrum(weight, bias, case.h + 2 * case.pad, case.w + 2 * case.pad, case.eps)
        return FlashS1.apply(x, k, l, case.pad, MODES[case.mode], ext)
    x0 = prior(case, x)
    k, l = kernel_spectrum(weight, bias, x0.shape[-2], x0.shape[-1], case.eps)
    return FlashScaled.apply(x, x0, k, l, case.scale, ext)


def flash_reg(case, ext, x, weight, bias):
    """flash() with the device-side two-term regularizer (s2/s3); s1 unchanged."""
    if case.scale == 1:
        return flash(case, ext, x, weight, bias)
    x0 = prior(case, x)
    k, _ = kernel_spectrum(weight, bias, x0.shape[-2], x0.shape[-1], case.eps)
    return FlashScaledReg.apply(x, x0, k, bias, case.eps, case.scale, ext)


def production(case, x, weight, bias):
    """Converse2D / ConvReverseDataNet routing: the circular s1 training entry, else pad -> forward -> crop."""
    if case.scale == 1 and case.pad and case.mode == 'circular' and torch.is_grad_enabled():
        return torch.ops.converse2d._training_circular_s1(x, weight, bias, case.pad, case.eps)
    xp = padded(case, x)
    return crop(case, torch.ops.converse2d.forward(xp, prior(case, xp), weight, bias, case.scale, case.eps))


def reference64(case, x, weight, bias, upstream=None):
    xd, wd, bd = (t.detach().double().requires_grad_(upstream is not None) for t in (x, weight, bias))
    xp = padded(case, xd)
    y = crop(case, converse2d_reference(xp, prior(case, xp), wd, bd, case.scale, case.eps))
    if upstream is None:
        return y
    return [y.detach(), *torch.autograd.grad(y, (xd, wd, bd), upstream.double())]


def supported(ext, case, device):
    return ext.supported(case.h, case.w, case.scale, case.pad, device)


def training_study(ext, names, args, report):
    device = torch.cuda.current_device()
    for name in names:
        case = CASES[name]
        row = dict(case=case._asdict(), path='fused' if supported(ext, case, device) else 'production',
                   plane_bytes=(case.scale * (case.h + 2 * case.pad)) ** 2 * 8)
        report['cases'][name] = row
        print(f"{name:32s} plane {row['plane_bytes']:7d} B -> {row['path']}", flush=True)
        if row['path'] != 'fused':
            continue
        x, weight, bias, upstream = make_inputs(case, 74001)

        def run(fn):
            out = fn(case, x, weight, bias) if fn is production else fn(case, ext, x, weight, bias)
            return [out.detach(), *torch.autograd.grad(out, (x, weight, bias), upstream)]

        try:
            candidate, baseline = run(flash), run(production)
            ref = reference64(case, x, weight, bias, upstream)
            row['accuracy'] = {}
            for label, cand, base, r in zip(LABELS, candidate, baseline, ref):
                acc = row['accuracy'][label] = comparison(cand, base, r)
                print(f"{'':32s} {label:11s} {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 "
                      f"{acc['candidate']['rel_l2']:.3e} (prod {acc['baseline']['rel_l2']:.3e}, "
                      f"{acc['ratio']['rel_l2'] or 0:.2f}x) max_abs {acc['ratio']['max_abs'] or 0:.2f}x", flush=True)
            again = run(flash)
            row['bitwise_repeatable'] = all(torch.equal(a.view(torch.int32), z.view(torch.int32))
                                            for a, z in zip(candidate, again))
        except Exception as error:  # keep failures in the report
            row['error'] = repr(error)
            print(f"{'':32s} ERROR {error!r}", flush=True)
            continue
        row['timing'] = paired({'production': lambda: run(production), 'flash': lambda: run(flash)},
                               args.rounds, args.iters)
        t = {k: v['median_us'] for k, v in row['timing'].items()}
        row['speedup'] = t['production'] / t['flash']
        print(f"{'':32s} fwd+vjp production {t['production']:8.1f}us | flash {t['flash']:8.1f}us "
              f"({row['speedup']:.2f}x) repeatable {row['bitwise_repeatable']}", flush=True)


def seed_sweep(ext, names, args, report):
    from numerical_policy import seeded_comparison
    device = torch.cuda.current_device()
    rows = []
    for name in names:
        case = CASES[name]
        if not supported(ext, case, device):
            print(f'{name}: not eligible here', flush=True)
            continue
        for seed in range(args.seeds):
            x, weight, bias, upstream = make_inputs(case, 75000 + seed)

            def run(fn):
                out = fn(case, x, weight, bias) if fn is production else fn(case, ext, x, weight, bias)
                return [out.detach(), *torch.autograd.grad(out, (x, weight, bias), upstream)]

            cand, base = run(flash), run(production)
            ref = reference64(case, x, weight, bias, upstream)
            row = dict(case=name, seed=seed, results={l: comparison(c, b, r) for l, c, b, r in zip(LABELS, cand, base, ref)})
            rows.append(row)
            failed = [k for k, v in row['results'].items() if not v['passed']]
            worst = {k: round(max(v['ratio']['rel_l2'] or 0, v['ratio']['max_abs'] or 0), 2) for k, v in row['results'].items()}
            print(f"{name:32s} seed {seed}: {'FAIL ' + ','.join(failed) if failed else 'pass'} worst ratio {worst}", flush=True)
    report['seed_rows'] = rows
    verdicts = {}
    for row in rows:
        for label, result in row['results'].items():
            verdicts.setdefault((row['case'], label), []).append(result)
    print('seeded budget (geometric mean over seeds):', flush=True)
    report['seeded'] = []
    for (case, label), results in verdicts.items():
        v = seeded_comparison(results)
        report['seeded'].append(dict(case=case, output=label, **v))
        print(f"  {case:32s} {label:11s} {'pass' if v['passed'] else 'FAIL'} rel-L2 geomean "
              f"{v['rel_l2']['geomean_ratio']:.2f} (max seed {v['rel_l2']['max_seed_ratio']:.2f}) max-abs "
              f"{v['max_abs']['geomean_ratio']:.2f} | single-run failures {v['single_run_failures']}/{v['seeds']}", flush=True)


def inference_study(ext, names, args, report):
    """Fused forward under no_grad against production's half-spectrum inference forward."""
    device = torch.cuda.current_device()
    for name in names:
        case = CASES[name]
        row = dict(case=case._asdict(), path='fused' if supported(ext, case, device) else 'production')
        report['cases'][name] = row
        print(f"{name:32s} -> {row['path']}", flush=True)
        if row['path'] != 'fused':
            continue
        x, weight, bias, _ = make_inputs(case, 74001)
        x, weight, bias = x.detach(), weight.detach(), bias.detach()
        with torch.no_grad():
            if case.scale == 1:
                H, W = case.h + 2 * case.pad, case.w + 2 * case.pad
                k, l = kernel_spectrum(weight, bias, H, W, case.eps)
                l = l.reshape(-1).contiguous()
                solve = lambda: ext.forward(x, k, l, case.pad, 1, MODES[case.mode])
                fused = lambda: ext.forward(x, *kernel_spectrum(weight, bias, H, W, case.eps)[:1],
                                            (torch.sigmoid(bias - 9.0) + case.eps).reshape(-1), case.pad, 1, MODES[case.mode])
            else:
                x0 = prior(case, x)
                k, l = kernel_spectrum(weight, bias, x0.shape[-2], x0.shape[-1], case.eps)
                l = l.reshape(-1).contiguous()
                solve = lambda: ext.scaled_forward(x, x0, k, l, case.scale)

                def fused():
                    x0 = prior(case, x)
                    kk, ll = kernel_spectrum(weight, bias, x0.shape[-2], x0.shape[-1], case.eps)
                    return ext.scaled_forward(x, x0, kk, ll.reshape(-1), case.scale)
            prod = lambda: production(case, x, weight, bias)
            candidate, baseline = fused(), prod()
            ref = reference64(case, x, weight, bias)
            acc = row['accuracy'] = comparison(candidate, baseline, ref)
            print(f"{'':32s} output      {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 {acc['candidate']['rel_l2']:.3e} "
                  f"(prod {acc['baseline']['rel_l2']:.3e}, {acc['ratio']['rel_l2'] or 0:.2f}x) "
                  f"max_abs {acc['ratio']['max_abs'] or 0:.2f}x", flush=True)
            row['timing'] = paired({'production': prod, 'flash': fused, 'flash_solve_only': solve}, args.rounds, args.iters)
        t = {k: v['median_us'] for k, v in row['timing'].items()}
        row['speedup'] = t['production'] / t['flash']
        print(f"{'':32s} fwd production {t['production']:8.1f}us | flash {t['flash']:8.1f}us ({row['speedup']:.2f}x) "
              f"| solve only {t['flash_solve_only']:8.1f}us ({t['production'] / t['flash_solve_only']:.2f}x)", flush=True)


def summary(report, mode):
    lines = [f"# Fused SRAM {mode}: {report['device']} (sm_{''.join(map(str, report['capability']))})", '',
             f"- torch {report['torch']} / CUDA {report['cuda']}; opt-in shared memory per block "
             f"{report['smem_optin_bytes']} B", '']
    if mode == 'seeds':
        lines += ['| case | output | seeded | rel-L2 geomean | max seed | max-abs geomean | single-run failures |',
                  '|---|---|---|---|---|---|---|']
        for v in report['seeded']:
            lines.append(f"| {v['case']} | {v['output']} | {'pass' if v['passed'] else 'FAIL'} | "
                         f"{v['rel_l2']['geomean_ratio']:.2f} | {v['rel_l2']['max_seed_ratio']:.2f} | "
                         f"{v['max_abs']['geomean_ratio']:.2f} | {v['single_run_failures']}/{v['seeds']} |")
    else:
        head = 'fwd+vjp' if mode == 'training' else 'fwd'
        lines += [f'| case | path | production {head} us | flash us | speedup | worst rel-L2 ratio | budgets | repeatable |',
                  '|---|---|---|---|---|---|---|---|']
        for name, row in report['cases'].items():
            t = {k: v['median_us'] for k, v in row.get('timing', {}).items()}
            acc = row.get('accuracy', {})
            if mode == 'inference' and acc:
                acc = {'output': acc}
            worst = max((a['ratio']['rel_l2'] or 0 for a in acc.values()), default=0)
            budgets = ('pass' if all(a['passed'] for a in acc.values()) else 'FAIL') if acc else row.get('error', '-')
            if 'flash' in t:
                extra = f" (solve only {t['flash_solve_only']:.0f})" if 'flash_solve_only' in t else ''
                lines.append(f"| {name} | {row['path']} | {t['production']:.0f} | {t['flash']:.0f}{extra} | "
                             f"{row['speedup']:.2f}x | {worst:.2f} | {budgets} | {row.get('bitwise_repeatable', '-')} |")
            else:
                lines.append(f"| {name} | {row['path']} | - | - | - | - | {budgets} | - |")
    lines += ['', 'Operator-level result; not a whole-model or convergence result.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--cases', nargs='*', default=list(CASES))
    parser.add_argument('--seeds', type=int, default=0, help='run the seeded gate over this many seeds')
    parser.add_argument('--inference', action='store_true', help='forward-only comparison under no_grad')
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
    mode = 'seeds' if args.seeds else ('inference' if args.inference else 'training')
    if mode == 'seeds':
        seed_sweep(ext, args.cases, args, report)
    elif mode == 'inference':
        inference_study(ext, args.cases, args, report)
    else:
        training_study(ext, args.cases, args, report)
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / f'{mode}.json').write_text(json.dumps(report, indent=1))
    text = summary(report, mode)
    (args.output / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
