"""Is the fused s2/s3 kernel/regularizer-gradient error a systematic excess or gate noise?

Per seed, grad_weight and grad_bias of three FP32 paths are compared with the FP64
reference: the fused kernels, production, and a control that runs production on the
transposed problem (x, x0, weight and the upstream gradient transposed, the result
transposed back). The control is mathematically identical to production and equally
accurate, but cuFFT and the reductions round differently: it is a second draw of
production's own noise. The single-run gate (rel-L2 ratio <= 1.25) is then applied to
fused/production and to control/production, and three statistics are reported per case
and output with bootstrap intervals over seeds:
- the single-run failure rate,
- the geometric-mean ratio (the seeded gate's statistic),
- the pooled ratio sqrt(sum ||err||^2) / sqrt(sum ||err_prod||^2) over all seeds, which has
  many more degrees of freedom than one seed's few effective channels.
A fused pooled ratio inside the control's interval means no systematic excess. Research only.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import study
import flash_study as fs
import ops_study as ops

CASES = ['forward_s2_b4_c64_32', 'forward_s2_b4_c64_48', 'forward_s3_b2_c32_32', 'forward_s3_b2_c32_24',
         'data_s2_b4_c64_48_k7', 'data_s3_b4_c64_32_k7']


def transposed(t):
    return t.detach().transpose(-1, -2).contiguous().requires_grad_(t.requires_grad)


def one_seed(name, ext, seed):
    """(weight, bias) gradients of fused, production, control and the FP64 reference."""
    if name in fs.CASES:
        b, c, h, w, s, pad = fs.CASES[name]
        torch.manual_seed(75000 + seed)  # seed_sweep.py inputs
        x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
        x0 = torch.randn(b, c, h * s, w * s, device='cuda', requires_grad=s > 1)
        weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
        bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
        g = torch.randn(b, c, h * s, w * s, device='cuda')

        def grads(fn, xx, xx0, ww, gg):
            return torch.autograd.grad(fn(xx, xx0, ww, bias, s, pad), (ww, bias), gg)

        fused = grads(lambda *a: fs.flash(ext, *a), x, x0, weight, g)
        fused_reg = grads(lambda *a: fs.flash_reg(ext, *a), x, x0, weight, g)
        prod = grads(fs.production, x, x0, weight, g)
        gw, gb = grads(fs.production, transposed(x), transposed(x0), transposed(weight), g.transpose(-1, -2).contiguous())
        ref = fs.reference64(x, x0, weight, bias, s, pad, g)[-2:]
    else:
        case = ops.CASES[name]
        x, weight, bias, g = ops.make_inputs(case, 75000 + seed)

        def grads(fn, xx, ww, gg):
            return torch.autograd.grad(fn(case, xx, ww, bias), (ww, bias), gg)

        fused = grads(lambda cs, xx, ww, bb: ops.flash(cs, ext, xx, ww, bb), x, weight, g)
        fused_reg = grads(lambda cs, xx, ww, bb: ops.flash_reg(cs, ext, xx, ww, bb), x, weight, g)
        prod = grads(ops.production, x, weight, g)
        gw, gb = grads(ops.production, transposed(x), transposed(weight), g.transpose(-1, -2).contiguous())
        ref = ops.reference64(case, x, weight, bias, g)[-2:]
    ctrl = (gw.transpose(-1, -2), gb)
    return dict(fused=fused, fused_reg=fused_reg, prod=prod, ctrl=ctrl), ref


def regularizer_check(ext):
    """Two-term device regularizer against FP64: relative errors of l and dl/dbias."""
    torch.manual_seed(1)
    bias = torch.randn(4096, device='cuda') * 3
    reg = ext.regularizer(bias, 1e-5).double()
    sigma = torch.sigmoid(bias.double() - 9)
    l, ds = sigma + 1e-5, sigma * (1 - sigma)
    err_l = ((reg[:, 0] + reg[:, 1] - l) / l).abs().max().item()
    err_ds = ((reg[:, 2] + reg[:, 3] - ds) / ds).abs().max().item()
    fp32 = torch.sigmoid(bias - 9.0).double()
    err_fp32 = ((fp32 - sigma) / sigma).abs().max().item()
    print(f'regularizer two-term: max rel error l {err_l:.1e}, dl/dbias {err_ds:.1e} (FP32 sigmoid: {err_fp32:.1e}), '
          f'bias in [{bias.min().item():.1f}, {bias.max().item():.1f}]', flush=True)
    return dict(l=err_l, ds=err_ds, fp32_sigmoid=err_fp32)


def rel_sq(a, ref):
    e = (a.double() - ref)
    return float((e * e).sum()), float((ref * ref).sum())


def bootstrap(values, statistic, n=2000, seed=0):
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(statistic(sample))
    draws.sort()
    return [draws[int(0.025 * n)], draws[int(0.975 * n)]]


def summarize(rows):
    def geo(v):
        return math.exp(sum(math.log(x) for x in v) / len(v))

    def pooled(v, key):
        return math.sqrt(sum(r[key] for r in v) / sum(r['prod'] for r in v))

    out = []
    for name in dict.fromkeys(r['case'] for r in rows):
        for q in ('bias', 'weight'):
            rs = [r['sq'][q] for r in rows if r['case'] == name]
            entry = dict(case=name, output=q, seeds=len(rs))
            for path in ('fused', 'fused_reg', 'ctrl'):
                ratios = [math.sqrt(r[path] / r['prod']) for r in rs]
                entry[path] = dict(
                    fail_rate=sum(x > 1.25 for x in ratios) / len(ratios),
                    fail_rate_ci=bootstrap(ratios, lambda v: sum(x > 1.25 for x in v) / len(v)),
                    geomean=geo(ratios), geomean_ci=bootstrap(ratios, geo),
                    pooled=pooled(rs, path), pooled_ci=bootstrap(rs, lambda v: pooled(v, path)),
                    median=sorted(ratios)[len(ratios) // 2])
            out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=64)
    parser.add_argument('--cases', nargs='*', default=CASES)
    parser.add_argument('--summarize', type=Path, help='re-evaluate an existing run without a GPU')
    args = parser.parse_args()
    if args.summarize:
        report = json.loads(args.summarize.read_text())
        print(render(summarize(report['rows']), report))
        return
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; keep earlier evidence')
    args.output.mkdir(parents=True)
    study.load_extension()
    ext = study.build()
    device = torch.cuda.current_device()
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, seeds=args.seeds,
                  started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), rows=[],
                  regularizer=regularizer_check(ext))
    for name in args.cases:
        spec = fs.CASES[name] if name in fs.CASES else ops.CASES[name]
        h, w, s, pad = (spec[2], spec[3], spec[4], spec[5]) if name in fs.CASES else (spec.h, spec.w, spec.scale, spec.pad)
        if not ext.supported(h, w, s, pad, device):
            print(f'{name}: not eligible here', flush=True)
            continue
        for seed in range(args.seeds):
            paths, ref = one_seed(name, ext, seed)
            row = dict(case=name, seed=seed, sq={}, rel={})
            for qi, q in enumerate(('weight', 'bias')):
                sq = {path: rel_sq(paths[path][qi], ref[qi])[0] for path in paths}
                sq['ref'] = rel_sq(paths['prod'][qi], ref[qi])[1]
                row['sq'][q] = sq
                row['rel'][q] = {path: math.sqrt(sq[path] / sq['ref']) for path in paths}
            report['rows'].append(row)
            rb, rw = row['rel']['bias'], row['rel']['weight']
            print(f"{name:22s} seed {seed:2d}: bias fused/prod {rb['fused'] / rb['prod']:.2f} reg {rb['fused_reg'] / rb['prod']:.2f} "
                  f"ctrl/prod {rb['ctrl'] / rb['prod']:.2f} | weight fused/prod {rw['fused'] / rw['prod']:.2f} "
                  f"reg {rw['fused_reg'] / rw['prod']:.2f} ctrl/prod {rw['ctrl'] / rw['prod']:.2f}", flush=True)
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    report['summary'] = summarize(report['rows'])
    (args.output / 'check.json').write_text(json.dumps(report, indent=1))
    text = render(report['summary'], report)
    (args.output / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


def render(summary, report):
    lines = [f"# s2/s3 kernel and regularizer gradients: fused and control against production "
             f"({report['device']}, {report['seeds']} seeds)", '',
             '| case | output | path | single-run failures | geomean ratio [95% CI] | pooled ratio [95% CI] | median |',
             '|---|---|---|---|---|---|---|']
    for e in summary:
        for path, label in (('fused', 'fused'), ('fused_reg', 'fused + device regularizer'), ('ctrl', 'control')):
            v = e.get(path)
            if v is None:
                continue
            lines.append(f"| {e['case']} | {e['output']} | {label} | {v['fail_rate'] * 100:.0f}% "
                         f"[{v['fail_rate_ci'][0] * 100:.0f}, {v['fail_rate_ci'][1] * 100:.0f}] | "
                         f"{v['geomean']:.2f} [{v['geomean_ci'][0]:.2f}, {v['geomean_ci'][1]:.2f}] | "
                         f"{v['pooled']:.2f} [{v['pooled_ci'][0]:.2f}, {v['pooled_ci'][1]:.2f}] | {v['median']:.2f} |")
    lines += ['', 'Ratios are rel-L2 against the FP64 reference, divided by production\'s. The control is production on the',
              'transposed problem: identical arithmetic in exact terms, different rounding. Research only.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
