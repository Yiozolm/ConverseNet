"""Accuracy-only sweep over seeds: is a budget failure systematic or seed-specific?"""
import argparse
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import torch
import study
import flash_study as fs

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seeds', type=int, default=8)
parser.add_argument('--cases', nargs='*', default=['forward_s2_b4_c64_32', 'forward_s2_b4_c64_48',
                                                   'forward_s3_b2_c32_32', 'forward_s3_b2_c32_24'])
args = parser.parse_args()
study.load_extension()
ext = study.build()
device = torch.cuda.current_device()
rows = []
for name in args.cases:
    b, c, h, w, scale, pad = fs.CASES[name]
    if not ext.supported(h, w, scale, pad, device):
        print(f'{name}: not eligible here')
        continue
    for seed in range(args.seeds):
        torch.manual_seed(75000 + seed)
        x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
        x0 = torch.randn(b, c, h * scale, w * scale, device='cuda', requires_grad=scale > 1)
        weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
        bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
        upstream = torch.randn(b, c, h * scale, w * scale, device='cuda')
        inputs = (x, weight, bias) if scale == 1 else (x, x0, weight, bias)
        labels = ['output', 'grad_x'] + ([] if scale == 1 else ['grad_x0']) + ['grad_weight', 'grad_bias']

        def run(fn):
            out = fn(x, x0, weight, bias, scale, pad)
            return [out.detach(), *torch.autograd.grad(out, inputs, upstream)]

        cand, base = run(lambda *a: fs.flash(ext, *a)), run(fs.production)
        ref = fs.reference64(x, x0, weight, bias, scale, pad, upstream)
        row = dict(case=name, seed=seed, results={})
        for label, cv, bv, rv in zip(labels, cand, base, ref):
            row['results'][label] = fs.comparison(cv, bv, rv)
        rows.append(row)
        failed = [k for k, v in row['results'].items() if not v['passed']]
        worst = {k: round(max(v['ratio']['rel_l2'] or 0, v['ratio']['max_abs'] or 0), 2) for k, v in row['results'].items()}
        print(f"{name:24s} seed {seed}: {'FAIL ' + ','.join(failed) if failed else 'pass'} worst ratio {worst}", flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise SystemExit(f'{args.output} exists')
args.output.write_text(json.dumps(rows, indent=1))
