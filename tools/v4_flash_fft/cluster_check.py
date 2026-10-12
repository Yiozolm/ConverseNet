"""Two-CTA cluster kernels against the single-CTA fused kernels, on the s2/s3 planes that
fit one SM: output, grad_x0 and grad_k are expected bitwise identical (every FFT line and
every alias group is computed exactly as before); grad_x and grad_bias go through a second
two-term reduction and are reported as relative differences. Then the planes that do not fit
one SM (s2 64 -> 128, s3 48 -> 144) run in clusters against production and the FP64 reference
with numerical_policy.comparison. Research only.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import study
from study import comparison
import flash_study as fs

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f'{args.output} exists; keep earlier evidence')
study.load_extension()
ext = study.build()
device = torch.cuda.current_device()
report = dict(device=torch.cuda.get_device_name(), smem_optin_bytes=ext.smem_capacity(device), cross={}, large={})


def run(case, mode, seed=78001):
    b, c, h, w, scale, pad = fs.CASES[case]
    torch.manual_seed(seed)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    x0 = torch.randn(b, c, h * scale, w * scale, device='cuda', requires_grad=True)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(b, c, h * scale, w * scale, device='cuda')
    ext.set_cluster_mode(mode)
    out = fs.flash(ext, x, x0, weight, bias, scale, pad)
    grads = torch.autograd.grad(out, (x, x0, weight, bias), upstream)
    return [out.detach(), *grads], (x, x0, weight, bias, upstream)


labels = ['output', 'grad_x', 'grad_x0', 'grad_weight', 'grad_bias']
for case in ('forward_s2_b4_c64_48', 'forward_s2_b4_c64_32', 'forward_s3_b2_c32_32', 'forward_s3_b2_c32_24'):
    single, _ = run(case, 2)
    cluster, _ = run(case, 1)
    row = report['cross'][case] = {}
    for label, a, z in zip(labels, single, cluster):
        same = bool(torch.equal(a.view(torch.int32), z.view(torch.int32)))
        rel = float((a.double() - z.double()).norm() / a.double().norm().clamp_min(1e-30))
        row[label] = dict(bitwise=same, rel_l2_diff=rel)
    def fmt(label):
        r = row[label]
        return f"{label} {'bitwise' if r['bitwise'] else format(r['rel_l2_diff'], '.1e')}"
    print(f"{case:28s} " + ' '.join(fmt(l) for l in labels), flush=True)

ext.set_cluster_mode(0)
for case in ('forward_s2_b4_c64_64', 'forward_s3_b2_c32_48'):
    b, c, h, w, scale, pad = fs.CASES[case]
    supported = ext.supported(h, w, scale, pad, device)
    row = report['large'][case] = dict(supported=supported)
    print(f'{case:28s} supported in clusters: {supported}', flush=True)
    if not supported:
        continue
    candidate, (x, x0, weight, bias, upstream) = run(case, 0)
    baseline = [t.detach() for t in [fs.production(x, x0, weight, bias, scale, pad)]]
    baseline += list(torch.autograd.grad(fs.production(x, x0, weight, bias, scale, pad), (x, x0, weight, bias), upstream))
    ref = fs.reference64(x, x0, weight, bias, scale, pad, upstream)
    row['accuracy'] = {}
    for label, cand, base, r in zip(labels, candidate, baseline, ref):
        acc = row['accuracy'][label] = comparison(cand, base, r)
        print(f"{'':28s} {label:11s} {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 {acc['ratio']['rel_l2'] or 0:.2f}x "
              f"max_abs {acc['ratio']['max_abs'] or 0:.2f}x", flush=True)
    again, _ = run(case, 0)
    row['bitwise_repeatable'] = all(torch.equal(a.view(torch.int32), z.view(torch.int32)) for a, z in zip(candidate, again))
    print(f"{'':28s} repeatable {row['bitwise_repeatable']}", flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=1))
