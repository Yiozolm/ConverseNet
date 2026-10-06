"""Save, or compare bit for bit, fused forward+VJP results for every eligible case.

Used to show that a scheduling change (e.g. launch geometry) keeps the arithmetic
identical: capture before the change, then run with --compare after it.
"""
import argparse
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import torch
import study
import flash_study as fs

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--compare', type=Path)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f'{args.output} exists')
study.load_extension()
ext = study.build()
device = torch.cuda.current_device()
results = {}
for name, (b, c, h, w, scale, pad) in fs.CASES.items():
    if not ext.supported(h, w, scale, pad, device):
        continue
    torch.manual_seed(77001)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    x0 = torch.randn(b, c, h * scale, w * scale, device='cuda', requires_grad=scale > 1)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(b, c, h * scale, w * scale, device='cuda')
    inputs = (x, weight, bias) if scale == 1 else (x, x0, weight, bias)
    out = fs.flash(ext, x, x0, weight, bias, scale, pad)
    results[name] = [t.detach().cpu() for t in (out, *torch.autograd.grad(out, inputs, upstream))]
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(results, args.output)
if args.compare:
    before = torch.load(args.compare)
    mismatched = []
    for name, tensors in results.items():
        same = name in before and all(torch.equal(a.view(torch.int32), z.view(torch.int32))
                                      for a, z in zip(tensors, before[name]))
        print(f'{name:28s} {"identical" if same else "DIFFERENT"}')
        if not same:
            mismatched.append(name)
    print(f'{len(results) - len(mismatched)}/{len(results)} cases bitwise identical')
    if mismatched:
        raise SystemExit(1)
