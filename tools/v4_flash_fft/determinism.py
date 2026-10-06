"""Two identical FlashS1 forward+VJP calls must agree bit for bit."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import torch
import study
import train_study

study.load_extension()
ext = study.build()
for name, (b, c, h, w, pad) in study.CASES.items():
    torch.manual_seed(73001)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(b, c, h, w, device='cuda')

    def run():
        out = train_study.flash(ext, x, weight, bias, pad)
        return [out.detach(), *torch.autograd.grad(out, (x, weight, bias), upstream)]

    first, second = run(), run()
    same = all(torch.equal(a.view(torch.int32), b_.view(torch.int32)) for a, b_ in zip(first, second))
    print(f'{name:28s} bitwise repeatable: {same}')
