"""Per-kernel CUDA time per forward+VJP call: production vs FlashS1."""
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
from torch.profiler import ProfilerActivity, profile
import study
import train_study

study.load_extension()
ext = study.build()
for name, (b, c, h, w, pad) in study.CASES.items():
    torch.manual_seed(72001)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(b, c, h, w, device='cuda')
    for label, fn in (('flash', lambda *a: train_study.flash(ext, *a)), ('production', study.production)):
        def run():
            return torch.autograd.grad(fn(x, weight, bias, pad), (x, weight, bias), upstream)
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                run()
            torch.cuda.synchronize()
        times = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time > 0:
                times[e.name[:80]] += e.device_time / 10
        print(f'{name} {label}: kernels {sum(times.values()):.1f}us')
        for k, v in times.most_common(5):
            print(f'    {v:8.1f}  {k}')
