"""Per-kernel CUDA time of the fused forward and the production forward (one profiled window)."""
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
from torch.profiler import ProfilerActivity, profile
import study

study.load_extension()
ext = study.build()
for name, (b, c, h, w, pad) in study.CASES.items():
    torch.manual_seed(71001)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.zeros(1, c, 1, 1, device='cuda', requires_grad=True)
    for label, fn in (('fused', lambda: study.fused(ext, x.detach(), weight.detach(), bias.detach(), pad, 2)),
                      ('production', lambda: study.production(x, weight, bias, pad))):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
        times = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time > 0:
                times[e.name[:90]] += e.device_time / 20
        print(f'{name} {label}: total {sum(times.values()):.1f}us')
        for k, v in times.most_common(6):
            print(f'    {v:8.1f}  {k}')
