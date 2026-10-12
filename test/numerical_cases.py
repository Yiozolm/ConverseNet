"""Reproducible release matrix; values originate in FP32 for all three runs."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class NumericalCase:
    scale: int
    kb: int = 1
    kc: int = 3
    layout: str = 'contiguous'
    distribution: str = 'normal'
    height: int = 5
    width: int = 7

    @property
    def name(self):
        return f's{self.scale}-k{self.kb}x{self.kc}-{self.layout}-{self.distribution}-{self.height}x{self.width}'

    @property
    def weak(self):
        return self.distribution.startswith('weak_')

    @property
    def eps(self):
        return 1e-8 if self.weak else 1e-5

    def fixture(self):
        s, h, w = self.scale, self.height, self.width
        g = torch.Generator().manual_seed(96013 + s + self.kb * 31 + self.kc * 7 + h + w)
        kh, kw = min(3, h*s), min(3, w*s)
        raw = [torch.randn(2, 3, h, w, generator=g),
               torch.randn(2, 3, h*s, w*s, generator=g),
               torch.rand(self.kb, self.kc, kh, kw, generator=g) / (kh*kw),
               torch.randn(1, 3, 1, 1, generator=g)]
        up = torch.randn(raw[1].shape, generator=g)
        kind = self.distribution
        if self.weak:
            raw[0].mul_(1e-5)
            raw[1].mul_(1e-5)
            raw[2].mul_(float(kind.removeprefix('weak_')))
            raw[3].fill_(-40)
            up.mul_(1e-5)
        elif kind == 'softmax':
            raw[2] = torch.randn(self.kb, self.kc, kh*kw, generator=g).softmax(-1).reshape_as(raw[2])
        elif kind == 'dynamic':
            raw[0].mul_(torch.logspace(-3, 3, w))
            raw[1].mul_(torch.logspace(3, -3, w*s))
        elif kind == 'near_zero_input':
            raw[0].mul_(1e-8)
            raw[1].mul_(1e-8)
        elif kind == 'near_zero_kernel':
            raw[2].mul_(1e-6)
        elif kind == 'cancellation':
            raw[2].zero_()
            raw[2][..., kh//2, kw//2] = 1
            raw[1] = raw[0].repeat_interleave(s, -2).repeat_interleave(s, -1) + raw[1] * 1e-6
        return raw, up


def numerical_cases(level='full'):
    if level not in ('fast', 'full'):
        raise ValueError('CONVERSE2D_NUMERICAL_LEVEL must be fast or full')
    for s in (1, 2, 3, 4):
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            for layout in (('contiguous',) if level == 'fast' else ('contiguous', 'strided', 'transpose')):
                for dist in (('normal',) if level == 'fast' else ('normal', 'weak_0', 'weak_1e-6', 'weak_1e-3')):
                    yield NumericalCase(s, kb, kc, layout, dist)
        if level == 'full':
            for dist in ('softmax', 'dynamic', 'near_zero_input', 'near_zero_kernel', 'cancellation'):
                yield NumericalCase(s, distribution=dist)
            yield NumericalCase(s, layout='strided', width=1)
            yield NumericalCase(s, layout='transpose', height=7, width=6)


def evaluate(fn, raw, up, case, *, device, dtype=torch.float32, mode='training'):
    from support import leaves
    training = mode == 'training'
    data = leaves([v.to(dtype) for v in raw], device,
                  strided=case.layout == 'strided', transpose=case.layout == 'transpose',
                  needs=(training or mode != 'frozen',) * 4)
    if training:
        out = fn(*data, case.scale, case.eps)
        return (out, *torch.autograd.grad(out, data, up.to(device=device, dtype=dtype)))
    context = {'no_grad': torch.no_grad, 'inference_mode': torch.inference_mode,
               'frozen': torch.enable_grad}[mode]
    with context():
        return (fn(*data, case.scale, case.eps),)
