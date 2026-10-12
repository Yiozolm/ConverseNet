"""Restricted follow-up to the failed two-direction v1 GEMM experiment.

Only the structural C128->64 direction may use GEMM. All C64->128 calls retain
native convolution; the original v1 failures remain recorded and unchanged.
The arithmetic implementation is the original v1 class, with native dx/dbias
and complete native higher-order fallback.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import types

import torch
import torch.nn.functional as F
import pointwise_wgrad as base

PointwiseWeightGradient = base.PointwiseWeightGradient
fp32_policy = base.fp32_policy


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def eligible(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return (base.eligible(x, weight, bias, stride, padding, dilation, groups)
            and (weight.shape[1], weight.shape[0]) == (128, 64))


def conv1x1(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if any(t is not None and t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError('FP32 tensors required; use a separate FP64 test reference')
    fp32_policy()
    if eligible(x, weight, bias, stride, padding, dilation, groups):
        return PointwiseWeightGradient.apply(x, weight, bias)
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


@contextmanager
def patch_modules(model, allowed_names):
    fp32_policy()
    previous = []
    try:
        for name, module in model.named_modules():
            if name not in allowed_names:
                continue
            if not (isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1)
                    and module.stride == (1, 1) and module.padding == (0, 0)
                    and module.dilation == (1, 1) and module.groups == 1
                    and module.padding_mode == 'zeros'
                    and (module.in_channels, module.out_channels) == (128, 64)):
                raise ValueError('Nonqualifying module in C128->64 admission: ' + name)
            previous.append((module, 'forward' in vars(module), vars(module).get('forward')))
            def forward(this, value):
                return conv1x1(value, this.weight, this.bias, this.stride,
                               this.padding, this.dilation, this.groups)
            module.forward = types.MethodType(forward, module)
        if len(previous) != len(allowed_names):
            raise ValueError('Admission includes unknown model module')
        yield model
    finally:
        for module, had_override, previous_forward in previous:
            if had_override:
                module.forward = previous_forward
            else:
                del module.forward
