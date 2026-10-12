"""Isolated v4 experiment: direct FP32 GEMM for 1x1 weight VJP only.

Import has no effect on production. Layout materialization belongs to backward
and is included in complete-layer timings. No spectrum, activation, or matrix
cache is retained. The entire higher-order VJP remains native ATen.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import types

import torch
import torch.nn.functional as F


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fp32_policy():
    if (torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32):
        raise RuntimeError('Direct FP32 GEMM requires autocast and both TF32 flags disabled')


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def eligible(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return (x.is_cuda and x.dtype == torch.float32 and weight.dtype == torch.float32
            and weight.device == x.device and x.ndim == weight.ndim == 4
            and not x.is_neg() and not weight.is_neg() and (bias is None or not bias.is_neg())
            and x.layout == weight.layout == torch.strided
            and (bias is None or (bias.dtype == torch.float32 and bias.device == x.device
                                 and bias.layout == torch.strided and bias.shape == (weight.shape[0],)))
            and weight.shape[-2:] == (1, 1) and x.shape[1] == weight.shape[1]
            and (weight.shape[1], weight.shape[0]) in ((64, 128), (128, 64))
            and _pair(stride) == (1, 1) and _pair(padding) == (0, 0)
            and _pair(dilation) == (1, 1) and groups == 1
            and x.shape[0] > 0 and x.shape[2] > 0 and x.shape[3] > 0
            and x.shape[0] * x.shape[2] * x.shape[3] >= 4096
            and torch.is_grad_enabled() and weight.requires_grad)


class PointwiseWeightGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.bias_sizes = [weight.shape[0]] if bias is not None else None
        ctx.set_materialize_grads(False)
        return F.conv2d(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        fp32_policy()
        x, weight = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if torch.is_grad_enabled():
            return torch.ops.aten.convolution_backward.default(
                grad_output, x, weight, ctx.bias_sizes, [1, 1], [0, 0], [1, 1],
                False, [0, 0], 1, list(needs))
        dx = dw = db = None
        if needs[0] or needs[2]:
            # Preserve the native bias reduction and original-layout dx in the
            # same native call. Do not replace db with Tensor.sum.
            dx, _, db = torch.ops.aten.convolution_backward.default(
                grad_output, x, weight, ctx.bias_sizes, [1, 1], [0, 0], [1, 1],
                False, [0, 0], 1, [needs[0], False, needs[2]])
        if needs[1]:
            output_matrix = grad_output.permute(1, 0, 2, 3).reshape(weight.shape[0], -1)
            input_matrix = x.permute(1, 0, 2, 3).reshape(weight.shape[1], -1)
            dw = (output_matrix @ input_matrix.t()).reshape_as(weight)
        return dx, dw, db


def conv1x1(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if any(t is not None and t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError('FP32 tensors required; use a separate FP64 test reference')
    fp32_policy()
    if eligible(x, weight, bias, stride, padding, dilation, groups):
        return PointwiseWeightGradient.apply(x, weight, bias)
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


@contextmanager
def patch_modules(model, allowed_names):
    """Explicit local experiment scope; the harness verifies gate provenance."""
    fp32_policy()
    previous = []
    try:
        for name, module in model.named_modules():
            if name not in allowed_names:
                continue
            if not (isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1)
                    and module.stride == (1, 1) and module.padding == (0, 0)
                    and module.dilation == (1, 1) and module.groups == 1
                    and module.padding_mode == 'zeros'):
                raise ValueError('Nonqualifying module in admission: ' + name)
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
