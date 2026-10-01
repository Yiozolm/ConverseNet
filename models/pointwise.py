"""FP32 pointwise weight VJP in the measured B4 C128->64 HR96 domain."""
import os

import torch
import torch.nn.functional as F


def _fp32_inputs(x, weight, bias):
    if any(t is not None and t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError('PointwiseConv2d requires FP32 tensors')


def _fp32_policy():
    if (torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32):
        raise RuntimeError('Direct FP32 GEMM requires autocast and both TF32 flags disabled')


def _eligible(x, weight, bias, module):
    tensors = (x, weight) if bias is None else (x, weight, bias)
    return (torch.is_grad_enabled() and weight.requires_grad and x.is_cuda
            and tuple(x.shape) == (4, 128, 96, 96)
            and tuple(weight.shape) == (64, 128, 1, 1)
            and (bias is None or tuple(bias.shape) == (64,))
            and all(t.device == x.device and t.dtype == torch.float32
                    and t.layout == torch.strided and t.is_contiguous()
                    and not t.is_neg() and not t.is_conj() for t in tensors)
            and module.stride == (1, 1) and module.padding == (0, 0)
            and module.dilation == (1, 1) and module.groups == 1
            and module.padding_mode == 'zeros')


class PointwiseWeightGradient(torch.autograd.Function):
    """Same arithmetic as the independently tested v1/v2 research candidate."""

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
        _fp32_policy()
        x, weight = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if torch.is_grad_enabled():
            return torch.ops.aten.convolution_backward.default(
                grad_output, x, weight, ctx.bias_sizes, [1, 1], [0, 0], [1, 1],
                False, [0, 0], 1, list(needs))
        dx = dw = db = None
        if needs[0] or needs[2]:
            dx, _, db = torch.ops.aten.convolution_backward.default(
                grad_output, x, weight, ctx.bias_sizes, [1, 1], [0, 0], [1, 1],
                False, [0, 0], 1, [needs[0], False, needs[2]])
        if needs[1]:
            output_matrix = grad_output.permute(1, 0, 2, 3).reshape(weight.shape[0], -1)
            input_matrix = x.permute(1, 0, 2, 3).reshape(weight.shape[1], -1)
            dw = (output_matrix @ input_matrix.t()).reshape_as(weight)
        return dx, dw, db


class PointwiseConv2d(torch.nn.Conv2d):
    """Conv2d checkpoint-compatible module with a narrowly measured weight VJP.

    Only B4/C128->64/96x96 contiguous CUDA FP32 tensors with differentiable
    weights select GEMM. Native convolution handles every other shape/layout,
    inference and frozen weights. The pytorch backend always remains native.
    """

    def __init__(self, *args, backend='auto', **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(backend, str) or backend.lower() not in ('auto', 'cuda', 'pytorch'):
            raise ValueError('backend must be auto, cuda or pytorch')
        self.backend = backend.lower()

    def _conv_forward(self, x, weight, bias):
        _fp32_inputs(x, weight, bias)
        backend = os.environ.get('CONVERSE2D_BACKEND', '') or self.backend
        if not isinstance(backend, str) or backend.lower() not in ('auto', 'cuda', 'pytorch'):
            raise ValueError('backend must be auto, cuda or pytorch')
        if backend.lower() != 'pytorch' and _eligible(x, weight, bias, self):
            _fp32_policy()
            return PointwiseWeightGradient.apply(x, weight, bias)
        return super()._conv_forward(x, weight, bias)
