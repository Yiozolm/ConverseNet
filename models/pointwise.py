"""FP32 pointwise (1x1) convolution with a deterministic split-K weight VJP."""
import os

import torch
import torch.nn.functional as F

CHANNELS = (64, 128)
_EXT_CANDIDATES = ('converse2d_ext', 'torch_converse2d.converse2d_ext', 'torch_converse2d')


def _fp32_inputs(x, weight, bias):
    if any(t is not None and t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError('PointwiseConv2d requires FP32 tensors')


def _fp32_policy():
    if (torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32):
        raise RuntimeError('Direct FP32 GEMM requires autocast and both TF32 flags disabled')


def _has_wgrad_op():
    if hasattr(torch.ops.converse2d, '_pointwise_wgrad'):
        return True
    for name in _EXT_CANDIDATES:
        try:
            __import__(name)
        except Exception:
            continue
        if hasattr(torch.ops.converse2d, '_pointwise_wgrad'):
            return True
    return False


def _eligible(x, weight, bias, module):
    tensors = (x, weight) if bias is None else (x, weight, bias)
    return (torch.is_grad_enabled() and weight.requires_grad and x.is_cuda
            and x.dim() == 4 and x.numel() > 0 and x.numel() * 4 <= 2**31 - 1
            and weight.dim() == 4 and tuple(weight.shape[2:]) == (1, 1)
            and weight.shape[0] in CHANNELS and weight.shape[1] in CHANNELS
            and weight.shape[1] == x.shape[1]
            and (bias is None or tuple(bias.shape) == (weight.shape[0],))
            and all(t.device == x.device and t.dtype == torch.float32
                    and t.layout == torch.strided and t.is_contiguous()
                    and not t.is_neg() and not t.is_conj() for t in tensors)
            and module.stride == (1, 1) and module.padding == (0, 0)
            and module.dilation == (1, 1) and module.groups == 1
            and module.padding_mode == 'zeros'
            and _has_wgrad_op())


class PointwiseWeightGradient(torch.autograd.Function):
    """Native forward/dx/dbias; the weight VJP is the fused split-K kernel.

    The kernel (`_pointwise_wgrad`) streams both NCHW operands once, folds
    short FMA chains into compensated (hi, lo) accumulators and reduces the
    per-split partials in index order, so the gradient is deterministic and
    its rounding error stays within the budget of the cuDNN deterministic
    direct kernel in both channel directions. Higher-order graphs keep the
    native ATen backward for every gradient.
    """

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
            upstream = grad_output.resolve_neg().resolve_conj().contiguous()
            dw = torch.ops.converse2d._pointwise_wgrad(upstream, x)
        return dx, dw, db


class PointwiseConv2d(torch.nn.Conv2d):
    """Conv2d checkpoint-compatible module with a measured weight VJP.

    Contiguous CUDA FP32 NCHW inputs with 64 or 128 input and output channels
    and differentiable weights select the split-K kernel for the weight
    gradient. Native convolution handles every other shape/layout, inference,
    frozen weights, higher-order graphs and the pytorch backend.
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
