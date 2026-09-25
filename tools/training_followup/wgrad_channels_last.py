"""Narrow research candidate: native convolution with NHWC-only weight VJP.

The candidate changes cuDNN's layout/algorithm and may change its reduction
order. No numerical-equivalence or speed claim is made by this implementation.
Forward/input VJP/bias VJP and higher-order fallback stay native. Copies required
for the weight-only cuDNN call occur inside backward. No production dispatch.
"""
import torch
import torch.nn.functional as F


def eligible(x, weight, bias):
    return (torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.float32
            and weight.dtype == torch.float32 and (bias is None or bias.dtype == torch.float32)
            and tuple(x.shape) == (4, 128, 96, 96) and tuple(weight.shape) == (64, 128, 1, 1)
            and all(t is None or (t.layout == torch.strided and t.is_contiguous()
                    and not t.is_neg() and not t.is_conj()) for t in (x, weight, bias))
            and weight.requires_grad)


def policy():
    if (torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32 or not torch.are_deterministic_algorithms_enabled()
            or not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark):
        raise RuntimeError('Requires deterministic FP32 cuDNN, TF32/AMP/benchmark disabled')


def native_vjp(g, x, weight, bias_sizes, needs):
    return torch.ops.aten.convolution_backward.default(g, x, weight, bias_sizes, [1, 1], [0, 0],
                                                       [1, 1], False, [0, 0], 1, list(needs))


class ChannelsLastWeightVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.bias_sizes = [weight.shape[0]] if bias is not None else None
        ctx.set_materialize_grads(False)
        return F.conv2d(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None
        policy()
        x, weight = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if torch.is_grad_enabled():
            return native_vjp(g, x, weight, ctx.bias_sizes, needs)
        # Preserve native input and bias VJPs in one original-layout invocation.
        # Bias is computed by the same ATen convolution backward reduction.
        if needs[0] or needs[2]:
            dx, _, db = native_vjp(g, x, weight, ctx.bias_sizes, (needs[0], False, needs[2]))
        else:
            dx = db = None
        dw = None
        if needs[1]:
            # Every required conversion belongs to the measured backward call.
            nhwc_x = x.contiguous(memory_format=torch.channels_last)
            nhwc_g = g.contiguous(memory_format=torch.channels_last)
            nhwc_w = weight.contiguous(memory_format=torch.channels_last)
            _, dw, _ = native_vjp(nhwc_g, nhwc_x, nhwc_w, None, (False, True, False))
            dw = dw.contiguous()
        return dx, dw, db


def conv1x1(x, weight, bias=None):
    if any(t is not None and t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError('Candidate accepts FP32 only; FP64 is an independent reference')
    if eligible(x, weight, bias):
        policy()
        return ChannelsLastWeightVJP.apply(x, weight, bias)
    return F.conv2d(x, weight, bias)
