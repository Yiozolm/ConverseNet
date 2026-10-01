"""Selected experimental FP16/circular/s1 pad-cast fusion, never auto-enabled.

Only input conversion plus padding changes. Weight, bias, all spectral compute,
and the output remain FP32/complex64. The final crop retains the original view.
"""
import math
import os

import torch

INT32_MAX = (1 << 31) - 1


def _original_forward(module, original_forward):
    if original_forward is None:
        if 'forward' in vars(module):
            raise ValueError('Pass the saved bound original_forward when module.forward is overridden')
        original_forward = module.forward
    if not callable(original_forward) or getattr(original_forward, '__self__', None) is not module:
        raise TypeError('original_forward must be a saved method bound to this module')
    return original_forward


def _eligible(module, x, original_forward, converse_type):
    backend = os.environ.get('CONVERSE2D_BACKEND', '') or module.backend
    if (type(module) is not converse_type or getattr(original_forward, '__func__', None) is not converse_type.forward
            or x.dtype != torch.float16 or not x.is_cuda or x.ndim != 4 or x.numel() == 0
            or x.layout != torch.strided or not x.is_contiguous() or x.is_neg() or x.is_conj()
            or type(module.scale) is not int or module.scale != 1 or module.variant != 'v7'
            or type(module.padding) is not int or module.padding <= 0 or module.padding_mode != 'circular'
            or not isinstance(backend, str) or backend.lower() not in ('auto', 'cuda')
            or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or (torch.is_grad_enabled() and any(value.requires_grad for value in (x, module.weight, module.bias)))):
        return False
    weight, bias = module.weight, module.bias
    if (weight.device != x.device or bias.device != x.device
            or weight.layout != torch.strided or bias.layout != torch.strided
            or weight.ndim != 4 or weight.shape[0] not in (1, x.shape[0])
            or weight.shape[1] not in (1, x.shape[1]) or bias.shape != (1, x.shape[1], 1, 1)
            or not isinstance(module.eps, (int, float)) or not math.isfinite(module.eps) or module.eps <= 0):
        return False
    h, w, p = x.shape[-2], x.shape[-1], module.padding
    if p > min(h, w) or x.numel() > INT32_MAX:
        return False
    oh, ow = h + 2 * p, w + 2 * p
    if oh > INT32_MAX or ow > INT32_MAX or x.shape[0] * x.shape[1] * oh * ow > INT32_MAX:
        return False
    if not (0 < weight.shape[2] <= oh and 0 < weight.shape[3] <= ow):
        return False
    return hasattr(torch.ops.converse2d, 'forward')


def mixed_module_forward(module, xlow, extension, *, original_forward=None):
    """Invoke one experimental module call, preserving the original fallback.

    Fast scope: exact Converse2D class/default forward, contiguous CUDA FP16,
    circular positive padding <= both input dimensions, s1/v7, FP32 weight and
    bias, valid INT32 indexing, and no differentiable input under GradMode.
    no_grad, inference_mode and fully frozen GradMode can use this path.

    BF16 and FP32 activations, CPU, noncircular padding, unsupported layouts,
    lazy input flags, p0/other scales, differentiable GradMode, explicit pytorch
    backend, or enabled autocast/TF32 delegate to saved_original(xlow.float()).
    Caller autocast/GradMode flags are never changed. Non-FP32 parameters and
    activation dtypes outside FP16/BF16/FP32 are rejected. Invalid padding keeps
    the original module's exception semantics through that same fallback.

    When temporarily overriding module.forward, supply its previously saved
    bound method as original_forward; this prevents recursive fallback. The
    caller must load checked production/research libraries before eligible use.
    No model patch, library load, spectrum cache, or persistent cast is created.
    """
    from models.util_converse import Converse2D
    if not isinstance(module, Converse2D):
        raise TypeError('mixed_module_forward requires a Converse2D module')
    if not isinstance(xlow, torch.Tensor):
        raise TypeError('xlow must be a tensor')
    if xlow.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Experimental activations must be FP16, BF16 or FP32')
    if module.weight.dtype != torch.float32 or module.bias.dtype != torch.float32:
        raise ValueError('Converse2D weight and bias must remain FP32')
    original = _original_forward(module, original_forward)
    if not _eligible(module, xlow, original, Converse2D):
        return original(xlow.float())
    if not callable(getattr(extension, 'pad_cast', None)):
        raise TypeError('A checked research extension with pad_cast is required')
    p = module.padding
    padded = extension.pad_cast(xlow, p, 'circular')
    expected = (*xlow.shape[:2], xlow.shape[2] + 2 * p, xlow.shape[3] + 2 * p)
    if (padded.dtype != torch.float32 or padded.device != xlow.device or tuple(padded.shape) != expected
            or not padded.is_contiguous() or padded.requires_grad):
        raise RuntimeError('pad_cast violated its contiguous FP32 inference output contract')
    # Preserve x/prior object identity for s1; the existing solver and parameter
    # objects are untouched. No extra unpadded FP32 activation is materialized.
    output = torch.ops.converse2d.forward(padded, padded, module.weight, module.bias,
                                         1, float(module.eps), module.variant)
    return output[..., p:-p, p:-p]
