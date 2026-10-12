"""Experimental low-precision I/O boundary around the unchanged FP32 solver.

This module has no registration, model patching, build or cache side effects.
The release operator and model APIs continue to accept FP32 tensors only.
"""
import math
from numbers import Real

import torch
from models import converse_core


def mixed_converse2d(x, prior, weight, bias, scale, eps=1e-5, *,
                     output_dtype=torch.float32, backend='cuda'):
    """Run experimental Level 1A/1B with a complete FP32 compute boundary.

    ``x`` and ``prior`` must share one storage dtype, float16 or bfloat16.
    ``weight`` may retain an FP32 master value or use that same low dtype;
    ``bias`` must remain FP32. ``output_dtype=torch.float32`` selects Level 1A;
    explicitly selecting ``x.dtype`` adds the final Level 1B output cast.

    ``backend='cuda'`` calls the existing ``torch.ops.converse2d.forward`` on
    CUDA tensors. The caller must load the checked extension beforehand; this
    function neither builds nor loads libraries. ``backend='pytorch'`` uses
    the current portable ``converse2d_fp32`` on CPU or CUDA.

    Autocast is disabled locally for casts and the entire FP32 solver. GradMode
    is preserved: any differentiable input still selects full-spectrum
    training, while frozen inputs/no_grad/inference_mode retain half-spectrum
    inference. Casts remain differentiable, including the explicit output cast.
    Repeated references to one input object share a single upcast per call, so
    ``prior is x`` stays aliased inside the solver. No detached values, cached
    spectra, hidden resizing, clipping, or persistent low-precision parameters
    are introduced. Input/gradient/output quantization must be measured by the
    experiment harness; this boundary alone does not grant numerical admission.
    """
    tensors = (x, prior, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError('x, prior, weight and bias must be tensors')
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Experimental inputs require float16 or bfloat16 storage')
    if prior.dtype != x.dtype:
        raise ValueError('prior must have the same low storage dtype as x')
    if weight.dtype not in (torch.float32, x.dtype):
        raise ValueError('weight must be FP32 or match the low input dtype')
    if bias.dtype != torch.float32:
        raise ValueError('bias must remain FP32')
    if output_dtype not in (torch.float32, x.dtype):
        raise ValueError('output_dtype must be FP32 or match the low input dtype')
    if any(value.layout != torch.strided for value in tensors):
        raise ValueError('Only strided tensors are supported')
    if x.device.type not in ('cpu', 'cuda') or any(value.device != x.device for value in tensors):
        raise ValueError('All tensors must be on one CPU or CUDA device')
    if type(scale) is not int or scale < 1:
        raise ValueError('scale must be a positive integer')
    if not isinstance(eps, Real) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be a finite positive real number')
    if not isinstance(backend, str) or backend.lower() not in ('cuda', 'pytorch'):
        raise ValueError("backend must be 'cuda' or 'pytorch'")
    backend = backend.lower()
    if backend == 'cuda' and not x.is_cuda:
        raise ValueError("backend='cuda' requires CUDA tensors")
    if x.is_cuda and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError('Disable both TF32 flags for this FP32-compute experiment')
    if backend == 'cuda' and not hasattr(torch.ops.converse2d, 'forward'):
        raise RuntimeError("Load the checked production extension before backend='cuda'")

    with torch.autocast(device_type=x.device.type, enabled=False):
        # This dictionary is local to the call and contains only input casts.
        # Identity is preserved even if a caller aliases more than x/prior.
        converted = {}
        def upcast(value):
            identity = id(value)
            if identity not in converted:
                converted[identity] = value.to(dtype=torch.float32)
            return converted[identity]
        x32, prior32, weight32, bias32 = (upcast(value) for value in tensors)
        converse_core.validate_inputs(x32, prior32, weight32, bias32, scale, float(eps))
        if backend == 'cuda':
            result = torch.ops.converse2d.forward(x32, prior32, weight32, bias32, scale, float(eps), 'v7')
        else:
            result = converse_core.converse2d_fp32(x32, prior32, weight32, bias32, scale, float(eps))
        return result if output_dtype == torch.float32 else result.to(dtype=output_dtype)
