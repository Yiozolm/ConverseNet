"""Pure building blocks for the mixed FP32 training candidate.

These functions do not select a backend, enable training, patch a model, or
cache tensors. Callers own dispatch (including no_grad/eval policy) and supply
the differentiable spectral callable. Existing model call sites do not use
this module yet.

Kernel generation/preparation uses differentiable FP64 arithmetic; spatial
activations, lambda, spectral inputs and solver outputs remain FP32. Parameter
storage is never converted in place. Operation order follows the validated
mixed-DataNet/shared-prior adapter; no alternative precision path is selected.
"""
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["prepare_kernel", "shared_s1", "mixed_data", "kernelnet_fp64",
           "projection_fp64", "circular_pad", "crop_view"]


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _tensor(value, name, dtypes, ndim=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.layout != torch.strided or value.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{name} requires CPU/CUDA strided storage")
    if value.dtype not in dtypes:
        raise ValueError(f"{name} has unsupported dtype {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if torch.is_autocast_enabled(value.device.type):
        raise ValueError("These training helpers require autocast to be disabled")


def _same_device(reference, *values):
    if any(value.device != reference.device for value in values if value is not None):
        raise ValueError("All inputs and parameters must be on the same device")


def _solver_inputs(x, weight, bias, scale, eps, kernel_dtype):
    _integer(scale, "scale")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive real scalar")
    _tensor(x, "activation", (torch.float32,), 4)
    _tensor(weight, "kernel", (kernel_dtype,), 4)
    _tensor(bias, "bias", (torch.float32,), 4)
    _same_device(x, weight, bias)
    batch, channels, height, width = x.shape
    if weight.shape[0] not in (1, batch) or weight.shape[1] not in (1, channels):
        raise ValueError("kernel must have batch/channel dimensions (1|B,1|C)")
    if weight.shape[2] > height * scale or weight.shape[3] > width * scale:
        raise ValueError("kernel must fit inside the output spatial dimensions")
    if bias.shape != (1, channels, 1, 1):
        raise ValueError("bias must have shape (1,C,1,1)")


def _spectrum_output(spectrum, activation, height, width):
    if not isinstance(spectrum, torch.Tensor) or spectrum.dtype != torch.complex64:
        raise RuntimeError("The supplied spectral callable must return complex64")
    if spectrum.device != activation.device or spectrum.shape != (*activation.shape[:2], height, width // 2 + 1):
        raise RuntimeError("The supplied spectral callable returned an invalid device or shape")


def prepare_kernel(weight, height, width):
    """Differentiable FP64 centered kernel FFT -> contiguous complex64.

    Accepts real FP32/FP64 [KB,KC,kh,kw] tensors. The separable-FFT gate and
    pad/roll/FFT order exactly match the established preparation path.
    """
    _tensor(weight, "kernel", (torch.float32, torch.float64), 4)
    _integer(height, "output height")
    _integer(width, "output width")
    kh, kw = weight.shape[-2:]
    if kh > height or kw > width:
        raise ValueError("kernel must fit inside the output spatial dimensions")
    filters, area = weight.shape[0] * weight.shape[1], height * width
    work = weight.double()
    if (area >= 16384 or filters * area >= 1048576) and kh <= height // 4:
        rows = torch.roll(F.pad(work, (0, width - kw)), -(kw // 2), -1)
        horizontal = torch.fft.rfft(rows, dim=-1)
        columns = torch.roll(F.pad(horizontal, (0, 0, 0, height - kh)), -(kh // 2), -2)
        spectrum = torch.fft.fft(columns, dim=-2)
    else:
        psf = F.pad(work, (0, width - kw, 0, height - kh))
        spectrum = torch.fft.rfft2(torch.roll(psf, (-(kh // 2), -(kw // 2)), (-2, -1)))
    return spectrum.to(dtype=torch.complex64, memory_format=torch.contiguous_format)


def shared_s1(x, weight, bias, eps, shared_spectral_op):
    """FP32 s=1 with the SAME activation as observation and prior.

    shared_spectral_op(y, kernel, lambda) is injected by the caller. This
    helper cannot represent an independent prior; use the general solver for
    that case. FFT/IFFT and all parameter casts remain in the autograd graph.
    """
    _solver_inputs(x, weight, bias, 1, eps, torch.float32)
    if not callable(shared_spectral_op):
        raise TypeError("shared_spectral_op must be callable")
    value = x.contiguous()
    y = torch.fft.rfft2(value)
    kernel = prepare_kernel(weight, *value.shape[-2:])
    regularizer = torch.sigmoid(bias.contiguous() - 9.) + eps
    spectrum = shared_spectral_op(y, kernel, regularizer)
    _spectrum_output(spectrum, value, *value.shape[-2:])
    return torch.fft.irfft2(spectrum, s=value.shape[-2:])


def mixed_data(x, kernel, bias, scale, eps, spectral_op):
    """FP32 DataNet activation/solve with an unrounded FP64 generated kernel.

    spectral_op(y, prior, kernel, lambda, H, W, scale) is injected. The prior
    is nearest(x); scale one reuses the identical activation FFT Tensor.
    Callers apply any outer padding/cropping before/after this function.
    """
    _solver_inputs(x, kernel, bias, scale, eps, torch.float64)
    if not callable(spectral_op):
        raise TypeError("spectral_op must be callable")
    value = x.contiguous()
    height, width = value.shape[-2:]
    regularizer = torch.sigmoid(bias.contiguous() - 9.) + eps
    prepared = prepare_kernel(kernel, height * scale, width * scale)
    y = torch.fft.rfft2(value)
    prior = y if scale == 1 else torch.fft.rfft2(F.interpolate(value, scale_factor=scale, mode="nearest"))
    corrected = spectral_op(y, prior, prepared, regularizer, height, width, scale)
    _spectrum_output(corrected, value, height * scale, width * scale)
    return torch.fft.irfft2(corrected, s=(height * scale, width * scale))


def kernelnet_fp64(value, kernelnet):
    """Evaluate the original three-Linear/two-GELU KernelNet in FP64.

    value and master parameters must be FP32. The output is FP64 with shape
    [B,16,kernel_size,kernel_size]. Module forwards/hooks are not invoked;
    this is the same functional evaluation used by the mixed adapter.
    """
    _tensor(value, "KernelNet input", (torch.float32,), 4)
    size = kernelnet.kernel_size
    _integer(size, "KernelNet kernel_size")
    if value.shape[1:] != (1, size, size):
        raise ValueError("KernelNet input must have shape (B,1,kernel_size,kernel_size)")
    layers = (kernelnet.fc1, kernelnet.fc2, kernelnet.fc3)
    shapes = ((64, size * size), (64, 64), (16 * size * size, 64))
    if not isinstance(kernelnet.gelu, nn.GELU) or kernelnet.gelu.approximate not in ("none", "tanh"):
        raise ValueError("KernelNet requires its configured GELU activation")
    for layer, shape in zip(layers, shapes):
        if not isinstance(layer, nn.Linear) or tuple(layer.weight.shape) != shape:
            raise ValueError("KernelNet requires the original three Linear layer shapes")
        _tensor(layer.weight, "Linear master weight", (torch.float32,), 2)
        if layer.bias is not None:
            _tensor(layer.bias, "Linear master bias", (torch.float32,), 1)
            if layer.bias.shape != (shape[0],):
                raise ValueError("Invalid Linear bias shape")
        _same_device(value, layer.weight, layer.bias)
    batch = value.shape[0]
    result = value.double().reshape(batch, -1)
    for index, layer in enumerate(layers):
        result = F.linear(result, layer.weight.double(), None if layer.bias is None else layer.bias.double())
        if index < 2:
            result = F.gelu(result, approximate=kernelnet.gelu.approximate)
    return result.view(batch, 16, size, size)


def projection_fp64(value, layer):
    """Evaluate the original 16->64 1x1 kernel projection without rounding.

    Input/output are FP64; the Conv2d master weight and optional bias remain
    FP32 and receive their gradients through differentiable .double() casts.
    """
    _tensor(value, "kernel projection input", (torch.float64,), 4)
    if not isinstance(layer, nn.Conv2d) or (
        layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
        layer.padding, layer.dilation, layer.groups, layer.padding_mode
    ) != (16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, "zeros"):
        raise ValueError("Kernel projection requires the original 16->64 1x1 Conv2d geometry")
    if value.shape[1] != 16 or layer.weight.shape != (64, 16, 1, 1):
        raise ValueError("Invalid kernel projection input/weight shape")
    _tensor(layer.weight, "Conv2d master weight", (torch.float32,), 4)
    if layer.bias is not None:
        _tensor(layer.bias, "Conv2d master bias", (torch.float32,), 1)
        if layer.bias.shape != (64,):
            raise ValueError("Invalid kernel projection bias shape")
    _same_device(value, layer.weight, layer.bias)
    return F.conv2d(value, layer.weight.double(), None if layer.bias is None else layer.bias.double(),
                    layer.stride, layer.padding, layer.dilation, layer.groups)


def circular_pad(x, padding):
    """One-wrap circular pad using ATen cat; p=0 preserves tensor identity."""
    _tensor(x, "padding input", (torch.float32, torch.float64), 4)
    _integer(padding, "padding", minimum=0)
    if padding == 0:
        return x
    if padding > min(x.shape[-2:]):
        raise ValueError("Circular padding cannot exceed either input spatial dimension")
    value = torch.cat((x[..., -padding:], x, x[..., :padding]), dim=-1)
    return torch.cat((value[..., -padding:, :], value, value[..., :padding, :]), dim=-2)


def _non_overlapping_positive_strides(x):
    # Sufficient metadata-only test; uncertain/overlapping layouts use native
    # slicing. No device synchronization or data-dependent condition is used.
    extent = 1
    for stride, size in sorted((stride, size) for stride, size in zip(x.stride(), x.shape) if size > 1):
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return True


def crop_view(x, padding):
    """Crop a positive interior with original strides/offset and ATen VJP.

    padding is in this tensor's pixel units; callers multiply an input padding
    by scale before cropping a super-resolved output. Overlapping layouts use
    the original two slices instead of an unproven as_strided fast path.
    """
    _tensor(x, "crop input", (torch.float32, torch.float64), 4)
    _integer(padding, "crop padding", minimum=0)
    if padding == 0:
        return x
    if 2 * padding >= min(x.shape[-2:]):
        raise ValueError("Crop must leave positive spatial dimensions")
    if not _non_overlapping_positive_strides(x):
        return x[..., padding:-padding, padding:-padding]
    return x.as_strided((*x.shape[:-2], x.shape[-2] - 2 * padding, x.shape[-1] - 2 * padding),
                        x.stride(), x.storage_offset() + padding * x.stride(-2) + padding * x.stride(-1))
