"""Isolated FP32 training experiments; none is a release backend.

All registry callables take ``(x, prior, weight, bias, scale=1, eps=1e-5)``.
The half-spectrum and spectrum-reuse experiments rely on this checkout's
explicit research exception. Numerical and quality admission is independent
of their mathematical equivalence or a successful short execution.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
from torch.nn import functional as F

from models.converse_core import alias_mean, validate_inputs


def _validate(x, prior, weight, bias, scale, eps):
    if any(value.dtype != torch.float32 for value in (x, prior, weight, bias)):
        raise ValueError("training research candidates require FP32 tensors")
    validate_inputs(x, prior, weight, bias, scale, eps)
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("training research candidates support CPU or CUDA only")
    if torch.is_autocast_enabled(x.device.type):
        raise RuntimeError("training research candidates require autocast disabled")
    if x.is_cuda and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError("training research candidates require CUDA matmul and cuDNN TF32 disabled")


def _shifted_psf(weight, height, width):
    kh, kw = weight.shape[-2:]
    padded = F.pad(weight, (0, width - kw, 0, height - kh))
    return torch.roll(padded, (-(kh // 2), -(kw // 2)), (-2, -1))


def _full_kernel(weight, height, width):
    return torch.fft.fft2(_shifted_psf(weight, height, width))


def _complete_half_spectrum(value, width):
    """Reconstruct omitted columns, including the reflected row index.

    DC and even-width Nyquist columns occur once. Real powers use the same
    row/column reflection without a conjugation operation. All operations
    remain differentiable ATen operations, including for higher derivatives.
    """
    tail = value[..., 1:(width + 1) // 2].flip((-2, -1)).roll(1, -2)
    return torch.cat((value, tail.conj() if value.is_complex() else tail), -1)


def half_spectrum_solve(x, prior, weight, bias, scale=1, eps=1e-5):
    """Differentiable RFFT/IRFFT candidate for real inputs and any prior.

    This ATen prototype materializes complete prediction/power aliases for
    s>1. It does not claim that half spectra halve peak memory or runtime.
    """
    _validate(x, prior, weight, bias, scale, eps)
    height, width = x.shape[-2:]
    hs, ws = height * scale, width * scale
    kernel = torch.fft.rfft2(_shifted_psf(weight, hs, ws))
    power = kernel.real.square() + kernel.imag.square()
    y = torch.fft.rfft2(x)
    p = y if prior is x else torch.fft.rfft2(prior)
    prediction = kernel * p
    if scale > 1:
        prediction = alias_mean(_complete_half_spectrum(prediction, ws), scale)[..., :width // 2 + 1]
        power = alias_mean(_complete_half_spectrum(power, ws), scale)[..., :width // 2 + 1]
    regularizer = torch.sigmoid(bias - 9.0) + eps
    correction = (y - prediction) / (power + regularizer)
    if scale > 1:
        correction = _complete_half_spectrum(correction, width).repeat(1, 1, scale, scale)[..., :ws // 2 + 1]
    return torch.fft.irfft2(p + kernel.conj() * correction, s=(hs, ws))


def _spatial_filter(weight, batch, channels):
    kh, kw = weight.shape[-2:]
    # Conv2d is correlation; reversing both axes reproduces the convolution
    # defined by the centered PSF FFT. Folding B into channels supports every
    # existing batch/channel kernel broadcast without a per-sample loop.
    filters = weight.expand(batch, channels, kh, kw).reshape(batch * channels, 1, kh, kw)
    return filters.flip((-2, -1))


def _spatial_padding(kh, kw):
    # Even supports need asymmetric padding to preserve roll(-floor(k/2))
    # followed by phase-zero strided sampling.
    return kw - 1 - kw // 2, kw // 2, kh - 1 - kh // 2, kh // 2


def _spatial_observation(prior, filters, scale, padding):
    batch, channels = prior.shape[:2]
    padded = F.pad(prior, padding, mode="circular")
    grouped = padded.reshape(1, batch * channels, *padded.shape[-2:])
    observed = F.conv2d(grouped, filters, stride=scale, groups=batch * channels)
    return observed.reshape(batch, channels, *observed.shape[-2:])


def _fold_circular_padding(padded, height, width, padding):
    """Adjoint of one-wrap circular padding using deterministic additions.

    Valid kernels fit the HR grid, so neither side wraps more than once.
    This explicit gather/fold uses no scatter or index_add accumulation.
    """
    left, right, top, bottom = padding
    rows = padded[..., top:top + height, :]
    if top:
        rows = rows + F.pad(padded[..., :top, :], (0, 0, height - top, 0))
    if bottom:
        rows = rows + F.pad(padded[..., top + height:, :], (0, 0, 0, height - bottom))
    result = rows[..., left:left + width]
    if left:
        result = result + F.pad(rows[..., :left], (width - left, 0))
    if right:
        result = result + F.pad(rows[..., left + width:], (0, width - right))
    return result


def _spatial_adjoint(correction, filters, scale, padding, height, width):
    batch, channels = correction.shape[:2]
    grouped = correction.reshape(1, batch * channels, *correction.shape[-2:])
    # The original circular-padded input has s-1 trailing positions beyond
    # the last sampled stencil. Recover their zeros before folding padding.
    padded = F.conv_transpose2d(grouped, filters, stride=scale,
                                output_padding=scale - 1, groups=batch * channels)
    padded = padded.reshape(batch, channels, *padded.shape[-2:])
    return _fold_circular_padding(padded, height, width, padding)


def mixed_spatial_solve(x, prior, weight, bias, scale=1, eps=1e-5):
    """Spatial A/A^T with an LR full-FFT solve and per-call HR kernel FFT.

    The solve is p + A^T (AA^T + lambda I)^-1 (x - A p). It supports
    overlapping kernels; it never assumes AA^T is a scalar identity.
    Grouped convolution and folding can change FP32 rounding and gradient
    accumulation order, so this is a route-B numerical research candidate.
    """
    _validate(x, prior, weight, bias, scale, eps)
    batch, channels, height, width = x.shape
    hs, ws = height * scale, width * scale
    kernel = _full_kernel(weight, hs, ws)
    power = kernel.real.square() + kernel.imag.square()
    regularizer = torch.sigmoid(bias - 9.0) + eps
    denominator = alias_mean(power, scale) + regularizer
    filters = _spatial_filter(weight, batch, channels)
    padding = _spatial_padding(*weight.shape[-2:])
    residual = x - _spatial_observation(prior, filters, scale, padding)
    correction = torch.fft.ifft2(torch.fft.fft2(residual) / denominator).real
    return prior + _spatial_adjoint(correction, filters, scale, padding, hs, ws)


@dataclass
class _SpectrumEntry:
    source: torch.Tensor
    spectrum: torch.Tensor


@dataclass
class SpectrumScope:
    """Scope counters remain readable after exit; graph references do not."""
    hits: int = 0
    misses: int = 0
    _entries: Dict[Tuple, _SpectrumEntry] = field(default_factory=dict, repr=False)

    @property
    def cached_entries(self):
        return len(self._entries)


_spectrum_scopes = ContextVar("converse_training_research_spectrum_scopes", default=())


@contextmanager
def kernel_spectrum_scope():
    """Explicitly delimit one model forward, with nested-scope isolation.

    No detach or copy is inserted between repeated uses and the shared FFT
    graph. Therefore VJPs can accumulate in complex64 before FFT backward;
    this is the numerical change the reuse experiment must assess.
    """
    state = SpectrumScope()
    token = _spectrum_scopes.set(_spectrum_scopes.get() + (state,))
    try:
        yield state
    finally:
        state._entries.clear()
        _spectrum_scopes.reset(token)


def _spectrum_key(weight, height, width):
    # Inference tensors have no version counter; safely avoid reuse for them.
    if torch.is_inference(weight):
        return None
    stream = torch.cuda.current_stream(weight.device).cuda_stream if weight.is_cuda else None
    return (id(weight), weight._version, weight.data_ptr(), weight.storage_offset(),
            tuple(weight.shape), tuple(weight.stride()), weight.dtype, weight.device,
            weight.requires_grad, weight.grad_fn, weight.is_conj(), weight.is_neg(),
            torch.is_grad_enabled(), stream, height, width)


def _scoped_full_kernel(weight, height, width):
    scopes = _spectrum_scopes.get()
    if not scopes:
        return _full_kernel(weight, height, width)
    state = scopes[-1]
    key = _spectrum_key(weight, height, width)
    entry = state._entries.get(key) if key is not None else None
    if entry is not None and entry.source is weight:
        state.hits += 1
        return entry.spectrum
    state.misses += 1
    spectrum = _full_kernel(weight, height, width)
    if key is not None:
        state._entries[key] = _SpectrumEntry(weight, spectrum)
    return spectrum


def reuse_solve(x, prior, weight, bias, scale=1, eps=1e-5):
    """Full-spectrum solve; optional reuse is restricted to the active scope."""
    _validate(x, prior, weight, bias, scale, eps)
    height, width = x.shape[-2:]
    kernel = _scoped_full_kernel(weight, height * scale, width * scale)
    power = kernel.real.square() + kernel.imag.square()
    y = torch.fft.fft2(x)
    p = y if prior is x else torch.fft.fft2(prior)
    regularizer = torch.sigmoid(bias - 9.0) + eps
    correction = (y - alias_mean(kernel * p, scale)) / (alias_mean(power, scale) + regularizer)
    if scale > 1:
        correction = correction.repeat(1, 1, scale, scale)
    return torch.fft.ifft2(p + kernel.conj() * correction).real


REGISTRY = {
    "half_spectrum_training": half_spectrum_solve,
    "mixed_spatial_training": mixed_spatial_solve,
    "within_forward_spectrum_reuse": reuse_solve,
}
CANDIDATE_SCOPES = {"within_forward_spectrum_reuse": kernel_spectrum_scope}
LIMITATIONS = {
    "half_spectrum_training": "Research exception only; ATen full alias materialization can offset half-spectrum savings.",
    "mixed_spatial_training": "Spatial grouped convolution/folding changes rounding; cyclic phase-zero sampling only; kernel FFT retained.",
    "within_forward_spectrum_reuse": "Research exception only; caller must scope exactly one forward; shared FFT changes VJP accumulation order.",
}
