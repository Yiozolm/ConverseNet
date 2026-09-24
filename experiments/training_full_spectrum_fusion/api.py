"""Opt-in FP32 full-spectrum training with the supplied original v7 fallback."""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.converse_core import validate_inputs


def spatial(core, x, prior, weight, bias, scale, eps=1e-5):
    """Differentiable FP32 FFT preparation around the isolated fused core."""
    validate_inputs(x, prior, weight, bias, scale, eps)
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise TypeError("spatial() requires CUDA float32 inputs; FusedRoute provides the original fallback")
    h, w = x.shape[-2:]
    kh, kw = weight.shape[-2:]
    kernel = torch.fft.fft2(
        F.pad(weight, (0, w * scale - kw, 0, h * scale - kh))
        .roll((-(kh // 2), -(kw // 2)), (-2, -1))
    )
    y = torch.fft.fft2(x)
    p = y if prior is x else torch.fft.fft2(prior)
    regularizer = torch.sigmoid(bias - 9.0) + eps
    spectrum = core.spectral(y, p, kernel, regularizer, scale)
    if not p.is_contiguous():
        # Match the final Python addition's layout before selecting an IFFT plan.
        spectrum = torch.empty_like(p).copy_(spectrum)
    return torch.fft.ifft2(spectrum).real


class FusedRoute:
    """Wrap an already loaded original operator namespace without patching it.

    Only grad-enabled CUDA float32 calls needing a gradient use the candidate.
    CPU, other dtypes and calls without gradient work use ops.forward(..., 'v7').
    """
    def __init__(self, ops, core):
        self.ops = ops
        self.core = core
        self.counts = {}

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def forward(self, x, prior, weight, bias, scale, eps=1e-5, variant="v7"):
        use_fused = (
            x.device.type == "cuda" and x.dtype == torch.float32
            and torch.is_grad_enabled()
            and any(value.requires_grad for value in (x, prior, weight, bias))
        )
        route = "fused_full" if use_fused else "original_v7"
        self.counts[route] = self.counts.get(route, 0) + 1
        if use_fused:
            return spatial(self.core, x, prior, weight, bias, scale, eps)
        return self.ops.forward(x, prior, weight, bias, scale, eps, "v7")
