"""Explicit nearest-prior inference; arbitrary priors are not accepted."""
import math

import torch


class NearestK2Inference:
    def __init__(self, extension):
        self.extension = extension

    @staticmethod
    def _check(x, weight, bias, eps):
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias)):
            raise RuntimeError("FFT-free nearest does not support differentiable GradMode inputs")
        if any(t.dtype != torch.float32 for t in (x, weight, bias)):
            raise ValueError("FFT-free nearest requires FP32 tensors")
        if any(not t.is_cuda or t.device != x.device for t in (x, weight, bias)):
            raise ValueError("all tensors must be on one CUDA device")
        if torch.is_autocast_enabled("cuda"):
            raise RuntimeError("autocast must be disabled")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if x.ndim != 4 or not all(x.shape):
            raise ValueError("expected nonempty NCHW input")
        if weight.ndim != 4 or weight.shape[-2:] != (2, 2):
            raise ValueError("only k2/s2 nearest upsampling is supported")
        if weight.shape[0] not in (1, x.shape[0]) or weight.shape[1] not in (1, x.shape[1]):
            raise ValueError("invalid kernel broadcasting")
        if bias.shape != (1, x.shape[1], 1, 1):
            raise ValueError("invalid bias shape")

    def __call__(self, x, weight, bias, eps=1e-5):
        self._check(x, weight, bias, eps)
        # The explicit nearest API also permits frozen inputs in GradMode. The
        # CUDA entry remains inference-only; no differentiable call reaches it.
        # Preserve the original ATen energy reduction on the original layout.
        with torch.no_grad():
            power = weight.square().sum((-2, -1), keepdim=True)
            regularizer = torch.sigmoid(bias - 9.0) + eps
            return self.extension.forward(x, weight, power + regularizer)
