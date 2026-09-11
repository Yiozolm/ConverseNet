"""Differentiable full-spectrum reference for the Converse2D closed form."""
import math

import torch


def alias_mean(a, scale):
    """Average corresponding entries in frequency blocks, not adjacent pixels."""
    if scale == 1:
        return a
    h, w = a.shape[-2:]
    return a.reshape(*a.shape[:-2], scale, h // scale, scale, w // scale).mean((-4, -2))


def validate_inputs(x, x0, weight, bias, scale, eps):
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.ndim != 4 or any(d == 0 for d in x.shape):
        raise ValueError("x must have nonempty shape (B,C,H,W)")
    b, c, h, w = x.shape
    if x0.shape != (b, c, h * scale, w * scale):
        raise ValueError("x0 must have shape (B,C,H*scale,W*scale)")
    if weight.ndim != 4 or weight.shape[:2] != (1, c):
        raise ValueError("weight must have shape (1,C,kh,kw)")
    if not (0 < weight.shape[2] <= h * scale and 0 < weight.shape[3] <= w * scale):
        raise ValueError("kernel must fit within the output spatial dimensions")
    if bias.shape != (1, c, 1, 1):
        raise ValueError("bias must have shape (1,C,1,1)")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("inputs must have real floating dtype")
    if any(t.device != x.device or t.dtype != x.dtype for t in (x0, weight, bias)):
        raise ValueError("all inputs must have the same device and dtype")


def converse2d_reference(x, x0, weight, bias, scale=1, eps=1e-5):
    """Stable solution with independent x0, including when scale == 1.

    Half/bfloat16 inputs compute in float32 and return the original dtype.
    """
    validate_inputs(x, x0, weight, bias, scale, eps)
    output_dtype = x.dtype
    same_prior = x0 is x
    if output_dtype in (torch.float16, torch.bfloat16):
        x, weight, bias = x.float(), weight.float(), bias.float()
        x0 = x if same_prior else x0.float()
    h, w = x.shape[-2:]
    kh, kw = weight.shape[-2:]
    psf = torch.nn.functional.pad(weight, (0, w * scale - kw, 0, h * scale - kh))
    fb = torch.fft.fft2(torch.roll(psf, (-(kh // 2), -(kw // 2)), (-2, -1)))
    power = fb.real.square() + fb.imag.square()
    fy = torch.fft.fft2(x)
    fx0 = fy if same_prior else torch.fft.fft2(x0)
    regularizer = torch.sigmoid(bias - 9.0) + eps
    correction = (fy - alias_mean(fb * fx0, scale)) / (alias_mean(power, scale) + regularizer)
    if scale != 1:
        correction = correction.repeat(1, 1, scale, scale)
    result = torch.fft.ifft2(fx0 + fb.conj() * correction).real
    return result.to(output_dtype)
