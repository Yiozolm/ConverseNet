"""CPU-only diagnostic mirror, never imported by the candidate or GPU gate.

NumPy scalar operations round to binary32; UCRT fmaf/expf supply CPU FP32 math.
The c_float result passes through Python exactly; no Python arithmetic is used
for the emulated floating-point operations. CUDA admission still needs its own
checked build and full gate, including the GPU FP32 sigmoid implementation.
"""
import ctypes
import os

import numpy as np
import torch


F32 = np.float32


def cpu_solve(data, eps):
    if os.name != "nt":
        raise RuntimeError("This diagnostic explicitly uses the Windows UCRT fmaf")
    if any(v.is_cuda or v.dtype != torch.float32 for v in data):
        raise ValueError("The diagnostic accepts CPU FP32 fixtures only")
    lib = ctypes.CDLL("ucrtbase.dll")
    intrinsic = lib.fmaf
    intrinsic.argtypes = [ctypes.c_float] * 3
    intrinsic.restype = ctypes.c_float
    exponential = lib.expf
    exponential.argtypes = [ctypes.c_float]
    exponential.restype = ctypes.c_float

    def add(a, b):
        return F32(F32(a) + F32(b))

    def sub(a, b):
        return F32(F32(a) - F32(b))

    def mul(a, b):
        return F32(F32(a) * F32(b))

    def div(a, b):
        return F32(F32(a) / F32(b))

    def fma(a, b, c):
        return F32(intrinsic(F32(a), F32(b), F32(c)))

    # This distinguishes fused multiplication from a rounded product plus add.
    if fma(F32(1.0000001192092896), F32(1.0000001192092896), F32(-1.000000238418579)) != F32(2.0 ** -46):
        raise AssertionError("UCRT fmaf failed the exact cancellation probe")

    def two_sum(a, b):
        hi = add(a, b)
        recovered_b = sub(hi, a)
        return hi, add(sub(a, sub(hi, recovered_b)), sub(b, recovered_b))

    def product(a, b):
        hi = mul(a, b)
        return hi, fma(a, b, -hi)

    def pair_add(a, b):
        high = two_sum(a[0], b[0])
        return two_sum(high[0], add(add(a[1], b[1]), high[1]))

    x, weight, bias = data
    # Mirror the new local formula without claiming UCRT expf matches CUDA expf.
    regularizer = np.empty(bias.shape, dtype=np.float32)
    for c in range(bias.shape[1]):
        shift = sub(bias.numpy()[0, c, 0, 0], F32(9))
        sigmoid = div(F32(1), add(F32(1), F32(exponential(-shift))))
        regularizer[0, c, 0, 0] = add(sigmoid, F32(eps))
    x, weight = x.numpy(), weight.numpy()
    batch, channels, height, width = x.shape
    output = np.empty((batch, channels, 2*height, 2*width), dtype=np.float32)
    zero = F32(0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for b in range(batch):
            for c in range(channels):
                weights = weight[0 if weight.shape[0] == 1 else b,
                                 0 if weight.shape[1] == 1 else c].reshape(-1)
                for h in range(height):
                    for w in range(width):
                        xi = x[b, c, h, w]
                        numerator = xi, zero
                        denominator = regularizer[0, c, 0, 0], zero
                        for value in weights:
                            term = product(value, xi)
                            numerator = pair_add(numerator, (-term[0], -term[1]))
                            denominator = pair_add(denominator, product(value, value))
                        qhi = div(numerator[0], denominator[0])
                        remainder = fma(-qhi, denominator[0], numerator[0])
                        remainder = add(remainder, numerator[1])
                        remainder = sub(remainder, mul(qhi, denominator[1]))
                        qlo = div(remainder, denominator[0])
                        for j, value in enumerate(weights):
                            total = pair_add((xi, zero), product(value, qhi))
                            total = pair_add(total, product(value, qlo))
                            output[b, c, 2*h+1-j//2, 2*w+1-j%2] = add(*total)
    return torch.from_numpy(output)
