"""CPU semantic reproduction of historical kernels, NOT a CUDA benchmark.

Run: python analysis/v2_v7_numerical_audit.py
Source: 207e214 (v2-v7), 2197378 (v7 ortho), 6353910 (weighted v7).
NumPy + SciPy FFT preserve float32/complex64 in the precision experiment.
"""
import json
from pathlib import Path

import numpy as np
from scipy import fft


def alias_mean(a, s):
    h, w = a.shape[-2:]
    return a.reshape(*a.shape[:-2], s, h // s, s, w // s).mean(axis=(-4, -2))


def local_mean(a, s):
    h, w = a.shape[-2:]
    return a.reshape(*a.shape[:-2], h // s, s, w // s, s).mean(axis=(-3, -1))


def tile(a, s):
    return np.tile(a, (s, s))


def adjacent_repeat(a, s):
    return a.repeat(s, axis=-2).repeat(s, axis=-1)


def weighted_local(a, s, full_w):
    h, wr = a.shape
    out = np.empty((h // s, (wr + s - 1) // s), dtype=a.dtype)
    weights = np.full(wr, 2.0)
    weights[0] = 1.0
    if full_w % 2 == 0:
        weights[-1] = 1.0
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            ws = weights[j * s:(j + 1) * s]
            block = a[i * s:(i + 1) * s, j * s:(j + 1) * s]
            out[i, j] = (block * ws).sum() / (s * ws.sum())
    return out


def solve(x, weight, s, mode="v2", dtype=np.float64, eps=1e-5):
    x, weight = x.astype(dtype), weight.astype(dtype)
    shape = (x.shape[0] * s, x.shape[1] * s)
    x0 = adjacent_repeat(x, s)
    sty = np.zeros(shape, dtype=dtype)
    sty[::s, ::s] = x
    otf = np.zeros(shape, dtype=dtype)
    kh, kw = weight.shape
    otf[:kh, :kw] = weight
    otf = np.roll(otf, (-(kh // 2), -(kw // 2)), axis=(-2, -1))
    lam = dtype(1 / (1 + np.exp(9.0)) + eps)
    real_fft = mode.startswith("v7")
    norm = "ortho" if mode == "v7_ortho" else "backward"
    forward = fft.rfft2 if real_fft else fft.fft2
    fb = forward(otf, norm=norm)
    fbc = fb.conj()
    f2b = np.abs(fb) ** 2
    if mode == "v6":
        f2b = fb.real ** 2 + fb.imag ** 2
    if mode == "stable":
        fx0 = fft.fft2(x0)
        correction = (fft.fft2(x) - alias_mean(fb * fx0, s)) / (alias_mean(f2b, s) + lam)
        return fft.ifft2(fx0 + fbc * tile(correction, s)).real
    fr = fbc * forward(sty, norm=norm) + forward(lam * x0, norm=norm)
    if mode in ("v7", "v7_ortho"):
        fbr = local_mean(fft.irfft2(fb * fr, s=shape, norm=norm), s)
        invw = local_mean(fft.irfft2(f2b, s=shape, norm=norm), s)
        q = forward(adjacent_repeat(fbr / (invw + lam), s), norm=norm)
    elif mode == "v7_weighted":
        reduce = (lambda a: a) if s == 1 else (lambda a: weighted_local(a, s, shape[1]))
        q = adjacent_repeat(reduce(fb * fr) / (reduce(f2b) + lam), s)
        q = q[:shape[0], :shape[1] // 2 + 1]
    else:
        reduce = alias_mean if mode in ("v2", "repeat_only") else local_mean
        expand = tile if mode in ("v2", "mean_only") else adjacent_repeat
        q = expand(reduce(fb * fr, s) / (reduce(f2b, s) + lam), s)
    fx = (fr - fbc * q) / lam
    return fft.irfft2(fx, s=shape, norm=norm) if real_fft else fft.ifft2(fx).real


def maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    b = np.array([[1, 2], [3, 4]])
    results = {"scope": "CPU formula/index emulation; not compiled CUDA results", "index_example": {
        "v2_mean": alias_mean(a, 2).tolist(), "v3_mean": local_mean(a, 2).tolist(),
        "v2_repeat": tile(b, 2).tolist(), "v3_expand": adjacent_repeat(b, 2).tolist(),
    }}
    rng = np.random.default_rng(20260911)
    x = rng.standard_normal((8, 10))
    weight = np.exp(rng.standard_normal((3, 3)))
    weight /= weight.sum()
    results["float64_max_abs_vs_v2"] = {}
    for s in (1, 2, 3):
        baseline = solve(x, weight, s)
        results["float64_max_abs_vs_v2"][str(s)] = {
            mode: maxdiff(solve(x, weight, s, mode), baseline)
            for mode in ("v3", "v6", "mean_only", "repeat_only", "v7", "v7_ortho", "v7_weighted", "stable")
        }
    results["float32_max_abs_vs_float64_stable"] = {}
    # Round inputs first so this experiment isolates arithmetic precision.
    x32, w32 = x.astype(np.float32), weight.astype(np.float32)
    for s in (1, 2, 3):
        reference = solve(x32, w32, s, "stable")
        results["float32_max_abs_vs_float64_stable"][str(s)] = {
            mode: maxdiff(solve(x32, w32, s, mode, np.float32), reference)
            for mode in ("v2", "stable")
        }
    results["lambda"] = float(1 / (1 + np.exp(9)) + 1e-5)
    results["inverse_lambda"] = 1 / results["lambda"]
    results["float32_inverse_9_error"] = float(np.float32(1 / 9)) - 1 / 9
    assert results["index_example"]["v2_mean"] == [[5.0, 6.0], [9.0, 10.0]]
    assert results["index_example"]["v3_mean"] == [[2.5, 4.5], [10.5, 12.5]]
    assert results["float64_max_abs_vs_v2"]["1"]["v3"] == 0
    assert results["float64_max_abs_vs_v2"]["2"]["v3"] > 1
    assert results["float64_max_abs_vs_v2"]["1"]["v7"] > 1
    assert all(row["stable"] < 1e-8 for row in results["float64_max_abs_vs_v2"].values())
    path = Path(__file__).with_name("v2_v7_numerical_results.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
