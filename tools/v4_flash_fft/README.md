# SRAM-resident fused s1 training forward (research only)

FlashAttention-style tiling for the Converse2D solve. In production, every stage writes
its result to DRAM, as the attention score matrix does in standard attention: the input FFT
row and column passes, the spectral solve, the inverse row and column passes. Here one
thread block owns one padded `(b, c)` plane. A 100x100 complex64 plane is 80 KB and fits
in shared memory, so the whole chain runs on chip:

```
load x (circular pad, (x, 0)) -> row FFT -> column FFT -> solve with k[c], l[c]
  -> inverse column FFT -> inverse row FFT -> 1/N, real part, crop -> store
```

Global traffic is one read of `x`, one read of the channel's `k` (from L2) and one write
of the output. The FFTs are mixed-radix (2/3/4/5) Stockham stages, in place in shared
memory via register staging. Twiddles come from a host table of exp(-2*pi*i*t/n): rounded
roots, no `__sincosf`, no fast math. The solve keeps production's FP32 operation
boundaries (`scale1_forward_planes` with p = y). It handles forward only, s = 1, weight
batch 1 and the shared prior.

Variants (`fused.cu`):
- `generic`: sizes, radices and strides at run time.
- `static`: all of them fixed at compile time; instantiated for 100x100, 96x96 and 64x64.
- `pair`: `static`, plus batch planes 2m and 2m+1 of one channel packed into one complex
  FFT as x_2m + i*x_2m+1. The solve is linear, so the real and imaginary outputs are the
  two results. FFT work is halved.

`study.py --output <json>` gates each variant through `numerical_policy.comparison`
(production FP32 forward as baseline, FP64 Python reference) and runs paired CUDA-event
timing. The fused timing includes the per-call kernel preparation it consumes (PSF
pad/roll, FP32 `fft2`, sigmoid). The production timing is its differentiable forward with
grad enabled. `profile_kernels.py` lists per-kernel times.

## RTX 5060 Ti (`artifacts/v4_flash_fft/forward_002.json`)

| Case | Production fwd | static | pair | static rel-L2 / max-abs vs prod | pair rel-L2 |
|---|---|---|---|---|---|
| circular s1 B4 C64 96 pad 2 | 421 us | 263 us (1.60x) | 188 us (2.23x) | 1.00x / 1.01x | 1.17x |
| circular s1 B4 C128 96 pad 2 | 1285 us | 530 us (2.42x) | 385 us (3.34x) | 1.00x / 1.01x | 1.22x |
| s1 B4 C128 100 | 1286 us | 546 us (2.35x) | 393 us (3.28x) | 1.00x / 1.00x | 1.24x |
| s1 B4 C64 96 | 400 us | 270 us (1.48x) | 190 us (2.11x) | 1.00x / 1.00x | **1.27x, fails** |
| s1 B4 C64 64 | 159 us | 140 us (1.14x) | 119 us (1.33x) | 1.00x / 1.00x | 1.14x |

`generic` was 0.56-1.43x. Runtime integer division and the radix switch in every
butterfly dominated it.

Findings:
- `static` passes every budget at production's own error level (1.00x) and is 2.4x faster
  at C=128.
- The gain is largest where production is DRAM-bound. At C=128 its 41 MB spectra exceed
  the 32 MB L2. At C=64 they stay in L2, which already acts as the on-chip tier.
- `pair` leaks each plane's rounding-level imaginary residue into its partner. It fails the
  1.25x normal rel-L2 budget once (1.27x), so it is not eligible as is.
- The fused forward saves nothing for backward beyond its inputs. Production keeps
  several full complex spectra per layer, so activation memory would also drop.

Limits and open work:
- 99 KB of shared memory per block on sm_120 holds at most ~12.6k complex elements. The s2
  (128x128) and s3 (144x144) planes do not fit. Options: split a plane across a thread
  block cluster with distributed shared memory, or a two-kernel split. A half spectrum would
  fit, but conflicts with the full-spectrum training policy.
- Backward is not implemented. Production forward is about a third of forward+VJP (1285 of
  3713 us), so the backward decides the training gain.
- One block per SM at 80 KB; global loads and FFT stages do not overlap; Stockham writes
  have bank conflicts.
- No A100 run, no release tests: this is not wired into the operator.
