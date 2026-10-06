# SRAM-resident fused training: s1, s2, s3 (research only)

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
- One block per SM at 80 KB; global loads and FFT stages do not overlap; Stockham writes
  have bank conflicts.
- No A100 run, no release tests: this is not wired into the operator.

## Backward (`static_s1_backward`, `train_study.py`)

FlashAttention-style recomputation: forward saves only `x`, `k` and `l`. Backward runs
one block per channel, with one launch per batch index; stream order keeps the
kernel-gradient accumulation deterministic. Per plane:

```
load x (circular pad) -> FFT2 -> Y, parked in a per-channel L2-resident scratch
load g embedded in the padded grid -> FFT2 -> G = (.)/N
pointwise, production's scale1_adjoint boundaries (shared prior):
  t = G*k, gy = t/d, gm = -gy, q = (Y - k*Y)/d, gd = Re(-t * conj(q/d))
  grad_Y = (G + gy) + gm*conj(k)                 -> stays in shared memory
  grad_k += conj(G*conj(q)) + gm*conj(Y) + 2*k*gd -> channel slice of grad_k (L2)
  grad_l += gd                                    -> block reduction
unnormalized IFFT2(grad_Y) -> real part -> fold circular margins -> grad_x
```

`FlashS1` (Python autograd Function) wraps both kernels. PSF pad/roll, the per-call kernel
`fft2` and `sigmoid(bias - 9) + eps` remain differentiable ATen operations, so grad_k and
grad_l reach weight and bias through autograd.

RTX 5060 Ti, forward+VJP for (x, weight, bias) (`artifacts/v4_flash_fft/train_003.json`):

| Case | Production | FlashS1 | Output, grad_x, grad_weight, grad_bias rel-L2 vs prod |
|---|---|---|---|
| circular s1 B4 C128 96 pad 2 | 3217 us | 1652 us (**1.95x**) | 1.00, 1.00, 0.99, 1.00 |
| s1 B4 C128 100 | 3091 us | 1685 us (**1.83x**) | 1.00, 1.00, 1.00, 1.00 |
| circular s1 B4 C64 96 pad 2 | 1169 us | 837 us (1.40x) | 1.00, 1.00, 1.00, 1.01 |
| s1 B4 C64 96 | 1138 us | 823 us (1.38x) | 1.00, 1.00, 1.00, 0.99 |
| s1 B4 C64 64 | 462 us | 464 us (0.99x) | 1.00, 1.00, 1.00, 1.00 |

Every output and gradient passes its normal budget; max-abs ratios are 0.99-1.04x.
Repeated calls are bitwise identical (`determinism.py`).

Implementation notes:
- The first backward looped over the batch inside the kernel. ptxas then spilled 1.2-2.6 KB
  per thread at 96x96 and 100x100, even with Y out of registers and with the twiddle
  pointers and thread index laundered. One launch per batch index removes the spills: the
  backward kernel at C128 went from 2012 us to 1010 us.
- At C128 the backward kernel is 1010 us against 380 us for the forward, though it does
  only 1.5x the FFT work. Launching C blocks per batch index leaves partial waves
  (128 blocks on 36 SMs), and the Y scratch and grad_k accumulation add L2 traffic.
- `torch.roll` in the PSF preparation costs about 65 us per call; production's fused
  `psf_pad_roll` kernel would remove most of that.

Not done: integration into the operator (routing, broadcast kernels KB > 1, independent
prior for s1, higher-order fallback), release tests, convergence.

## s2/s3 and the capacity switch (`flash_study.py`, `seed_sweep.py`, `a100_colab.ipynb`)

`static_s_forward` / `static_s_backward` (S = 2, 3) keep the high-res (S*H) x (S*W) plane in
shared memory. The low-res spectrum of x is parked in an L2-resident scratch, so the plane is
the only large shared allocation. Alias groups {(h + a*H, w + b*W)} are solved in place:
production's means (factor H*W/N), `gm = (-t/d)*(1/S^2, 0)`, and the power term
`2k*gd/S^2`. The backward recomputes Y = FFT(x) and P = FFT(x0) on chip and returns grad_x,
grad_x0, grad_k and grad_l. Large planes run each FFT stage in line chunks: at most about 20
complex values are staged per thread, so 128x128 takes 2 chunks and 144x144 takes 3; s1 sizes
still run in one.

Capacity switch: `ext.supported(h, w, scale, pad, device)` is true only when the shape is
compiled and plane + 64 B static shared memory <= `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
`flash_study.py` sends every other case to the production operator and records its path.

| Plane | Size | RTX 5060 Ti (101376 B) | A100 (166912 B) |
|---|---|---|---|
| s1 100x100 / 96x96 | 80 / 72 KB | fused | fused |
| s2 64 -> 128x128 | 128 KB | production | fused |
| s3 48 -> 144x144 | 162 KB | production | fused (about 1 KB spare) |

Compiled shapes: s1 planes 100/96/64; s2 low-res 64/48/32; s3 low-res 48/32/24. The smaller
s2/s3 shapes validate the same kernels on GPUs that cannot hold the production planes.

RTX 5060 Ti (`artifacts/v4_flash_fft/flash_rtx5060ti_002/summary.md`), forward+VJP:

| Case | Path | Production | Fused | Worst rel-L2 ratio | Budgets |
|---|---|---|---|---|---|
| circular s1 B4 C128 96 pad 2 | fused | 3219 us | 1660 us (1.94x) | 1.00 | pass |
| s1 B4 C128 100 | fused | 3093 us | 1692 us (1.83x) | 1.00 | pass |
| s2 B4 C64 64 (128x128) | production | 2958 us | - | - | - |
| s2 B4 C64 48 (96x96) | fused | 1242 us | 797 us (1.56x) | 0.88 | pass |
| s2 B4 C64 32 (64x64) | fused | 542 us | 403 us (1.34x) | 2.00 (grad_bias) | FAIL |
| s3 B2 C32 48 (144x144) | production | 602 us | - | - | - |
| s3 B2 C32 32 (96x96) | fused | 637 us | 466 us (1.36x) | 0.81 | pass |
| s3 B2 C32 24 (72x72) | fused | 497 us | 405 us (1.23x) | 1.17 | pass |

s2/s3 accuracy over 8 seeds x 4 shapes (`seed_sweep_001.json`):
- output, grad_x and grad_x0 are 0.64-0.93x production's rel-L2 in every run;
- grad_weight always passes;
- grad_bias ranges from 0.31x to 2.75x and fails the 1.25x normal budget in 4 of 32 runs.

grad_bias is one sum per channel of B*H*W cancelling `gd` terms, each built from FFT spectra.
Its error is a few ulps and moves with any change in FFT rounding; the geometric mean ratio is
about 0.95. A two-component FP32 (TwoSum) reduction did not change this (3 of 32 failures,
`seed_sweep_002.json`), so the noise is in the terms, not the summation; the reduction was
reverted. This is an open accuracy item for any integration under the current policy.

### Running on an A100

Push `claude/v4-dev`, open `tools/v4_flash_fft/a100_colab.ipynb` in Colab with an A100
runtime, run all cells and send back the zip. It runs `flash_study.py` (all cases, now
including the 128x128 and 144x144 planes) and `seed_sweep.py` on the production and small
s2/s3 shapes.
