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
s2/s3 shapes, then `ops_study.py` (training gate and its 8-seed sweep), `half_study.py`,
`s23_error_check.py` (32 seeds, all eight s2/s3 cases), `efficiency.py` and `error_anatomy.py`
(all planes, the data terms and the device-regularizer path). About an hour, half of it the
two builds.

## Memory and FLOP efficiency (`efficiency.py`)

FlashAttention-style accounting against hardware peaks
(`artifacts/v4_flash_fft/efficiency_rtx5060ti_003/efficiency.md`). Peaks are derived from device
properties: FP32 = 36 SMs x 128 lanes x 2 x 2.63 GHz = 24.3 TFLOP/s and DRAM =
2 x 14 GHz x 128 bit = 448 GB/s; a 1 GiB device copy measures 390 GB/s.

- Memory comes from the allocator: `saved` is what autograd keeps after the forward, `peak`
  is over forward + VJP.
- DRAM bytes and executed FP32 FLOPs (fadd + fmul + 2 ffma) are Nsight Compute counters over
  one call.
- Rates divide counters by the uninstrumented CUDA-event time. SASS instruction counting
  instruments the kernels, and ncu's own kernel durations come out 10-40x too long.
- Model FFT FLOPs (5 N log2 N for the standard algorithm's transforms) are the same for both
  paths, so the fused recomputation is not credited.

| Case | Path | fwd+VJP | Saved | Peak | DRAM / call | DRAM BW | Measured FP32 | Model FFT |
|---|---|---|---|---|---|---|---|---|
| s1 B4 C128 100 | production | 3.09 ms | 55.6 MiB | 230 MiB | 1172 MB | 380 GB/s (85%) | 3.7% | 2.0% |
| s1 B4 C128 100 | fused | 1.69 ms | 10.2 MiB | 70 MiB | 292 MB | 173 GB/s (39%) | 6.7% | 3.7% |
| circular s1 B4 C64 96 pad 2 | production | 1.16 ms | 27.8 MiB | 115 MiB | 442 MB | 380 GB/s (85%) | 4.9% | 2.7% |
| circular s1 B4 C64 96 pad 2 | fused | 0.82 ms | 4.9 MiB | 33 MiB | 41 MB | 51 GB/s (11%) | 6.9% | 3.9% |
| s2 B4 C64 48 (96x96) | production | 1.40 ms | 32.1 MiB | 151 MiB | 493 MB | 352 GB/s (79%) | 2.9% | 2.3% |
| s2 B4 C64 48 (96x96) | fused | 0.82 ms | 4.5 MiB | 36 MiB | 98 MB | 119 GB/s (27%) | 5.3% | 3.8% |
| s3 B2 C32 32 (96x96) | production | 0.62 ms | 8.4 MiB | 41 MiB | 49 MB | 79 GB/s (18%) | 1.7% | 1.3% |
| s3 B2 C32 32 (96x96) | fused | 0.47 ms | 2.3 MiB | 12 MiB | 18 MB | 38 GB/s (9%) | 2.3% | 1.8% |

Reading:
- Production is bandwidth-bound wherever its spectra exceed L2: 79-86% of peak DRAM
  bandwidth, 97% of the measured copy rate. Its FP32 use is 2-5% of peak, so it cannot go
  faster without moving fewer bytes.
- Fused moves 2.7-11x fewer DRAM bytes per call (1172 -> 292 MB at s1 C128) and keeps
  3.4-7.7x less memory for backward (all eight cases with a fused path). Peak memory is 3.3-4.2x lower.
- Fused is neither DRAM-bound (9-39% of peak) nor FLOP-bound (2-7% of FP32 peak). Its limit is
  on-chip: one 80 KB block per SM, a barrier per Stockham stage, shared-memory bank
  conflicts, and global loads that do not overlap compute.
- Of the fused s1 C128 traffic, the forward kernel moves 41 MB, the ideal x read plus output
  write. The four backward launches move 212 MB against about 72 MB of necessary traffic:
  the parked Y spectrum and the grad_k read-modify-write get written back to DRAM.
- Small s3 shapes fit in L2 for both paths, so the fused gain there (1.3x) comes from fewer
  kernels and passes, not from DRAM.

## A100-SXM4-40GB (`artifacts/v4_flash_fft/flash-NVIDIA_A100-SXM4-40GB-20261006T143334`)

Colab, torch 2.11+cu130, opt-in shared memory 166912 B: every case ran fused, including the
128x128 s2 and 144x144 s3 planes. Peaks: 19.5 TFLOP/s FP32, 1555 GB/s DRAM (copy 1373 GB/s).

| Case | Production fwd+VJP | Fused | DRAM MB/call prod -> fused | Prod DRAM BW | Saved MiB prod -> fused | Peak MiB |
|---|---|---|---|---|---|---|
| s1 B4 C128 100 | 1.11 ms | 1.22 ms (0.91x) | 1212 -> 295 | 71% | 55.6 -> 10.2 | 230 -> 70 |
| circular s1 B4 C128 96 pad 2 | 1.17 ms | 1.20 ms (0.97x) | - | - | 55.6 -> 9.8 | 230 -> 66 |
| s2 B4 C64 64 (128x128) | 1.05 ms | 0.98 ms (1.07x) | 1035 -> 464 | 63% | 57.0 -> 8.0 | 268 -> 63 |
| s3 B2 C32 48 (144x144) | 0.83 ms | 0.89 ms (0.93x) | 196 -> 105 | 15% | 17.8 -> 5.1 | 91 -> 26 |
| circular s1 B4 C64 96 pad 2 | 0.68 ms | 0.78 ms (0.87x) | 474 -> 60 | 46% | 27.8 -> 4.9 | 115 -> 33 |

Reading:
- Memory gains are architecture-independent: 3.4-7.7x less saved for backward, 3.3-4.3x
  lower peak, 1.2-8x fewer DRAM bytes (4.1x at s1 C128).
- Speed does not carry over. A100 DRAM is 3.5x faster than the 5060 Ti's, so production is
  less DRAM-bound (71% of peak at s1 C128, 46-63% elsewhere). The fused kernels are on-chip
  bound (4-12% of FP32 peak), so moving fewer bytes no longer pays.
- The fused backward wastes most of the A100: one launch per batch index with C blocks
  each. C=128 on 108 SMs takes 2 waves with the second 19% full; C=64 leaves 44 SMs idle.
  At s1 C128 the fused forward wins (0.33 vs 0.40 ms) and the backward loses (0.89 vs 0.71 ms).
- Small cases (s3 24/32, s2 32, and s1 C64 production at 0.67 ms) sit on a host-side
  launch floor of about 0.7 ms fwd+VJP on Colab. Their ratios compare launch counts, not
  GPU work.
- Accuracy: output, grad_x and grad_x0 again 0.67-0.94x production's rel-L2. The seed sweep
  failed 10 of 32 runs (first reported as 11, a miscount), on grad_bias (up to 4.17x) and
  grad_weight (up to 2.06x). Median
  errors are equal or lower than production's (geometric-mean ratios 0.73-0.98, except s3
  48 grad_bias 1.26 and s3 24 grad_bias 1.13); all are about 3e-7, a few ulps.

Next, for the A100: one B*C-block backward launch with per-batch grad_k partials reduced in
fixed batch order (waves 8 -> 5 at s1 C128); larger radix codelets (100 = 10x10,
144 = 12x12) to cut shared-memory passes and barriers; 2 blocks per SM for 80 KB planes.

## Single-launch backward (one block per plane)

Both backward kernels now run all B*C planes in one launch. Each block writes its grad_k
term and grad_l block sum to per-plane partials (B, C, ...). `reduce_batches` then sums them
as `((p0 + p1) + p2) + ...`, the same batch order and roundings as the earlier one-launch-
per-batch accumulation. The recomputed spectrum (Y for s1, P for s2/s3) is parked in the
plane's own partial slot: each thread reads it before overwriting that slot with its grad_k
term. s2/s3 park the low-res Y in a per-plane scratch.

- `capture.py`: all 8 locally eligible cases are bitwise identical to the per-batch-launch
  version (`capture_per_batch_launch.pt` vs `capture_plane_launch.pt`). Only scheduling
  changed, so accuracy results carry over unchanged.
- ptxas: no spills in any backward instantiation; the s2/s3 ones previously spilled 12-40 B.
- RTX 5060 Ti (`flash_rtx5060ti_003`): s1 C128 2.12x (circular) and 1.99x, up from 1.94x
  and 1.83x; C64 1.43x; s2/s3 small shapes 1.28-1.56x.
- Cost (`efficiency_rtx5060ti_004`): peak memory at s1 C128 rises from 70 to 100 MiB (the
  40 MB partial buffer), still 2.3x below production's 230 MiB. DRAM per call 292 -> 276 MB
  at s1 C128 and 98 -> 124 MB at s2 48 (per-plane scratch).
- Expected on A100: s1 C128 backward goes from 8 waves (4 launches x 2, the second 19% full)
  to 5 (512 blocks on 108 SMs); C64 no longer leaves 44 SMs idle. Not measured yet.

### A100 with the single-launch backward (`flash-NVIDIA_A100-SXM4-40GB-20261006T150610`)

The accuracy metrics (flash study and all 32 seed-sweep runs, candidate and baseline) are
identical to the earlier A100 run, as expected for bitwise-identical arithmetic: same 10/32
seed-sweep failures and the same s3 144x144 grad_bias failure in the study.

| Case | Production | Fused before | Fused now | Peak MiB prod / now |
|---|---|---|---|---|
| circular s1 B4 C128 96 pad 2 | 1.17 ms | 1.20 ms (0.97x) | 0.90 ms (**1.30x**) | 230 / 96 |
| s1 B4 C128 100 | 1.11 ms | 1.22 ms (0.91x) | 0.92 ms (**1.20x**) | 230 / 100 |
| s2 B4 C64 64 (128x128) | 1.06 ms | 0.98 ms (1.07x) | 0.90 ms (**1.18x**) | 268 / 93 |
| s3 B2 C32 48 (144x144) | 0.86 ms | 0.89 ms (0.93x) | 0.76 ms (**1.14x**) | 91 / 32 |
| circular s1 B4 C64 96 pad 2 | 0.68 ms | 0.78 ms (0.87x) | 0.70 ms (0.97x) | 115 / 48 |

- At s1 C128 the fused backward fell from 0.89 to 0.59 ms (production 0.71 ms), close to
  the predicted 0.55 ms from 8 -> 5 waves. Fused FP32 use rose to 15.4% of peak at 19% of
  DRAM bandwidth; production runs at 70% of DRAM bandwidth.
- C64 and the small s2/s3 shapes sit on the ~0.7 ms Colab launch floor (both paths); their
  1.0-1.24x ratios mostly reflect fewer launches.
- Saved-for-backward memory is unchanged (3.4-7.7x less than production); peak is now
  2.3-2.9x lower at the production-size cases, against 3.3-4.3x before.

## Anatomy of the grad_bias / grad_weight budget failures (`error_anatomy.py`, `gd_rounding.py`)

`error_anatomy.py` re-evaluates the kernel- and regularizer-gradient formulas in FP64 using
FP32 spectra from four sources: exact, production's cuFFT (`fft2` of (x, 0), `fft2(g)/N`),
cuFFT on transposed planes, and the fused kernels' own FFT (`ext.debug_fft2`). The transposed
cuFFT is mathematically identical and equally accurate but rounds differently; it is the
control "second FFT". Each path's error then splits into:
- shared: FP32 k and l, common to both paths;
- spectrum: the path's FP32 spectra;
- arithmetic: FP32 pointwise math, reductions and the sigmoid-backward chain.

RTX 5060 Ti, 5 shapes x 8 seeds (`error_anatomy_rtx5060ti_003.json`):

| | fused / production, geometric mean | runs > 1.25x (bias) |
|---|---|---|
| spectrum-only error, fused FFT | 0.59-0.88 | 3 / 40 |
| spectrum-only error, control (transposed cuFFT) | 0.94-1.17 | 13 / 40 |
| actual gate ratio, fused | bias 0.75-1.00, weight 0.84-1.00 | 4 / 40 |
| gate ratio if fused arithmetic were exact | 0.58-0.78 (s2/s3) | 1 / 32 (weight 0 / 32) |

- The fused FFT is more accurate than cuFFT in these sums. Every failing run has a fused
  spectrum ratio below 1.
- Failures come from the arithmetic component, a few ulps from several roughly equal sources.
  Rounding one step at a time in FP64 (`gd_rounding_001.json`), each pointwise step of gd
  adds 1.7-2.8e-8 to grad_l, all of them together 4.5e-8, and the FP32 sum 0.8-1.6e-7. The
  measured arithmetic component is 1-5e-7 in both paths.
- Why a few ulps can fail a 1.25x budget: each channel's sum cancels 17-71x, and the error
  concentrates in an effective 2-9 of 32-64 channels. The gate compares two independent
  noise draws with about 4 degrees of freedom. For two equally accurate implementations,
  P(ratio > 1.25) = P(F(4,4) > 1.5625) = 0.34. The control's spectrum-only rate is 13/40.
- s1 never fails: the shared FP32 kernel-spectrum error (about 1e-5) dominates both paths and
  pins the ratio at 1.00.
- Making the fused arithmetic exact (compensated FP32 products, alias sums and grad_l sum)
  would bound the bias failure rate near 1/32 here. It cannot reach zero: the budget is
  relative to production's own noise realization, and the FP32 sigmoid-backward rounding
  stays outside the kernel.

## Compensated backward arithmetic and the seeded gate

**Kernels.** Both backward kernels now evaluate the pointwise adjoint in two-term FP32
("double-float": TwoSum/TwoProduct via FMA, as in `inference/nearest_k2_s2.cu`). No FP64
runs on the device. Two-term values carry the 1/N scale and |k|^2; one reciprocal of d
replaces six divisions. The alias sums and products, the cancelling real part of gd,
grad_Y, the grad_k term, the per-channel grad_l sum and its batch reduction are two-term
too, each rounded to FP32 once. The transforms assume finite, non-overflowing
intermediates.

**Gate.** `test/numerical_policy.seeded_comparison` is additive: `comparison`, `BUDGETS` and
the release tests are unchanged. It is meant for outputs whose FP64 error is reduction-order
noise. Each seed contributes `u = candidate / max(baseline, floor / factor)`, so one seed
passes `comparison` exactly when u <= factor. The output passes when the geometric mean of u
over at least 8 seeds is within the same factor, for rel-L2 and max-abs separately, with
every seed finite. There is no per-seed cap: two equally accurate implementations exceed
2x on about 10% of single seeds, so some seed in 8 exceeds it with probability ~57%.
Tests are in `test/test_numerical_policy.py` (14/14 pass). `seed_sweep.py` prints the
verdict, and `--summarize` re-evaluates old sweeps offline.

RTX 5060 Ti, s2/s3 shapes x 8 seeds:

| | Before (`seed_sweep_001`) | Compensated (`seed_sweep_004_compensated_rcp`) |
|---|---|---|
| grad_bias single-run failures | 4 / 32 | 1 / 32 |
| grad_bias seeded geomean | 0.75-0.96, all pass | 0.73-0.88, all pass |
| grad_weight seeded geomean | 0.84-0.96 | 0.82-0.94 |
| grad_x seeded geomean | 0.65-0.90 | 0.60-0.85 |
| output, grad_x0 | unchanged | unchanged |

- Remaining arithmetic error (about 2e-7) is mostly the FP32 sigmoid-backward of the bias,
  outside the kernel and the same in both paths.
- The earlier A100 sweep under the seeded gate: 7/8 case/gradient pairs pass. s3 144x144
  grad_bias fails at a geomean of 1.26 (4/8 single-run failures), a mild systematic excess
  beyond noise on that plane. The compensated kernels need an A100 rerun to show whether
  they fix it.
- Cost (`flash_rtx5060ti_005_compensated_rcp`): s1 C128 fwd+VJP 1.92x (circular) and 1.81x
  vs production, from 2.12x and 1.99x uncompensated; C64 1.28x from 1.43x. With six
  two-term divisions it was 1.82x and 1.71x.
- The flash study's single fixed seed still fails s2 32 grad_bias (1.46x); its 8-seed
  geomean is 0.73.

### s1 under the same scrutiny (`seed_sweep_005_s1.json`, `error_anatomy_rtx5060ti_005_s1.json`)

Four s1 cases (circular C64/C128 96 pad 2, unpadded C128 100 and C64 96) x 8 seeds, with
the compensated kernels:
- 0 of 128 single-run budget failures (4 outputs x 32 runs). Every seeded geomean is
  1.00 (rel-L2 and max-abs, max seed 1.01).
- Why: the shared FP32 kernel spectrum and regularizer error is 6e-6 to 2e-5 relative and
  dominates both paths. The spectrum (1-3e-7) and arithmetic (2-4e-7) parts are 30-100x
  smaller. The fused spectrum part is again 0.70-0.90x cuFFT's; the transposed-cuFFT control
  is 0.95-1.03x.
- Each channel's grad_l sum cancels only 6-10x in s1, against 17-71x in s2/s3.

## More operations: kernel batches, padding modes, inference (`ops_study.py`)

The studies above cover one 3x3 kernel per channel with circular (s1) or no (s2/s3) padding
and, for s2/s3, an independent prior. The models also run:
- the USRNet data term (`ConvReverseDataNet`): one 7x7 kernel per (b, c) plane (weight batch
  B), eps 1e-3, no padding, with the nearest-upsampled prior built from x. The recipe
  (`--scale 3`, patch 96) runs it once at s3 (32 -> 96) and four times at s1 (96x96) per
  forward, next to 35 p-block calls at circular pad 2;
- Converse2D's replicate (ConverseMSRResNet: kernel 5, pad 4), reflect and zero padding at s1.

Kernel changes (`fused.cu`), both run-time arguments, so no new instantiations:
- `KB`: the kernel plane of (b, c) is `c` (KB == 1) or `b*C + c`. In the backward, the
  per-plane grad_k partial buffer is grad_k itself when KB == B; only grad_l is batch-reduced.
- `mode` (s1): `pad_source` maps a padded position to its interior index (circular wraps,
  replicate clamps, reflect mirrors about the edge pixel, zeros reads nothing). The pad
  adjoint sums the padded gradient plane over `fold_set`: an optional position below, a
  contiguous range and an optional position above, ascending; for circular that is the
  earlier dr, dq = -1, 0, +1 order.
- `compare_commit.py`: all 8 locally eligible `flash_study` cases are bitwise identical to a
  build of the sources at 17dd026, so the earlier paths' arithmetic is untouched. ptxas: no
  spills in any s1 kernel before or after; the s2 48 backward gained 36 B and the s2 64
  forward 16 B (64 -> 80 B); the s3 48 forward's 72 B is unchanged.

RTX 5060 Ti, training fwd+VJP (`ops_rtx5060ti_001`). Absolute times in this run are about
1.3x those of `flash_rtx5060ti_005` on both sides (desktop GPU load); ratios are comparable.

| Case | Production | Fused | rel-L2 ratio: output / grad_x / grad_weight / grad_bias | Budgets |
|---|---|---|---|---|
| data term s1 B4 C64 96, k7, KB 4 | 1620 us | 1398 us (1.16x) | 1.00 / 1.00 / 1.00 / 0.99 | pass |
| data term s3 B4 C64 32 -> 96, k7, KB 4, nearest | 2058 us | 1367 us (**1.50x**) | 0.80 / 0.82 / 0.86 / 0.72 | pass |
| data term s2 B4 C64 48 -> 96, k7, KB 4, nearest | 1752 us | 1480 us (1.18x) | 0.83 / 0.84 / 0.91 / **1.38** | FAIL (grad_bias) |
| replicate s1 B4 C64 92 pad 4, k5 | 1667 us | 1130 us (1.47x) | 1.00 / 1.00 / 1.00 / 1.00 | pass |
| reflect s1 B4 C64 96 pad 2 | 1731 us | 1122 us (1.54x) | 1.00 / 1.00 / 1.00 / 1.00 | pass |
| zeros s1 B4 C64 96 pad 2 | 1654 us | 1131 us (1.46x) | 1.00 / 1.00 / 1.00 / 1.00 | pass |
| circular s1 B4 C64 96 pad 2 (control) | 1503 us | 1140 us (1.32x) | 1.00 / 1.00 / 1.00 / 0.99 | pass |
| circular s1 B4 C128 96 pad 2 (control) | 4136 us | 2130 us (1.94x) | 1.00 / 1.00 / 1.00 / 1.00 | pass |

Seeded gate, 8 seeds (`ops_rtx5060ti_002_seeds`): all 32 case/output pairs pass. Single-run
failures: s2 data-term grad_bias 3/8 (geomean 1.14), s3 data-term grad_bias 1/8 (geomean 0.99,
one seed at 2.71); every other output 0/8 at geomeans 0.80-1.00. The s2/s3 grad_bias max-abs
geomeans of 0.01-0.04 are relative to the policy floor: both paths' worst element errors are
1e-8 to 4e-8, under the 1e-6 floor, not a production defect.

Reading:
- The padding modes are free: the same kernel with another index map. They gain more than
  circular (1.46-1.54x against 1.32x) because production serves them with F.pad + forward +
  crop: a separate pad kernel, no training-callback fusion, and ATen's replicate-pad backward
  accumulates with atomics. The fused fold is deterministic.
- The s1 data term gains only 1.16x: production's KB == B adjoint already writes grad_k per
  plane, the kernel FFT preparation (B*C planes, 4x the KB == 1 work) is the same on both
  sides, and the fused kernels are on-chip bound at C64.
- The s3 data term, the recipe's first call, gains 1.50x with 0.72-0.86x of production's
  error. The s2 data term's grad_bias shows the known reduction noise and passes the seeded
  gate at 1.14.

Inference (`ops_rtx5060ti_003_inference`, `--inference`): the fused full-spectrum forward
under no_grad against production's half-spectrum inference forward (rfft2, correction kernel,
irfft2; kernel spectrum cached per weight tensor, so production's time excludes the kernel
preparation):

| Case | Production fwd | Fused, kernel prep included | Fused, k and l given | Output rel-L2 ratio |
|---|---|---|---|---|
| data term s1 C64 96, KB 4 | 262 us | 533 us (0.49x) | 283 us (0.93x) | 0.79 |
| data term s3 32 -> 96, KB 4 | 342 us | 524 us (0.65x) | 223 us (1.54x) | 0.70 |
| data term s2 48 -> 96, KB 4 | 344 us | 547 us (0.63x) | 256 us (1.34x) | 0.70 |
| replicate / reflect / zeros / circular s1 C64 96 | 243-249 us | 329-361 us (0.68-0.74x) | 198-205 us (1.21-1.23x) | 0.77-0.85 |
| circular s1 C128 96 pad 2 | 687 us | 618 us (1.11x) | 470 us (1.46x) | 0.79 |

Reading:
- Accuracy: the fused forward has 0.70-0.85x of the half-spectrum path's error everywhere.
- Speed: the per-call kernel preparation (PSF pad/roll, `fft2` of (KB, C, H, W), sigmoid)
  costs 125-300 us here and erases the gain. With k given, as production's cache provides it
  for parameter kernels, the full-spectrum fused forward is 1.2-1.5x faster. The data term's
  kernels are activations, which production never caches either, so there the fair comparison
  needs production's preparation on the clock too (not measured).
- The fused forward does the full-spectrum transforms; inference needs half. A half-spectrum
  fused forward (R2C rows, C2C columns on H x (W/2+1), C2R rows) keeps a 100x100 plane in
  41 KB (two blocks per SM), fits the s2 128x128 and s3 144x144 inference planes in 67 and 84
  KB under the 99 KB limit, and halves the column passes: see the next section.

Still not covered: s2/s3 with padding (ConverseMSRResNet's k2 s2 pad 2 upsamplers; their
inference already has the FFT-free `_nearest_k2_s2` path), independent prior at s1, sizes
that are not compiled or not 2/3/5-smooth, and the operator integration items listed above.

## Half-spectrum fused inference forward (`half_study.py`)

`static_s1_half_forward` and `static_s_half_forward` (S = 2, 3) keep an H x (W/2 + 1) plane in
shared memory for the whole no_grad chain, with the kernel batch and s1 padding modes of the
training kernels:
- Rows transform as real data through a complex FFT of half length on the packed samples
  z[n] = x[2n] + i x[2n+1], then the split X[k] = E + W^k O, X[M-k] = conj(E - W^k O) with
  E = (Z[k] + conj Z[M-k]) / 2 and O = (Z[k] - conj Z[M-k]) / 2i; W^k comes from the existing
  host table for length W. The inverse undoes the split (its factor 2 folds into the 1/N) before
  the inverse half-length FFT; x[2n] and x[2n+1] are the real and imaginary outputs.
- Columns are the existing C2C Stockham passes over the W/2 + 1 lines.
- The solve follows production's inference kernels in c10::complex: `correction_scale_one` at
  s1; `alias_correction` with the inline power sum and `apply_correction` at s2/s3, reading
  mirrored frequencies as conj F[-h, -w] from the stored half planes (`read_frequency`).
- Capacity: 100x100 takes 40.8 KB. With `__launch_bounds__(512, 2)` the s1 kernels use 62-64
  registers with no spills at 64/96/100, and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  reports 2 blocks per SM. The s2 128x128 plane (66.5 KB, 128 registers, 32 B spill) and the s3
  144x144 plane (84 KB, 128 registers, no spill) fit the 99 KB limit, so the inference planes
  that the training kernels cannot hold here run fused (`half_supported`).
- `debug_rfft2` against `torch.fft.rfft2` with an FP64 reference: 0.68-0.89x cuFFT's rel-L2 at
  all nine sizes from 24 to 144.

RTX 5060 Ti, idle GPU (`half_rtx5060ti_002`), forward under no_grad with k and l given to every
candidate, as production's cache provides them for parameter kernels. Production is the
half-spectrum inference path (rfft2, correction kernel, irfft2); its medians agree with
`ops_rtx5060ti_003_inference` within 5%. "Full" is the training-plane fused forward.

| Case | Half plane | Production fwd | Half | Full | Half rel-L2 ratio |
|---|---|---|---|---|---|
| p-block s1 B4 C128 96 pad 2 | 40 KB | 721 us | 201 us (**3.59x**) | 509 us (1.42x) | 1.00 |
| p-block s1 B4 C64 96 pad 2 | 40 KB | 258 us | 99 us (2.60x) | 203 us (1.27x) | 1.01 |
| data term s1 B4 C64 96, k7, KB 4 | 37 KB | 265 us | 81 us (3.26x) | 294 us (0.90x) | 1.00 |
| replicate s1 B4 C64 92 pad 4, k5 | 40 KB | 245 us | 97 us (2.53x) | 191 us (1.28x) | 1.00 |
| s1 B4 C64 64 | 17 KB | 133 us | 39 us (3.37x) | 54 us (2.45x) | 1.00 |
| data term s3 B4 C64 32 -> 96, KB 4 | 37 KB | 312 us | 142 us (2.20x) | 239 us (1.30x) | 0.83 |
| data term s2 B4 C64 48 -> 96, KB 4 | 37 KB | 343 us | 152 us (2.26x) | 270 us (1.27x) | 0.85 |
| s2 B4 C64 64 -> 128, KB 4 | 65 KB | 590 us | 318 us (1.85x) | not eligible (128 KB) | 0.78 |
| s3 B2 C32 48 -> 144, KB 2 | 82 KB | 200 us | 68 us (2.93x) | not eligible (162 KB) | 0.85 |

Reading:
- Every output passes its budget. At s1 the half kernel's error is 1.00x production's (the
  shared FP32 kernel spectrum dominates both), at s2/s3 0.78-0.85x. Repeated calls are bitwise
  identical.
- 2.2-3.6x against production's inference forward on equal terms, where the full-spectrum
  fused forward gave 0.9-1.4x: the half plane halves the column FFTs and the solve, and two
  blocks per SM let one block's global loads overlap the other's FFT stages. The USRNet
  inference call (p-block s1 C128 100x100, 35 per forward) goes from 721 to 201 us.
- `half_rtx5060ti_001` ran while another process held the GPU at 98%: its production medians
  were 1.4-10x inflated and are superseded by `_002`; the accuracy results are identical.

Not covered: the kernel preparation (rfft2 of the PSF) stays an ATen call, which production
caches for parameter kernels and both sides would pay for data-term kernels; model-level
inference, CUDA-graph capture, the A100.

## s2/s3 kernel and regularizer gradients: is the fused error a systematic excess? (`s23_error_check.py`)

The fused grad_bias failed the single-run budget on some s2/s3 seeds (1/32 locally after the
compensated adjoint, 3/8 on the s2 data term, 4/8 on the A100 s3 144x144 plane). Two questions:
is any of it an excess of the fused kernels, and what sets the floor.

Method: 64 seeds x 6 cases (the four local s2/s3 `flash_study` cases and the two data terms),
three FP32 paths against the FP64 reference: the fused kernels, production, and a control that
runs production on the transposed problem (x, x0, weight and the upstream gradient transposed,
the result transposed back). The control is mathematically identical to production and equally
accurate, but cuFFT and the reductions round differently: it is a second draw of production's own
noise. Per case and output, with bootstrap 95% intervals over seeds: the single-run failure rate
(rel-L2 ratio > 1.25), the geometric-mean ratio (the seeded gate's statistic) and the pooled
ratio. RTX 5060 Ti, `s23_check_rtx5060ti_002_reg`:

| Case, grad_bias | fused: failures, geomean [CI] | control: failures, geomean [CI] | fused + device regularizer |
|---|---|---|---|
| s2 B4 C64 32 | 22%, 0.90 [0.81, 0.99] | 22%, 1.03 [0.94, 1.14] | 8%, 0.61 [0.54, 0.70] |
| s2 B4 C64 48 | 6%, 0.85 [0.78, 0.91] | 27%, 0.94 [0.83, 1.04] | 3%, 0.66 [0.60, 0.72] |
| s3 B2 C32 32 | 8%, 0.79 [0.73, 0.86] | 27%, 1.03 [0.96, 1.11] | 3%, 0.59 [0.53, 0.65] |
| s3 B2 C32 24 | 14%, 0.86 [0.80, 0.94] | 27%, 1.01 [0.93, 1.10] | 16%, 0.75 [0.68, 0.84] |
| data term s2 48, k7, KB 4 | 14%, 0.98 [0.92, 1.03] | 25%, 1.03 [0.94, 1.12] | 5%, 0.87 [0.82, 0.93] |
| data term s3 32, k7, KB 4 | 6%, 0.89 [0.84, 0.96] | 16%, 1.02 [0.95, 1.09] | 3%, 0.72 [0.67, 0.77] |

grad_weight: fused 0.79-0.94 with intervals of +-0.02 and no single-run failure in 384 runs; the
control 1.00-1.01.

Findings:
- No systematic excess. Every fused geomean has an upper bound at or below 1.03, and grad_weight
  is better in every case. The control sets the floor: production against a differently rounded
  copy of itself fails the single-run grad_bias budget in 16-27% of seeds (the README's F(4, 4)
  estimate of 34% was for equal implementations; the fused path fails 6-22%, less than the control
  in every case). An 8-seed geomean has a standard deviation of about 0.4 in log, so the earlier
  8-seed readings of 1.14 (s2 data term) and 1.26 (A100 s3 144, uncompensated) are inside that
  noise; the A100 case still needs a run with the compensated kernels.
- Where the error comes from (`error_anatomy_rtx5060ti_006_reg.json`, geomeans over 8 seeds,
  rel-L2 of grad_bias): the shared FP32 kernel spectrum and regularizer, 1.4-2.0e-7 for the 3x3
  kernels and 3.2-4.4e-7 for the 7x7 data-term kernels; the spectra, cuFFT 1.7-2.8e-7 against the
  fused FFT 1.0-1.9e-7; and the arithmetic, production 2.0-2.6e-7 against fused 1.8-2.2e-7. The
  fused arithmetic figure is not the kernel: it is the FP32 sigmoid chain outside it (the FP32
  `sigmoid` value, off by up to 1e-6 relative near bias -10; grad_l rounded to FP32; ATen's
  sigmoid backward), which a CPU estimate puts at 2e-7 on its own and which both paths share.
- Device regularizer (`regularizer_prep`, `scaled_backward_reg`, `FlashScaledReg`): the kernel
  takes bias and eps, forms l = sigmoid(bias - 9) + eps and dl/dbias as two-term FP32 on the
  device (exp by range reduction with a three-part ln2 and a degree-13 series in two-term
  arithmetic; 7e-15 relative against FP64), uses the two-term l in the backward's denominator and
  returns grad_bias itself. The arithmetic component drops to 2-3.5e-8, the geomean to 0.59-0.87
  and the single-run failures to 3-16%. Output, grad_x, grad_x0 and grad_weight are unchanged
  within their budgets and repeated calls are bitwise identical. On the study seed two grad_bias
  single runs still exceed 1.25 (s2 32 at 1.37, data term s2 at 1.48): single seeds remain draws.
- What remains is common to both paths: the FP32 kernel spectrum from ATen's `fft2` of the PSF,
  which dominates the 7x7 data terms (fused total 4.4e-7 against production's 4.7e-7). Production's
  own cuFFT-derived terms partly cancel against it (its total is below the quadrature sum of its
  components), which the fused FFT's independent rounding cannot do; that is why the fused path
  with the FP32 sigmoid chain read 1.14 on the s2 data term despite smaller components. Removing
  it would take a two-term kernel FFT on chip; not done.
- For the gate: a single-run grad_bias comparison on s2/s3 cannot distinguish the two paths;
  the seeded gate with enough seeds can, and the device regularizer moves the fused path from
  "equal within noise" to about 0.6-0.85x of production's error.

## A100 with the compensated backward (`flash-NVIDIA_A100-SXM4-40GB-20261007T064737`)

Colab, torch 2.11+cu130, branch at 06b831c. The run started four minutes after that push and
before the notebook gained the new studies, so it holds the original four studies on the current
kernels: the two-term backward, kernel batches and padding modes (run-time arguments), the
half-spectrum and regularizer kernels compiled in, and the anatomy with the device-regularizer path.

Accuracy: all 10 cases pass the single-seed gate (the s3 144x144 plane at a worst ratio of 0.87,
against 1.42 and FAIL with the uncompensated kernels) and all 40 seeded case/output pairs pass.

| Case | Worst single-seed rel-L2 ratio | grad_bias seeded geomean (single-run failures) | grad_weight seeded geomean |
|---|---|---|---|
| s1, all four cases | 1.00 | 1.00 (0/8) | 1.00 (0/8) |
| s2 B4 C64 64 -> 128 | 0.88 | 0.81 (1/8) | 0.94 (1/8, max seed 1.37) |
| s2 B4 C64 32 | 1.16 | 0.73 (0/8) | 0.85 (1/8) |
| s3 B2 C32 48 -> 144 | 0.87 | 1.01 (2/8), was 1.26 (4/8) | 0.83 (0/8) |
| s3 B2 C32 24 | 0.89 | 0.99 (1/8) | 0.92 (0/8) |

The anatomy on the A100 repeats the RTX picture: with the device regularizer the grad_bias gate
geomean is 0.62-0.95 against 0.73-1.01 for the FP32 sigmoid chain, and the transposed-cuFFT control
exceeds 1.25 in 1-3 of 8 seeds per s2/s3 case while fused does in 0-2.

Timing, forward+VJP (production is within 2% of the earlier run):

| Case | Production | Fused, this run | Fused, 20261006T150610 (uncompensated) |
|---|---|---|---|
| circular s1 B4 C64 96 pad 2 | 679 us | 985 us (0.69x) | 699 us (0.97x) |
| circular s1 B4 C128 96 pad 2 | 1169 us | 1295 us (0.90x) | 903 us (1.30x) |
| s1 B4 C128 100 | 1108 us | 1300 us (0.85x) | 923 us (1.20x) |
| s1 B4 C64 96 | 726 us | 927 us (0.78x) | 745 us (1.05x) |
| s2 B4 C64 64 -> 128 | 1058 us | 1210 us (0.87x) | 900 us (1.18x) |
| s2 B4 C64 48 | 889 us | 892 us (1.00x) | 779 us (1.24x) |
| s2 B4 C64 32 | 884 us | 736 us (1.20x) | 765 us (1.20x) |
| s3 B2 C32 48 -> 144 | 819 us | 736 us (1.11x) | 758 us (1.14x) |
| s3 B2 C32 32 / 24 | 880 / 822 us | 730 / 734 us (1.21x / 1.12x) | 741 / 749 us |

Reading:
- The compensated backward costs the A100 its speed advantage at the production sizes. From the
  efficiency counters at s1 C128 100: the fused forward is 0.33 -> 0.37 ms, the backward 0.59 ->
  0.93 ms; DRAM per call is unchanged at 272 MB while the executed FP32 rate rose from 3.0 to 5.2
  TFLOP/s (15 -> 27% of peak). The two-term pointwise phase roughly doubles the FP32 work, and the
  A100 (64 FP32 lanes per SM, 19.5 TFLOP/s) is FP32-bound in it. s2 128x128: backward 0.60 -> 0.91
  ms, FP32 2.0 -> 3.9 TFLOP/s. On the RTX 5060 Ti the same change cost 10% (2.12x -> 1.92x) because
  production is DRAM-bound there and the fused path has FP32 to spare.
- Memory is as before: 3.4-7.7x less saved for backward, peak 2.3-2.9x lower. The small cases sit
  on the Colab launch floor.
- The accuracy gain came from the compensation (the A100's seeded failures went from 10/32 and the
  s3 144 excess to none), so the trade is real. The next step is partial compensation: keep the
  two-term arithmetic for the terms that feed grad_k and grad_l (the gd dot product, the grad_k
  term, the grad_l sum) and return grad_Y, which feeds grad_x and never failed a budget
  uncompensated, to plain FP32. Not done.
- This zip has no ops, half-spectrum or s2/s3-check results; the updated notebook (510dedb) runs them.

## Partial compensation of the backward (`flash_rtx5060ti_006_partial`)

The A100 run showed the two-term backward is FP32-bound there. The pointwise adjoint now keeps
two-term arithmetic only where it bought accuracy, and cheaper two-term arithmetic at that:
- `df_adds`: the high parts are summed exactly (TwoSum), the low parts in one FP32 add, about
  2^-47 of the larger operand; the pointwise helpers (complex products, sums, |k|^2, the gd dot
  product) use it. The block and batch reductions keep the accurate `df_add`.
- `df_rcp`: 1/d from one IEEE reciprocal and a Newton correction (about 2^-46) in place of the
  three divisions of `df_div(1, d)`.
- The gradient spectrum enters t and the grad_k term as production's FP32 value, FFT(g) * (1/N)
  as the store callback rounds it; its FFT error (1e-7) dwarfs that rounding.
- grad_Y (s1) and grad_p (s2/s3), which feed grad_x and never failed a budget uncompensated,
  are assembled in FP32 with production's boundaries: (G + gy) + gm conj(k), and G + gm conj(k).
q, t, gy, gm, the gd term, the grad_k term and the grad_l sum stay two-term, so the kernel and
regularizer gradients keep their accuracy. The pointwise phase does about half the FP32
operations of the full two-term version. ptxas: no spills in any s1 backward; the s2 48 backward
spill fell from 36 to 12 B.

Accuracy is unchanged within the intervals. `s23_check_rtx5060ti_003_partial` (64 seeds; the
previous values from `_002_reg` in parentheses):

| Case, grad_bias geomean | fused | fused + device regularizer | grad_weight, fused |
|---|---|---|---|
| s2 32 | 0.90 (0.90) | 0.61 (0.61) | 0.83 (0.83) |
| s2 48 | 0.86 (0.85) | 0.68 (0.66) | 0.83 (0.84) |
| s3 32 | 0.79 (0.79) | 0.58 (0.59) | 0.78 (0.79) |
| s3 24 | 0.85 (0.86) | 0.73 (0.75) | 0.92 (0.94) |
| data term s2 48 | 0.98 (0.98) | 0.88 (0.87) | 0.89 (0.90) |
| data term s3 32 | 0.89 (0.89) | 0.73 (0.72) | 0.84 (0.85) |

Single-run failure rates are within 3 points of the previous run; grad_weight never fails. The
s1 seed sweep (`seed_sweep_006_partial.json`, four s1 cases plus s2 48 and s3 32) passes every
output at geomean 1.00 for s1 and 0.58-0.88 for s2/s3 with no single-run failure; the ops gate
(`ops_rtx5060ti_004_partial`) repeats the earlier single-seed picture, including the s2 data
term's fixed-seed grad_bias draw of 1.38.

RTX 5060 Ti, forward+VJP (desktop load of about 14% in this run; compare ratios):

| Case | Uncompensated (`_003`) | Full two-term (`_005`) | Partial (`_006`) |
|---|---|---|---|
| circular s1 B4 C128 96 pad 2 | 2.12x | 1.92x | **2.18x** |
| s1 B4 C128 100 | 1.99x | 1.81x | **2.06x** |
| circular s1 B4 C64 96 pad 2 | 1.43x | 1.28x | 1.48x |
| s1 B4 C64 96 | - | - | 1.48x |
| s2 B4 C64 48 | 1.56x | - | 1.57x |
| s2 32 / s3 32 / s3 24 | 1.34x / 1.36x / 1.23x | - | 1.29x / 1.36x / 1.22x |
| ops: padding modes | - | 1.46-1.54x | 1.69-1.73x |
| ops: data term s1 / s2 / s3 | - | 1.16x / 1.18x / 1.50x | 1.27x / 1.27x / 1.60x |

On the A100 the backward's pointwise FP32 work halves; the expected effect at s1 C128 is a
backward near the uncompensated 0.6 ms instead of 0.93 ms. Needs the notebook rerun.
