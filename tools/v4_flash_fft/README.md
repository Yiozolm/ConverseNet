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
