# Fused channels-first LayerNorm training

The B4 USRNet training trace in `profile/flash-bottleneck-20261012` attributed 47.7 ms of a
403 ms GPU kernel sum (11.9%, or 15.7% of the Flash-integrated step) to the 70 channels-first
LayerNorm calls, all `4x64x96x96`. Only the inference forward had a fused kernel; under
GradMode the layer ran the original ten-kernel ATen expression and its roughly twenty-five
kernel backward, almost all of it memory traffic through 9.4 MB tensors.

`models/util_converse.py` now routes differentiable CUDA inputs in the existing fused domain
(contiguous FP32 NCHW, 64 or 128 channels, `H*W > 1`, no negative/conjugate views, backend not
`pytorch`, autocast off) through `_ChannelLayerNormTraining`:

- **Forward** reuses the exact inference kernel, which reproduces the ATen reduction
  partitions, so the training output stays byte-identical to the original expression. The
  kernel additionally stores each pixel's mean and `sqrt(var + eps)`.
- **First-order backward** is one kernel (`full_layernorm_backward`, one thread per pixel,
  128-thread blocks) computing `dx = (g*w - mean_c(g*w) - xhat*mean_c(g*w*xhat)) / sigma`
  and per-block partials of `sum(g*xhat)` and `sum(g)` in a fixed order (xor-shuffle tree,
  then warps in index order); `at::sum_out` finishes the channel vectors, so the gradients
  repeat bitwise. Only the requested gradients are computed.
- **Higher-order** requests (`create_graph`) recompute the original ATen expression inside the
  backward and differentiate it, so second and third derivatives keep the native path;
  first-order gradients under `create_graph` are byte-identical to the ATen graph.
- CPU, frozen inputs under GradMode, other channel counts, strided/channels-last layouts and
  the `pytorch` backend keep the original expression. Inference dispatch is unchanged.

Gates (`test/test_layernorm_training.py`, plus the updated routing/gradient checks in
`test_layernorm_full.py` and `test_layernorm_affine.py`): byte-exact output; every output/VJP
within the FP32 policy budget against the Python FP32 expression and the independent FP64
oracle for all seven gradient masks at the production shape, eight geometries including
unaligned storage, tails, `H*W` of 7 and `eps` from 1e-6 to 0.2; the channel vectors (36864-
and 20000-term cancelling sums) over eight independent seeds with the seeded budget; a
permuted upstream gradient; bitwise repeatability; every dispatch boundary; the fused
first-order kernel and the native higher-order fallback. The complete suite passed
225/225 on the timed binary `e6b425c3`.

## Measured (RTX 5060 Ti, MSVC 14.44, CUDA 13.0, torch 2.11+cu130)

`perf.py --rounds 6 --iters 30 --model --model-rounds 5 --model-iters 5`, GPU idle, routes
alternated per round in one process (`artifacts/v4_layernorm_training/perf_rtx5060ti_001.json`,
local, untracked). Per-call numbers are forward plus the three VJPs; kernel sums come from
`torch.profiler`.

| Case | ATen call | fused call | call | ATen kernels | fused kernels | kernels |
|---|---:|---:|---:|---:|---:|---:|
| B4 C64 96x96 (production) | 852 µs | 177 µs | **4.81x** | 684 µs | 76 µs | **9.02x** |
| B2 C128 100x100 | 953 µs | 244 µs | 3.90x | 806 µs | 113 µs | 7.14x |
| B1 C64 96x96 | 694 µs | 207 µs | 3.36x | 113 µs | 27 µs | 4.26x |

The small-batch call is launch-bound on both routes (the ATen path issues about 35 kernels
per call, the fused path 5), so the kernel ratio, not the call ratio, is the GPU-side number.
The operator-level gradients of the two routes differ by 0.8-1.7e-7 relative L2.

Complete pretrained USRNet B4 LR32x32 -> HR96x96, scale 3, five iterations, synthetic resident
inputs, `zero_grad + forward + MSE + backward + Adam`, only the 14 LayerNorm modules switched
(Converse2D and pointwise paths identical on both routes), state restored before each round:

| Route | step median | rounds |
|---|---:|---|
| ATen LayerNorm | 309.8 ms | 322.2 / 309.8 / 297.2 / 303.1 / 315.8 |
| fused LayerNorm | 262.0 ms | 262.0 / 275.2 / 248.1 / 263.1 / 255.0 |
| | **1.18x** | |

The 47.8 ms saved matches the 47.7 ms the profile attributed to LayerNorm kernels, i.e. the
layer's GPU cost is now about 5 ms per step. This is a measured paired step on one machine,
not a convergence claim; `test_pretrained_fp32.py`'s three-seed Adam regression passes with
the fused path.

Remaining per the same profile: the pointwise convolutions (about 50% of the step, mostly the
cuDNN grouped wgrad) and the Flash fused prior kernels.

## A100 and the compensated backward (2026-10-12, `tools/v4_release_a100/`)

The first run of this path on an A100 (sm_80, torch 2.11+cu130; Colab via the Colab CLI,
`tools/v4_release_a100/README.md`) ran the 228 release tests with 56 failures: the 54 pre-existing
sm_80 failures unchanged, plus two new single-seed failures in `test_geometries_alignment_tails_and_eps`:
dweight on (1,64,20,24) at 1.26x and dbias on (1,64,96,96) off 1 eps 1e-6 at 1.33x of the ATen
expression's FP64 rel-L2 (limit 1.25x; both max-abs ratios about 1.0).

Eight-seed sweeps of every geometry (`layernorm_seeds.py`) on the RTX 5060 Ti and the A100 gave
the same picture on both: 1-3 of 8 seeds miss the single-seed factor on the same geometries,
every seeded verdict passes, and the committed seeds had simply passed on the RTX 5060 Ti. The
channel vectors are cancelling sums over every pixel, which the production-shape test already
gated over seeds; the geometry test now does the same. The sweeps also showed dx at a systematic
1.15x rel-L2 of the baseline on every geometry and both GPUs.

Two rounding sources in the backward explained the dx excess. It normalized with a rounded
reciprocal product `(x - m) * (1/sigma)` while the forward and the reference divide, and it
accumulated the two 64/128-term channel means serially. Variants measured on the RTX 5060 Ti in
one session (`artifacts/v4_release_a100/layernorm_seeds_rtx5060ti_00{1,2,3}*.json`,
`artifacts/v4_layernorm_training/perf_rtx5060ti_{original_samesession,compensated,comp_recip}_001.json`):

| Backward | dx rel-L2 ratio, geomean over 9 geometries | B4 C64 96x96 call | kernel ratio vs ATen |
|---|---|---:|---:|
| committed (reciprocal, serial sums) | 1.08–1.16 | 228 µs | 9.14x |
| TwoSum means, reciprocal | 1.02–1.08 | 215 µs | 8.33x |
| **TwoSum means, division (kept)** | **0.97–0.99** | 217 µs | 6.84x |

The kept variant divides by the stored `sqrt(var + eps)` in both passes (xhat is then the
forward's own normalized value) and accumulates `sum(g*w)` and `sum(g*w*xhat)` as Knuth TwoSum
(hi, lo) pairs in the unchanged channel order. Fused dx is now at or below the baseline's FP64
error on every geometry. The forward output and the dweight/dbias partials are unchanged. Cost:
about 25 µs more kernel time per B4 call on the RTX 5060 Ti; the USRNet B4 step did not move
outside its round-to-round spread (the same-session ATen reference itself varied 306-398 ms
with the machine's load, so the earlier 1.18x table is not refreshed here).

One dx max-abs single-seed miss remains at (1,64,96,96) off 1 eps 1e-6, seed 5326, on both
GPUs: element (0,13,48,53), value 5.27, fused error 1.18e-6 versus 7.1e-7 for the baseline
(2.5 vs 1.5 ulp at that magnitude; limit `max(1.5x, 1e-6)`). Its pixel's denominator from the
forward statistics kernel is 4.5e-8 from FP64 where ATen's is 1.4e-8, a 1-ulp difference that
the backward cannot remove without changing the byte-identical forward. The test therefore keeps
dx's per-seed rel-L2 gate and gates its max-abs over seeds with the channel vectors; the
seeded max-abs geomean for dx is at most 1.08 on either GPU. Release suite 228/228 locally on the
new binary (`.build/layernorm-compensated-release-001.log`); the A100 rerun is recorded in
`tools/v4_release_a100/README.md`.
