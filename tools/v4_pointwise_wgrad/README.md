# Split-K pointwise weight gradient

`profile/sm120-bound-refresh-20261012` (HEAD `1c2faf9`, RTX 5060 Ti) attributed 111.8 ms of a
301.6 ms GPU kernel sum (37.1%) of the B4 USRNet training step to cuDNN's
`wgrad2d_grouped_direct_kernel`: the deterministic FP32 weight gradient of the 70 `C64->128`
pointwise convolutions in the prior (1.42 ms per call at `[4,64,96,96]`). NCU showed the kernel
at 78% SM throughput but only 12% DRAM throughput and about 2% of the FP32 roofline: address
arithmetic of a generic direct convolution, not matrix arithmetic. The earlier
[cuBLAS GEMM candidate](../v4_experiments/README_c128_to64.md) had been rejected for this
direction because its FP64 error exceeded the direct kernel's by up to 2.5x (63 failed tensors in
`artifacts/v4_campaign/wgrad_gate_v1.json`); only `C128->64` had been accepted, through two
layout copies plus cuBLAS.

## Kernel

`Converse2D/torch_converse2d/peripheral/pointwise_wgrad.cu`, op `converse2d::_pointwise_wgrad`:

- `dW[co][ci] = sum_{b,p} G[b][co][p] * X[b][ci][p]` directly on the NCHW operands; the
  contraction index `k = b*HW + p` is the contiguous one of each row, so neither operand is
  copied or permuted. The operand with more channels fills the 128-row side of a 128x64 tile.
- Split-K: a 256-thread block owns one output tile and 16 K tiles of 16 (256 terms), streamed
  through a three-stage `cp.async` pipeline into padded shared rows (stride 17, conflict-free
  column reads). Rows past the operand and columns past K are zero-filled, so any batch, any
  `H*W` and any channel count are correct (channels above the tile use more grid tiles).
- Each thread accumulates 8x4 outputs with FMA chains of 8 terms and folds every chain sum into
  a `(hi, lo)` pair with Knuth TwoSum. Per-split `(hi, lo)` partials are reduced in index order
  by a second kernel with the same compensation. The result is deterministic, and the chunking
  depends only on K, so a geometry rounds identically on every GPU.
- 128 registers, no spills, 40 KB shared, two blocks per SM; 72 blocks for the B4 geometry.

`models/pointwise.py` routes contiguous CUDA FP32 NCHW inputs with 64 or 128 input and output
channels and trainable weights through the kernel for the weight gradient only; forward,
`dx`/`dbias` (one `convolution_backward` call) and every higher-order VJP stay native ATen.
Inference, frozen weights, other channel counts, strided layouts, negative/conjugate views and the
`pytorch` backend keep native convolution. `ConverseBlockAlphaVariant` now uses `PointwiseConv2d`
for both pointwise convolutions of each branch (28 modules in USRNet); state-dict keys are unchanged.

## Gate (`gate.py`, `artifacts/v4_pointwise_wgrad/gate_rtx5060ti_001.json`)

Candidate against the deterministic cuDNN weight gradient (frozen FP32 baseline) and FP64
`F.conv2d`, normal budget (rel-L2 1.25x, max-abs 1.50x, floors 1e-7 / 1e-6):

| Fixture | Direction | Cases | Baseline rel-L2 (median) | Candidate rel-L2 (median) | Ratio rel-L2 max | Ratio max-abs max |
|---|---|---:|---:|---:|---:|---:|
| `wgrad_capture_v1.pt` (real activations) | C64->128 | 70/70 | 1.08e-7 | 2.57e-8 | 0.374 | 0.659 |
| `wgrad_capture_v1.pt` | C128->64 | 70/70 | 4.69e-7 | 2.54e-8 | 0.069 | 0.106 |
| `wgrad_capture_seed29.pt` | C64->128 | 70/70 | 1.08e-7 | 2.55e-8 | 0.368 | 0.564 |
| `wgrad_capture_seed29.pt` | C128->64 | 70/70 | 4.29e-7 | 2.54e-8 | 0.087 | 0.106 |

The candidate error is at the level of the final FP32 rounding of each element. All 14 synthetic
geometries (B1, 64x64, 48x48, K tails such as `3x64x7x5`, `H*W = 1`, channel counts 3/5/130/200
with partial and multiple tiles) pass with two seeds each, every call repeats bitwise, and the
production module returns exactly the operator result with the custom autograd node active.

## Measured (`perf.py --model`, `artifacts/v4_pointwise_wgrad/perf_rtx5060ti_001.json`)

RTX 5060 Ti, MSVC 14.44, CUDA 13.0, torch 2.11+cu130, GPU idle (7% desktop), deterministic
algorithms, cuDNN deterministic, TF32 off, 6 rounds x 30 iterations, routes alternated per round.
Medians of CUDA-event call time / profiler kernel sum, µs:

| Geometry | cuDNN deterministic wgrad | previous cuBLAS path | split-K kernel | Speedup vs cuDNN |
|---|---:|---:|---:|---:|
| `4x64x96x96 -> 128` | 1418 / 1525 | n/a (rejected) | 145 / 144 | **9.8x** / 10.6x |
| `4x128x96x96 -> 64` | 284 / 304 | 234 / 233 | 143 / 143 | 2.0x / 2.1x |
| `1x64x96x96 -> 128` | 345 / 369 | n/a | 45 / 31 | 7.6x / 12.0x |
| `1x128x96x96 -> 64` | 359 / 367 | 51 / 22 | 45 / 31 | 7.9x / 12.0x |

Complete `PointwiseConv2d` forward + `(dx, dweight, dbias)` VJP, `pytorch` route (native
convolution) against `auto`: `4x64x96x96->128` 1655 -> 498 µs (3.32x), `4x128x96x96->64`
739 -> 495 µs (1.49x), B1 511/525 -> 298/305 µs (1.7x).

Complete pretrained USRNet B4 LR32->HR96 s3 training step (zero_grad, forward, MSE, backward,
Adam; only the 28 `PointwiseConv2d` modules switched, every other Converse operator on the same
CUDA binary; 5 rounds x 5 steps, checkpoint and optimizer state restored before each round):

| Route | Step median | Rounds |
|---|---:|---|
| `pytorch` (native cuDNN weight gradients) | 389.3 ms | 367.2, 388.7, 389.3, 402.0, 400.7 |
| `auto` (split-K kernel) | 282.7 ms | 262.0, 282.7, 284.0, 281.7, 282.7 |

Step speedup **1.377x**; both routes reach the same first-step loss. The `pytorch` control uses
native convolution for both directions; against the previous production state (native
`C64->128`, cuBLAS `C128->64`) the gain is about 6 ms smaller, roughly 1.35x. These are short
timing windows on one machine, not convergence evidence.

The B4 kernel reads 28.3 MB per call, so its 145 µs is about 2.3x the 448 GB/s floor; the 4-byte
`cp.async` granularity forced by the stride-17 rows and the TwoSum folds (448 FADD per 512 FFMA
in the SASS) are the next things to examine with NCU. The remaining pointwise time is now cuDNN's
`dgrad_alg1_engine` for `dx` (about 20 ms per step) and the forward `implicit_convolve_sgemm`
(about 14 ms), both also 1x1 GEMMs.

## Verification

- Checked build of the complete source set; `test_pointwise_wgrad.py` 9/9 (both directions,
  all gradient masks, B1 and other geometries, raw-operator tails/bounds/determinism and
  contracts, native higher-order fallback, checkpoint compatibility, backend isolation, USRNet
  module routing).
- Complete unmodified release suite on the same binary: 228 tests, all passed
  (`.build/pointwise-wgrad-release-001.log`).
- Reproduce (Python arguments to `tools/run.ps1`; outputs must be new paths):

```text
test/extension_loader.py
tools/v4_pointwise_wgrad/gate.py --output artifacts/v4_pointwise_wgrad/gate_<gpu>_<nnn>.json
tools/v4_pointwise_wgrad/perf.py --output artifacts/v4_pointwise_wgrad/perf_<gpu>_<nnn>.json --model
```
