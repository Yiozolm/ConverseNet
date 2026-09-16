# Low-precision operator work — first stage

This document records the first-stage implementation and measurements. The
subsequent [training fusion stage](TRAINING_FUSION.md) now implements the
trainable spectral forward/VJP that was pending in this snapshot.

Design: [low-precision operator research and unified-entry agreement](https://chatgpt.com/s/cx_6aa41acec2f48191a0c03ce16a69aab4),
read on 2026-09-11. The replacement link confirms one public entry, dispatch by
activation dtype, FP32 master parameters, and separate training/inference
execution with a shared mathematical definition. This stage implements the AMP
boundary, FP32 normalization statistics, low-precision inference tiling and
output fusion. Training spectral fusion and full ConverseBlock fusion are
subsequent stages, not completed by this change.

## Implementation

The public entry/cache remains in `converse2d.cpp`. `converse2d_fp32.h` holds
the FP32/FP64 operator; `converse2d_low_precision.h` and
`converse2d_low_precision.cu` hold FP16/BF16 execution and storage kernels.
The internal implementation headers share one C++ translation unit so cache
ownership and registration remain unique. CUDA kernels compile separately.

- Accept FP16/BF16 activations with FP32 master weights and regularizers without
  rounding those parameters down. Return the activation dtype. Keep the FP64
  contract strict, and reject unrelated dtype mismatches.
- Make both Python and native model paths usable under AMP for training and
  inference. Explicit CPU/CUDA autocast fallthrough registrations preserve the
  actual activation dtype (including FP32/FP64); surrounding Conv/Linear select
  their AMP dtype. Existing checkpoint keys and operator signatures are unchanged.
- Compute channels-first LayerNorm statistics in FP32 for FP16/BF16 activations,
  retaining its previous affine/output promotion rule and parameter names.
- Reuse the FP32 L2 tile budget for FP16/BF16 inference, including incomplete
  last tiles, independent priors, nearest priors and dynamic/broadcast filters.
  Promote only one input/prior tile and write directly to the final low-precision
  output slice, avoiding full-batch FP32 input and output buffers on tiled calls.
- Promote the LR activation before differentiable nearest interpolation, reducing
  temporary allocation and performing the interpolation's backward reduction in
  FP32. Independent priors retain their own gradient path.
- Fuse post-C2R FP32 scaling and the low-precision output store. FFT normalization
  order and spatial sizes remain unchanged. No fast-math is enabled.
- Preserve eager cache invalidation, direct-capture fallback and graph-owned plan
  lifetime rules. The optional USRNet graph runner's dtype restrictions are unchanged.
- Include the local FFT header hash in JIT build flags so header-only changes
  rebuild correctly even when localized MSVC output hides Ninja include dependencies.

## Numerical and performance checks

```sh
python test/test_low_precision.py
python test/test_low_precision.py --cpu
python test/benchmark_low_precision.py --iters 20 --rounds 5
```

The dedicated suite covers mixed parameter precision, all filter broadcast modes,
independent and nearest priors, noncontiguous inputs, cache mutations and dtype
transitions, first/second derivatives, extreme input magnitudes, direct CUDA Graph
capture and graph-owned plans. The AMP integration case runs a two-iteration USRNet
through loss scaling, unscaling, backward and a verified optimizer update. All
three pretrained models also run FP16/BF16 inference against the Python backend.
CPU checks cover the operator; CUDA is used for end-to-end AMP because this
machine's oneDNN reports unsupported BF16 convolution backward.

The benchmark builds commit `85b9f80fa40dceedd88a5f543fd727617c7cf7bc` in an isolated
namespace and alternates baseline/current order. Both use the same low-precision
inputs and parameters. It measures inference and forward plus gradients of every
input; training timings exclude the optimizer. JSON includes CUDA-event and wall
times, incremental peak allocation, FP64-reference error and forward bitwise
agreement. It does not measure dataset PSNR or time to convergence.

Results are written to `artifacts/low_precision_benchmark.json` and are specific
to the GPU, CUDA/PyTorch versions, shapes and repetition counts recorded there.
Small differences near 1x should be treated as measurement noise.

## Measured first-stage results (2026-09-11)

RTX 5060 Ti, PyTorch 2.11.0+cu130, CUDA 13.0; 9 alternating rounds of 100 calls,
same inputs and original baseline commit in one process. The following are
operator inference CUDA-event medians, not whole-network timings:

| Dtype | Input B,C,H,W | Scale | Baseline ms | Current ms | Speedup | Peak-extra MiB, baseline → current |
|---|---|---:|---:|---:|---:|---:|
| FP16 | 16,32,64,64 | 2 | 0.9041 | 0.4386 | 2.062x | 80.75 → 56.38 |
| BF16 | 16,32,64,64 | 2 | 0.9130 | 0.4413 | 2.069x | 80.75 → 56.38 |
| FP16 | 17,32,64,64 | 2 | 0.9904 | 0.5238 | 1.891x | 85.80 → 57.38 |
| BF16 | 17,32,64,64 | 2 | 0.9876 | 0.5083 | 1.943x | 85.80 → 57.38 |

All 16 measured forward outputs (10 inference, 6 training) matched the frozen
baseline bit for bit on this environment. This is a measured result, not a
cross-GPU/cuFFT guarantee. FP64-reference comparisons passed independently.
The six forward/backward speed ratios ranged from 0.954x to 1.040x, with unchanged
peak-extra allocation: no material training speedup is claimed at this stage.
Small inference cases ranged from 0.895x to 1.220x in the full run; see the raw
rounds and the focused repeat below rather than assuming every shape speeds up.

Validation after the file split: 13 CUDA low-precision tests passed; the CPU
suite passed 8 tests with 5 CUDA-specific skips; 12 original correctness tests
and 6 FFT-batching tests passed. Additional C2R, nearest-prior, batched-kernel
and pretrained FP32/FP64 regression suites passed during implementation.
The low-precision suite includes pretrained DnCNN, SRResNet and USRNet AMP
inference, and an actual USRNet optimizer update; it does not establish dataset
quality or training convergence.

Full run: `artifacts/low_precision_benchmark.json`.
The initial, noisier pre-tiling experiment is retained separately in
`artifacts/low_precision_benchmark_initial.json` and is not the final code's result.

The FP16 small scale-3 outlier can be reproduced in isolation with:

```sh
python test/benchmark_low_precision.py --mode inference --dtype float16 --case small_s3 --iters 200 --rounds 15 --output artifacts/low_precision_small_s3_repeat.json
```

That independently seeded same-shape repeat measured 0.1665 → 0.1617 ms
(1.030x), with bitwise-equal output. The full-run 0.895x result was therefore
not stable across these runs; neither result establishes a reliable small-case
speedup. Both raw measurements are retained.

## Remaining stages

1. Fuse the trainable half-spectrum solve and its matching VJP, with correct
   broadcast/Hermitian reductions and a verified second-derivative fallback.
2. Extend nearest-prior phase synthesis to training and fuse the full
   ConverseBlock's normalization, pointwise operations and residual boundaries.
3. Coordinate whole-network workspace/graphs and measure full training steps,
   convergence and dataset PSNR/SSIM against the FP32 model.
4. Investigate native low-precision FFT only for supported shapes and controlled
   amplitudes, with automatic FP32 fallback and unchanged spatial semantics.

## Precision policy references

- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html): mixed precision and
  gradient scaling, including FP32 handling for FFTs.
- [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/index.html): native half/BF16
  transform restrictions. This stage keeps the existing FP32/FP64 FFT path.
