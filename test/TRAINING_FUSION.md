# CUDA training: fused half-spectrum forward and VJP

This stage follows the [unified dtype-dispatch design](https://chatgpt.com/s/cx_6aa41acec2f48191a0c03ce16a69aab4).
It replaces the FP16/BF16 **CUDA training** spectral solve with a differentiable
custom operator. Public signatures and activation/master-parameter contracts
are unchanged. FP32 CUDA training now also selects this fused solver. FP64,
CPU, and inference keep their existing paths. FP32 FFTs always use FP32,
including inside an experimental `native_fft()` scope. The measurements below
describe the original low-precision stage; the newer FP32 paired results and
reproduction commands are in `artifacts/fp32_training/report.md`.

## Files and execution

- `Converse2D/torch_converse2d/converse2d_training.h`: custom autograd node,
  validated internal spectral entry, FP32 spatial/FFT preparation and exact
  higher-order fallback.
- `Converse2D/torch_converse2d/converse2d_training.cu`: half-spectrum forward
  and VJP kernels, including deterministic broadcast reductions.
- `converse2d.cpp` and `converse2d_low_precision.h`: route FP32 and low-precision
  CUDA calls with autograd enabled to this backend. CPU low-precision calls
  still use the FP32 reference core.

The forward performs alias prediction and power accumulation in one kernel and
applies the correction in another. The first backward gathers the correction's
adjoint, computes prior/filter gradients, and reduces the regularizer gradient.
Kernel gradients use a fixed summation order over broadcast batch/channels;
no expanded per-example filter gradient or floating-point atomic buffer is used.
When a filter has no broadcast dimensions, its gradient is fused with the prior
gradient. Unrequested prior/filter/regularizer gradients are not materialized.

FFT, PSF padding/centering, nearest interpolation, sigmoid and their derivatives
remain ATen operations. They run in FP32 for low-precision activations. The
training backend does not read detached inference caches.

## Derivative definition

Write `A` for alias averaging on a Hermitian half spectrum, `E` for alias
expansion with conjugate reads, and `K`, `P`, `Y` for kernel, prior and observation:

```
D = A(|K|²) + lambda
Q = (Y - A(K P)) / D
Z = P + conj(K) E(Q)
```

For the complex upstream gradient `G`, using the real inner product on complex
storage, the VJP is:

```
R        = E*(K G) / D
T        = -Re(conj(R) Q)
grad_Y   = R
grad_P   = G - conj(K) A*(R)
grad_K   = broadcast_sum(conj(G) E(Q) - conj(P) A*(R) + 2 K A*_real(T))
grad_lam = sum_over_batch_and_frequency(T)
```

The adjoints gather both direct and conjugated aliases. Missing columns reflect
**both** row and column; DC and even-size Nyquist have different multiplicities
from interior columns. The tests use arbitrary complex spectra that deliberately
violate real-FFT boundary symmetry, so the derivative is checked for all stored
components rather than relying on a special upstream gradient.

For a shared scale-1 observation/prior, autograd adds both formal-argument
gradients. For nearest interpolation it also differentiates the prior through
the FP32 interpolation graph. FP32 kernel/bias gradients stay FP32.

## Higher-order derivatives

The first-order CUDA VJP is not itself differentiable. With `create_graph=True`,
backward recomputes the ATen spectral expression from saved original inputs and
uses autograd to obtain a differentiable VJP. Separate connected views are used
for each formal argument, preventing duplicate gradients when `Y` and `P` share
the same tensor. Saved tensor version checks continue to reject in-place edits.

The recomputation intentionally restores full-spectrum/repeat temporaries for
higher-order work; reported speedups apply to ordinary first-order training.
The internal spectral entry supports FP64 so complex gradcheck/gradgradcheck
can validate the CUDA VJP and this fallback independently of activation rounding.

## Reproduction and scope

```sh
python test/test_training_fusion.py
python test/test_fp32_training.py
python test/test_low_precision.py
python test/benchmark_training_fusion.py --iters 20 --rounds 5
```

The benchmark freezes commit `c4b950908645e616fedd24c84ea3c845d051126c` (the completed
mixed-precision/inference stage), in a separate namespace. Operator cases use
the same low activations, **FP32 master parameters**, scale, prior and upstream
gradient for both implementations. They separately time forward, retained-graph
backward, and forward+backward; the latter does not include an optimizer.
The synthetic USRNet case uses two iterations and one block, times complete
SGD/AMP training steps, and compares a short loss/parameter-update trace.

JSON includes all timing rounds, wall times, incremental peak allocated memory,
FP64-reference output/gradient errors, zero-gradient fractions and loss traces.
These results do not establish full-size model convergence or dataset PSNR/SSIM.
Nearest phase synthesis in training, full ConverseBlock fusion and native
low-precision FFT remain later work.

## Measured results — 2026-09-11

Final run: RTX 5060 Ti, PyTorch 2.11.0+cu130, CUDA 13.0; 7 alternating rounds of
30 calls. Builds and other test jobs had completed before this run. The numbers
are medians on this workstation, not guarantees for other GPUs or workloads.

| Dtype | B,C,H,W | Scale/prior | Forward+backward ms, old → new | Speedup | Peak-extra MiB, old → new |
|---|---|---|---:|---:|---:|
| FP16 | 1,32,64,80 | 1 / nearest | 1.5629 → 0.9184 | 1.702x | 8.93 → 6.69 |
| FP16 | 1,32,64,80 | 2 / independent | 2.5446 → 1.0358 | 2.457x | 28.41 → 22.58 |
| FP16 | 4,32,32,40 | 3 / nearest, per-example filter | 2.1142 → 0.9116 | 2.319x | 63.44 → 51.63 |
| FP16 | 8,32,64,64 | 2 / nearest | 3.9750 → 1.7817 | 2.231x | 149.66 → 116.41 |
| BF16 | 1,32,64,80 | 1 / nearest | 1.3048 → 0.7775 | 1.678x | 8.93 → 6.69 |
| BF16 | 1,32,64,80 | 2 / independent | 2.0544 → 0.8395 | 2.447x | 29.16 → 23.33 |
| BF16 | 4,32,32,40 | 3 / nearest, per-example filter | 2.4146 → 1.0705 | 2.255x | 63.06 → 50.78 |
| BF16 | 8,32,64,64 | 2 / nearest | 3.9753 → 1.7674 | 2.249x | 149.66 → 116.41 |

The synthetic **two-iteration, one-block** USRNet case (`B=2`, RGB `16x20`,
scale 2) includes zeroing gradients, forward, loss, backward, SGD update and
FP16 GradScaler operations:

| Dtype | Full step ms, old → new | Speedup | Peak-extra MiB, old → new |
|---|---:|---:|---:|
| FP16 | 13.0629 → 10.3960 | 1.257x | 48.95 → 41.26 |
| BF16 | 14.9436 → 11.4253 | 1.308x | 48.95 → 41.26 |

All gradients and final model parameters were finite. Against FP64 references
on the same quantized activations, the largest relative L2 error across the
measured FP32 master-parameter gradients was `2.31e-6`. Maximum activation-gradient
relative L2 errors were `2.08e-4` (FP16) and `1.66e-3` (BF16), including the final
gradient's activation-dtype rounding.

After the 10-step paired trace, maximum parameter differences were `4.39e-6`
(FP16) and `2.94e-4` (BF16). After the timed training steps, evaluation losses on
the synthetic fixed batch were `0.42528 / 0.42565` (baseline/current FP16) and
`0.46145 / 0.46474` (BF16). This checks finite, comparable short training behavior;
it is not a dataset convergence or reconstruction-quality result.

Validation: 9 new fusion tests, 13 mixed-precision tests, 12 existing correctness
tests and 8 batched-kernel tests passed on CUDA. CPU passed 8 applicable
mixed-precision tests, with 5 CUDA-specific skips. New tests include complex
gradcheck/gradgradcheck, all four filter broadcast modes, all 15 nonempty
requested-gradient combinations, conjugate/strided views, shared FY/prior,
weak regularization, zero filters, nondefault streams and saved-input mutations.

Full distributions and errors: `artifacts/training_fusion_benchmark.json`.
Preliminary development timings are retained in
`artifacts/training_fusion_benchmark_initial.json` and are not used in the table.

## References

- [PyTorch C++ custom autograd](https://docs.pytorch.org/cppdocs/api/autograd/custom_functions.html)
- [PyTorch complex autograd convention](https://docs.pytorch.org/docs/stable/notes/autograd)
- [PyTorch double-backward requirements](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)
