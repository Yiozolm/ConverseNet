# FP32 release branch

- Public tensors are FP32, spectra complex64; reject other dtypes.
- GradMode plus any differentiable input selects full-spectrum training.
  no_grad/inference_mode and frozen inputs use half-spectrum inference.
- Preserve Python FP32 accuracy, broadcast reductions, shared-input gradient order,
  per-call differentiable kernel FFT, and higher-order ATen fallback.
- No fast-math, AMP, TF32, or training spectrum reuse for speed.
- FP64 is only an independent Python test reference, not a production backend.
- Run checked builds and release tests. Short trajectories do not prove convergence.
- Historical source and all successes/failures are preserved at commit 1b579ea
  and on codex/training-operator-optimization. Do not relabel old failures.
