# FP32 release branch

- Public tensors are FP32, spectra complex64; reject other dtypes.
- GradMode plus any differentiable input selects full-spectrum training.
  no_grad/inference_mode and frozen inputs use half-spectrum inference.
- Validate arithmetic changes against the frozen FP32 baseline and independent
  Python FP64 reference using test/numerical_policy.py. Normal relative-L2 may
  grow by 1.25x, weak by 1.50x; max-abs by 1.50x, with documented floors.
  Bitwise identity is an optional arithmetic diagnostic, not the accuracy gate.
- Preserve broadcast reductions, shared-input gradient order,
  per-call differentiable kernel FFT, and higher-order ATen fallback.
- No fast-math, AMP, TF32, or training spectrum reuse for speed.
- FP64 is only an independent Python test reference, not a production backend.
- Run checked builds and release tests. Short trajectories do not prove convergence.
- Historical source and all successes/failures are preserved at commit 1b579ea
  and on codex/training-operator-optimization. Do not relabel old failures.
