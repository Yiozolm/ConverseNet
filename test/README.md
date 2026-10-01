# FP32 regression tests

Run from the repository root:

```sh
python -m unittest discover -s test -p 'test_*.py' -v
```

On Windows, use `./tools/run.ps1` instead of `python` to initialize the compiler.
Set `CONVERSE2D_CPU_ONLY=1` for a CPU build: all CUDA-specific cases are skipped,
while build checks, the CPU extension and the portable Python fallback still run.

| File | Coverage |
| --- | --- |
| `test_build_layout.py` | Source selection and transitive fingerprints; no Torch/compiler required |
| `test_full_spectrum_default.py` | Full/half routing; budgeted FP32 output/VJP; shared/broadcast kernels, gradient masks, weak regularization and strided layouts |
| `test_fp32_release.py` | Frozen FP32 baseline, independent FP64 budgets, denominator statistics, training and three inference contexts, higher derivatives, cache/streams and dtype contracts |
| `test_numerical_policy.py` | Budget boundaries, improved-but-different candidates, nonfinite rejection, zero baseline, outliers and report preservation |
| `test_gradient_mask.py` | All gradient subsets, broadcast reductions, higher derivatives and masked backward Graph replay; verifies unused reductions are skipped |
| `test_inference_p0.py` | Frozen-input cache reuse without changing ATen arithmetic, cross-GradMode cache hits and layout invalidation |
| `test_negative_metadata.py` | Lazy negative kernels, cold PSF materialization and same-pointer/version cache invalidation across all three GradMode contexts |
| `test_psf_preparation.py` | PSF boundary/layout cases, shared-ancestor gradient order, second/third derivatives, streams and per-call kernel FFT |
| `test_scale2_kernel_fusion.py` | Budgeted non-broadcast s2 kernel VJPs, weak regularization, gradient subsets, conjugated spectra and Graph replay |
| `test_scale3_fusion.py` | Budgeted nine-alias cancellation, large mean counts, broadcast VJPs, strided/weak-regularization cases and guarded fallback |
| `test_batch_reduce.py` | B2/B4 shared-kernel batch reductions, original accumulation order, gradient masks, fallback geometry, higher derivatives and Graph replay |
| `test_peripheral_fusion.py` | Alpha residual FP32 multiply/add boundaries, gradient subsets, aliases, higher derivatives, layouts and streams |
| `test_layernorm_affine.py` | Private affine values/gradients, full-inference dispatch precedence and unchanged training statistics |
| `test_layernorm_full.py` | Six ATen reduction geometries, alignment/tails, byte and FP64 gates, unsupported-input fallbacks, higher derivatives, streams and Graph replay |
| `test_peripheral_policy.py` | CPU-only automatic dispatch boundaries for large inference tensors, frozen/differentiable inputs and all three GradMode contexts |
| `test_cuda_graph.py` | Graph ownership, replay, invalidation and training transitions |
| `test_pretrained_fp32.py` | DnCNN/SRResNet/USRNet pretrained outputs and three-seed, three-step USRNet/Adam regression |

`support.py` owns shared fixtures, byte comparisons and CPU/CUDA policy.
`extension_loader.py` builds locally under `.build/` and verifies source/header
and binary fingerprints. Tests never import fixtures from another test module.
Existing fixtures retain their shapes and random seeds. Public operator spatial
comparisons use FP64 error budgets, including shared-ancestor gradients; byte
checks still protect copying, cache consistency and private peripheral operations.
Three additional CUDA Graph tests cover the optional execution switch, CPU
passthrough and training passthrough inherited from main.

Release benchmarks, quality campaigns, snapshots and note generation are local
tools under `tools/release/`, excluded by `.gitignore`; they are not needed for
this suite. Their measured source versions remain in commit `0a99235` under the
old `test/` paths. Experiment reports and measured results remain in Git history
at commit `8750661`; they are excluded from the release tree.

The tracked `tools/benchmark_fp32_p0.py` compares isolated checked builds for
this optimization round. It records complete operator/VJP timing, memory,
FP64 errors and tensor hashes; `--include-model` adds full USRNet inference
and Adam steps.

`tools/benchmark_fp32_psf.py` measures the next PSF/s2 training-fusion batch.
Its explicit `--deterministic-algorithms` lane makes external replicate-padding
backward repeatable for exact comparisons. Default-mode results and their
non-repeatable input gradients are preserved in the historical experiment reports.

`tools/benchmark_fp32_roadmap.py` extends complete operator and model measurements
to s3, peripheral expressions and Graph miss/hit lifetimes.
`tools/benchmark_peripheral_inference.py` alternates original and fused inference
expressions and compares both Graph runners using the same model object.
`tools/profile_fp32_roadmap.py` records a fresh full-model Torch or Nsight trace.
The independently checked real-photo fine-tuning and resume tools live under
`tools/roadmap_quality/`; their results do not replace numerical release gates.

## v4 numerical acceptance

`fp32_baseline.py` is an unchanged copy of `models/converse_core.py` from
`0d636215e23d472825e09cc340a99f1d2ae2b78c`, with a newline-normalized SHA-256
guard in `numerical_policy.py`. It is test-only and must not follow candidate
arithmetic changes. Training uses its full-spectrum FP32 expression; inference
uses its half-spectrum FP32 expression, so FFT representation differences do not
silently become a new accuracy allowance. Both are compared against the current
independent Python full-spectrum FP64 reference using identical FP32 input values
promoted to FP64, including identical upstream gradients.

Every output and each of dx/dprior/dweight/dbias must be finite and pass:

| Regime | Relative L2 limit | Maximum absolute error limit |
| --- | --- | --- |
| Normal (`eps=1e-5`) | `max(1.25 * baseline, 1e-7)` | `max(1.50 * baseline, 1e-6)` |
| Weak (`eps=1e-8`, bias=-40) | `max(1.50 * baseline, 1e-7)` | `max(1.50 * baseline, 1e-6)` |

These are initial project budgets, not IEEE guarantees. The supplied specification
does not fix the weak floor or a weak max-abs bound; this implementation starts
with the normal floor and retains a 1.50x max-abs safety gate. There is no automatic
2x stress exemption. Normal production inference also retains `atol=rtol=1e-5`
smoke checks. The dynamic-range stress fixture (input multipliers 1e-3 to 1e3)
records allclose diagnostically because a fixed spatial absolute tolerance is
not scale invariant; it still must pass BOTH normal FP64 budgets. Initial
calibration on the unchanged CUDA implementation exposed six such smoke failures
while all 1,760 FP64 comparisons passed; the original failed report is retained.
Floors apply separately to each tensor; relative percentiles use a diagnostic
near-zero floor and never decide admission. A ratio with a zero baseline and
nonzero candidate is JSON `null`; the explicit budget still decides pass/fail.

The full matrix covers s1–s4, all four kernel broadcasts, contiguous/sliced/
transposed inputs, odd/even extents, width one, partial block tails, normalized
kernels, dynamic range, near-zero inputs/kernels, cancellation, and weak kernel
amplitudes 0, 1e-6 and 1e-3. Each fixture exercises training, no_grad,
inference_mode, and GradMode with frozen inputs. The CPU fallback runs the fast
matrix independently; CUDA absence is a skip, never a CUDA validation success.

```sh
# Fast PR gate: 16 cases, all scales/broadcasts, full gradients and inference.
CONVERSE2D_NUMERICAL_LEVEL=fast python -m unittest discover -s test -p test_fp32_release.py -v
# Arithmetic changes and release: full matrix (default), plus all semantic tests.
python -m unittest discover -s test -p 'test_*.py' -v
# Optional historical bit identity in public spatial comparisons.
CONVERSE2D_EXACT_REGRESSION=1 python -m unittest discover -s test -p test_full_spectrum_default.py -v
```

On PowerShell set `$env:CONVERSE2D_NUMERICAL_LEVEL='fast'` (remove it for a full
release run) and invoke `./tools/run.ps1` with the same Python arguments.
Use `CONVERSE_PYTHON` to select an existing PyTorch environment when needed.

Reports are uniquely named JSON files under `artifacts/fp32_release/` (override
with `CONVERSE2D_REPORT_DIR`). They preserve failed rows, max-abs/relative-L2,
P50/P90/P99/P99.9/max diagnostic relative error, effective limits and ratios,
FP64 denominator min/P01/median/max, baseline identity, source/binary checked-build
manifest, Git state, device and PyTorch/CUDA versions. Keep the initial run and
each optimization's reports together with complete-operator timing results.
Never overwrite a failed experiment or refresh the baseline to make it pass.

Pretrained DnCNN/SRResNet and USRNet s1–s4 record model output max-abs/relative-L2
against Python FP32 and keep the 1e-5 spatial gate. No PSNR/SSIM claim is made
without a ground-truth evaluation dataset. The short USRNet training test uses
two iterations/one block, fixed data and initial weights for seeds 17/29/43,
three Adam steps, and records output, loss, gradients, updated parameters and
optimizer tensors with a 3e-5 model smoke tolerance. It is not proof
of convergence; long training/quality campaigns remain necessary for release
claims involving convergence.

Private peripheral and preparation tests still carry their own exact contracts.
An optimization that changes those operations must supply its own independent
FP64 budget test; do not globally weaken `assert_bytes_equal`. Cache and stream
consistency, spectrum routing, dtype rejection, higher derivatives and per-call
training kernel FFT checks remain required. Fast-math, AMP and TF32 stay disabled.
