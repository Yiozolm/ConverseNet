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
| `test_full_spectrum_default.py` | Full/half routing; exact FP32 output/VJP; shared/broadcast kernels, gradient masks, weak regularization and strided layouts |
| `test_fp32_release.py` | Independent FP64 noninferiority, higher derivatives, cache/streams, dtype contracts and CPU fallback |
| `test_gradient_mask.py` | All gradient subsets, broadcast reductions, higher derivatives and masked backward Graph replay; verifies unused reductions are skipped |
| `test_inference_p0.py` | Frozen-input cache reuse without changing ATen arithmetic, cross-GradMode cache hits and layout invalidation |
| `test_negative_metadata.py` | Lazy negative kernels, cold PSF materialization and same-pointer/version cache invalidation across all three GradMode contexts |
| `test_psf_preparation.py` | PSF boundary/layout cases, shared-ancestor gradient order, second/third derivatives, streams and per-call kernel FFT |
| `test_scale2_kernel_fusion.py` | Exact non-broadcast s2 kernel VJPs, weak regularization, gradient subsets, conjugated spectra and Graph replay |
| `test_scale3_fusion.py` | Exact nine-alias reduction order, separate mean factors, broadcast VJPs, strided/weak-regularization cases and guarded fallback |
| `test_batch_reduce.py` | B2/B4 shared-kernel batch reductions, original accumulation order, gradient masks, fallback geometry, higher derivatives and Graph replay |
| `test_peripheral_fusion.py` | Alpha residual FP32 multiply/add boundaries, gradient subsets, aliases, higher derivatives, layouts and streams |
| `test_layernorm_affine.py` | Private affine values/gradients, full-inference dispatch precedence and unchanged training statistics |
| `test_layernorm_full.py` | Six ATen reduction geometries, alignment/tails, byte and FP64 gates, unsupported-input fallbacks, higher derivatives, streams and Graph replay |
| `test_peripheral_policy.py` | CPU-only automatic dispatch boundaries for large inference tensors, frozen/differentiable inputs and all three GradMode contexts |
| `test_cuda_graph.py` | Graph ownership, replay, invalidation and training transitions |
| `test_pretrained_fp32.py` | DnCNN/SRResNet checkpoint and inference compatibility |

`support.py` owns shared fixtures, byte comparisons and CPU/CUDA policy.
`extension_loader.py` builds locally under `.build/` and verifies source/header
and binary fingerprints. Tests never import fixtures from another test module.
The FP32 tests retain their numerical thresholds, shapes and random seeds.
Three additional CUDA Graph tests cover the optional execution switch, CPU
passthrough and training passthrough inherited from main.

Release benchmarks, quality campaigns, snapshots and note generation are local
tools under `tools/release/`, excluded by `.gitignore`; they are not needed for
this suite. Their measured source versions remain in commit `0a99235` under the
old `test/` paths. Release notes and measured results stay under `docs/`.

The tracked `tools/benchmark_fp32_p0.py` compares isolated checked builds for
this optimization round. It records complete operator/VJP timing, memory,
FP64 errors and tensor hashes; `--include-model` adds full USRNet inference
and Adam steps. See `docs/fp32_p0.md` for the measured scope and rejected candidate.

`tools/benchmark_fp32_psf.py` measures the next PSF/s2 training-fusion batch.
Its explicit `--deterministic-algorithms` lane makes external replicate-padding
backward repeatable for exact comparisons. Default-mode results and their
non-repeatable input gradients are preserved separately; see `docs/fp32_psf.md`.

`tools/benchmark_fp32_roadmap.py` extends complete operator and model measurements
to s3, peripheral expressions and Graph miss/hit lifetimes.
`tools/benchmark_peripheral_inference.py` alternates original and fused inference
expressions and compares both Graph runners using the same model object.
`tools/profile_fp32_roadmap.py` records a fresh full-model Torch or Nsight trace.
The independently checked real-photo fine-tuning and resume tools live under
`tools/roadmap_quality/`; their results do not replace numerical release gates.
