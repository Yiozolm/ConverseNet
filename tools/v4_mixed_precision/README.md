# FP16 / BF16 boundary experiments

This is an opt-in research suite for the user's v4.0.0 low-precision study.
The production Python and C++ APIs still require FP32. Existing FP32 release
budgets, checked build flags, and historical experiment decisions are unchanged.

The supplied `converse2d_fp16_bf16_plan.md` (SHA256
`9fb6a7b79743a1f797f11df04cfcba85f56b37de9d2d2bff3109cf6cfcd57127`)
is the design reference. This first round covers its Level 1A and 1B: low
precision spatial inputs and optional weights, FP32 solver, optional output
cast. It does not implement native low-precision FFT or compressed spectra.

## Fixed experimental contract

- Both FP16 and BF16 are measured. Master bias, sigmoid, epsilon, power,
  denominator, reduction, division, FFT and IFFT remain FP32/complex64.
- The experimental adapter preserves shared input/prior identity, differentiable
  casts and the existing full-training / half-inference routing. It does not
  cache training spectra, alter FFT sizes, enable AMP/TF32, or change CUDA flags.
- An FP32 master weight may remain FP32 or have an explicitly measured low
  precision storage copy. Casts happen on every call and count toward timing.
- Four references separate original FP32 error, input representation error,
  implementation error, and total error. Independent FP64 references are test
  code only. The frozen FP32 reference is never recalibrated to a candidate.
- Normal total relative-L2 and max-abs errors must be at most 1.25 times the
  quantized reference error; weak cases allow 1.50. Extra implementation error
  must be at most 0.25 times the quantized error. Near-zero absolute floors are
  relative-L2 `1e-7` and max-abs `1e-6`, fixed before running the matrix.
- Output casting is recorded separately, and cannot rescue a failing FP32
  internal output. Low precision leaf gradient casting is also separated from
  the core solver's FP32 gradient error. All nonfinite values fail safety.
- These numerical gates establish implementation correctness **relative to
  quantization**. They do not establish acceptable restoration quality or
  convergence. Pretrained PSNR/SSIM distributions must be measured before
  adopting a production quality threshold.

## Programs

- `adapter.py`: explicit experimental call boundary; the caller loads the
  checked extension before selecting `backend='cuda'`.
- `policy.py`, `gate.py`: four-reference operator matrix and numerical policy.
- `model_study.py`: pretrained DnCNN, MSRResNet and USRNet sensitivity study.
  Quantize/dequantize hooks simulate boundary representation loss while model
  computation remains FP32. These are not complete AMP model benchmarks.
- `perf.py`: actual low precision input storage passed through the adapter;
  complete-call AB/BA CUDA-event and synchronized wall timing, input/output cast
  probes, allocator peak increments and profiler component attribution.
- `training.py`: three seeds and three Adam steps with FP32 masters/state,
  manual activation casts, FP16 loss scaling and FP32 output. This is an
  operator diagnostic, not whole-model mixed-precision training.
- `profile_worker.py`, `profile_ncu.py`: a separate complete-call DRAM counter
  diagnostic using application replay. The three dtypes share source/build and
  original input identities. Profiler kernel times are not latency benchmarks.

Use the repository's checked build and `tools/run.ps1` with the project PyTorch
environment. Every report requires a fresh path and preserves failed rows.
Do not run competing GPU jobs. Model quality evaluation and profiling run
separately from latency measurements. Source/build hashes identify the exact
experiment; an archived result is not a substitute for a fresh run.

Logical storage bytes and allocator peaks are not measured DRAM traffic.
Level 1 still reads and writes complex64 spectra, so a smaller external tensor
does not imply a faster spectral kernel. In particular, casting a low precision
weight each call changes its identity and prevents warm spectrum reuse; this is part of the cost
being measured, not a reason to hide its cast outside timing.

No Phase 3–5 kernel is selected solely because the boundary experiments pass.
Any later compressed-storage kernel needs its own correctness, quality,
complete-call performance and measured bandwidth evidence.

## Reproduction

After configuring `CONVERSE_PYTHON` and building the checked extension as in
`test/README.md`, run these sequentially. Choose fresh output paths each time.

```powershell
$env:CONVERSE2D_SKIP_BUILD='1'
./tools/run.ps1 tools/v4_mixed_precision/gate.py --output artifacts/v4_campaign/mixed_gate_NEW.json
./tools/run.ps1 tools/v4_mixed_precision/model_study.py --output artifacts/v4_campaign/mixed_models_NEW.json --samples 6
./tools/run.ps1 tools/v4_mixed_precision/model_study.py --output artifacts/v4_campaign/mixed_models_usr_s1_NEW.json --models usrnet --usr-scale 1 --samples 6
./tools/run.ps1 tools/v4_mixed_precision/training.py --output artifacts/v4_campaign/mixed_training_NEW.json
./tools/run_affinity.ps1 -Mask 0xC03C03 -MetadataPath artifacts/v4_campaign/mixed_perf_NEW.affinity.json tools/v4_mixed_precision/perf.py --gate artifacts/v4_campaign/mixed_gate_NEW.json --output artifacts/v4_campaign/mixed_perf_NEW.json
```

The affinity mask above describes the measured machine; use a valid, explicitly
recorded mask on another host. Models require an audited image manifest (pass
`--manifest`); the local default is `artifacts/v4_campaign/dataset_absolute.json`.
`--synthetic` is an explicit alternative for numerical sensitivity without GT
quality claims. No images or model weights are downloaded by these programs.

The gate intentionally rejects range probes, so its overall `passed` is false
and its numerical-rejection exit status is nonzero even when classification
works correctly. Check `status`, `complete`, `range_probe_classification_passed`
and the separate admitted precision/output partitions. Performance measures
only partitions that passed their inference gate; failures stay in the report.
Do not call all 972 cases a pass or reinterpret a completed run as admission.

Actual DRAM-counter reproduction on the installed Nsight Compute host:

```powershell
./tools/run.ps1 tools/v4_mixed_precision/profile_ncu.py --dtype fp32 --output artifacts/v4_campaign/mixed_ncu_fp32_NEW
./tools/run.ps1 tools/v4_mixed_precision/profile_ncu.py --dtype fp16 --identity-from artifacts/v4_campaign/mixed_ncu_fp32_NEW/launcher.json --output artifacts/v4_campaign/mixed_ncu_fp16_NEW
./tools/run.ps1 tools/v4_mixed_precision/profile_ncu.py --dtype bf16 --identity-from artifacts/v4_campaign/mixed_ncu_fp32_NEW/launcher.json --output artifacts/v4_campaign/mixed_ncu_bf16_NEW
```

The default NCU path is specific to this Windows host; `--ncu` can select an
installed executable. The initial mismatched deterministic-fill profiler
protocol is retained under `history/` and excluded from traffic conclusions.
See [RESULTS.md](RESULTS.md) for measured outcomes and admission decisions.
