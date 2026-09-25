# Current FP32 paired quality checks

These tools restore the audited image sampling and RGB/Y metric definitions
from commit `1b579ea`. Original file identities are in `restored_sources.json`.
The current training worker uses a checked source checkout, with a new output
directory for every run. It does not load an old half-spectrum training backend.

The default protocol is full pretrained USRNet, 5 iterations/7 blocks, HR96/s3,
effective batch 4, Adam 1e-5, seeds 17/29/43, and 250 updates. Every evaluation
uses all 100 fixed held-out center crops from the existing 900/100 manifest.
This is short paired fine-tuning, not convergence or an independent test set.
All inputs and model operations remain FP32. Diagnostic gradient norms are now
FP32 as well; they do not affect the model update, but timings must not be mixed
with the old worker's FP64 diagnostic norms.

Before using the GPU, rerun the CPU protocol checks with the current NumPy,
SciPy and Pillow versions. They verify encoded image hashes, all 1000 decoded
RGB hashes, asymmetric-kernel convolution/phase, sampling, and holdout isolation:

```powershell
.venv/Scripts/python.exe tools/roadmap_quality/test_usrnet_training_data.py --output artifacts/roadmap_quality/data_protocol_checks.json
```

Use a separate process and the same build environment for each GPU run. First
perform a capacity pilot on both checkouts; previous half-spectrum capacity
measurements do not establish the current full-spectrum memory requirement.
Changing microbatch size changes the arithmetic path and must be identical
for both variants. No batch size, learning rate or stopping rule is adjusted
automatically.

```powershell
pwsh -NoProfile -File tools/run.ps1 tools/roadmap_quality/train_usrnet_dataset.py --root 'baseline checkout' --variant before --purpose pilot --steps 5 --eval-every 5 --run-dir artifacts/roadmap_quality/pilot_before
pwsh -NoProfile -File tools/run.ps1 tools/roadmap_quality/train_usrnet_dataset.py --variant current --purpose pilot --steps 5 --eval-every 5 --run-dir artifacts/roadmap_quality/pilot_current
```

The default verifies an existing source/binary manifest. Add `--build` only
when a checked build is needed. `--root` selects the code; `--variant` only
labels it. `--data-root`, manifest and checkpoint default to this repository
so that a baseline checkout can share the same read-only dataset and initial
checkpoint.

Run both variants for each seed, alternating their order between seeds:

```powershell
pwsh -NoProfile -File tools/run.ps1 tools/roadmap_quality/train_usrnet_dataset.py --root 'baseline checkout' --variant before --seed 17 --run-dir artifacts/roadmap_quality/before_seed17
pwsh -NoProfile -File tools/run.ps1 tools/roadmap_quality/train_usrnet_dataset.py --variant current --seed 17 --run-dir artifacts/roadmap_quality/current_seed17
```

Repeat for seeds 29 and 43. Each run records all batch hashes, finite checks,
parameter counts, evaluations, checkpoints, source identities and synchronized
phase times. The final source and binary identities must still match the start.
Use `--deterministic-algorithms` on **both** variants for the separate strict
algorithm lane. Default-lane failures are retained rather than relabelled.

The offline audit requires all six complete runs, matched recipes and data,
and each seed's final RGB/Y PSNR drop no greater than 0.05 dB and SSIM drop no
greater than 0.001. No average can rescue a failed seed:

```powershell
.venv/Scripts/python.exe tools/roadmap_quality/summarize_usrnet_training.py --root artifacts/roadmap_quality --output artifacts/roadmap_quality/paired_summary.json --markdown artifacts/roadmap_quality/paired_summary.md
```

Pass the same explicit `--steps`, `--eval-every`, `--batch-size`,
`--microbatch-size`, `--patch-size` and `--deterministic-algorithms` to the
summary when running a protocol that differs from the defaults. A formal
summary needs more than five steps. It refuses incomplete or mismatched runs
and never overwrites the historical dataset-training results by default.

`evaluate_usrnet_quality.py` also retains the standalone full-image evaluator
for supplied HR/LR pairs and shared/per-image kernels. It requires explicit
input directories and `--output`; it does not synthesize or silently resize
external evaluation data. No external-data claim is made merely by restoring it.

## Opt-in long runs and safe resume

`--stop-when-stable` keeps the original optimizer/data recipe, requires at least
1000 updates, and evaluates every 250 updates. All four RGB/Y metrics must meet
their window limits over the last five consecutive evaluations: PSNR span
strictly below 0.02 dB and SSIM span strictly below 0.0005. A satisfied window is
recorded as `status="stable"`; it is not a claim of convergence or external
quality. If the maximum `--steps` is reached first, status is
`max_steps_reached`, with the unsatisfied criterion retained.

```powershell
pwsh -NoProfile -File tools/run.ps1 tools/roadmap_quality/train_usrnet_dataset.py --root 'baseline checkout' --variant before --seed 17 --stop-when-stable --steps 10000 --max-wall-seconds 3600 --run-dir artifacts/roadmap_quality/long_before17_part1
```

`--max-wall-seconds` includes setup for this invocation. An optional
`--deadline-utc` accepts a timezone-aware ISO timestamp; the first budget
condition reached wins. A running optimizer step completes before stopping.
Validation can stop between batches, and a partial validation is never included
in the metric history. A budget stop saves `latest.pth` and `final.pth` with
`status="budget_stopped"`, not `complete`. Checkpoint writing itself must finish
to preserve a resumable state. The final evaluated step is recorded separately
because a budget stop may happen between scheduled evaluations.

All checkpoints now contain CPU/CUDA Torch RNG, Python RNG, NumPy RNG,
`next_data_step`, optimizer state and the complete mean-metric trajectory.
Resume is explicit, requires a new output directory, and rejects a different
source, checked-build inputs/binary, data split/kernel identity, optimizer/data recipe, environment or
algorithm lane. The step ceiling and wall/deadline budgets may be extended;
learning rate, sampling, batch/microbatch settings and evaluation cadence may
not change under the name of resume.

```powershell
pwsh -NoProfile -File tools/run.ps1 tools/roadmap_quality/train_usrnet_dataset.py --root 'baseline checkout' --variant before --seed 17 --stop-when-stable --steps 10000 --max-wall-seconds 3600 --resume artifacts/roadmap_quality/long_before17_part1/final.pth --run-dir artifacts/roadmap_quality/long_before17_part2
```

Use the same `--deterministic-algorithms` option on every session if it was
enabled initially. Restoring RNG does not remove nondeterminism from a default
CUDA algorithm. Each run reports per-session timings, global update counters,
its parent checkpoint SHA and the inherited metric trajectory; it does not
invent an aggregate time-to-quality value. Long/resumed sessions use these
`run.json` fields and are explicitly rejected by the six fixed-run short-study
summarizer.

The CPU control-flow checks use a tiny mocked CPU model to verify budget-stop
checkpointing and resumed updates/RNG against an uninterrupted run, plus strict
identity, window and deadline tests:

```powershell
.venv/Scripts/python.exe tools/roadmap_quality/test_run_state.py
```

These checks do not replace a real CUDA pause/resume pilot or the full model's
numerical and quality gates.

## Offline long-run audit and explicit failure recovery

`summarize_long_training.py` audits the six selected long-run trajectories and
loads linked parent sessions automatically. Pass explicit leaf directories when
failed attempts or competing branches also exist; the auditor does not choose a
preferred branch. All reports and plots require fresh output filenames. Inspect
closed runs only. During a Windows run, use its append-only progress log and wait
for process exit before opening `run.json` or checkpoints, so a reader does not
overlap the worker's atomic metadata replacement.

Ordinary failed sessions remain inadmissible. `--recovery-manifest` supports one
narrow, explicit recovery: a `PermissionError` / `WinError 5` replacing exactly
`run.json.tmp` with `run.json`, after the complete evaluated checkpoint boundary
was already saved. The manifest authorizes an exact parent/child pair; it is not
a general permission to ignore an error. The auditor independently requires:

- The original parent is still `failed`, its only outstanding findings are the
  failed closure and missing terminal `final.pth`, and the authorized child has
  closed successfully after completing additional updates.
- The exact parent and child paths, seed, variant and checkpoint boundary match.
  Parent endpoint, latest checkpoint, final completed evaluation and child start
  agree. Every training row and scheduled evaluation is present, without gaps,
  rewind, duplicate steps or replay. A failed terminal session cannot pass.
- The manifest binds `run.json`, `training.jsonl`, `evaluations.jsonl`,
  `latest.pth` and `initial.pth` by SHA256, plus the original error, experimental
  identity, model, Adam, RNG state and failed-session process cost. Different
  failures, paths, boundaries or changed evidence are rejected.
- The latest checkpoint and the child's initial checkpoint have identical model
  tensors, complete FP32 Adam moments/counters/hyperparameters and RNG state.
  Source, checked build, recipe, data and environment identities must agree.
  These CPU checkpoint checks remain mandatory with
  `--skip-checkpoint-comparison`; that flag only skips the separate final
  before/current state comparison.

The original failed files and status are never edited or relabelled. The audit's
`failure_recoveries` entry records the exact two waived closure findings and the
verified checkpoint prefix. The failed session and its original error remain
visible under `sessions`; ordinary integrity checks and all four per-seed
quality gates still apply to the resulting trajectory.

For the recorded seed-43 recovery, the immutable manifest is
`artifacts/fp32_roadmap/quality_before43_recovery_manifest.json`. After all
selected processes have exited, its explicit six-leaf audit is:

```powershell
.venv/Scripts/python.exe tools/roadmap_quality/summarize_long_training.py `
  artifacts/fp32_roadmap/quality_long_before17 artifacts/fp32_roadmap/quality_long_current17 `
  artifacts/fp32_roadmap/quality_long_before29 artifacts/fp32_roadmap/quality_long_current29 `
  artifacts/fp32_roadmap/quality_long_before43_recovered artifacts/fp32_roadmap/quality_long_current43 `
  --recovery-manifest artifacts/fp32_roadmap/quality_before43_recovery_manifest.json `
  --output artifacts/fp32_roadmap/quality_final_audit.json
```

Timing uses the entire failed process plus the entire recovered process,
including repeated setup. In this case that is
`1762.548218 + 589.659853 = 2352.208071` seconds. The separate observed restart
interval is `282.197191` seconds (4.703 minutes), from the supervisor's failure
exit event at `10:32:13.701338 UTC` to the recovery affinity wrapper's start at
`10:36:55.8985294 UTC`, both on 2026-09-25. Evidence and hashes are recorded in
`artifacts/fp32_roadmap/quality_recovery_timing.json` and its Markdown companion.
No missing `completed_utc` is invented for the failed parent. Process sums
exclude this restart interval and must not be described as time to quality
including human recovery waits. The event interval and process timers have
different start/end scopes, so they are reported separately rather than summed
into an exact elapsed-time claim. Evaluation, checkpoint and setup scopes
overlap other timers and must not be added again.

CPU-only recovery regression tests cover accepted continuation and rejection of
tampered manifests/checkpoints, malformed evidence, terminal failures, changed
errors or paths, gaps/replay, incomplete Adam state and changed RNG continuation:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tools/roadmap_quality -p test_summarize_long_training.py -v
.venv/Scripts/python.exe -m unittest discover -s tools/roadmap_quality -p test_failed_checkpoint_recovery.py -v
```
