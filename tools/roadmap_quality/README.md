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
