# Paired long-training audit

Status: **quality_gates_passed**. This report makes no convergence claim.

## Evidence limitations

- H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before43: Historical failed session retained; explicit manifest verified its exact checkpoint prefix and full process cost
- Intersession timestamp interval unavailable: H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before43_recovered

## Explicit failure recovery

- verified_checkpoint_prefix: H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before43 -> H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before43_recovered; checkpoint update 3000. Original status remains failed; recorded failed-session process cost 1762.548 s is retained.

## Trajectories

| Seed | Variant | Sessions | Updates | Last evaluated | Status | Window satisfied | Process seconds |
|---:|---|---:|---:|---:|---|---|---:|
| 29 | current | 1 | 4000 | 4000 | max_steps_reached | False | 2412.212 |
| 17 | before | 1 | 4000 | 4000 | max_steps_reached | False | 2661.630 |
| 17 | current | 1 | 4000 | 4000 | max_steps_reached | False | 2672.117 |
| 43 | before | 2 | 4000 | 4000 | max_steps_reached | False | 2352.208 |
| 43 | current | 1 | 4000 | 4000 | max_steps_reached | False | 2493.539 |
| 29 | before | 1 | 4000 | 4000 | max_steps_reached | False | 2347.406 |

## Per-seed output-quality gates

Each of the four metrics must pass independently; terminal updates may differ.

| Seed | Before/current evaluated updates | RGB PSNR delta | RGB SSIM delta | Y PSNR delta | Y SSIM delta | Passed |
|---:|---|---:|---:|---:|---:|---|
| 17 | 4000/4000 | 0.000 | 0.000 | 0.000 | 0.000 | True |
| 29 | 4000/4000 | 0.000 | 0.000 | 0.000 | 0.000 | True |
| 43 | 4000/4000 | 0.000 | 0.000 | 0.000 | 0.000 | True |

## Data and checkpoint comparisons

- Seed 17: exact matching data prefix 4000 updates; before-only/current-only updates 0/0; checkpoint comparison: compared_on_cpu.
  Model tensors differing: 0/133; optimizer tensors differing: 0/399. These counts do not decide quality admission.
- Seed 29: exact matching data prefix 4000 updates; before-only/current-only updates 0/0; checkpoint comparison: compared_on_cpu.
  Model tensors differing: 0/133; optimizer tensors differing: 0/399. These counts do not decide quality admission.
- Seed 43: exact matching data prefix 4000 updates; before-only/current-only updates 0/0; checkpoint comparison: compared_on_cpu.
  Model tensors differing: 0/133; optimizer tensors differing: 0/399. These counts do not decide quality admission.

## Session timing

| Session | Training step s | Data s | Evaluation s | Checkpoint s | Setup s | Loop s | Process s |
|---|---:|---:|---:|---:|---:|---:|---:|
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_current43 | 2325.152 | 112.766 | 38.821 | 0.876 | 2.641 | 2490.899 | 2493.539 |
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_current29 | 2256.009 | 105.993 | 35.484 | 0.829 | 2.992 | 2409.220 | 2412.212 |
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_current17 | 2498.009 | 114.806 | 42.209 | 0.860 | 2.847 | 2669.270 | 2672.117 |
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before43_recovered | 546.082 | 26.273 | 11.231 | 0.238 | 2.806 | 586.854 | 589.660 |
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before43 | 1635.983 | 78.619 | 35.189 | 0.668 | 2.714 | 1759.832 | 1762.548 |
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before29 | 2181.039 | 105.948 | 44.298 | 0.859 | 2.734 | 2344.672 | 2347.406 |
| H:\Python\ConverseNet\artifacts\fp32_roadmap\quality_long_before17 | 2475.921 | 114.156 | 54.178 | 0.865 | 3.595 | 2658.034 | 2661.630 |

Setup/checkpoint work repeats across resumed sessions. These scopes overlap; only the recorded process times are summed as process elapsed time.
Summed process times exclude waits between sessions. Inter-session timestamp gaps in JSON are metadata intervals, not exact downtime, because created_utc is recorded after initial preparation.

## Interpretation

- Quality gates are per seed and per RGB/Y metric: PSNR drop <=0.05 dB and SSIM drop <=0.001.
- Source/build/data/recipe/environment identity must remain fixed within each variant's resumed chain; different production sources are allowed between before/current.
- State and optimizer equality are independent diagnostics. Permitted rounding differences never establish or override quality admission.
- All shared data-step hashes are shown. Holes, rewinds, overlaps, missing parent sessions and competing branches are not silently repaired.
- An explicit failure-recovery manifest can admit only a verified checkpoint-closed non-terminal prefix; the original failed status/error and its entire process cost remain visible.
- Endpoint update counts can differ. Their extra steps and the common exact-data prefix are reported separately; full elapsed ratios would not be equal-work speedups.
- Session process elapsed times are summed honestly, including repeated setup. Training-step and data preparation are separate; evaluation/checkpoint/setup/loop are overlapping scopes and must not be added to process totals.
- stable is only the declared five-evaluation four-metric window; max_steps_reached, budget_stopped and unstable trajectories are not convergence evidence.
