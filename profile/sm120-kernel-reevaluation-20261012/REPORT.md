# SM120 kernel reevaluation after split-K pointwise weight gradients

Measured 2026-10-12 at HEAD `5c4a6897898fb84a03dbee0641fed7ffb2a06253`. RTX 5060 Ti 16 GB, compute capability 12.0, 36 SMs, driver 617.14, Windows WDDM, PyTorch 2.11.0+cu130, CUDA toolkit 13.0, MSVC 14.44, NCU 2026.3.1.

**Prior solver fusion is now the largest production opportunity.** The old grouped direct weight-gradient bottleneck was removed by the split-K kernel. Production prior solves and their surrounding work consume 55.9% of training GPU work; all pointwise work consumes 23.5%. With the experimental fused s1 prior, the shares are 32.9% and 35.9%, respectively. The s1 substitution measures **1.56x** on the complete resident-data training step.

**Production passed 228/228 release tests.** The fused FFT research path passed all s1 fixtures, but retained five individual s2/s3 budget failures in the fresh 160-fixture sweep. No kernel, dispatch, numerical policy, or production arithmetic was changed during this reevaluation.

![Fresh model work attribution](analysis/current_model_work.png)

## Measured scope and provenance

The model workload is pretrained USRNet, B4 LR32x32 to HR96x96, scale 3, five iterations. A training step contains zero_grad, forward, MSE, backward and Adam (`foreach=False`, `fused=False`); inputs are already resident on the GPU. The research model substitutes only the seven prior modules with `FlashS1`, leaving the data solver and current production pointwise/LayerNorm paths in place. The 35 prior calls still perform a differentiable kernel FFT on every invocation.

Full-model paired measurements use six ABBA/BAAB rounds, two steps per timed block, restoring checkpoint parameters and resetting Adam before each block, followed by one warm step. The production/research controls share AMP-off, both TF32 flags off, deterministic algorithms and cuDNN deterministic settings, and CPU affinity `0xC03C03`. Model traces and paired-model measurements retain PyTorch deterministic memory filling; the pointwise performance tool explicitly disables it for both of its routes. Compare routes within one measurement, not absolute times across these tools.

CUDA-event timing is collected separately from profiler/NCU instrumentation. GPU kernel sum is the attribution denominator; its union agrees to numerical precision in all three traces. Kernel span, host annotation durations and NCU replay durations are not used as end-to-end speedups. Clocks and caches were uncontrolled, with desktop activity; single traces and short timing windows have run-to-run spread.

- Production: a native `sm_120` checked rebuild and the unmodified full release suite. Binary SHA256 `928a1e6694e9798d0501a033e57d2ccfca273b4b1b794fb215198b9578ebbad7`.
- Research source SHA256 `a2ddd155b6367943ac377176e0eec2567bb6ef2e57d7693f0269450b4cfe7b0a`; binary SHA256 `ccfbbbe2bed55d35f686830281d625424713a3542281a94712a3f98ea9d4a69c`.
- Fresh research recompilation failed twice with CUDA 13.0 `cudafe++` ACCESS_VIOLATION. Fresh measurements reuse the exact earlier native-SM120 binary after byte checks against the source/binding snapshots and the recorded binary hash. This is a verified reuse, not a successful fresh research rebuild.
- Initial Windows compiler/temp-directory failures, both research compiler failures, and NCU launch/Ninja failures are preserved under `reports/`. The successful release run used the recorded MSVC environment outside the sandbox. NCU subsequently loaded the checked binary without rebuilding.

## Whole-model timing and current optimization budget

| Comparison (only the named route changes) | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| Current production prior vs fused s1 prior | 332.87 | 213.62 | 1.558x |
| ATen LN vs fused LN, both with fused s1 prior | 260.16 | 215.90 | 1.205x |
| Native pointwise weight VJPs vs split-K (current production solvers) | 350.01 | 246.76 | 1.418x |
| ATen LN vs fused LN (current production solvers) | 282.16 | 244.52 | 1.154x |

These ratios are separate experiments and must not be multiplied into a measured combined speedup. The paired prior comparison is the relevant measured incremental gain over the current production branch.

| GPU work group | Production training | Fused-s1 training | Production inference |
|---|---:|---:|---:|
| Prior solves + surrounding work | 55.94% | 32.92% | 43.94% |
| All pointwise convolutions | 23.54% | 35.86% | 35.12% |
| GELU | 7.91% | 10.53% | 8.44% |
| All LayerNorm calls | 4.90% | 8.66% | 5.21% |
| Data solves | 3.43% | 5.22% | 2.71% |
| Fused FFT core (subset of prior) | 0.00% | 20.83% | 0.00% |
| Split-K weight VJP (subset of pointwise) | 6.80% | 10.58% | 0.00% |
| Named native dgrad kernels (subset of pointwise) | 6.49% | 9.80% | 0.00% |

The training trace has 140 split-K partials and 140 deterministic reductions; the seven remaining native direct wgrad launches are small boundary convolutions and account for only 0.10% of production GPU work. The old claim that grouped direct wgrad dominates the step no longer describes this branch.

Conditional bounds on the fused-s1 model, holding the other GPU work fixed: doubling all pointwise work would improve total GPU work by at most 1.219x; doubling just split-K wgrad by 1.056x; doubling the FFT core by 1.116x; doubling the complete prior group by 1.197x. These are Amdahl bounds, not measured wall-time gains.

65/70 training LayerNorm calls use fused kernels; five strided first-block calls retain ATen. Their associated GPU work is 3.53 ms in production and 4.75 ms in the research trace. The fused LN kernels themselves are only 3.53% of the research trace. Inference remains the production half-spectrum path; its trace is 43.94% prior work and 35.12% pointwise work. The full-spectrum training candidate is not an inference promotion under the current spectrum policy.

## Complete operator/VJP timing

FFT timing includes per-call differentiable PSF preparation, kernel FFT, regularization and all VJPs. Eight rounds of 20 iterations, paired in one process. The accuracy column below uses that tool's seed 74001 and production baseline; the independent frozen-baseline sweep follows separately.

| Case | Production us | Fused us | Speedup | Timing fixture accuracy |
|---|---:|---:|---:|---|
| circular_s1_b4_c64_96_pad2 | 1243 | 686 | 1.81x | pass |
| circular_s1_b4_c128_96_pad2 | 3425 | 1195 | 2.87x | pass |
| forward_s1_b4_c128_100 | 3294 | 1186 | 2.78x | pass |
| forward_s1_b4_c64_96 | 1235 | 673 | 1.84x | pass |
| forward_s2_b4_c64_64 | 3135 | 1471 | 2.13x | pass |
| forward_s2_b4_c64_48 | 1289 | 856 | 1.51x | pass |
| forward_s2_b4_c64_32 | 575 | 485 | 1.19x | **FAIL** |
| forward_s3_b2_c32_48 | 649 | 611 | 1.06x | pass |
| forward_s3_b2_c32_32 | 603 | 478 | 1.26x | pass |
| forward_s3_b2_c32_24 | 538 | 465 | 1.16x | pass |

All ten timing-fixture results repeat bitwise. On the four s2/s3 planes that fit one CTA, forced two-CTA clusters remain bitwise identical to the single-CTA kernels in output and all four VJPs (`analysis/cluster_check.json`). This diagnostic does not override accuracy failures.

| Pointwise geometry | Native wgrad us | Split-K us | Wgrad speedup | Complete layer speedup |
|---|---:|---:|---:|---:|
| 4x64x96x96->128 | 1311 | 144 | 9.13x | 3.24x |
| 4x128x96x96->64 | 278 | 142 | 1.95x | 1.41x |
| 1x64x96x96->128 | 342 | 45 | 7.61x | 1.59x |
| 1x128x96x96->64 | 341 | 46 | 7.48x | 1.62x |

## Frozen-baseline accuracy and retained failures

The fresh FFT sweep has ten compiled geometries, eight seeds each (75000-75007), normal `eps=1e-5` and weak `eps=1e-8`, bias=-40, kernel amplitude scaled by 1e-3. Every output and VJP is compared with `test/fp32_baseline.py` in FP32 and the independent Python FP64 expression using identical FP32 operands/upstream gradients promoted to FP64. The same `test/numerical_policy.py` relative-L2/max-abs factors and floors apply, including shared-input differentiation through padding.

**731/736 fused output/VJP comparisons pass; 736/736 production comparisons pass.** All 160 fused fixtures repeat bitwise. All s1 comparisons pass in both regimes. Five failed rows are:

| Case | Regime | Seed | Tensor | Failed metric | Candidate error / permitted limit |
|---|---|---:|---|---|---|
| forward_s2_b4_c64_64 | normal | 75001 | grad_bias | rel_l2 | 4.45058050e-07 / 3.10722315e-07 |
| forward_s2_b4_c64_64 | normal | 75007 | grad_bias | rel_l2 | 5.14299315e-07 / 3.74690627e-07 |
| forward_s2_b4_c64_32 | weak | 75005 | grad_bias | rel_l2 | 2.71619235e-07 / 2.53983869e-07 |
| forward_s3_b2_c32_24 | normal | 75003 | grad_bias | rel_l2 | 4.80328529e-07 / 3.93087432e-07 |
| forward_s3_b2_c32_24 | weak | 75004 | grad_weight | max_abs | 2.41185403e+01 / 2.40587664e+01 |

All supplemental eight-seed aggregates pass, but individual failures above remain failures under the requested per-output gates. The separate performance fixture also retains its s2 32 grad_bias failure. No earlier failure was relabeled and no baseline was recalibrated.

The production pointwise gate independently passed all 280 captured pointwise calls, 28 synthetic geometry/seed pairs, deterministic repeats and both production routing probes. Captured C64->128 worst rel-L2 ratio is 0.374; C128->64 is 0.087. The unmodified release suite covers dtype rejection, broadcast/shared gradients, spectrum routing, per-call kernel FFT, higher-order fallback, streams, Graphs, LayerNorm and pretrained smoke tests. Neither the seed sweep nor short Adam timing proves convergence.

## Fresh kernel counters and code evidence

The counter table uses the separate `--set full` collection (`reports/current_full.ncu-repz`). An additional full+SourceCounters collection is preserved for instruction/source evidence; it inflated s1 forward from about 251 us to 1257 us and LN forward from about 42 us to 467 us. Its durations are excluded from the table and from all speedup claims. Both collections use the same binary/operands.

| Kernel / geometry | NCU replay us | Registers | Active warps % | DRAM % | Issue-active % | MIO / barrier / wait stalls |
|---|---:|---:|---:|---:|---:|---|
| s1 forward B4 C128 100 | 250.7 | 67 | 48.8 | 42.3 | 54.9 | 1.23 / 1.82 / 1.68 |
| s1 backward B4 C128 100 | 572.5 | 119 | 33.2 | 43.9 | 63.2 | 0.59 / 0.38 / 0.96 |
| LN forward B4 C64 96 | 41.5 | 38 | 85.2 | 82.9 | 28.7 | 0.10 / 2.78 / 2.11 |
| LN backward B4 C64 96 | 79.7 | 40 | 63.5 | 90.1 | 15.2 | 1.31 / 3.17 / 1.36 |
| Split-K partials B4 C64->128 96 | 132.1 | 128 | 30.6 | 66.1 | 55.9 | 1.30 / 0.29 / 0.34 |
| Split-K reduction | 8.5 | 40 | 15.9 | 2.5 | 30.4 | 0.00 / 0.00 / 0.70 |
| s2 128 cluster forward | 400.2 | 94 | 32.7 | 32.4 | 18.1 | 6.00 / 2.09 / 0.99 |
| s2 128 cluster backward | 736.1 | 127 | 33.2 | 35.4 | 28.8 | 3.03 / 1.15 / 0.50 |
| s3 144 cluster forward | 146.0 | 128 | 33.3 | 32.4 | 20.8 | 5.22 / 3.63 / 0.90 |
| s3 144 cluster backward | 276.8 | 128 | 33.9 | 32.7 | 29.0 | 2.79 / 2.42 / 0.61 |

Stall ratios are per issued instruction, not percentages. Source samples and native SASS are preserved in `analysis/ncu_full_summary.json`, `analysis/ncu_summary.json`, and the two `.sass` dumps.

- **s1 FFT:** forward uses 67 registers, backward 119; no local-memory reads. Backward long-scoreboard is only 0.39, while wait is 0.96 and MIO 0.59. Its 15.41% model share is the largest individual research kernel opportunity. Shared FFT round trips, the compensated pointwise phase, register dependencies and store layout are better targets than treating it as a saturated DRAM kernel.
- **Split-K wgrad:** 66.1% DRAM, roughly 49.7% FMA-heavy pipeline activity, 128 registers, 30.6% active warps, MIO 1.30, long-scoreboard 0.31. This is a mixed memory/shared-issue/register limit, unlike the old address-heavy direct wgrad. Source hotspots are shared operand loads (line 136), 4-byte cp.async (line 44), and FFMA (line 142). Native SASS confirms LDGSTS, FFMA/FADD, and no tensor MMA in either pointwise kernel. Layout/vectorized staging is worth testing; removing compensation or using TF32 is outside policy.
- **LayerNorm:** forward/backward reach 82.9%/90.1% DRAM, with long-scoreboard 31.4/40.0. The fused kernels are bandwidth-limited. The five strided ATen fallbacks and surrounding reductions have more plausible model-level room than arithmetic changes to an already bandwidth-saturated kernel.
- **Clusters:** s2 forward MIO 6.00 and barrier 2.09; s3 forward MIO 5.22 and barrier 3.63. Source samples concentrate at row/column exchanges (lines 1159/1181) and fused shared FFT passes. s3 forward also records 16,384 local-load sectors; the other sampled kernels record zero. Its 128-register limit and small plane count make a larger block risky without spill checks.

## Priority for the next implementation round

1. **Prepare a guarded s1 integration first.** It has the largest production budget, passes the fresh normal/weak sweep, and shows a measured 1.56x full-step opportunity. Integration still needs the public routing/broadcast/gradient-mask/higher-order/Graph contracts and long quality validation. The research wrapper alone does not establish those contracts.
2. **Then target the fused prior as a whole and pointwise dx/forward.** FFT backward is 15.41% and preparation/VJP/reduction overhead another 12.09% of research GPU work; all pointwise work is 35.86%, but split-K wgrad alone is only 10.58%. Preserve per-call differentiable kernel FFT and shared-input accumulation order while testing memory/layout changes.
3. **Keep s2/s3 promotion gated.** s2 128 is 2.13x faster in the operator experiment but has two normal grad_bias seed failures. s3 144 is only 1.06x faster here; solving the retained s2/s3 accuracy items takes precedence over broader cluster dispatch.
4. **Deprioritize standalone LN arithmetic.** Check the five training layout fallbacks with complete copy-inclusive VJP timings and independent accuracy gates. Optimizing only the two fused LN kernels has a 1.037x ideal elimination ceiling in the research model.

## Reproduction and preserved evidence

Use `H:/Python/ConverseNet/.venv/Scripts/python.exe -B -u` with `harness/bootstrap.py` to restore the recorded local toolchain/affinity, then pass the desired script. `bootstrap.py release` runs the unmodified release suite. Other scripts require the exact checked production manifest and the recorded research binary/source snapshots. Use new output directories for reruns; existing evidence must not be overwritten.

- `analysis/fft_gate.json`: all frozen-baseline rows, individual failures and supplemental aggregates.
- `analysis/fft_perf/flash.json`, `pointwise_perf.json`, `layernorm_perf.json`, `paired_model.json`: raw timing rounds.
- `analysis/model_{production,flash,inference}/`: fresh traces, module routing/strides and summaries.
- `analysis/model_attribution.json`: denominator checks and conditional bounds.
- `reports/current_full.ncu-repz`, `current_kernels.ncu-repz`: native counter/source reports.
- `analysis/production_manifest.json`, `sass_verification.json`, `evidence_manifest.json`: checked identities and hashes.
- `reports/release.log`, `pointwise_gate.log`, and retained failed logs: complete outcomes.

Architecture cross-check: LeetCUDA profiling and SM120 chapters (local reference under `.build/LeetCUDA`). Its generic table says SM120 has no clusters; that entry conflicts with this GPU's successful two-CTA launches, cross-checks and NCU reports. Runtime evidence governs this reevaluation. No SM100 TMEM/tcgen05, reduced-precision tensor path, fast-math, AMP or training-spectrum reuse was introduced.
