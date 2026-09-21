# Tests and profiling

Run commands from the repository root with PyTorch, Ninja and a configured
compiler. Tests build the current extension under `.build/`; a global extension
installation is not required. Set `CONVERSE2D_CPU_ONLY=1` for a CPU-only build.

All generated results go under `artifacts/`, which is ignored by Git.

## Correctness

```sh
python test/test_error.py --device cuda
python test/test_error.py --device cpu
python test/test_batched_kernels.py
python test/test_batched_kernels.py --cpu
python test/test_cache.py
python test/test_cuda_graph.py
python test/test_spectral_io.py
python test/test_fp32_training.py
python test/test_training_fusion.py
python test/test_training_scale3_batch.py
python test/test_training_refinements.py
python test/test_training_scope.py
```

- `test_correctness.py` (also exposed by `test_error.py`): independent dense
  spatial solves, forward/gradient agreement, first/second derivatives, odd/even
  and singleton sizes, low precision, padding, mutation-aware caches and streams.
- `test_batched_kernels.py`: per-sample/channel-shared kernels, dynamic-kernel
  gradients, DataNet mixed precision and USRNet end-to-end training.
- `test_cache.py`: focused cache invalidation and inference/training transitions.
- `test_cuda_graph.py`: capture with a warm eager cache, graph eviction, changed
  inputs and weights, shape/batch/scale/dtype changes, independent outputs and
  sequential calls on different CUDA streams.
- `test_spectral_io.py`: v7 spectral I/O changes against the frozen pre-change
  kernel and float64 reference, including subnormal inputs, rectangular PSFs,
  broadcast kernels and bitwise-equivalent PSF spectra. Requires CUDA and Git.
- `test_fp32_training.py`: production CUDA FP32 output and gradients against
  an independent full-FFT FP64 reference, weak regularization, shared priors,
  selective gradients, higher derivatives, streams, cache transitions and
  short USRNet SGD/gradient accumulation against an FP64 model.
- `test_training_fusion.py`: arbitrary complex half spectra, broadcast VJPs,
  Hermitian boundaries, gradcheck/gradgradcheck, conjugate views and saved-input
  mutation checks through the internal production spectral entry.
- `test_training_scale3_batch.py`: scale 3 and B8/B32 against FP64, including
  dynamic/broadcast kernels, shared input and kernel gradients, higher
  derivatives and weak regularization. Writes per-comparison absolute,
  relative and pointwise error-budget metrics to
  `artifacts/training_scale3_batch_validation.json`.

These use Python's standard unittest; pytest is not required.

`test_training_refinements.py` covers the large s1 dispatch boundary, selective
gradients, weak regularization, dynamic-kernel work thresholds and higher
derivatives. `test_training_scope.py` checks the opt-in per-forward cache,
including invalidation, FP64 gradient accumulation, nesting, streams, threads
and complete-model equivalence with reuse disabled.

The separate strict quality audit is deliberately **not a passing gate yet**:

```sh
python test/test_full_usrnet_precision.py
```

It retains fixed FP64 pointwise thresholds, seed, alpha and epsilon, and returns
a nonzero status on the known full-model gradient discrepancy. The frozen
pre-refinement source exhibits the same failing indices; this is not skipped
or marked as an expected test success. See [the report](../docs/training_refinements.md).

## Supplied real-data quality evaluation

```sh
python test/evaluate_usrnet_quality.py --hr-dir DATA/HR --lr-dir DATA/LR --kernel DATA/kernel.npy --scale 3 --backend both
```

This loads the full pretrained USRNet and reports RGB/Y PSNR, SSIM, paired
backend output differences, timing and allocated/reserved peaks. HR/LR pairs
and 7x7 kernels must match exactly; no resizing or implicit degradation is
performed. `--kernel-dir` supports per-image kernels. Dataset files are not tracked
in Git, and a synthetic I/O smoke is not dataset-quality evidence.

## Real-photo FP32 fine-tuning

The user's 1000 images under `dataset/` have a hash-verified 900/100 local split
at `artifacts/dataset_training/split_900_100.json`. They are unpaired photographs;
`usrnet_training_data.py` explicitly synthesizes LR input using the five repository
7x7 kernels, circular convolution, phase-zero s3 sampling and noise sigma=.01.
Training uses random HR96 crops/augmentation; validation uses fixed center crops
and kernel/noise independent of the training seed. Source images are read-only.

```powershell
& ./experiments/training_speed/run.ps1 test/train_usrnet_dataset.py --variant current --seed 17 --run-dir artifacts/dataset_training/repeat_current17
& ./experiments/training_speed/run.ps1 test/train_usrnet_dataset.py --variant before --seed 17 --run-dir artifacts/dataset_training/repeat_before17
```

Both commands strictly load the same full pretrained 5-iteration/7-block model.
Defaults are FP32 MSE/Adam1e-5, batch4, 250 steps and all 100 validation crops at
steps 0/125/250. No gates/lambda are reinitialized. `before` uses the verified
pre-refinement source and frozen Python models. `reuse` is an explicit candidate;
the current production run leaves it off. Existing run directories are rejected.
Checkpoints contain `state_dict`, optimizer state and provenance; per-step tensor
hashes permit exact paired-data checks. Short fine-tuning is not convergence proof.
See [protocol and results](../docs/dataset_finetuning.md).

For the **same-GPU Python/ATen training baseline**, use:

```powershell
& ./experiments/training_speed/run.ps1 test/benchmark_python_training.py --batch-size 4 --microbatch-size 4 --rounds 4 --steps 8 --warmup 5 --output artifacts/python_training_comparison/repeat_timing.json
& ./experiments/training_speed/run.ps1 test/train_usrnet_python_comparison.py --backend pytorch --seed 17 --run-dir artifacts/python_training_comparison/repeat_pytorch17
```

The Python baseline is `models.converse_core.converse2d_reference` with CUDA FP32
tensors and full-spectrum FFT/autograd. It is not CPU execution or the frozen
pre-refinement C++ backend. The timing script alternates Python/current, restores
identical model and Adam state after warmup and keeps one GPU model resident.
The separate quality adapter preserves the original worker and verifies all 40
reference calls per full forward plus the first training graph. Both paths retain
the same physical batch4; no OOM workaround changes the comparison. See
[Python baseline comparison](../docs/python_training_comparison.md).

## Pretrained models

```sh
python test/test_pretrained_smoke.py
python test/test_usrnet.py
```

These load the repository's DnCNN, SRResNet and USRNet checkpoints and compare
Python/CUDA outputs in float32/float64 with TF32 disabled. They verify integration
on deterministic sample inputs, not dataset PSNR. Add `--installed` to test a
package built in `Converse2D/` instead of the local JIT build.

## Performance

```sh
python test/benchmark_training_refinements.py --iters 20 --rounds 6
python test/training_s1_ablation.py --iters 20 --rounds 6
python test/profile_training_refinements.py --case usrnet-b16-s3 --variant both
```

These measure the latest FP32 changes against the manifest-verified source and
Python-model snapshot in `artifacts/training_refinements/source_before`.
Preserve that snapshot to reproduce this incremental comparison. The s1
ablation builds current code with only its selector disabled. Profiler captures
are separate from timings. `--reuse-spectra` explicitly evaluates the disabled
reuse candidate; do not combine its results with the default path. The explicit
`--case usrnet-default-tiny --reuse-spectra` full-model stress configuration
currently fails its short-training gate and must not be treated as validated
training performance. FP16/BF16 work is deferred at the user's request.

```sh
python test/benchmark_fp32_training.py --iters 20 --rounds 6
python test/benchmark_fp32_training.py --suite scale3-batch --iters 20 --rounds 6
```

This eager FP32 training benchmark builds the pre-integration dev commit
`b850e38` from Git in a separate namespace and compares it with current
production. It measures operator forward/backward and complete SGD steps for
operators, ConverseBlock and a reduced USRNet, including dynamic kernel
preparation. Each alternating sample starts from identical parameters and
optimizer state; short trajectory checks precede timing. CUDA event time,
synchronized wall time and total PyTorch peak allocated/reserved memory are
reported with one model resident at a time. Data loading/H2D, AMP, CUDA Graphs
and dataset convergence are outside this benchmark. Results default to
`artifacts/training_operator_optimization/benchmark.json`.

The `scale3-batch` suite adds scale 3, B8/B32 at 64x80, B4 at 256x256,
and larger-batch ConverseBlock/USRNet complete steps. USRNet is evaluated at
both scales 2 and 3. Its default output is the separate
`artifacts/training_operator_optimization/benchmark_scale3_batch.json`.

```sh
python test/test_speed.py
python test/test_speed.py --training
python test/test_speed.py --single --variant v7 --B 1 --C 32 --H 128 --W 128 --scale 2
```

`test_speed.py` invokes `benchmark.py`. The default grid compares the current v2,
v6 and v7 paths using identical inputs and a float64 reference check. Timings
include CUDA events, wall time and incremental peak allocated memory. Training
includes forward plus gradients for x/x0/weight/bias.

Use `--iters` to control repetitions and `--output` to select a result path.
Defaults are `artifacts/benchmark.json` and `artifacts/benchmark_training.json`.

```sh
python test/benchmark_cuda_graph.py
```

This compares the pretrained USRNet eager path with the full graph runner on
32x40 and 64x80 inputs at scale 2, FP32 and TF32 disabled. The graph timing
includes signature checks, input copies, replay and an independent output copy.
Five alternating rounds of 20 calls report both CUDA-event and wall medians;
first-call setup and memory snapshots are recorded separately. Two changed
input/kernel pairs per size are checked against eager output. Results go to
`artifacts/benchmark_cuda_graph.json`; `--iters`, `--rounds` and `--output` can
override the defaults. This measures steady-state inference, not training or
dataset PSNR.

```sh
python test/benchmark_spectral_io.py
python test/benchmark_spectral_io.py --operators-only --iters 100 --rounds 7
```

These build the baseline from commit `19c1bfc` in an isolated namespace and
alternate baseline/current timing on identical inputs in the same process.
They cover cached and dynamic kernels, odd sizes, and pretrained USRNet eager
and graph inference. Baseline sources/build files stay under `.build/`;
results default to `artifacts/spectral_io_benchmark.json`. A `--profile` mode
emits baseline/optimized NVTX ranges for dynamic s2 inside a CUDA profiler range.

## Nsight

For the current **FP32 training** path, use the dedicated capture script:

```powershell
$env:CONVERSE2D_SKIP_BUILD='1'
& ./experiments/training_speed/run.ps1 test/profile_nsight_training.py --tool nsys --case op-b4-256-s3
& ./experiments/training_speed/run.ps1 test/profile_nsight_training.py --tool ncu --case op-b4-256-s3 --steps 1
& ./experiments/training_speed/run.ps1 test/profile_nsight_training.py --tool nsys --case usrnet-b16-s3
```

Build with `test/extension_loader.py` first if the source-validated binary is
missing or stale. Cases cover B1/B4 C32 256x256 at s1/s3, B32 C32 64x80 at s3,
and reduced USRNet B16 s3. Operator sizes describe the **low-resolution input**;
256x256 s3 produces 768x768. Operator captures include forward and x/kernel/bias
VJP, with a nearest prior, but no loss or optimizer. The USRNet case includes a
complete SGD step for the 2-iteration/1-block fixture; it is not the full model
or a convergence test. Spectrum reuse remains disabled.

The launcher excludes compilation, fixture creation, five warmup iterations and
state reset from the CUDA profiler range. Systems records three iterations with
CUDA/NVTX, including autograd ranges. Compute filters the actual training kernel
names and collects launch, occupancy, memory, scheduler and warp-state sections.
It uses kernel replay, `--cache-control none` and `--clock-control none`; these
counter captures do not reproduce an unprofiled training timeline. Use a fresh
`--output-dir` for each repeat. Reports, exact commands and source hashes default
to `artifacts/nsight_training/`. See [current analysis](../docs/nsight_training_analysis.md).

Offline exports can be summarized without running GPU work:

```powershell
python test/summarize_nsight_training.py 'artifacts/nsight_training/final/*.sqlite' --output artifacts/nsight_training/final/nsys_summary.json --markdown artifacts/nsight_training/final/nsys_summary.md
python test/summarize_ncu_training.py artifacts/nsight_training/op-b4-256-s3.ncu.ncu-rep --export --output artifacts/nsight_training/one_ncu_summary.json
```

The Systems parser checks unique process/correlation matches and partitions
kernel totals by step and phase. Cross-thread autograd phase assignment is
explicitly marked temporal; unassigned events remain visible. FFT preparation
tags overlap the mutually exclusive kernel classes and must not be added to them.

The older wrapper below is for the inference benchmark and uses different kernel
filters; it does not substitute for a training capture.

```sh
python test/profile_nsight.py --kind systems --variant v7 --scale 2
python test/profile_nsight.py --kind compute --variant v7 --scale 2
python test/profile_nsight.py --kind compute --set full --C 64 --scale 2
python test/profile_nsight.py --kind compute --set basic --C 64 --scale 2 --cache-control none --replay-mode application --output artifacts/profiles/warm
```

The wrapper builds first, then profiles the warmed CUDA/NVTX region. Reports go
to `artifacts/profiles/`. Use `--tool` for an explicit profiler executable and
`--C`, `--H`, `--W` for the workload. Nsight Compute requires access to the GPU
performance counters; the scripts do not modify system permissions.

Compute defaults to `--set basic`, kernel replay, cache flushing and
`--clock-control none` (leaves GPU clock policy unchanged). Use `--set full` for
memory, instruction and warp-stall analysis. Application replay with
`--cache-control none` preserves the workload's preceding cache activity, at the
cost of rerunning the process per pass. Compare like-for-like collection settings;
NCU kernel durations are not end-to-end inference timings. Use distinct output
directories to retain multiple shapes or collection policies for the same variant
and scale.

`CONVERSE2D_SKIP_BUILD=1` loads the existing local extension without building,
after checking its source/header and binary fingerprints. Rebuild after source
changes; stale binaries are rejected. This is useful inside a profiler.
# 完整训练重新定位（2026-09-18）

见 [调研报告](../docs/training_research.md)。以下是新增的独立研究入口，不改变旧质量/计时脚本或生产默认：

- `benchmark_sustained_training.py`：current / 同 GPU Python，完整预训练模型、真实数据连续100步AB/BA；先检查新旧循环数值一致。
- `profile_full_training.py --tool torch`：完整训练CPU/CUDA轨迹、40次solver的模块导航证据。
- `profile_full_training_nsight.py --tool nsys|ncu`：使用source/header/torch/binary哈希验证过的现有构建，保存bootstrap、命令和明确成功/失败状态，避免无关的编译器版本探测。
- `summarize_full_training_trace.py`：只以原始kernel事件作分母，关联CPU模块/自动反向并保留未归属项。
- `verify_full_training_trace.py`：验证指定技能生成的紧凑视图可逆、原trace不变及Perfetto实际导入轨道深度；Perfetto依赖路径见其参数。
- `probe_pointwise_training.py`：真实特征形状的1×1卷积/等价matmul独立FP64筛选和前向+全部VJP计时；局部结果不代表整网质量或速度。

所有计时关闭profiler。新输出目录/文件拒绝覆盖；原始报告在`artifacts/training_research/`。

## Converse2D / 原生 ConvTranspose2d 对照（2026-09-18）

见[结果与适用范围](../docs/converse_deconv_target.md)和[编译候选使用说明](../experiments/training_nonoverlap/README.md)。以下均为显式实验，不改变生产默认；数学不同的原生转置卷积仅作速度标杆，Converse仍使用独立FP64检查。

- `benchmark_native_deconv.py`、`benchmark_converse_candidates.py`：原生基线及s1/边界组合的同fixture对照。
- `train_usrnet_converse_candidate.py --candidate current|combined`、`summarize_converse_candidate_training.py`：完整模型真实图三种子短训练及批次、checkpoint、指标审计。
- `probe_nearest_fused_training.py`、`probe_nearest_geometry_training.py`：频域nearest融合和仅几何phase复用，包含动态核准备成本。
- `probe_spatial_nonoverlap_fast.py`、`probe_compiled_nonoverlap.py`：nearest k3/s3精确空间特例与自动编译。该特例不命中当前完整USRNet。
- `check_compiled_autograd_contract.py --cuda`：关闭donated_buffer后的自动高阶回退契约；CPU/GPU覆盖分别记录。
- `benchmark_deconv_target_final.py`：默认B32/C32/64×80/s3/k3，五路同确定性/精度/编译设置，包含真实封装开销、独立FP64门槛、热测无重编译和首次准备成本。默认六轮、每轮30次；拒绝覆盖已有输出。

原始结果位于`artifacts/native_deconv_target/`；旧失败、负收益实验和编译成本均保留。示例：

```powershell
& ./experiments/training_speed/run.ps1 test/check_compiled_autograd_contract.py --cuda --output artifacts/native_deconv_target/new_contract.json
& ./experiments/training_speed/run.ps1 test/run_verified_cuda.py test/benchmark_deconv_target_final.py --output artifacts/native_deconv_target/new_final.json
```
