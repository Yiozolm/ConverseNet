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
```

- `test_correctness.py` (also exposed by `test_error.py`): independent dense
  spatial solves, the six-argument operator API, forward/gradient agreement,
  first/second derivatives, odd/even
  and singleton sizes, low precision, padding, mutation-aware caches and streams.
- `test_batched_kernels.py`: per-sample/channel-shared kernels, dynamic-kernel
  gradients, DataNet mixed precision and USRNet end-to-end training.
- `test_cache.py`: focused cache invalidation and inference/training transitions.
- `test_cuda_graph.py`: capture with a warm eager cache, graph eviction, changed
  inputs and weights, shape/batch/scale/dtype changes, independent outputs and
  sequential calls on different CUDA streams; optional enable/disable, cache
  release, and CPU/training passthrough when disabled.
- `test_spectral_io.py`: v7 spectral I/O changes against the frozen pre-change
  kernel and float64 reference, including subnormal inputs, rectangular PSFs,
  broadcast kernels and bitwise-equivalent PSF spectra. Requires CUDA and Git.

These use Python's standard unittest; pytest is not required.

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
python test/test_speed.py
python test/test_speed.py --training
python test/test_speed.py --single --B 1 --C 32 --H 128 --W 128 --scale 2
```

`test_speed.py` invokes `benchmark.py`. The default grid measures the single
fused solver across shapes and scales, with a float64 reference check. Timings
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

```sh
python test/profile_nsight.py --kind systems --scale 2
python test/profile_nsight.py --kind compute --scale 2
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
directories to retain multiple shapes or collection policies for the same scale.

`CONVERSE2D_SKIP_BUILD=1` loads the existing local extension without building.
Only use it after compiling the current sources, for example inside a profiler.
