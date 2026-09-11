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
```

- `test_correctness.py` (also exposed by `test_error.py`): independent dense
  spatial solves, forward/gradient agreement, first/second derivatives, odd/even
  and singleton sizes, low precision, padding, mutation-aware caches and streams.
- `test_batched_kernels.py`: per-sample/channel-shared kernels, dynamic-kernel
  gradients, DataNet mixed precision and USRNet end-to-end training.
- `test_cache.py`: focused cache invalidation and inference/training transitions.

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
python test/test_speed.py --single --variant v7 --B 1 --C 32 --H 128 --W 128 --scale 2
```

`test_speed.py` invokes `benchmark.py`. The default grid compares the current v2,
v6 and v7 paths using identical inputs and a float64 reference check. Timings
include CUDA events, wall time and incremental peak allocated memory. Training
includes forward plus gradients for x/x0/weight/bias.

Use `--iters` to control repetitions and `--output` to select a result path.
Defaults are `artifacts/benchmark.json` and `artifacts/benchmark_training.json`.

## Nsight

```sh
python test/profile_nsight.py --kind systems --variant v7 --scale 2
python test/profile_nsight.py --kind compute --variant v7 --scale 2
```

The wrapper builds first, then profiles the warmed CUDA/NVTX region. Reports go
to `artifacts/profiles/`. Use `--tool` for an explicit profiler executable and
`--C`, `--H`, `--W` for the workload. Nsight Compute requires access to the GPU
performance counters; the scripts do not modify system permissions.

`CONVERSE2D_SKIP_BUILD=1` loads the existing local extension without building.
Only use it after compiling the current sources, for example inside a profiler.
