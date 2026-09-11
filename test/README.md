# Validation and profiling

Run from the repository root with PyTorch and Ninja available. The scripts build
the current checkout under `.build/`; no global extension install is required.
On Windows, use the same Developer shell / CUDA toolkit configuration as for
installation. Python's standard `unittest` is used; pytest is not required.

```sh
python test/test_error.py --device cuda
python test/test_error.py --device cpu
python test/test_cache.py
python test/test_pretrained_smoke.py
```

`test_error.py` is the entry point for `test_correctness.py`. The suite compares
all v2–v7 labels on even/odd and singleton dimensions, scales 1/2/3, plus the
generic scale=4 path. It checks a dense spatial linear-system solution independent
of FFT, gradients of x/x0/weight/bias, gradcheck, gradgradcheck, low-precision
promotion, rectangular kernels, noncontiguous inputs, padding, cache mutation,
repeated training and non-default CUDA streams. Results are written to
`analysis/correctness_cuda.json` and `analysis/correctness_cpu.json`.

To test the build without CUDA kernels, set `CONVERSE2D_CPU_ONLY=1` before running
the CPU suite. This builds a separate `.build/cpu` extension.

The pretrained smoke test loads the repository's DnCNN and SRResNet checkpoints,
then compares complete-model outputs in float32 and float64 using seeded random
inputs. It disables cuDNN/matmul TF32 for a strict arithmetic comparison; it does
not measure dataset PSNR. `--installed` tests an extension built in `Converse2D/`.

## Comparable benchmarks

```sh
python test/test_speed.py --legacy
python test/test_speed.py --legacy --training --iters 10 --output analysis/gpu_training_benchmark.json
```

The optional legacy baseline is read from Git commit `c7eb880` and compiled under
a separate operator namespace. It receives exactly the same tensors and padding
as the corrected versions. No branch switch is performed. The benchmark covers
v2, v6 (shared v3–v6 implementation), v7, and optionally legacy. Every corrected
forward is checked against a float64 reference before timing.

Timings are medians of five batches with CUDA events, with host wall time and
incremental peak allocated memory also recorded. Training timing includes forward
and gradients for all four independent inputs. Results are measurements on the
current machine, not universal speedup guarantees.

The old `grad_ok` speed-test field did not compare gradients and is no longer used
as a correctness claim. The old grid/variant-per-build CLI is replaced by these
commands; use `--help` for supported options.

## Nsight

```sh
python test/profile_nsight.py --kind systems --variant v7 --scale 2
python test/profile_nsight.py --kind systems --variant legacy --scale 2
python test/profile_nsight.py --kind compute --variant v7 --scale 2
```

Use `--tool` for an explicit profiler executable, and `--C`, `--H`, `--W` to
change the workload. The wrapper builds first, then profiles only the warmed
CUDA/NVTX region using `cudaProfilerStart/Stop`. Reports and kernel/API summaries
are saved under `analysis/profiles/` (ignored by Git).

`CONVERSE2D_SKIP_BUILD=1` loads an existing local build; use it only after building
the current sources, particularly when invoking a profiler manually.

On the tested machine Nsight Systems collected CUDA traces successfully. Nsight
Compute connected but returned `ERR_NVGPUCTRPERM`; hardware-counter results are
unavailable until the machine's NVIDIA performance-counter policy permits them.
The scripts do not change system profiling permissions.
