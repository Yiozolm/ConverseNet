# v3.0.0 `scale1_forward` bound check on Colab

This checks whether the recorded RTX 5060 Ti result also holds on another GPU. On the
5060 Ti, the full-spectrum s1 training `scale1_forward` was DRAM-bound at B4/C128/100x100
with a shared prior and kernel `[1,128,3,3]`: 88.34% DRAM (395.4 GB/s) and 31.23% SM.

Open `kernel_bound_colab.ipynb` in Colab (GitHub tab, branch `colab/kernel-bound`),
pick an A100/L4/H100 runtime and run all cells. Or run it on any Linux CUDA host:

```bash
pip install ninja
python -u tools/colab_kernel_bound/run.py --output results/<gpu>-001
```

`run.py` builds the unmodified v3.0.0 production sources and records whether
`Converse2D/`, `models/` and `test/` still match `v3.0.0`. It then writes
`summary.json`/`summary.md` with:

- copy-bandwidth and FP32 GEMM probes (TF32 off),
- torch.profiler kernel times, plus a lower-bound achieved bandwidth for `scale1_forward`
  that needs no hardware counters,
- NCU capture A, which matches the original command (`--set full`, `scale1_forward`,
  launch-count 1, default clock/cache control),
- NCU capture B, which records SpeedOfLight for every kernel in the same training call.

If the host forbids GPU counters (`ERR_NVGPUCTRPERM`), the run reports
`counters_unavailable` instead of NCU numbers. The timing estimate is then the only
evidence. Profiler durations are not benchmarks. One shape on one GPU does not
classify all calls.
