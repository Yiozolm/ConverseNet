# FP32 regression tests

Run from the repository root:

```sh
python -m unittest discover -s test -p 'test_*.py' -v
```

On Windows, use `./tools/run.ps1` instead of `python` to initialize the compiler.
Set `CONVERSE2D_CPU_ONLY=1` for a CPU build: all CUDA-specific cases are skipped,
while build checks, the CPU extension and the portable Python fallback still run.

| File | Coverage |
| --- | --- |
| `test_build_layout.py` | Source selection and transitive fingerprints; no Torch/compiler required |
| `test_full_spectrum_default.py` | Full/half routing; exact FP32 output/VJP; shared/broadcast kernels, gradient masks, weak regularization and strided layouts |
| `test_fp32_release.py` | Independent FP64 noninferiority, higher derivatives, cache/streams, dtype contracts and CPU fallback |
| `test_cuda_graph.py` | Graph ownership, replay, invalidation and training transitions |
| `test_pretrained_fp32.py` | DnCNN/SRResNet checkpoint and inference compatibility |

`support.py` owns shared fixtures, byte comparisons and CPU/CUDA policy.
`extension_loader.py` builds locally under `.build/` and verifies source/header
and binary fingerprints. Tests never import fixtures from another test module.
The 32 test methods retain their numerical thresholds, shapes and random seeds.

Release benchmarks, quality campaigns, snapshots and note generation are local
tools under `tools/release/`, excluded by `.gitignore`; they are not needed for
this suite. Their measured source versions remain in commit `0a99235` under the
old `test/` paths. Release notes and measured results stay under `docs/`.
