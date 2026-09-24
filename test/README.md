# FP32 release validation

Run `python -m unittest discover -s test -p 'test_*.py' -v` (or the repository's
Windows `tools/run.ps1` launcher). The loader checks all production source/header
hashes and builds locally, avoiding stale installed extensions.

- `test_build_layout.py`: production source set and transitive fingerprints.
- `test_full_spectrum_default.py`: FFT routing, byte-identical Python FP32 output
  and VJP checks, broadcast/shared kernels, gradient subsets, weak regularization,
  noncontiguous inputs and singleton widths. Unsupported dtype/version rejection.
- `test_fp32_release.py`: independent FP64 max-absolute/relative-L2 noninferiority,
  higher derivatives, cache mutation, streams, and module dtype contracts.
- `test_cuda_graph.py`: graph ownership, invalidation, replay and transitions.
- `test_pretrained_fp32.py`: DnCNN/SRResNet checkpoint and FP32 inference checks.
- `release_snapshot.py`: separate-process pre/post cleanup comparison of inference
  and full pretrained USRNet tensor hashes, including gradients and Adam states.

Example snapshot:

```sh
python test/release_snapshot.py --root /path/to/before --output before.json
python test/release_snapshot.py --output after.json --compare before.json
```

The trajectory uses three seeds and three Adam updates each, HR24 crops from the
included real image, full pretrained USRNet. It verifies cleanup preservation;
it is not a training recipe, dataset-quality benchmark, or convergence study.

Historical experiments, failures, benchmarks and old tests are preserved in
pre-cleanup commit `1b579ea`, rather than being included in the release test suite.
