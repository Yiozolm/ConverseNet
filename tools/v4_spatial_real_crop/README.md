# Fused real-part VJP for the unpadded public forward

`6ba4f5f` added `real_crop`, which keeps the native `at::real` view metadata and replaces
only its first-order VJP. Native backward does a zero fill plus a strided copy into the real
component. `real_crop` writes `(g, +0)` in one kernel. Until now only the circular-padded
s1 entry used it. The general `spatial()` path, which serves s1/s2/s3 differentiable
training, still ended in `at::real`. It now returns `real_crop(z, 0)`. No arithmetic changes.

- `capture.py` checks 213 public-forward cases (s1/s2/s3, B1-4, shared and independent
  prior, KB/KC broadcast, plain/transposed/expanded/negative-zero upstream, the production
  shapes, and one double-backward case). For every output and gradient it records bytes,
  shape, stride and storage offset.
- `perf.py` runs one paired timing round for s1/s2/s3 configurations. It also records
  kernels per call and the real-part VJP kernels by name.

Recorded on the RTX 5060 Ti, MSVC 14.44, CUDA 13.0 (local `artifacts/v4_spatial_real_crop/`),
against baseline `4ac6847`:

- Byte capture: 213/213 identical, including strides and offsets. Checked on candidate
  binaries `0889b3b7` and `4548b2bf`. The 814-case spectral capture from
  `tools/v4_scale1_planes/` is also identical.
- Release suite: 206/206, including the new public-forward test in `test_real_crop.py`.
- Kernels per call: one fewer at every scale. `FillFunctor` plus the copy is replaced by a
  single `real_crop_backward_kernel` (s1 29 -> 28, s2/s3 38 -> 37).
- Paired timing, 6 alternating rounds, medians of summed kernel time per call:
  s1 B4/C128/100x100 1.046x, s2 B4/C64/64x64 1.062x, s3 B2/C32/48x48 0.998x (small shape,
  within noise). Round-to-round spread is large (for example s1 candidate 4.18-5.03 ms),
  so these are indicative only, not an admission-grade or whole-model result.
