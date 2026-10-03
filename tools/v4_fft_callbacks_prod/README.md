# Production cuFFT callback FFTs

These are the production counterparts of the research sites in `tools/v4_cufft_callbacks/`:
`fwd_load_real`, `fwd_load_circular`, `inv_store_scaled` and `vjp_crop_embed` with
`inv_store_scaled`. They live in `training/full_spectrum/fft_callbacks.{h,cpp}` and
`fft_callbacks_autograd.h`.

- NVRTC compiles the callbacks to LTO-IR for the current device. The extension links
  `cufft` and `nvrtc`, which resolve to the libraries torch already loads.
- **Bit-identity admission.** On first use, each plan runs once on a random probe next
  to the ATen expression it replaces. It is kept only if the bits match; otherwise that
  shape uses ATen. On the RTX 5060 Ti the circular-s1 100x100 transforms keep every
  callback. The unpadded 96x96 inverse and crop-embed plans are rejected; their forward
  load is kept.
- **Other ATen fallbacks:** planes with a side below 16, plans cuFFT cannot create,
  stream capture without a cached plan, forward-mode AD tangents, and
  `CONVERSE2D_FFT_CALLBACKS=0`.
- **Gradients.** Every VJP is the ATen expression of the replaced graph. Higher-order
  gradients use ATen ops. The fused real/crop+IFFT node keeps native view metadata and
  routes to the solved spectrum. The forward-FFT node sits where the replaced nodes were,
  which keeps backward execution and shared-ancestor accumulation order.

Recorded on the RTX 5060 Ti, MSVC 14.44, CUDA 13.0 (local `artifacts/v4_fft_callbacks_prod/`):

- All checks below used candidate binary `2560b85b`, with no rebuild in between.
- Byte captures against the `claude/v4-dev` baseline binary `4548b2bf`: 213/213
  public-forward cases and 814/814 spectral cases identical, including gradient strides
  and offsets.
- Release suite: 212/212, including `test/test_fft_callbacks.py`.
  - Before bit-identity admission, the first candidate failed one new budget case:
    s2 48 -> 96x96, weak regularization, bias-gradient rel-L2 2.40x > 1.50x.
  - That candidate also differed in bytes for the two 96x96 module captures.
  - Both are preserved in `fft-callbacks-release-002.log` and `module_001.json`.
- Paired timing, 6 alternating fresh-process rounds, `CONVERSE2D_FFT_CALLBACKS=0` vs
  default in the same binary. Every callback round beat every ATen round:

  | Config | Kernels/call | Call median | Speedup |
  |---|---|---|---|
  | circular s1 B4/C128 96 pad 2 | 29 -> 25 | 5.20 -> 3.91 ms | 1.33x |
  | forward s1 B4/C128 100 | 28 -> 24 | 5.09 -> 3.81 ms | 1.34x |
  | circular s1 B4/C64 96 pad 2 | 29 -> 25 | 1.41 -> 1.29 ms | 1.10x |
  | forward s2 B4/C64 64 -> 128 | 37 -> 35 | 3.89 -> 3.66 ms | 1.06x |
  | forward s3 B2/C32 48 -> 144 | 37 -> 32 | 0.98 -> 0.85 ms | 1.15x |

These are operator-level timings, not a whole-model or training-convergence result.

Not yet covered: the Linux/Colab build (`-lcufft -lnvrtc`), multi-GPU plan use, and
plan-cache growth. Plans and their callerInfo live for the process, one per distinct shape.

## Other GPUs (Colab)

Open `a100_colab.ipynb` from branch `claude/v4-dev`. It runs `run_colab.py`, which does the
following:

1. checked Linux build;
2. release suite;
3. callbacks-off vs callbacks-on byte captures in one binary;
4. `admission.py`;
5. paired timing.

Byte baselines do not carry across GPUs.

Admission counts per training call on the RTX 5060 Ti (`admission.py`). `lto_fft` is the
number of callback-linked FFT kernels; ATen 1/N and real_crop kernels mark sites that
stayed on ATen.

| Shape | lto_fft | ATen 1/N | ATen real_crop |
|---|---|---|---|
| circular s1 96 pad 2 (100x100), 48 pad 1 (50x50) | 4 | 0 | 0 |
| s1 100, 64, 33x17 | 4 | 0 | 0 |
| s1 96 | 1 | 2 | 1 |
| s2 48 (96x96), s2 64 (128x128), s3 32 (96x96) | 2 | 2 | 1 |
| s3 48 (144x144) | 5 | 0 | 0 |
