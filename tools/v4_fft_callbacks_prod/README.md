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

### First A100 run (A100-SXM4-40GB, torch 2.11 / CUDA 13.0, git eaa49e2, binary 2163ec77)

- The Linux build linked `-lcufft -lnvrtc` without problems.
- Callbacks off vs on in one binary: 213/213 module and 814/814 spectral captures identical.
- Admission differs from the RTX 5060 Ti. At 64x64 only one callback FFT was admitted
  (`forward_s1_64`: lto 1, ATen 1/N 1, ATen real_crop 1). The 96x96 rows match the 5060 Ti.
- Release suite: 212 run, 55 failures, recorded unchanged.
  - One was `test_callback_transforms_are_selected_only_at_and_above_side_16` at 64x64. It
    assumed the 5060 Ti admission set and now only requires some callback at side 16 or more.
  - The other 54 are FP64-budget failures on planes below 16, which never use callbacks.
    Examples: 5x7 s1/s2/s3/s4 training, PSF preparation, and s2/s3 fusion. Ratios are
    1.3-2.3x against 1.25/1.50 limits. Attribution is open; `followup.py` reruns the
    suite with callbacks off and at control refs.
- Paired timing (call medians, ATen -> callbacks):

  | Config | Call speedup | Kernel-time speedup |
  |---|---|---|
  | circular s1 B4/C64 96 pad 2 | 0.95x | 0.81x |
  | circular s1 B4/C128 96 pad 2 | 0.91x | 0.91x |
  | s1 B4/C128 100 | 1.05x | 1.06x |
  | s2 B4/C64 64 | 1.02x | 1.03x |
  | s3 B2/C32 48 | 1.13x | 0.91x |

  The pad, scale and crop passes that the callbacks remove are cheap at A100 bandwidth.
  The callback-linked FFT kernels cost more than those passes saved.

### A100 follow-up (`a100_followup.ipynb` -> `followup.py`)

- Release suite here with `CONVERSE2D_FFT_CALLBACKS=0` and with the default.
- Release suite at 5a5f0ba (before callbacks), 347b040 (codex/v4.0.0) and v3.0.0, each
  built in its own git worktree.
  - A failure at HEAD that is missing at a control ref may be a test that does not exist
    there; check the logs.
- Per-site timing with per-kernel-name times. `CONVERSE2D_FFT_CALLBACKS` accepts a comma
  list of `real,circular,inverse,crop_embed` to enable only those sites; `0` disables all
  and unset or `1` enables all.
