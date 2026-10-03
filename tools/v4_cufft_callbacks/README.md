# cuFFT LTO callbacks: P0 feasibility (research only)

This prototype folds the work next to each FFT into the FFT's own load and store. It is
not wired into production.

- `callbacks.cu` defines three callbacks:
  - `load_real`: real FP32 -> `(x, +0)`, replacing ATen's promote kernel.
  - `load_circular`: the production circular-pad index map plus `(x, +0)`, replacing
    `circular_pad_complex_forward_kernel`.
  - `store_scaled`: ATen's `complex<float>(float(1/N), 0) * z`, replacing the separate
    normalization pass after `ifft2`.
- `ext.cpp` creates its own C2C plans. They match ATen's simple-layout plan: null embeds,
  unit stride, dense batch.
- `loader.py` compiles the callbacks to an LTO-IR fatbin and builds the extension.
- `study.py` runs the byte checks against ATen, FP64 errors when bytes differ, sentinel
  checks on the real input buffer, kernel lists and paired timing.

Platform facts (Windows, torch 2.11+cu130):
- LTO callbacks work with the dynamic cuFFT. torch loads `torch/lib/cufft64_12.dll`
  (cuFFT 12.0) and `nvJitLink_130_0.dll`. The extension links the toolkit `cufft.lib`,
  which resolves to the same DLL.
- Callbacks must be plain `__device__` functions, looked up by their unmangled source name.
  With `extern "C"`, every callback plan failed with `CUFFT_INTERNAL_ERROR` (5).
- An LTO-IR for `compute_75` linked and ran on sm_120 with identical bytes to `lto_120`
  (8x8 probe). One low-arch fatbin may therefore serve all GPUs; only the 8x8 probe
  checked this.

Results, RTX 5060 Ti (`artifacts/v4_cufft_callbacks/p0_002.json`), 7 shapes plus 4 padded
shapes, each with randn, special-value and 1e18-scaled inputs:

| Check | Result |
|---|---|
| Our plans vs ATen, forward and unscaled inverse | 21/21 and 21/21 byte-identical |
| `load_real` vs ATen promote + `fft2` | 21/21 byte-identical; real input buffer and sentinel tail intact |
| `store_scaled` vs ATen `ifft2` | 18/21 identical; all 3 differences are 96x96 |
| `load_circular` vs pad + `fft2` | 9/12 identical; all 3 differences are 8x6 padded by 3 (14x14) |
| Production size 100x100 (B4/C128, padded USRNet) | identical for all callbacks |

The differing cases change the FFT itself (millions of components at 96x96), not just the
scaling. Callback plans can use different first-pass kernels (`body_lto_fft`). Against the
FP64 FFT, the callback path had a lower relative L2 in all 6 cases. Max-abs was lower in 5;
in one (scaled, 14x14) it was 1.26x the ATen value. This is inside the policy's 1.50x
max-abs allowance, but no formal `numerical_policy.py` gate has been run.

Paired CUDA-event timing, 6 alternating rounds x 30 iterations, B4/C128:

| Pair | ATen median | Callback median | Speedup |
|---|---|---|---|
| promote + fft2, 100x100 | 647 us | 415 us | 1.56x |
| circular pad (production kernel) + fft2, 96 -> 100 | 640 us | 410 us | 1.56x |
| ifft2 with 1/N scaling, 100x100 | 694 us | 463 us | 1.50x |

Not covered yet: the backward FFTs, the autograd/higher-order fallback, workspace from
the PyTorch allocator, CUDA Graph capture, plan-cache lifetime, Linux build, and the
first-plan JIT cost (cached by the driver afterwards).
