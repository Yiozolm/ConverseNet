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

## P1: backward sites and formal gate (`gate.py`)

Two more callbacks cover the backward FFTs:

- `load_crop_embed` with `store_scaled` handles the VJP of `real_crop(ifft2(Z), pad)`, which
  is `(1/N) * FFT(embed(g))`. It reads the real gradient through its strides into the
  padded grid, zeros outside the crop as in `real_crop_backward_kernel`, and scales at the
  store.
- `store_real` handles the VJP of `fft2(real x)`. It writes only the real part of the
  unnormalized inverse. Its FP32 destination is passed in callerInfo, so cuFFT's own
  output buffer stays complex-sized.

`gate.py` runs every site over 12 shape/pad combinations, 3 input kinds, and for B1 both
contiguous and transposed gradients. Each case is compared on bytes and through
`numerical_policy.comparison` (normal regime) against a NumPy FP64 FFT of the same FP32
inputs. Results on the RTX 5060 Ti (`artifacts/v4_cufft_callbacks/p1_gate_004.json`):

| Site | Passed | Byte-identical | Notes |
|---|---|---|---|
| F1 `load_real` | 21/21 | 21 | |
| F2 `load_circular` | 15/15 | 12 | differs only at 14x12 |
| F3 `store_scaled` | 36/36 | 30 | differs at 96x96 (more accurate) and 14x12 |
| B1 crop-embed + scaled | **65/72** | 54 | 6 plan-creation errors (`CUFFT_INTERNAL_ERROR`) at the degenerate 1x5 transform; 1 budget failure at 14x12, max-abs 1.605x > 1.50x |
| B2 `store_real` | 36/36 | 30 | differs at 96x96 (more accurate) and 14x12 |

All callbacks are byte-identical at the 100x100 production size. At 96x96 the callback plans
are more accurate than ATen (rel-L2 ratio about 0.94). The 14x12 failure moved between runs
when the input stream changed (`p1_gate_003` failed a different 14x12 B1 case at 1.524x).
At that small size the callback plan is a different algorithm and its max-abs error can
exceed the budget. Plan-creation failures and budget failures are both counted as failed
cases, never skipped.

During debugging, single probes could not reproduce the 1x5 plan error. It appeared only
when that shape ran inside the full gate, which is why the gate records per-case errors
instead of aborting.

Backward-site timing, 6 alternating rounds x 30 iterations, B4/C128:

| Pair | ATen median | Callback median | Speedup |
|---|---|---|---|
| B1: VJP of `real_crop(ifft2(Z), 2)`, 96 -> 100 | 912 us | 432 us | 2.13x |
| B2: VJP of `fft2(real x)`, 100x100 | 516 us | 400 us | 1.27x |

ATen's B2 VJP returns a strided real view without a copy. The callback writes a
contiguous FP32 tensor, so its gain is the halved store traffic only.

Not covered yet: autograd integration with the higher-order ATen fallback, fallback on
plan-creation failure, workspace from the PyTorch allocator, CUDA Graph capture (the
prototype updates per-call callerInfo with a host-to-device copy), plan-cache lifetime,
the padded B2 fold, Linux build, and the first-plan JIT cost.
