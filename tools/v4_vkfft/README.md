# VkFFT (CUDA backend) vs cuFFT: feasibility (research only)

`loader.py` clones VkFFT v1.3.4 (`066a17c1`) into `.build/VkFFT` and builds `ext.cpp`, a
wrapper for batched 2-D FP32 C2C, R2C and C2R transforms on the current torch stream. Plans
are NVRTC-compiled by VkFFT on first use and cached by shape. Nothing is wired into
production.

`study.py --output <json>` measures:
- accuracy against an FP64 FFT of the same FP32 input, through
  `test/numerical_policy.comparison` with ATen/cuFFT as the FP32 baseline. This is a
  per-transform proxy, not the operator release gate;
- paired CUDA-event timing, with ATen and VkFFT alternating in each of 6 rounds x 30 calls.

Twiddles: VkFFT's FP32 CUDA default (`useLUT=-1`) computes them with `__sincosf`, a
fast-math intrinsic. The FP32 policy excludes it. `vk_lut` (`useLUT=1`) uses a twiddle table
precomputed on the host and is the only eligible mode. `vk_sincosf` is measured only for
comparison.

## RTX 5060 Ti, torch 2.11+cu130 (`artifacts/v4_vkfft/study_001.json`)

Accuracy, 90 rows (randn / 1e18-scaled / special-value inputs x 9 shapes x 4 transforms):

| Mode | Failed budgets | Notes |
|---|---|---|
| `vk_lut` | 7 / 90 | All on 97x101 or 33x17, which use Bluestein/Rader. On every production shape (48-144, smooth), rel-L2 is 0.69-0.95x cuFFT's |
| `vk_sincosf` | 81 / 90 | rel-L2 up to 2.2x on production shapes, 3.5x on prime shapes |

Median us per call. Column ratios are ATen/VkFFT, so a value above 1 means VkFFT is faster:

| Shape (B, C, H, W) | fft2 | ifft2 drop-in | ifft2 in place | rfft2 | irfft2 |
|---|---|---|---|---|---|
| 4,128,100,100 | 425 / 422 (1.01x) | 636 / 631 (1.01x) | 423 (1.50x) | 1.04x | 1.07x |
| 4,128,96,96 | 387 / 387 (1.00x) | 1.00x | 1.50x | 1.19x | 1.33x |
| 4,64,100,100 | 1.08x | 1.00x | 2.36x | 1.05x | 1.30x |
| 4,64,128,128 | 334 / 337 (0.99x) | 0.98x | 1.80x | 0.91x | 0.99x |
| 2,32,144,144 | 1.05x | 1.09x | 1.60x | 1.03x | 1.29x |
| 4,64,64,64 | 1.09x | 1.13x | 1.57x | 1.12x | 1.44x |
| 16,64,256,256 | 1.00x | 1.00x | 1.50x | 1.00x | 1.32x |

Reading:
- Training C2C transforms run at the memory roofline on this GPU. For 4x128x100x100, two
  read+write passes over 41 MB take 424 us: about 386 GB/s of the card's 448 GB/s. That
  is true for both libraries, so VkFFT cannot help these transforms here.
- The "in place" inverse gains come from dropping ATen's separate 1/N pass. The production
  `store_scaled` cuFFT callback already removes that pass for the training IFFT, so
  production has already captured this gain.
- Production's load callbacks (real promotion, circular padding, crop embedding) have no
  VkFFT equivalent in v1.3.4. Swapping in VkFFT would bring those separate copy kernels
  back.
- The one real VkFFT gain is C2R: `irfft2` is 1.29-1.44x faster on five of seven shapes
  (1.07x at 4x128x100x100, 0.99x at 128x128), with matching or better accuracy. Only the half-spectrum inference path calls
  `irfft2` (`operator.cpp`).

Not covered: A100/other GPUs (VkFFT's published gains are largest on high-bandwidth
cards), NVRTC plan-compile latency, CUDA graph capture, whole-operator or model timing.
