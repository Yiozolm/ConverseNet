#pragma once
#include <ATen/ATen.h>

// cuFFT LTO callbacks for the full-spectrum training FFTs. The work next to
// each FFT runs inside the FFT's own loads and stores:
//   fwd_load_real / fwd_load_circular: real x (optionally circularly padded)
//       read as (x, +0), replacing the promote or circular-pad kernel;
//   inv_store_scaled: ATen's complex<float>(float(1/N), 0) * z at the IFFT store;
//   vjp_crop_embed + store_scaled: the real/crop VJP embedding (g, +0) read
//       through the gradient's strides, then (1/N) * FFT, as one transform.
// ready_* create or look up the plan and return false whenever the callback
// path is unavailable; callers then use the unchanged ATen path. Small planes,
// failed plan creation, stream capture without a cached plan and
// CONVERSE2D_FFT_CALLBACKS=0 all select ATen.
namespace converse2d::full_training::fft_callbacks {
bool ready_real(const at::Tensor& x, int64_t pad);
at::Tensor fft2_real(const at::Tensor& x, int64_t pad);
bool ready_inverse(const at::Tensor& z);
at::Tensor ifft2_scaled(const at::Tensor& z);
bool ready_crop_embed(const at::Tensor& gradient, int64_t c, int64_t h, int64_t w, int64_t pad);
at::Tensor crop_embed_fft2_scaled(const at::Tensor& gradient, int64_t c, int64_t h, int64_t w,
                                  int64_t pad);
} // namespace converse2d::full_training::fft_callbacks
