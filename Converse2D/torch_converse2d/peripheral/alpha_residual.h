#pragma once
#include <ATen/ATen.h>

namespace converse2d::peripheral {

// FP32 alpha * branch + residual, with original ATen fallback for layouts
// outside the narrow CUDA forward specialization. No parameter/state changes.
at::Tensor alpha_residual(const at::Tensor& alpha, const at::Tensor& branch,
                          const at::Tensor& residual);
at::Tensor channel_affine(const at::Tensor& scale, const at::Tensor& input,
                          const at::Tensor& bias);

#ifdef CONVERSE2D_WITH_CUDA
at::Tensor alpha_residual_cuda(const at::Tensor& alpha, const at::Tensor& branch,
                               const at::Tensor& residual);
at::Tensor channel_affine_cuda(const at::Tensor& scale, const at::Tensor& input,
                              const at::Tensor& bias);
#endif

} // namespace converse2d::peripheral
