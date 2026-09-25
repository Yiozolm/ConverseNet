#pragma once
#include <ATen/ATen.h>

namespace converse2d::peripheral {

at::Tensor channel_layernorm(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias, double eps);

#ifdef CONVERSE2D_WITH_CUDA
at::Tensor channel_layernorm_cuda(const at::Tensor& input, const at::Tensor& weight,
                                  const at::Tensor& bias, double eps);
#endif

} // namespace converse2d::peripheral
