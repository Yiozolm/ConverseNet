#pragma once
#include <ATen/ATen.h>

namespace converse2d::peripheral {

// FP32 weight gradient of a 1x1 convolution (stride 1, no padding, groups 1):
// grad_output is NCHW [B, Cout, H, W], input is NCHW [B, Cin, H, W]; the
// result is [Cout, Cin, 1, 1]. Deterministic split-K SIMT GEMM with
// compensated accumulation; CUDA only. Routing policy lives in the caller.
at::Tensor pointwise_wgrad(const at::Tensor& grad_output, const at::Tensor& input);

#ifdef CONVERSE2D_WITH_CUDA
at::Tensor pointwise_wgrad_cuda(const at::Tensor& grad_output, const at::Tensor& input);
#endif

} // namespace converse2d::peripheral
