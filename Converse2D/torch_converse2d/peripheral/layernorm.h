#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace converse2d::peripheral {

at::Tensor channel_layernorm(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias, double eps);

// Training entry points: the forward also returns the per-pixel mean and
// standard-deviation denominator so the fused backward can avoid a second
// statistics pass. The backward returns (dx, dweight, dbias); outputs that
// are not requested are undefined tensors.
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_layernorm_forward_stats(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_layernorm_backward(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& denominator, const at::Tensor& weight,
    bool needs_input, bool needs_weight, bool needs_bias);

#ifdef CONVERSE2D_WITH_CUDA
at::Tensor channel_layernorm_cuda(const at::Tensor& input, const at::Tensor& weight,
                                  const at::Tensor& bias, double eps,
                                  at::Tensor* mean_out = nullptr, at::Tensor* denominator_out = nullptr);
std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_layernorm_backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& denominator, const at::Tensor& weight,
    bool needs_input, bool needs_weight, bool needs_bias);
#endif

} // namespace converse2d::peripheral
