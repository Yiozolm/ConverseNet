#include "layernorm.h"
#include <ATen/core/grad_mode.h>
#include <climits>
#include <cmath>
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

namespace converse2d::peripheral {
namespace {
at::Tensor reference(const at::Tensor& input, const at::Tensor& weight,
                     const at::Tensor& bias, double eps) {
    const auto mean = at::mean(input, {1}, true);
    const auto variance = at::mean(at::pow(input - mean, 2), {1}, true);
    // Keep the two original subtractions as separate autograd operations.
    const auto normalized = (input - mean) / at::sqrt(variance + eps);
    return weight.unsqueeze(1).unsqueeze(2) * normalized + bias.unsqueeze(1).unsqueeze(2);
}

void check_contract(const at::Tensor& input, const at::Tensor& weight,
                    const at::Tensor& bias, double eps) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                bias.scalar_type() == at::kFloat, "channel LayerNorm accepts FP32 only");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 1 && bias.dim() == 1 &&
                weight.numel() == input.size(1) && bias.sizes() == weight.sizes(), "expected NCHW and channel vectors");
    TORCH_CHECK(input.device() == weight.device() && input.device() == bias.device(), "device mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be finite and positive");
}

#ifdef CONVERSE2D_WITH_CUDA
bool plain_cuda(const at::Tensor& t) {
    return t.is_cuda() && t.layout() == c10::kStrided && t.is_contiguous() &&
           !t.is_neg() && !t.is_conj() && t.scalar_type() == at::kFloat;
}

// Shared CUDA domain of the fused inference forward, the training forward
// with statistics and the fused backward.
bool fused_domain(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias) {
    return plain_cuda(input) && plain_cuda(weight) && plain_cuda(bias) &&
        (input.size(1) == 64 || input.size(1) == 128) && input.size(2) * input.size(3) > 1 &&
        input.numel() > 0 && input.numel() <= INT_MAX / int64_t(sizeof(float));
}
#endif
} // namespace

at::Tensor channel_layernorm(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias, double eps) {
    check_contract(input, weight, bias, eps);
#ifdef CONVERSE2D_WITH_CUDA
    if (!at::GradMode::is_enabled() && fused_domain(input, weight, bias)) {
        c10::cuda::CUDAGuard guard(input.device());
        return channel_layernorm_cuda(input, weight, bias, eps);
    }
#endif
    return reference(input, weight, bias, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_layernorm_forward_stats(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, double eps) {
    check_contract(input, weight, bias, eps);
#ifdef CONVERSE2D_WITH_CUDA
    TORCH_CHECK(fused_domain(input, weight, bias),
                "channel LayerNorm training statistics require contiguous CUDA FP32 NCHW with 64 or 128 channels");
    c10::cuda::CUDAGuard guard(input.device());
    at::Tensor mean, denominator;
    auto output = channel_layernorm_cuda(input, weight, bias, eps, &mean, &denominator);
    return {output, mean, denominator};
#else
    TORCH_CHECK(false, "channel LayerNorm training statistics require the CUDA build");
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> channel_layernorm_backward(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& denominator, const at::Tensor& weight,
    bool needs_input, bool needs_weight, bool needs_bias) {
#ifdef CONVERSE2D_WITH_CUDA
    TORCH_CHECK(fused_domain(input, weight, weight) && plain_cuda(grad_output) &&
                plain_cuda(mean) && plain_cuda(denominator) &&
                grad_output.sizes() == input.sizes() &&
                mean.numel() == input.size(0) * input.size(2) * input.size(3) &&
                denominator.sizes() == mean.sizes() &&
                grad_output.device() == input.device() && mean.device() == input.device() &&
                denominator.device() == input.device() && weight.device() == input.device(),
                "channel LayerNorm backward requires the tensors produced by its forward");
    if (!needs_input && !needs_weight && !needs_bias) return {};
    c10::cuda::CUDAGuard guard(input.device());
    return channel_layernorm_backward_cuda(grad_output, input, mean, denominator, weight,
                                           needs_input, needs_weight, needs_bias);
#else
    TORCH_CHECK(false, "channel LayerNorm backward requires the CUDA build");
#endif
}

} // namespace converse2d::peripheral
