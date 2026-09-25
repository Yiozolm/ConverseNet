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
} // namespace

at::Tensor channel_layernorm(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias, double eps) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                bias.scalar_type() == at::kFloat, "channel LayerNorm accepts FP32 only");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 1 && bias.dim() == 1 &&
                weight.numel() == input.size(1) && bias.sizes() == weight.sizes(), "expected NCHW and channel vectors");
    TORCH_CHECK(input.device() == weight.device() && input.device() == bias.device(), "device mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be finite and positive");
#ifdef CONVERSE2D_WITH_CUDA
    const bool eligible = !at::GradMode::is_enabled() && input.is_cuda() &&
        input.layout() == c10::kStrided && weight.layout() == c10::kStrided && bias.layout() == c10::kStrided &&
        input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
        !input.is_neg() && !weight.is_neg() && !bias.is_neg() &&
        !input.is_conj() && !weight.is_conj() && !bias.is_conj() &&
        (input.size(1) == 64 || input.size(1) == 128) && input.size(2) * input.size(3) > 1 &&
        input.numel() > 0 && input.numel() <= INT_MAX / int64_t(sizeof(float));
    if (eligible) {
        c10::cuda::CUDAGuard guard(input.device());
        return channel_layernorm_cuda(input, weight, bias, eps);
    }
#endif
    return reference(input, weight, bias, eps);
}

} // namespace converse2d::peripheral
