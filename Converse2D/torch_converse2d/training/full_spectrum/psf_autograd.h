#pragma once
#include <torch/csrc/autograd/custom_function.h>
#include <c10/cuda/CUDAGuard.h>

namespace converse2d::full_training {
at::Tensor psf_pad_roll_cuda(const at::Tensor& weight, int64_t h, int64_t w);
at::Tensor psf_pad_roll_backward_cuda(const at::Tensor& gradient, int64_t kh, int64_t kw);

class PSFPadRoll : public torch::autograd::Function<PSFPadRoll> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                              at::Tensor weight, int64_t h, int64_t w) {
        ctx->saved_data["kh"] = weight.size(2);
        ctx->saved_data["kw"] = weight.size(3);
        return psf_pad_roll_cuda(weight, h, w);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list incoming) {
        const auto kh = ctx->saved_data["kh"].toInt();
        const auto kw = ctx->saved_data["kw"].toInt();
        const auto& g = incoming[0];
        c10::cuda::CUDAGuard guard(g.device());
        if (at::GradMode::is_enabled()) {
            // Match RollBackward's reversed dimension order, then PadBackward's
            // negative padding. constant_pad_nd clones even for a pure crop or
            // zero padding; a slice view would change the upstream layout.
            auto shifted = at::roll(g, {kw / 2, kh / 2}, {-1, -2});
            auto result = at::constant_pad_nd(shifted, {0, kw - g.size(3), 0, kh - g.size(2)}, 0);
            return {result, at::Tensor(), at::Tensor()};
        }
        return {psf_pad_roll_backward_cuda(g, kh, kw), at::Tensor(), at::Tensor()};
    }
};
} // namespace converse2d::full_training
