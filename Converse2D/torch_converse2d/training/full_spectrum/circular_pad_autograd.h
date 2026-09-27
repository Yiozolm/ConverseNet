#pragma once
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>

namespace converse2d::full_training {
at::Tensor circular_pad_complex_cuda(const at::Tensor& input, int64_t pad);
at::Tensor circular_pad_complex_backward_cuda(
    const at::Tensor& gradient, int64_t h, int64_t w, int64_t pad);

inline at::Tensor circular_pad_complex_backward_aten(
    const at::Tensor& gradient, int64_t h, int64_t w, int64_t pad) {
    auto real = at::real(gradient);
    // Circular-pad forward copies interior, left, right, top, then bottom.
    // Its VJP therefore folds bottom, top, right, then left. The zero adds
    // reproduce the intervening full-sized slice-backward additions, including
    // signed zeros. Each ATen operation remains differentiable for higher orders.
    auto vertical = at::add(real.slice(2, pad, pad + h),
        at::constant_pad_nd(real.slice(2, pad + h, pad + h + pad), {0, 0, 0, h - pad}, 0));
    vertical = at::add(vertical,
        at::constant_pad_nd(at::add(real.slice(2, 0, pad), 0.0), {0, 0, h - pad, 0}, 0));
    auto horizontal = at::add(vertical.slice(3, pad, pad + w),
        at::constant_pad_nd(vertical.slice(3, pad + w, pad + w + pad), {0, w - pad}, 0));
    return at::add(horizontal,
        at::constant_pad_nd(at::add(vertical.slice(3, 0, pad), 0.0), {w - pad, 0}, 0));
}

class CircularPadComplex : public torch::autograd::Function<CircularPadComplex> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                              at::Tensor input, int64_t pad) {
        // This map is linear; backward only needs shape metadata, not input values.
        auto output = circular_pad_complex_cuda(input, pad);
        ctx->saved_data["h"] = input.size(2);
        ctx->saved_data["w"] = input.size(3);
        ctx->saved_data["pad"] = pad;
        ctx->set_materialize_grads(false);
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list incoming) {
        const auto& gradient = incoming[0];
        if (!gradient.defined() || !ctx->needs_input_grad(0))
            return {at::Tensor(), at::Tensor()};
        const auto h = ctx->saved_data["h"].toInt();
        const auto w = ctx->saved_data["w"].toInt();
        const auto pad = ctx->saved_data["pad"].toInt();
        c10::cuda::CUDAGuard guard(gradient.device());
        if (at::GradMode::is_enabled())
            return {circular_pad_complex_backward_aten(gradient, h, w, pad), at::Tensor()};
        return {circular_pad_complex_backward_cuda(gradient, h, w, pad), at::Tensor()};
    }
};
} // namespace converse2d::full_training
