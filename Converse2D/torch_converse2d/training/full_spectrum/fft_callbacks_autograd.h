#pragma once
#include "circular_pad_autograd.h"
#include "fft_callbacks.h"
#include "real_crop_autograd.h"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>

// Training FFTs through cuFFT callbacks. Forward values come from one fused
// transform; each VJP is exactly the ATen expression of the replaced graph,
// so first-order and higher-order gradients keep their original operations.
namespace converse2d::full_training {
// ATen fft_norm_mode: 0 none, 2 by_n. fft2/ifft2 use dims (-2, -1).
inline at::Tensor fft2_vjp(const at::Tensor& g) { return at::_fft_c2c(g, {2, 3}, 0, false); }
inline at::Tensor ifft2_vjp(const at::Tensor& g) { return at::_fft_c2c(g, {2, 3}, 2, true); }

// fft2(x) for real x, or fft2(circular_pad_complex(x, pad)).
class RealFFT2 : public torch::autograd::Function<RealFFT2> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor x, int64_t pad) {
        ctx->saved_data["h"] = x.size(2);
        ctx->saved_data["w"] = x.size(3);
        ctx->saved_data["pad"] = pad;
        ctx->set_materialize_grads(false);
        return fft_callbacks::fft2_real(x, pad);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list incoming) {
        const auto& gradient = incoming[0];
        if (!gradient.defined() || !ctx->needs_input_grad(0))
            return {at::Tensor(), at::Tensor()};
        const auto pad = ctx->saved_data["pad"].toInt();
        c10::cuda::CUDAGuard guard(gradient.device());
        auto full = fft2_vjp(gradient);
        // Unpadded: the promote's VJP is the native real view.
        if (pad == 0)
            return {at::real(full), at::Tensor()};
        const auto h = ctx->saved_data["h"].toInt(), w = ctx->saved_data["w"].toInt();
        if (at::GradMode::is_enabled())
            return {circular_pad_complex_backward_aten(full, h, w, pad), at::Tensor()};
        return {circular_pad_complex_backward_cuda(full, h, w, pad), at::Tensor()};
    }
};

// Forward FFT of a real training input, through callbacks when available.
inline at::Tensor training_fft2(const at::Tensor& x, int64_t pad) {
    if (!x._fw_grad(0).defined() && fft_callbacks::ready_real(x, pad))
        return RealFFT2::apply(x, pad);
    return at::fft_fft2(pad ? CircularPadComplex::apply(x, pad) : x);
}

// ifft2 with ATen's 1/N normalization applied at the transform's store.
class ScaledIFFT2 : public torch::autograd::Function<ScaledIFFT2> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor z) {
        ctx->set_materialize_grads(false);
        return fft_callbacks::ifft2_scaled(z);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list incoming) {
        const auto& gradient = incoming[0];
        if (!gradient.defined() || !ctx->needs_input_grad(0))
            return {at::Tensor()};
        c10::cuda::CUDAGuard guard(gradient.device());
        return {ifft2_vjp(gradient)};
    }
};

// VJP of real_crop(ifft2(solved), pad) with respect to solved. Like
// RealCropBackward it saves no values; the IFFT VJP is linear.
class RealCropIFFTBackward final : public torch::autograd::Node {
public:
    RealCropIFFTBackward(int64_t b, int64_t c, int64_t h, int64_t w, int64_t pad)
        : b_(b), c_(c), h_(h), w_(w), pad_(pad) {}

    std::string name() const override { return "RealCropIFFTBackward"; }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& incoming) override {
        TORCH_CHECK(incoming.size() == 1, "real/crop IFFT backward expects one gradient");
        const auto& gradient = incoming[0];
        if (!gradient.defined() || !task_should_compute_output(0))
            return {at::Tensor()};
        c10::cuda::CUDAGuard guard(gradient.device());
        if (at::GradMode::is_enabled() || gradient._fw_grad(0).defined())
            return {ifft2_vjp(real_crop_backward_aten(gradient, b_, c_, h_, w_, pad_))};
        auto g = gradient.resolve_conj().resolve_neg();
        if (fft_callbacks::ready_crop_embed(g, c_, h_, w_, pad_))
            return {fft_callbacks::crop_embed_fft2_scaled(g, c_, h_, w_, pad_)};
        return {ifft2_vjp(real_crop_cuda_backward(gradient, b_, c_, h_, w_, pad_))};
    }

private:
    const int64_t b_, c_, h_, w_, pad_;
};

// real_crop(ifft2(solved), pad): the output keeps the native real/crop view of
// the IFFT result; its edge goes straight to solved's producer.
inline at::Tensor ifft2_real_crop(const at::Tensor& solved, int64_t pad) {
    const bool forward_ad = solved._fw_grad(0).defined();
    auto z = !forward_ad && fft_callbacks::ready_inverse(solved) ? ScaledIFFT2::apply(solved)
                                                                 : at::fft_ifft2(solved);
    if (!z.is_cuda() || !z.is_contiguous() || z.is_neg() || z.is_conj() || z.is_inference() ||
        c10::InferenceMode::is_enabled() || !torch::autograd::compute_requires_grad(solved) ||
        forward_ad || z._fw_grad(0).defined())
        return real_crop(z, pad);
    const auto h = z.size(2), w = z.size(3);
    TORCH_CHECK(pad >= 0 && pad <= (h - 1) / 2 && pad <= (w - 1) / 2,
                "real/crop padding must be nonnegative and leave a nonempty interior");
    auto output = at::real(z);
    if (pad > 0)
        output = output.slice(2, pad, h - pad).slice(3, pad, w - pad);
    auto node = std::make_shared<RealCropIFFTBackward>(z.size(0), z.size(1), h, w, pad);
    node->set_next_edges(torch::autograd::collect_next_edges(solved));
    // Same edge replacement as real_crop: the view metadata still points at z,
    // so an in-place rebase reconstructs the native chain through z's IFFT node.
    torch::autograd::set_history(output, node);
    return output;
}
} // namespace converse2d::full_training
