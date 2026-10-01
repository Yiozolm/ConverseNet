#pragma once
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <cstdint>
#include <memory>
#include <string>

namespace converse2d::full_training {
at::Tensor real_crop_cuda_backward(
    const at::Tensor& gradient, int64_t b, int64_t c,
    int64_t h, int64_t w, int64_t pad);

inline at::Tensor real_crop_backward_aten(
    const at::Tensor& gradient, int64_t b, int64_t c,
    int64_t h, int64_t w, int64_t pad) {
    auto full = gradient;
    if (pad > 0) {
        // Undo the width slice before the height slice, as in the native graph.
        full = at::slice_backward(
            full, {b, c, h - 2 * pad, w}, 3, pad, w - pad, 1);
        full = at::slice_backward(full, {b, c, h, w}, 2, pad, h - pad, 1);
    }
    // Native real is view_as_real followed by select of the real component.
    return at::view_as_complex(at::select_backward(full, {b, c, h, w, 2}, 4, 0));
}

// This Node replaces only the linear VJP of the native view chain. It must not
// save z's values: backward has no value or version dependency on its input.
class RealCropBackward final : public torch::autograd::Node {
public:
    RealCropBackward(int64_t b, int64_t c, int64_t h, int64_t w, int64_t pad)
        : b_(b), c_(c), h_(h), w_(w), pad_(pad) {}

    std::string name() const override {
        return "RealCropBackward";
    }

    torch::autograd::variable_list apply(
        torch::autograd::variable_list&& incoming) override {
        TORCH_CHECK(incoming.size() == 1, "real/crop backward expects one gradient");
        const auto& gradient = incoming[0];
        if (!gradient.defined() || !task_should_compute_output(0))
            return {at::Tensor()};
        if (at::GradMode::is_enabled() || gradient._fw_grad(0).defined())
            return {real_crop_backward_aten(gradient, b_, c_, h_, w_, pad_)};
        c10::cuda::CUDAGuard guard(gradient.device());
        return {real_crop_cuda_backward(gradient, b_, c_, h_, w_, pad_)};
    }

private:
    const int64_t b_, c_, h_, w_, pad_;
};

at::Tensor real_crop(at::Tensor z, int64_t pad) {
    TORCH_CHECK(
        z.defined() && z.scalar_type() == at::kComplexFloat &&
            z.dim() == 4 && z.numel() > 0,
        "real/crop requires a nonempty 4D complex64 tensor");
    const auto h = z.size(2), w = z.size(3);
    TORCH_CHECK(
        pad >= 0 && pad <= (h - 1) / 2 && pad <= (w - 1) / 2,
        "real/crop padding must be nonnegative and leave a nonempty interior");

    // Keep the exact native metadata, including the cross-dtype ViewFunc.
    // A custom Function returning this view would instead mark it
    // IN_CUSTOM_FUNCTION and reject otherwise-valid view+inplace operations.
    auto output = at::real(z);
    if (pad > 0)
        output = output.slice(2, pad, h - pad).slice(3, pad, w - pad);

    if (!z.is_cuda() || !z.is_contiguous() || z.is_neg() || z.is_conj() ||
        z.is_inference() || c10::InferenceMode::is_enabled() ||
        !torch::autograd::compute_requires_grad(z) || z._fw_grad(0).defined())
        return output;

    auto node = std::make_shared<RealCropBackward>(
        z.size(0), z.size(1), h, w, pad);
    node->set_next_edges(torch::autograd::collect_next_edges(z));
    // Only replace the current edge; preserve DifferentiableViewMeta and its
    // CreationMeta. A subsequent inplace rebase can then reconstruct the
    // original native view backward, which is equivalent to this linear VJP.
    torch::autograd::set_history(output, node);
    return output;
}
} // namespace converse2d::full_training
