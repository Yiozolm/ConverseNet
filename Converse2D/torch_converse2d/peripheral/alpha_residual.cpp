#include "alpha_residual.h"
#include <ATen/ExpandUtils.h>
#include <ATen/core/grad_mode.h>
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#endif

namespace converse2d::peripheral {
namespace {

at::Tensor reference(const at::Tensor& alpha, const at::Tensor& branch,
                     const at::Tensor& residual) {
    return at::add(at::mul(alpha, branch), residual);
}

#ifdef CONVERSE2D_WITH_CUDA
bool supported(const at::Tensor& alpha, const at::Tensor& branch,
               const at::Tensor& residual) {
    if (!branch.is_cuda() || branch.layout() != c10::kStrided ||
        alpha.layout() != c10::kStrided || residual.layout() != c10::kStrided ||
        branch.dim() != 4 || alpha.dim() != 4 ||
        branch.sizes() != residual.sizes() ||
        alpha.size(0) != 1 || alpha.size(1) != branch.size(1) ||
        alpha.size(2) != 1 || alpha.size(3) != 1 ||
        !branch.is_contiguous() || !alpha.is_contiguous() ||
        alpha.is_neg() || branch.is_neg() || residual.is_neg() ||
        alpha.is_conj() || branch.is_conj() || residual.is_conj()) return false;

    // Preserve the original two-node accumulation order for direct aliases.
    // Ordinary model branches can still have a shared differentiable ancestor;
    // their VJPs retain the same ATen products and broadcast reduction below.
    if (alpha.is_alias_of(branch) || alpha.is_alias_of(residual) ||
        branch.is_alias_of(residual)) return false;
    for (const auto stride : residual.strides()) if (stride < 0) return false;
    return true;
}

class AlphaResidual : public torch::autograd::Function<AlphaResidual> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                              at::Tensor alpha, at::Tensor branch, at::Tensor residual, bool channel_bias) {
        // Save only operands actually needed by a derivative. In particular,
        // residual-only differentiation must not introduce extra version checks
        // for frozen alpha/branch values that the original AddBackward did not save.
        ctx->save_for_backward({branch.requires_grad() ? alpha : at::Tensor(),
                               alpha.requires_grad() ? branch : at::Tensor()});
        ctx->saved_data["alpha_shape"] = alpha.sizes().vec();
        ctx->saved_data["branch_shape"] = branch.sizes().vec();
        ctx->saved_data["residual_shape"] = residual.sizes().vec();
        ctx->saved_data["channel_bias"] = channel_bias;
        ctx->set_materialize_grads(false);
        return channel_bias ? channel_affine_cuda(alpha, branch, residual)
                            : alpha_residual_cuda(alpha, branch, residual);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, torch::autograd::variable_list incoming) {
        const auto& g = incoming[0];
        if (!g.defined()) return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        c10::cuda::CUDAGuard guard(g.device());
        // autograd.grad can request only residual even when all inputs require
        // grad. The original engine then never executes MulBackward at all.
        at::Tensor gr;
        if (ctx->needs_input_grad(2)) gr = ctx->saved_data["channel_bias"].toBool()
            ? at::sum_to(g, ctx->saved_data["residual_shape"].toIntVector()) : g;
        if (!ctx->needs_input_grad(0) && !ctx->needs_input_grad(1))
            return {at::Tensor(), at::Tensor(), gr, at::Tensor()};
        const auto saved = ctx->get_saved_variables();
        at::Tensor ga, gb;
        // Deliberately keep ATen MulBackward's arithmetic and sum_to reduction.
        // These same differentiable ATen expressions are the higher-order
        // fallback; no handwritten reduction, atomic add or fused VJP is used.
        if (ctx->needs_input_grad(0))
            ga = at::sum_to(at::mul(g, saved[1]), ctx->saved_data["alpha_shape"].toIntVector());
        if (ctx->needs_input_grad(1))
            gb = at::sum_to(at::mul(g, saved[0]), ctx->saved_data["branch_shape"].toIntVector());
        return {ga, gb, gr, at::Tensor()};
    }
};
#endif

} // namespace

at::Tensor alpha_residual(const at::Tensor& alpha, const at::Tensor& branch,
                          const at::Tensor& residual) {
    TORCH_CHECK(alpha.scalar_type() == at::kFloat && branch.scalar_type() == at::kFloat &&
                residual.scalar_type() == at::kFloat, "alpha_residual requires FP32 tensors");
    TORCH_CHECK(alpha.device() == branch.device() && residual.device() == branch.device(),
                "alpha_residual inputs must be on the same device");
#ifdef CONVERSE2D_WITH_CUDA
    if (supported(alpha, branch, residual)) {
        c10::cuda::CUDAGuard guard(branch.device());
        if (at::GradMode::is_enabled() &&
            (alpha.requires_grad() || branch.requires_grad() || residual.requires_grad()))
            return AlphaResidual::apply(alpha, branch, residual, false);
        return alpha_residual_cuda(alpha, branch, residual);
    }
#endif
    return reference(alpha, branch, residual);
}

at::Tensor channel_affine(const at::Tensor& scale, const at::Tensor& input,
                          const at::Tensor& bias) {
    TORCH_CHECK(scale.scalar_type() == at::kFloat && input.scalar_type() == at::kFloat &&
                bias.scalar_type() == at::kFloat, "channel_affine requires FP32 tensors");
    TORCH_CHECK(scale.device() == input.device() && bias.device() == input.device(),
                "channel_affine inputs must be on the same device");
#ifdef CONVERSE2D_WITH_CUDA
    const bool eligible = input.is_cuda() && input.layout() == c10::kStrided &&
        scale.layout() == c10::kStrided && bias.layout() == c10::kStrided &&
        input.dim() == 4 && input.is_contiguous() &&
        scale.sizes() == at::IntArrayRef({input.size(1), 1, 1}) && bias.sizes() == scale.sizes() &&
        scale.is_contiguous() && bias.is_contiguous() && !input.is_neg() && !scale.is_neg() && !bias.is_neg() &&
        !input.is_alias_of(scale) && !input.is_alias_of(bias) && !scale.is_alias_of(bias);
    if (eligible) {
        c10::cuda::CUDAGuard guard(input.device());
        if (at::GradMode::is_enabled() && (scale.requires_grad() || input.requires_grad() || bias.requires_grad()))
            return AlphaResidual::apply(scale, input, bias, true);
        return channel_affine_cuda(scale, input, bias);
    }
#endif
    return reference(scale, input, bias);
}

} // namespace converse2d::peripheral
