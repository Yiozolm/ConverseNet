// The study copies the current production implementation into the build folder
// and changes only its registration namespace. No production files are edited.
#include "baseline.cpp"

namespace nearest_cpp_experiment {

Tensor forward_nearest(Tensor x, Tensor weight, Tensor bias, int64_t scale,
                       double eps, const std::string& variant) {
    TORCH_CHECK(variant == "v2" || variant == "v3" || variant == "v4" ||
                variant == "v5" || variant == "v6" || variant == "v7",
                "unknown Converse2D variant");
    TORCH_CHECK(scale >= 1 && std::isfinite(eps) && eps > 0,
                "scale >= 1 and finite eps > 0 required");
    TORCH_CHECK(x.dim() == 4 && x.numel() > 0, "x must be nonempty (B,C,H,W)");
    const auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H <= INT64_MAX / scale && W <= INT64_MAX / scale,
                "output size overflow");
    const int64_t Hs = H * scale, Ws = W * scale;
    TORCH_CHECK(weight.dim() == 4 && (weight.size(0) == 1 || weight.size(0) == B) &&
                (weight.size(1) == 1 || weight.size(1) == C) &&
                weight.size(2) > 0 && weight.size(3) > 0 &&
                weight.size(2) <= Hs && weight.size(3) <= Ws,
                "weight must be (1|B,1|C,kh,kw) and kernel must fit output");
    TORCH_CHECK(bias.sizes() == at::IntArrayRef({1,C,1,1}), "bias must be (1,C,1,1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble ||
                x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "unsupported input dtype");
    for (const auto& t : {weight, bias}) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                    "all tensors must have the same device and dtype");
    }

    // Match F.interpolate(mode="nearest", scale_factor=scale) in the original
    // dtype, including its low-precision backward reduction. Keeping x itself
    // at scale=1 also preserves the solver's shared-prior FFT shortcut.
    auto x0 = scale == 1 ? x : at::upsample_nearest2d(
        x, at::IntArrayRef({Hs, Ws}), double(scale), double(scale));
    return converse2d_forward(x, x0, weight, bias, scale, eps, variant);
}

} // namespace nearest_cpp_experiment

TORCH_LIBRARY_FRAGMENT(converse2d_nearest_experiment, m) {
    m.def("forward_nearest(Tensor x, Tensor weight, Tensor bias, int scale, "
          "float eps=1e-5, str variant='v7') -> Tensor");
}
TORCH_LIBRARY_IMPL(converse2d_nearest_experiment, CompositeImplicitAutograd, m) {
    m.impl("forward_nearest", TORCH_FN(nearest_cpp_experiment::forward_nearest));
}
