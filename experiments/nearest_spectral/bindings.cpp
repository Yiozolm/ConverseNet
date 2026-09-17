// Generated from the current production source with a private dispatcher
// namespace. It supplies the baseline, PSF preparation and identical cache.
#include "baseline.cpp"

Tensor nearest_phase_cuda(const Tensor&, int64_t, int64_t, int64_t);
Tensor nearest_spectral_cuda(const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, int64_t, int64_t, int64_t);

namespace nearest_spectral_experiment {

Tensor forward_nearest(Tensor x, Tensor weight, Tensor bias,
                       int64_t scale, double eps) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "nearest spectral experiment is inference-only; use no_grad or inference_mode");
    TORCH_CHECK(x.is_cuda(), "nearest spectral experiment requires CUDA");
    TORCH_CHECK(scale >= 1 && std::isfinite(eps) && eps > 0,
                "scale >= 1 and finite eps > 0 required");
    TORCH_CHECK(x.dim() == 4 && x.numel() > 0, "x must be nonempty (B,C,H,W)");
    const auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H <= INT64_MAX / scale && W <= INT64_MAX / scale,
                "output size overflow");
    const int64_t Hs = H * scale, Ws = W * scale;
    TORCH_CHECK(Hs <= INT64_MAX / Ws && Hs * Ws <= INT64_MAX / B / C &&
                Hs <= INT64_MAX - Ws, "output size overflow");
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
    if (scale == 1) return converse2d_forward(x, x, weight, bias, scale, eps, "v7");

    c10::cuda::CUDAGuard device_guard(x.device());
    const auto output_dtype = x.scalar_type();
    const auto compute_dtype = output_dtype == at::kDouble ? at::kDouble : at::kFloat;
    auto source = weight;
    x = x.to(compute_dtype).contiguous();
    weight = weight.to(compute_dtype);
    bias = bias.to(compute_dtype).contiguous();
    auto lambda = at::sigmoid(bias - 9.0) + eps;
    auto spectra = spectrum(source, weight, Hs, Ws, scale, true);
    auto fy = at::fft_rfft2(x);
    // nearest(x)'s spectrum is a tiled LR spectrum times two finite geometric
    // sums. The correction kernels evaluate it on demand: neither the spatial
    // HR prior nor its HR spectrum is allocated or transformed.
    auto phase = nearest_phase_cuda(x, Hs, Ws, scale);
    auto fx = nearest_spectral_cuda(fy, spectra.first, spectra.second,
                                   lambda, phase, H, W, scale);
    // Keep the same post-IFFT normalization and output dtype as the baseline.
    return at::fft_irfft2(fx, at::IntArrayRef({Hs, Ws})).to(output_dtype);
}

} // namespace nearest_spectral_experiment

TORCH_LIBRARY_FRAGMENT(converse2d_nearest_spectral_experiment, m) {
    m.def("forward_nearest(Tensor x, Tensor weight, Tensor bias, int scale, "
          "float eps=1e-5) -> Tensor");
}
TORCH_LIBRARY_IMPL(converse2d_nearest_spectral_experiment, CompositeImplicitAutograd, m) {
    m.impl("forward_nearest", TORCH_FN(nearest_spectral_experiment::forward_nearest));
}
