#include "operator.h"
#include "inference/inference.h"
#include "inference/nearest_k2_s2.h"
#include "reference/reference.h"
#include "training/full_spectrum/full_fusion.h"
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <climits>
#include <cmath>
using at::Tensor;
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
Tensor converse2d_forward(Tensor x, Tensor x0, Tensor weight, Tensor bias,
                          int64_t scale, double eps,
                          const std::string &variant) {
    TORCH_CHECK(variant == "v7", "FP32 release supports only variant v7");
    TORCH_CHECK(scale >= 1 && std::isfinite(eps) && eps > 0,
                "scale >= 1 and finite eps > 0 required");
    TORCH_CHECK(x.dim() == 4 && x.numel() > 0, "x must be nonempty (B,C,H,W)");
    auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H <= INT64_MAX / scale && W <= INT64_MAX / scale,
                "output size overflow");
    const int64_t Hs = H * scale, Ws = W * scale;
    TORCH_CHECK(x0.sizes() == at::IntArrayRef({B, C, Hs, Ws}),
                "x0 must be (B,C,H*scale,W*scale)");
    TORCH_CHECK(weight.dim() == 4 &&
                    (weight.size(0) == 1 || weight.size(0) == B) &&
                    (weight.size(1) == 1 || weight.size(1) == C) &&
                    weight.size(2) > 0 && weight.size(3) > 0 &&
                    weight.size(2) <= Hs && weight.size(3) <= Ws,
                "weight must be (1|B,1|C,kh,kw) and kernel must fit output");
    TORCH_CHECK(bias.sizes() == at::IntArrayRef({1, C, 1, 1}),
                "bias must be (1,C,1,1)");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "Converse2D requires FP32 tensors");
    for (const auto &t : {x0, weight, bias}) {
        TORCH_CHECK(t.device() == x.device() &&
                        t.scalar_type() == x.scalar_type(),
                    "all tensors must have the same device and dtype");
    }
#ifdef CONVERSE2D_WITH_CUDA
    c10::cuda::OptionalCUDAGuard device_guard;
    if (x.is_cuda())
        device_guard.set_index(x.get_device());
#endif
#ifdef CONVERSE2D_WITH_CUDA
    // Preserve the FP32 autograd graph before changing layout.
    if (x.is_cuda() && at::GradMode::is_enabled() &&
        (x.requires_grad() || x0.requires_grad() || weight.requires_grad() ||
         bias.requires_grad())) {
        return converse2d::full_training::spatial(x, x0, weight, bias, scale,
                                                  eps);
    }
#endif
    const bool training = at::GradMode::is_enabled() &&
                          (x.requires_grad() || x0.requires_grad() ||
                           weight.requires_grad() || bias.requires_grad());
    const bool same_prior = x.is_same(x0);
    auto source = weight;
    x = x.contiguous();
    x0 = same_prior ? x : x0.contiguous();
    bias = bias.contiguous();
    const bool real_fft = !training; // CPU training uses full spectra too.
    auto lambda = at::sigmoid(bias - 9.0) + eps;
    auto spectra = spectrum(source, weight, Hs, Ws, scale, real_fft);
    auto fb = spectra.first, invw = spectra.second;
    auto fy = real_fft ? at::fft_rfft2(x) : at::fft_fft2(x);
    auto fx0 =
        same_prior ? fy : (real_fft ? at::fft_rfft2(x0) : at::fft_fft2(x0));
    Tensor fx;
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !at::GradMode::is_enabled()) {
        fx = converse_spectral_cuda(fy, fx0, fb, invw, lambda, H, W, scale,
                                    real_fft);
    } else
#endif
    {
        fx = converse2d::reference::spectral(fy, fx0, fb, invw, lambda, W, Ws,
                                             scale, real_fft);
    }
    auto out = real_fft ? at::fft_irfft2(fx, at::IntArrayRef({Hs, Ws}))
                        : at::real(at::fft_ifft2(fx));
    return out;
}

// An explicit nearest-prior API: ordinary forward(x, x0, ...) never guesses
// prior provenance or changes its arithmetic. CPU and every differentiable
// GradMode call retain the existing full operator and its autograd graph.
Tensor converse2d_nearest_k2_s2(Tensor x, Tensor weight, Tensor bias,
                               double eps, const std::string& variant) {
    TORCH_CHECK(variant == "v7", "FP32 release supports only variant v7");
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be finite and positive");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                    bias.scalar_type() == at::kFloat,
                "Converse2D requires FP32 tensors");
    TORCH_CHECK(x.layout() == c10::kStrided && weight.layout() == c10::kStrided &&
                    bias.layout() == c10::kStrided, "strided tensors required");
    TORCH_CHECK(x.device() == weight.device() && x.device() == bias.device(), "device mismatch");
    TORCH_CHECK(x.dim() == 4 && x.numel() > 0, "x must be nonempty (B,C,H,W)");
    const auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(x.numel() <= INT64_MAX / 4 && H <= INT64_MAX / 2 && W <= INT64_MAX / 2,
                "output size overflow");
    TORCH_CHECK(weight.dim() == 4 && weight.size(2) == 2 && weight.size(3) == 2 &&
                    (weight.size(0) == 1 || weight.size(0) == B) &&
                    (weight.size(1) == 1 || weight.size(1) == C),
                "explicit nearest k2/s2 requires weight (1|B,1|C,2,2)");
    TORCH_CHECK(bias.sizes() == at::IntArrayRef({1, C, 1, 1}), "bias must be (1,C,1,1)");
    const bool differentiable = at::GradMode::is_enabled() &&
        (x.requires_grad() || weight.requires_grad() || bias.requires_grad());
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !differentiable) {
        TORCH_CHECK(!at::autocast::is_autocast_enabled(at::kCUDA), "autocast must be disabled");
        // Check differentiability above, before changing GradMode. Frozen
        // GradMode inputs are inference too; requires_grad leaves under
        // no_grad/inference_mode are equally eligible.
        at::NoGradGuard no_grad;
        return converse_nearest_k2_s2_cuda(x, weight, bias, eps);
    }
#endif
    auto prior = at::upsample_nearest2d(x, at::IntArrayRef({2 * H, 2 * W}));
    return converse2d_forward(x, prior, weight, bias, 2, eps, variant);
}
