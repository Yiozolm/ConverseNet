#include <ATen/core/grad_mode.h>
#include "operator.h"
#include "training/full_spectrum/full_fusion.h"
#include "inference/inference.h"
#include "reference/reference.h"
#include <cmath>
#include <climits>
#include <ATen/ATen.h>
using at::Tensor;
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#endif
Tensor converse2d_forward(Tensor x, Tensor x0, Tensor weight, Tensor bias,
                          int64_t scale, double eps, const std::string& variant) {
    TORCH_CHECK(variant == "v2" || variant == "v3" || variant == "v4" ||
                variant == "v5" || variant == "v6" || variant == "v7", "unknown Converse2D variant");
    TORCH_CHECK(scale >= 1 && std::isfinite(eps) && eps > 0, "scale >= 1 and finite eps > 0 required");
    TORCH_CHECK(x.dim() == 4 && x.numel() > 0, "x must be nonempty (B,C,H,W)");
    auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H <= INT64_MAX / scale && W <= INT64_MAX / scale, "output size overflow");
    const int64_t Hs = H * scale, Ws = W * scale;
    TORCH_CHECK(x0.sizes() == at::IntArrayRef({B,C,Hs,Ws}), "x0 must be (B,C,H*scale,W*scale)");
    TORCH_CHECK(weight.dim() == 4 && (weight.size(0) == 1 || weight.size(0) == B) &&
                (weight.size(1) == 1 || weight.size(1) == C) &&
                weight.size(2) > 0 && weight.size(3) > 0 && weight.size(2) <= Hs && weight.size(3) <= Ws,
                "weight must be (1|B,1|C,kh,kw) and kernel must fit output");
    TORCH_CHECK(bias.sizes() == at::IntArrayRef({1,C,1,1}), "bias must be (1,C,1,1)");
    TORCH_CHECK(x.is_floating_point() && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble ||
                x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16), "unsupported input dtype");
    for (const auto& t : {x0, weight, bias}) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                    "all tensors must have the same device and dtype");
    }
#ifdef CONVERSE2D_WITH_CUDA
    c10::cuda::OptionalCUDAGuard device_guard;
    if (x.is_cuda()) device_guard.set_index(x.get_device());
#endif
#ifdef CONVERSE2D_WITH_CUDA
    // Match the validated Python FP32 full-spectrum path before changing any
    // caller layout. Inference and non-FP32/non-v7 fallbacks remain below.
    if (x.is_cuda() && x.scalar_type() == at::kFloat && variant == "v7" &&
        at::GradMode::is_enabled() &&
        (x.requires_grad() || x0.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
        return converse2d::full_training::spatial(x, x0, weight, bias, scale, eps);
    }
#endif
    const auto output_dtype = x.scalar_type();
    const auto compute_dtype = output_dtype == at::kDouble ? at::kDouble : at::kFloat;
    const bool same_prior = x.is_same(x0);
    auto source = weight;
    x = x.to(compute_dtype).contiguous();
    x0 = same_prior ? x : x0.to(compute_dtype).contiguous();
    weight = weight.to(compute_dtype);
    bias = bias.to(compute_dtype).contiguous();
    const bool real_fft = variant == "v7";
    auto lambda = at::sigmoid(bias - 9.0) + eps;
    auto spectra = spectrum(source, weight, Hs, Ws, scale, real_fft);
    auto fb = spectra.first, invw = spectra.second;
    auto fy = real_fft ? at::fft_rfft2(x) : at::fft_fft2(x);
    auto fx0 = same_prior ? fy : (real_fft ? at::fft_rfft2(x0) : at::fft_fft2(x0));
    Tensor fx;
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !at::GradMode::is_enabled() && variant != "v2") {
        fx = converse_spectral_cuda(fy, fx0, fb, invw, lambda, H, W, scale, real_fft);
    } else
#endif
    {
        fx = converse2d::reference::spectral(fy,fx0,fb,invw,lambda,W,Ws,scale,real_fft);
    }
    auto out = real_fft ? at::fft_irfft2(fx, at::IntArrayRef({Hs,Ws})) : at::real(at::fft_ifft2(fx));
    return out.to(output_dtype);
}
