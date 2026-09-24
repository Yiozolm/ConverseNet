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
    TORCH_CHECK(variant == "v7", "FP32 release supports only variant v7");
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
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Converse2D requires FP32 tensors");
    for (const auto& t : {x0, weight, bias}) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                    "all tensors must have the same device and dtype");
    }
#ifdef CONVERSE2D_WITH_CUDA
    c10::cuda::OptionalCUDAGuard device_guard;
    if (x.is_cuda()) device_guard.set_index(x.get_device());
#endif
#ifdef CONVERSE2D_WITH_CUDA
    // Preserve the FP32 autograd graph before changing layout.
    if (x.is_cuda() &&
        at::GradMode::is_enabled() &&
        (x.requires_grad() || x0.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
        return converse2d::full_training::spatial(x, x0, weight, bias, scale, eps);
    }
#endif
    const bool training = at::GradMode::is_enabled() &&
        (x.requires_grad() || x0.requires_grad() || weight.requires_grad() || bias.requires_grad());
    const bool same_prior = x.is_same(x0);
    auto source = weight;
    x = x.contiguous();
    x0 = same_prior ? x : x0.contiguous();
    bias = bias.contiguous();
    const bool real_fft = !training;  // CPU training uses full spectra too.
    auto lambda = at::sigmoid(bias - 9.0) + eps;
    auto spectra = spectrum(source, weight, Hs, Ws, scale, real_fft);
    auto fb = spectra.first, invw = spectra.second;
    auto fy = real_fft ? at::fft_rfft2(x) : at::fft_fft2(x);
    auto fx0 = same_prior ? fy : (real_fft ? at::fft_rfft2(x0) : at::fft_fft2(x0));
    Tensor fx;
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !at::GradMode::is_enabled()) {
        fx = converse_spectral_cuda(fy, fx0, fb, invw, lambda, H, W, scale, real_fft);
    } else
#endif
    {
        fx = converse2d::reference::spectral(fy,fx0,fb,invw,lambda,W,Ws,scale,real_fft);
    }
    auto out = real_fft ? at::fft_irfft2(fx, at::IntArrayRef({Hs,Ws})) : at::real(at::fft_ifft2(fx));
    return out;
}
