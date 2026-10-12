#include "full_fusion.h"
#ifdef CONVERSE2D_WITH_CUDA
#include "full_fusion.cpp"
#include "psf_autograd.h"

#include "circular_pad_autograd.h"
#include "real_crop_autograd.h"
#include <cmath>

namespace converse2d::full_training {
static at::Tensor spatial_complex(at::Tensor x, at::Tensor prior, at::Tensor weight,
                   at::Tensor bias, int64_t scale, double eps) {
    const auto h = x.size(2), w = x.size(3);
    // Independent per-call FP32 preparation preserves the original graph and
    // gradient accumulation order. The legacy half-spectrum scope is not used.
    auto psf = PSFPadRoll::apply(weight, h * scale, w * scale);
    auto k = at::fft_fft2(psf);
    auto y = at::fft_fft2(x);
    auto p = x.is_same(prior) ? y : at::fft_fft2(prior);
    auto regularizer = at::sigmoid(bias - 9.0) + eps;
    auto solved = full_spectral(y, p, k, regularizer, scale);
    // The Python final addition follows the prior spectrum's dense layout.
    // cuFFT can choose a different numerical plan for a transposed spectrum;
    // restore that layout before IFFT instead of silently changing its order.
    if (!p.is_contiguous()) {
        auto laid_out = at::empty_like(p);
        laid_out.copy_(solved);
        solved = laid_out;
    }
    return at::fft_ifft2(solved);
}

at::Tensor spatial(at::Tensor x,at::Tensor prior,at::Tensor weight,at::Tensor bias,int64_t scale,double eps) {
    // Same native real view; its VJP embeds (g,+0) in one kernel instead of a
    // zero fill plus strided copy. Falls back to at::real for ineligible z.
    return real_crop(spatial_complex(x,prior,weight,bias,scale,eps),0);
}

at::Tensor circular_pad_complex(at::Tensor x,int64_t padding) {
    return CircularPadComplex::apply(x,padding);
}

at::Tensor circular_s1(at::Tensor x,at::Tensor weight,at::Tensor bias,int64_t padding,double eps) {
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kFloat&&x.dim()==4&&x.numel()>0,
        "circular s1 training requires nonempty CUDA FP32 NCHW input");
    TORCH_CHECK(x.is_contiguous()&&!x.is_neg()&&!x.is_conj(),"circular s1 input must be contiguous without lazy flags");
    const int64_t b=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3);
    TORCH_CHECK(padding>0&&padding<=h&&padding<=w,"positive circular padding must not exceed the input dimensions");
    TORCH_CHECK(std::isfinite(eps)&&eps>0,"finite eps > 0 required");
    for(const auto& t:{weight,bias})
        TORCH_CHECK(t.device()==x.device()&&t.scalar_type()==at::kFloat,"all tensors must have the same device and FP32 dtype");
    TORCH_CHECK(weight.dim()==4&&(weight.size(0)==1||weight.size(0)==b)&&
        (weight.size(1)==1||weight.size(1)==c)&&weight.size(2)>0&&weight.size(3)>0&&
        weight.size(2)<=h+2*padding&&weight.size(3)<=w+2*padding,"weight shape/kernel size invalid");
    TORCH_CHECK(bias.sizes()==at::IntArrayRef({1,c,1,1}),"bias must be (1,C,1,1)");
    TORCH_CHECK(at::GradMode::is_enabled()&&(x.requires_grad()||weight.requires_grad()||bias.requires_grad()),
        "circular s1 entry is for differentiable training only");
    c10::cuda::CUDAGuard guard(x.device());
    auto padded=CircularPadComplex::apply(x,padding);
    // Preserve shared y/prior identity and the per-call differentiable kernel FFT.
    auto result=spatial_complex(padded,padded,weight,bias,1,eps);
    // Native view metadata is preserved; only the ordinary VJP is fused.
    return real_crop(result,padding);
}
} // namespace converse2d::full_training
#endif
