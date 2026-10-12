#include "full_fusion.h"
#ifdef CONVERSE2D_WITH_CUDA
#include "full_fusion.cpp"
#include "psf_autograd.h"

#include "circular_pad_autograd.h"
#include "fft_callbacks_autograd.h"
#include "real_crop_autograd.h"
#include <cmath>

namespace converse2d::full_training {
// Returns the solved spectrum; pad > 0 circularly pads x, which is then also
// the prior. The x and prior FFTs may run through cuFFT callbacks.
static at::Tensor spatial_solved(at::Tensor x, at::Tensor prior, at::Tensor weight,
                   at::Tensor bias, int64_t scale, double eps, int64_t pad) {
    TORCH_CHECK(pad == 0 || x.is_same(prior), "circular padding requires the shared prior");
    const auto h = x.size(2) + 2 * pad, w = x.size(3) + 2 * pad;
    // Node creation order fixes backward execution order, and with it the
    // accumulation order at shared ancestors: the padded spectrum is created
    // first, where the circular pad node was; the unpadded one after k.
    at::Tensor y;
    if (pad > 0)
        y = training_fft2(x, pad);
    // Independent per-call FP32 preparation preserves the original graph and
    // gradient accumulation order. The legacy half-spectrum scope is not used.
    auto psf = PSFPadRoll::apply(weight, h * scale, w * scale);
    auto k = at::fft_fft2(psf);
    if (pad == 0)
        y = training_fft2(x, 0);
    auto p = x.is_same(prior) ? y : training_fft2(prior, 0);
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
    return solved;
}

at::Tensor spatial(at::Tensor x,at::Tensor prior,at::Tensor weight,at::Tensor bias,int64_t scale,double eps) {
    // Same native real view of ifft2(solved); see ifft2_real_crop.
    return ifft2_real_crop(spatial_solved(x,prior,weight,bias,scale,eps,0),0);
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
    // Preserve shared y/prior identity and the per-call differentiable kernel FFT.
    auto solved=spatial_solved(x,x,weight,bias,1,eps,padding);
    // Native view metadata is preserved; only the ordinary VJP is fused.
    return ifft2_real_crop(solved,padding);
}
} // namespace converse2d::full_training
#endif
