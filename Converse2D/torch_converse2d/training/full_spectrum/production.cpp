#include "full_fusion.h"
#ifdef CONVERSE2D_WITH_CUDA
// Embed the checked implementation without registering a second library or
// Python module. The standalone experiment compiles full_fusion.cpp directly.
#define CONVERSE_FULL_SPECTRUM_EMBEDDED 1
#include "full_fusion.cpp"
#undef CONVERSE_FULL_SPECTRUM_EMBEDDED

namespace converse2d::full_training {
at::Tensor spatial(at::Tensor x, at::Tensor prior, at::Tensor weight, at::Tensor bias, int64_t scale, double eps) {
    const auto h=x.size(2), w=x.size(3), kh=weight.size(2), kw=weight.size(3);
    // Independent per-call FP32 preparation preserves the original graph and
    // gradient accumulation order. The legacy half-spectrum scope is not used.
    auto psf=at::constant_pad_nd(weight,{0,w*scale-kw,0,h*scale-kh},0);
    auto k=at::fft_fft2(at::roll(psf,{-(kh/2),-(kw/2)},{-2,-1}));
    auto y=at::fft_fft2(x);
    auto p=x.is_same(prior)?y:at::fft_fft2(prior);
    auto regularizer=at::sigmoid(bias-9.0)+eps;
    auto solved=full_spectral(y,p,k,regularizer,scale);
    // The Python final addition follows the prior spectrum's dense layout.
    // cuFFT can choose a different numerical plan for a transposed spectrum;
    // restore that layout before IFFT instead of silently changing its order.
    if (!p.is_contiguous()) {
        auto laid_out=at::empty_like(p);
        laid_out.copy_(solved);
        solved=laid_out;
    }
    return at::real(at::fft_ifft2(solved));
}
}
#endif
