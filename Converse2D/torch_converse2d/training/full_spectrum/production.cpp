#include "full_fusion.h"
#ifdef CONVERSE2D_WITH_CUDA
#include "full_fusion.cpp"
#include "psf_autograd.h"

namespace converse2d::full_training {
at::Tensor spatial(at::Tensor x, at::Tensor prior, at::Tensor weight, at::Tensor bias, int64_t scale, double eps) {
    const auto h=x.size(2), w=x.size(3);
    // Independent per-call FP32 preparation preserves the original graph and
    // gradient accumulation order. The legacy half-spectrum scope is not used.
    auto psf=PSFPadRoll::apply(weight,h*scale,w*scale);
    auto k=at::fft_fft2(psf);
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
