#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <climits>

using at::Tensor;

static Tensor alias_mean(const Tensor& a, int64_t s) {
    if (s == 1) return a;
    return a.reshape({a.size(0),a.size(1),s,a.size(2)/s,s,a.size(3)/s}).mean({2,4});
}
static Tensor full_spectrum(const Tensor& half, int64_t width) {
    auto tail=half.slice(-1,1,(width+1)/2).flip({-2,-1}).roll({1},{-2});
    if (half.is_complex()) tail=tail.conj();
    return at::cat({half,tail},-1);
}

// The spatial entry is deliberately FP32-only. These adapters let the shared
// header compile without linking the unrelated native-low-precision FFT policy.
namespace converse2d::native_fft {
static Tensor real_fft(const Tensor& t, at::ScalarType) { return at::fft_rfft2(t); }
static Tensor real_ifft(const Tensor& t,int64_t h,int64_t w,at::ScalarType) {
    return at::fft_irfft2(t,at::IntArrayRef({h,w}));
}
}
#include "converse2d_training.h"
#include "optimized_training.h"

static Tensor forward(Tensor x,Tensor x0,Tensor weight,Tensor bias,int64_t s,double eps,
                      bool nearest,bool optimized) {
    TORCH_CHECK(s>=1 && std::isfinite(eps) && eps>0,"positive scale/eps required");
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.dim()==4 && x.numel()>0,
                "training experiment expects nonempty FP32 CUDA BCHW tensors");
    const auto B=x.size(0),C=x.size(1),H=x.size(2),W=x.size(3);
    TORCH_CHECK(H<=INT64_MAX/s && W<=INT64_MAX/s,"output size overflow");
    TORCH_CHECK(nearest || x0.sizes()==at::IntArrayRef({B,C,H*s,W*s}),"invalid prior shape");
    TORCH_CHECK(weight.dim()==4 && (weight.size(0)==1 || weight.size(0)==B) &&
                (weight.size(1)==1 || weight.size(1)==C) && weight.size(2)>0 && weight.size(3)>0 &&
                weight.size(2)<=H*s && weight.size(3)<=W*s,"invalid filter shape");
    TORCH_CHECK(bias.sizes()==at::IntArrayRef({1,C,1,1}),"invalid bias shape");
    for(const auto& t:{x0,weight,bias})
        TORCH_CHECK(t.device()==x.device() && t.scalar_type()==at::kFloat,"FP32 device mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    return optimized?converse2d::training_optimized::forward(x,x0,weight,bias,s,eps,nearest):
                     converse2d::training::forward(x,x0,weight,bias,s,eps,nearest);
}
static Tensor spectral(Tensor y,Tensor p,Tensor k,Tensor lambda,int64_t H,int64_t W,int64_t s,
                       bool optimized) {
    return optimized?converse2d::training_optimized::spectral(y,p,k,lambda,H,W,s):
                     converse2d::training::spectral(y,p,k,lambda,H,W,s);
}
TORCH_LIBRARY(training_speed,m) {
    m.def("forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int scale, float eps, bool nearest=False, bool optimized=False) -> Tensor");
    m.def("spectral(Tensor y, Tensor p, Tensor k, Tensor regularizer, int H, int W, int scale, bool optimized=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(training_speed,CompositeImplicitAutograd,m) {
    m.impl("forward",TORCH_FN(forward));
    m.impl("spectral",TORCH_FN(spectral));
}
