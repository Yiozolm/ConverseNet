#include "inference.h"
#include "launchers.cuh"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>

at::Tensor converse_spectral_cuda(const at::Tensor& fy, const at::Tensor& fx0,
    const at::Tensor& fb, const at::Tensor& invw, const at::Tensor& lambda,
    int64_t H, int64_t W, int64_t s, bool half) {
    // FFT output stride is not assumed: PyTorch may return transposed FFT storage.
    auto y = fy.contiguous(), prior = fx0.contiguous(), kernel = fb.contiguous();
    auto denom = invw.defined() ? invw.contiguous() : at::Tensor();
    auto out = at::empty(prior.sizes(), prior.options());
    auto stream = c10::cuda::getCurrentCUDAStream(prior.get_device());
    auto q = s == 1 ? at::Tensor() : at::empty(y.sizes(), y.options());
    if (s==1) launch_inference_scale1(y,prior,kernel,denom,lambda,q,out,H,W,s,half,stream);
    else if (s==2) launch_inference_scale2(y,prior,kernel,denom,lambda,q,out,H,W,s,half,stream);
    else if (s==3) launch_inference_scale3(y,prior,kernel,denom,lambda,q,out,H,W,s,half,stream);
    else launch_inference_generic(y,prior,kernel,denom,lambda,q,out,H,W,s,half,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
