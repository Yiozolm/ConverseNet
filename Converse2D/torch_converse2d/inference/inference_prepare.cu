#include "../common/fp32_dispatch.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>

template <typename T, typename I>
__global__ void prepare_psf(const T* weight, T* otf, I total, I H, I W, I kh, I kw) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I w = (i % W + kw / 2) % W;
    const I h = ((i / W) % H + kh / 2) % H;
    otf[i] = h < kh && w < kw ? weight[(i / (H*W) * kh + h) * kw + w] : T(0);
}

at::Tensor converse_psf_cuda(const at::Tensor& weight, int64_t h, int64_t w) {
    auto source = weight.contiguous();
    auto out = at::empty({weight.size(0), weight.size(1), h, w}, weight.options());
    const auto n = out.numel(), kh = weight.size(2), kw = weight.size(3);
    auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());
    CONVERSE_DISPATCH_FP32(weight.scalar_type(), "converse_psf", [&] {
        if (n <= INT_MAX - 256 && h <= INT_MAX / 2 && w <= INT_MAX / 2) {
            prepare_psf<scalar_t,int><<<(n+255)/256,256,0,stream>>>(
                source.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),int(n),int(h),int(w),int(kh),int(kw));
        } else {
            prepare_psf<scalar_t,int64_t><<<(n+255)/256,256,0,stream>>>(
                source.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),n,h,w,kh,kw);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
