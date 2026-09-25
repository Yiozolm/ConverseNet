#include "alpha_residual.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace converse2d::peripheral {
namespace {

__global__ void alpha_residual_forward(
    const float* alpha, const float* branch, const float* residual, float* output,
    int64_t total, int64_t channels, int64_t height, int64_t width,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += int64_t(blockDim.x) * gridDim.x) {
        const int64_t x = i % width;
        const int64_t y = (i / width) % height;
        const int64_t c = (i / (height * width)) % channels;
        const int64_t n = i / (channels * height * width);
        const int64_t offset = n * stride_n + c * stride_c + y * stride_h + x * stride_w;
        // Both Python operations round to FP32. Explicit intrinsics prevent
        // contraction into an FMA even with the normal optimizing compiler.
        const float product = __fmul_rn(alpha[c], branch[i]);
        output[i] = __fadd_rn(product, residual[offset]);
    }
}

} // namespace

template<bool ChannelBias, bool Vectorized>
__global__ void affine_plane(const float* scale, const float* input, const float* residual,
                             float* output, int channels, int hw) {
    const int channel = blockIdx.y;
    const int64_t base = (int64_t(blockIdx.z)*channels+channel)*hw;
    const float a = scale[channel];
    const int elements = Vectorized ? hw/4 : hw;
    for (int j = blockIdx.x*blockDim.x+threadIdx.x;
         j < elements; j += blockDim.x*gridDim.x) {
        if constexpr(Vectorized) {
            const float4 v = reinterpret_cast<const float4*>(input+base)[j];
            float4 r;
            if constexpr(ChannelBias) {
                const float b = residual[channel];
                r = make_float4(b,b,b,b);
            } else r = reinterpret_cast<const float4*>(residual+base)[j];
            reinterpret_cast<float4*>(output+base)[j] = make_float4(
                __fadd_rn(__fmul_rn(a,v.x),r.x), __fadd_rn(__fmul_rn(a,v.y),r.y),
                __fadd_rn(__fmul_rn(a,v.z),r.z), __fadd_rn(__fmul_rn(a,v.w),r.w));
        } else {
            const float r = ChannelBias ? residual[channel] : residual[base+j];
            output[base+j] = __fadd_rn(__fmul_rn(a,input[base+j]),r);
        }
    }
}

template<bool ChannelBias>
bool launch_planes(const at::Tensor& scale, const at::Tensor& input,
                   const at::Tensor& residual, at::Tensor& output) {
    const auto b=input.size(0), c=input.size(1), hw=input.size(2)*input.size(3);
    if(b>65535 || c>65535 || hw>INT_MAX-1024*256) return false;
    if constexpr(!ChannelBias) { if(!residual.is_contiguous()) return false; }
    const auto aligned=[](const at::Tensor& t){return (reinterpret_cast<uintptr_t>(t.const_data_ptr())&15)==0;};
    const bool vectorized = hw%4==0 && aligned(input) && (ChannelBias || aligned(residual));
    const auto stream=c10::cuda::getCurrentCUDAStream(input.get_device());
    const auto run=[&](auto tag){
        constexpr bool V=decltype(tag)::value;
        const int n=int(V?hw/4:hw);
        const dim3 grid(std::min((n+255)/256,1024),unsigned(c),unsigned(b));
        affine_plane<ChannelBias,V><<<grid,256,0,stream>>>(scale.const_data_ptr<float>(),input.const_data_ptr<float>(),
            residual.const_data_ptr<float>(),output.data_ptr<float>(),int(c),int(hw));
    };
    if(vectorized)run(std::true_type{});else run(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

__global__ void channel_affine_forward(const float* scale, const float* input, const float* bias,
                                       float* output, int64_t total, int64_t channels, int64_t hw) {
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i < total; i += int64_t(blockDim.x)*gridDim.x) {
        const int64_t c = (i/hw)%channels;
        output[i] = __fadd_rn(__fmul_rn(scale[c], input[i]), bias[c]);
    }
}

at::Tensor channel_affine_cuda(const at::Tensor& scale, const at::Tensor& input,
                              const at::Tensor& bias) {
    auto output = at::empty(input.sizes(), input.options());
    const auto n = output.numel();
    if (!n) return output;
    if (launch_planes<true>(scale,input,bias,output)) return output;
    const int blocks = static_cast<int>(std::min<int64_t>((n+255)/256, 4096));
    channel_affine_forward<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream(input.get_device())>>>(
        scale.const_data_ptr<float>(), input.const_data_ptr<float>(), bias.const_data_ptr<float>(),
        output.data_ptr<float>(), n, input.size(1), input.size(2)*input.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor alpha_residual_cuda(const at::Tensor& alpha, const at::Tensor& branch,
                               const at::Tensor& residual) {
    auto output = at::empty(branch.sizes(), branch.options());
    const auto total = output.numel();
    if (total == 0) return output;
    if (launch_planes<false>(alpha,branch,residual,output)) return output;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 4096));
    const auto stream = c10::cuda::getCurrentCUDAStream(branch.get_device());
    alpha_residual_forward<<<blocks, threads, 0, stream>>>(
        alpha.const_data_ptr<float>(), branch.const_data_ptr<float>(), residual.const_data_ptr<float>(),
        output.data_ptr<float>(), total, branch.size(1), branch.size(2), branch.size(3),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace converse2d::peripheral
