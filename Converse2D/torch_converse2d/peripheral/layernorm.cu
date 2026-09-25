#include "layernorm.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace converse2d::peripheral {
namespace {

__device__ __forceinline__ float left_combine(float a,float b,float c,float d) {
    return __fadd_rn(__fadd_rn(__fadd_rn(a,b),c),d);
}

template<bool Variance>
__device__ __forceinline__ float term(const float* input,int index,float mean) {
    if constexpr(Variance) {
        const float delta=__fsub_rn(input[index],mean);
        return __fmul_rn(delta,delta);
    }
    return input[index];
}

template<int Channels,int Stride,bool Variance>
__device__ __forceinline__ float lane_reduce(const float* input,int base,int hw,int lane,float mean) {
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
    #pragma unroll 1
    for(int c=lane;c<Channels;c+=4*Stride) {
        a0=__fadd_rn(a0,term<Variance>(input,base+c*hw,mean));
        a1=__fadd_rn(a1,term<Variance>(input,base+(c+Stride)*hw,mean));
        a2=__fadd_rn(a2,term<Variance>(input,base+(c+2*Stride)*hw,mean));
        a3=__fadd_rn(a3,term<Variance>(input,base+(c+3*Stride)*hw,mean));
    }
    return left_combine(a0,a1,a2,a3);
}

template<int Channels,bool Variance>
__device__ __forceinline__ float partial(const float* input,int base,int hw,int stride,int lane,float mean) {
    if(stride==4)return lane_reduce<Channels,4,Variance>(input,base,hw,lane,mean);
    if(stride==8)return lane_reduce<Channels,8,Variance>(input,base,hw,lane,mean);
    return lane_reduce<Channels,1,Variance>(input,base,hw,lane,mean);
}

int floor_power_two(int value) {
    int result=1;
    while(result<=value/2)result*=2;
    return result;
}

// Same source-derived ReduceConfig as the frozen serial variant.
int reduction_stride(int channels,int pixels,int hw,uintptr_t pointer) {
    int vector=4;
    while((pointer/sizeof(float))%vector || hw%vector)vector/=2;
    const int max_threads=512/vector;
    const int dim0=std::min(floor_power_two(pixels/vector),max_threads);
    const int dim1=std::min(floor_power_two(channels),max_threads);
    const int width=std::min(dim0,32);
    const int height=std::min(dim1,max_threads/width);
    return channels>=std::min(height*16,256) ? height : 1;
}

template<int Channels>
__global__ void full_layernorm_parallel_two_pass(
    const float* input,const float* weight,const float* bias,float* output,
    int pixels,int hw,int mean_stride,int variance_stride,float factor,float eps) {
    // A warp spans adjacent pixels; y lanes reproduce ATen's input partitions.
    // Every thread in a tail CTA participates in every barrier.
    __shared__ float scratch[8][32];
    __shared__ float means[32];
    __shared__ float denominators[32];
    const int x=threadIdx.x,y=threadIdx.y;
    const int pixel=int(blockIdx.x)*32+x;
    const bool valid=pixel<pixels;
    const int base=valid ? (pixel/hw)*Channels*hw+pixel%hw : 0;

    float local=0.f;
    if(valid && y<mean_stride)
        local=partial<Channels,false>(input,base,hw,mean_stride,y,0.f);
    scratch[y][x]=local;
    __syncthreads();
    for(int offset=mean_stride/2;offset>0;offset>>=1) {
        if(y<offset)scratch[y][x]=__fadd_rn(scratch[y][x],scratch[y+offset][x]);
        __syncthreads();
    }
    if(y==0)means[x]=valid ? __fmul_rn(scratch[0][x],factor) : 0.f;
    __syncthreads();

    const float mean=means[x];
    local=0.f;
    if(valid && y<variance_stride)
        local=partial<Channels,true>(input,base,hw,variance_stride,y,mean);
    scratch[y][x]=local;
    __syncthreads();
    for(int offset=variance_stride/2;offset>0;offset>>=1) {
        if(y<offset)scratch[y][x]=__fadd_rn(scratch[y][x],scratch[y+offset][x]);
        __syncthreads();
    }
    if(y==0) {
        if(valid) {
            const float variance=__fmul_rn(scratch[0][x],factor);
            denominators[x]=__fsqrt_rn(__fadd_rn(variance,eps));
        } else denominators[x]=1.f;
    }
    __syncthreads();

    // Only after the final barrier may invalid pixels leave. They performed
    // no global load/store in either reduction and perform none here.
    if(!valid)return;
    const float denominator=denominators[x];
    #pragma unroll 1
    for(int c=y;c<Channels;c+=int(blockDim.y)) {
        const int index=base+c*hw;
        const float normalized=__fdiv_rn(__fsub_rn(input[index],mean),denominator);
        output[index]=__fadd_rn(__fmul_rn(weight[c],normalized),bias[c]);
    }
}

} // namespace

at::Tensor channel_layernorm_cuda(const at::Tensor& input,const at::Tensor& weight,
                                       const at::Tensor& bias,double eps) {
    auto output=at::empty(input.sizes(),input.options());
    const auto hw=input.size(2)*input.size(3);
    const auto pixels=input.size(0)*hw;
    const float factor=static_cast<float>(pixels)/static_cast<float>(input.numel());
    const int mean_stride=reduction_stride(int(input.size(1)),int(pixels),int(hw),
        reinterpret_cast<uintptr_t>(input.const_data_ptr()));
    const int variance_stride=reduction_stride(int(input.size(1)),int(pixels),int(hw),0);
    const int lanes=std::max(mean_stride,variance_stride);
    TORCH_CHECK((mean_stride==1 || mean_stride==4 || mean_stride==8) &&
                (variance_stride==1 || variance_stride==4 || variance_stride==8),"Unexpected reduction configuration");
    const dim3 block(32,unsigned(lanes));
    const dim3 grid(unsigned((pixels+31)/32));
    const auto stream=c10::cuda::getCurrentCUDAStream(input.get_device());
    if(input.size(1)==64)
        full_layernorm_parallel_two_pass<64><<<grid,block,0,stream>>>(input.const_data_ptr<float>(),weight.const_data_ptr<float>(),
            bias.const_data_ptr<float>(),output.data_ptr<float>(),int(pixels),int(hw),mean_stride,variance_stride,factor,float(eps));
    else
        full_layernorm_parallel_two_pass<128><<<grid,block,0,stream>>>(input.const_data_ptr<float>(),weight.const_data_ptr<float>(),
            bias.const_data_ptr<float>(),output.data_ptr<float>(),int(pixels),int(hw),mean_stride,variance_stride,factor,float(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace converse2d::peripheral
