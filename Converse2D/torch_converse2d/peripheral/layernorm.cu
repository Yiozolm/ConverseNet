#include "layernorm.h"
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>
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
    float* mean_out,float* denominator_out,
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
            const float denominator=__fsqrt_rn(__fadd_rn(variance,eps));
            denominators[x]=denominator;
            // Training keeps the per-pixel statistics for the fused backward.
            if(denominator_out) {
                mean_out[pixel]=mean;
                denominator_out[pixel]=denominator;
            }
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

// Fused first-order VJP. One thread owns one pixel and walks its channels, so
// a warp reads 32 adjacent pixels of each channel plane in one transaction.
// dx = (1/sigma) * (g*w - mean_c(g*w) - xhat * mean_c(g*w*xhat)); the channel
// vectors dweight = sum(g*xhat) and dbias = sum(g) are reduced across the
// block in a fixed order (xor-shuffle tree, then warps in index order) and
// handed back as per-block partials so the host finishes with one ATen sum.
constexpr int kBackwardThreads=128;
constexpr int kBackwardWarps=kBackwardThreads/32;

template<int Channels>
__global__ void __launch_bounds__(kBackwardThreads) full_layernorm_backward(
    const float* __restrict__ grad,const float* __restrict__ input,
    const float* __restrict__ mean,const float* __restrict__ denominator,
    const float* __restrict__ weight,float* __restrict__ grad_input,
    float* __restrict__ partials,int pixels,int hw,bool needs_input,bool needs_params) {
    __shared__ float w[Channels];
    __shared__ float sums[kBackwardWarps][2][Channels];
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    for(int c=tid;c<Channels;c+=kBackwardThreads)w[c]=weight[c];
    __syncthreads();
    const int pixel=int(blockIdx.x)*kBackwardThreads+tid;
    const bool valid=pixel<pixels;
    const int base=valid ? (pixel/hw)*Channels*hw+pixel%hw : 0;
    const float m=valid ? mean[pixel] : 0.f;
    const float r=valid ? __fdiv_rn(1.f,denominator[pixel]) : 0.f;
    float sum_g=0.f,sum_gx=0.f;
    #pragma unroll 4
    for(int c=0;c<Channels;++c) {
        float g=0.f,xhat=0.f;
        if(valid) {
            const int i=base+c*hw;
            g=grad[i];
            xhat=__fmul_rn(__fsub_rn(input[i],m),r);
        }
        const float gw=__fmul_rn(g,w[c]);
        sum_g=__fadd_rn(sum_g,gw);
        sum_gx=__fadd_rn(sum_gx,__fmul_rn(gw,xhat));
        if(needs_params) {
            float pw=__fmul_rn(g,xhat),pb=g;
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1) {
                pw=__fadd_rn(pw,__shfl_xor_sync(0xffffffffu,pw,offset));
                pb=__fadd_rn(pb,__shfl_xor_sync(0xffffffffu,pb,offset));
            }
            if(lane==0) {
                sums[warp][0][c]=pw;
                sums[warp][1][c]=pb;
            }
        }
    }
    if(needs_input && valid) {
        constexpr float factor=1.f/Channels;
        const float a=__fmul_rn(sum_g,factor),b=__fmul_rn(sum_gx,factor);
        #pragma unroll 4
        for(int c=0;c<Channels;++c) {
            const int i=base+c*hw;
            const float xhat=__fmul_rn(__fsub_rn(input[i],m),r);
            const float gw=__fmul_rn(grad[i],w[c]);
            grad_input[i]=__fmul_rn(r,__fsub_rn(__fsub_rn(gw,a),__fmul_rn(xhat,b)));
        }
    }
    if(needs_params) {
        __syncthreads();
        for(int t=tid;t<2*Channels;t+=kBackwardThreads) {
            const int k=t/Channels,c=t%Channels;
            float s=sums[0][k][c];
            #pragma unroll
            for(int i=1;i<kBackwardWarps;++i)s=__fadd_rn(s,sums[i][k][c]);
            partials[(size_t(blockIdx.x)*2+k)*Channels+c]=s;
        }
    }
}

} // namespace

at::Tensor channel_layernorm_cuda(const at::Tensor& input,const at::Tensor& weight,
                                       const at::Tensor& bias,double eps,
                                       at::Tensor* mean_out,at::Tensor* denominator_out) {
    auto output=at::empty(input.sizes(),input.options());
    const auto hw=input.size(2)*input.size(3);
    const auto pixels=input.size(0)*hw;
    float* mean_pointer=nullptr;
    float* denominator_pointer=nullptr;
    if(mean_out && denominator_out) {
        *mean_out=at::empty({pixels},input.options());
        *denominator_out=at::empty({pixels},input.options());
        mean_pointer=mean_out->data_ptr<float>();
        denominator_pointer=denominator_out->data_ptr<float>();
    }
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
            bias.const_data_ptr<float>(),output.data_ptr<float>(),mean_pointer,denominator_pointer,
            int(pixels),int(hw),mean_stride,variance_stride,factor,float(eps));
    else
        full_layernorm_parallel_two_pass<128><<<grid,block,0,stream>>>(input.const_data_ptr<float>(),weight.const_data_ptr<float>(),
            bias.const_data_ptr<float>(),output.data_ptr<float>(),mean_pointer,denominator_pointer,
            int(pixels),int(hw),mean_stride,variance_stride,factor,float(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> channel_layernorm_backward_cuda(
    const at::Tensor& grad_output,const at::Tensor& input,const at::Tensor& mean,
    const at::Tensor& denominator,const at::Tensor& weight,
    bool needs_input,bool needs_weight,bool needs_bias) {
    const auto channels=input.size(1);
    const auto hw=input.size(2)*input.size(3);
    const auto pixels=input.size(0)*hw;
    const bool needs_params=needs_weight || needs_bias;
    at::Tensor grad_input,grad_weight,grad_bias,partials;
    if(needs_input)grad_input=at::empty(input.sizes(),input.options());
    const int blocks=int((pixels+kBackwardThreads-1)/kBackwardThreads);
    if(needs_params)partials=at::empty({blocks,2,channels},input.options());
    const auto stream=c10::cuda::getCurrentCUDAStream(input.get_device());
    const auto launch=[&](auto tag) {
        constexpr int C=decltype(tag)::value;
        full_layernorm_backward<C><<<blocks,kBackwardThreads,0,stream>>>(
            grad_output.const_data_ptr<float>(),input.const_data_ptr<float>(),
            mean.const_data_ptr<float>(),denominator.const_data_ptr<float>(),
            weight.const_data_ptr<float>(),needs_input ? grad_input.data_ptr<float>() : nullptr,
            needs_params ? partials.data_ptr<float>() : nullptr,
            int(pixels),int(hw),needs_input,needs_params);
    };
    if(channels==64)launch(std::integral_constant<int,64>{});
    else launch(std::integral_constant<int,128>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(needs_params) {
        // Fixed-order ATen reduction over the per-block partials; separate
        // storage so neither accumulated parameter gradient aliases the other.
        if(needs_weight) {
            grad_weight=at::empty({channels},input.options());
            at::sum_out(grad_weight,partials.select(1,0),{0});
        }
        if(needs_bias) {
            grad_bias=at::empty({channels},input.options());
            at::sum_out(grad_bias,partials.select(1,1),{0});
        }
    }
    return {grad_input,grad_weight,grad_bias};
}

} // namespace converse2d::peripheral
