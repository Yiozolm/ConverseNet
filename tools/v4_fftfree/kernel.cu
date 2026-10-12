#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>

namespace {
using I = int64_t;

__global__ void nearest_k2(const float* x,const float* weight,const float* denominator,
                          float* output,I n,I channels,I height,I width,I kb,I kc) {
    const I step=I(blockDim.x)*gridDim.x;
    for(I i=I(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=step) {
        const I hw=height*width,bc=i/hw,b=bc/channels,c=bc%channels;
        const I kbc=(kb==1?0:b)*kc+(kc==1?0:c);
        const float a=weight[4*kbc],b01=weight[4*kbc+1];
        const float b10=weight[4*kbc+2],b11=weight[4*kbc+3],xi=x[i];
        // Match disjoint_k2_s2's four independent multiply/add operations.
        // Summing weights first or contracting to an FMA changes FP32 bits.
        float predicted=__fadd_rn(0.0f,__fmul_rn(a,xi));
        predicted=__fadd_rn(predicted,__fmul_rn(b01,xi));
        predicted=__fadd_rn(predicted,__fmul_rn(b10,xi));
        predicted=__fadd_rn(predicted,__fmul_rn(b11,xi));
        const float d=denominator[(kb==1?0:b)*channels+c];
        const float residual=__fdiv_rn(__fsub_rn(xi,predicted),d);
        const I h=(i/width)%height,w=i%width;
        const I base=bc*(4*hw)+(2*h)*(2*width)+2*w;
        // roll(-1,-1) and phase-zero sampling assign coefficient (a,c) to
        // output phase (1-a,1-c). Every output has exactly one writer.
        output[base+2*width+1]=__fadd_rn(xi,__fmul_rn(a,residual));
        output[base+2*width]=__fadd_rn(xi,__fmul_rn(b01,residual));
        output[base+1]=__fadd_rn(xi,__fmul_rn(b10,residual));
        output[base]=__fadd_rn(xi,__fmul_rn(b11,residual));
    }
}
}

at::Tensor nearest_k2_cuda(at::Tensor x0,at::Tensor weight0,at::Tensor denominator0) {
    TORCH_CHECK(!at::GradMode::is_enabled(),"FFT-free nearest is inference-only: use no_grad or inference_mode");
    TORCH_CHECK(x0.is_cuda()&&weight0.is_cuda()&&denominator0.is_cuda(),"expected CUDA tensors");
    TORCH_CHECK(x0.scalar_type()==at::kFloat&&weight0.scalar_type()==at::kFloat&&denominator0.scalar_type()==at::kFloat,"FP32 tensors required");
    TORCH_CHECK(x0.device()==weight0.device()&&x0.device()==denominator0.device(),"device mismatch");
    TORCH_CHECK(x0.dim()==4&&x0.numel()>0,"expected nonempty NCHW input");
    TORCH_CHECK(x0.numel()<=INT64_MAX/4&&x0.size(2)<=INT64_MAX/2&&x0.size(3)<=INT64_MAX/2,"output dimensions overflow");
    TORCH_CHECK(weight0.dim()==4&&weight0.size(2)==2&&weight0.size(3)==2,"only k2/s2 is supported");
    TORCH_CHECK((weight0.size(0)==1||weight0.size(0)==x0.size(0))&&(weight0.size(1)==1||weight0.size(1)==x0.size(1)),"invalid kernel broadcast");
    TORCH_CHECK(denominator0.sizes()==at::IntArrayRef({weight0.size(0),x0.size(1),1,1}),"invalid denominator shape");
    const c10::cuda::CUDAGuard guard(x0.device());
    auto x=x0.resolve_neg().contiguous(),weight=weight0.resolve_neg().contiguous();
    auto denominator=denominator0.resolve_neg().contiguous();
    auto output=at::empty({x.size(0),x.size(1),2*x.size(2),2*x.size(3)},x.options());
    const int blocks=static_cast<int>(std::min<I>((x.numel()+255)/256,65535));
    nearest_k2<<<blocks,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),weight.data_ptr<float>(),denominator.data_ptr<float>(),output.data_ptr<float>(),
        x.numel(),x.size(1),x.size(2),x.size(3),weight.size(0),weight.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
