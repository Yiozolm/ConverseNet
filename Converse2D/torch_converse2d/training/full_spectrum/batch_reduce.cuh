#include "../../common/fp32_dispatch.h"
// Included inside converse2d::full_training, after the generic CUDA helpers.
// Only s1/B2 or B4/KB1/KC=C with an input VJP and a kernel VJP may use this.

template<class T>__global__ void scale1_batch_prepare(
    const Z<T>*g,const Z<T>*p,const Z<T>*k,const Z<T>*y,const T*d,Z<T>*gy,Z<T>*gd,I n,I m) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const Z<T>t=product(g[i],k[i%m]),den(d[i%m],0);
    // These are exactly the original scale1_adjoint expressions. The gy
    // output may temporarily be gp; the second launch then completes gp.
    gy[i]=t/den;
    const Z<T> qi=scale1_recompute_q(y[i],p[i],k[i%m],d[i%m]);
    gd[i]=product(-t,cj(qi/den));
}

template<class T,int Batch>__global__ void scale1_batch_kernel(
    const Z<T>*g,const Z<T>*p,const Z<T>*k,const Z<T>*y,const T*d,
    const Z<T>*gy,Z<T>*gp,const T*power,Z<T>*out,I m,bool shared) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=m)return;
    const Z<T>zero(0,0),ki=k[i];
    Z<T>direct[4]={zero,zero,zero,zero};
    Z<T>prediction[4]={zero,zero,zero,zero};
    #pragma unroll
    for(int b=0;b<Batch;++b) {
        const I j=I(b)*m+i;
        const Z<T>gi=g[j],gyi=gy[j],gm=-gyi;
        // sum_to's ATen reduction is non-fastest-axis, with four independent
        // accumulators initialized to +0. Do not combine direct and prediction
        // before their reductions, and do not conjugate before summing direct.
        const Z<T> qi=scale1_recompute_q(y[j],p[j],ki,d[i]);
        direct[b]=add(zero,product(gi,cj(qi)));
        prediction[b]=add(zero,product(gm,cj(p[j])));
        if(gp) {
            // A unique (C,H,W) owner consumes every batch's intermediate gy
            // before writing that element's final gp, even when gy==gp.
            gp[j]=add(shared?add(gi,-gm):gi,product(gm,cj(ki)));
        }
    }
    // B2 must also execute the trailing +0,+0 combinations; omitting them
    // changes signed-zero behavior relative to Reduce.cuh::thread_reduce_impl.
    const Z<T>sum_direct=add(add(add(direct[0],direct[1]),direct[2]),direct[3]);
    const Z<T>sum_prediction=add(add(add(prediction[0],prediction[1]),prediction[2]),prediction[3]);
    const T v=power[i];
    const Z<T>power_value(mul_rn(v,mul_rn(T(2),ki.real())),
                           mul_rn(v,mul_rn(T(2),ki.imag())));
    out[i]=add(add(add(cj(sum_direct),sum_prediction),Z<T>(0,power_value.imag())),
               Z<T>(power_value.real(),0));
}

Scale1BatchAdjoint full_scale1_batch_prepare_cuda(
    Tensor g0,Tensor p0,Tensor k0,Tensor y0,Tensor d0,bool need_independent_y,bool need_prior) {
    TORCH_CHECK(need_independent_y||need_prior,"batch reduction needs a reusable input-gradient output");
    Scale1BatchAdjoint stage;
    // Materialize once, and retain all four read-only inputs until stage two.
    stage.g=plain(g0);stage.p=plain(p0);stage.k=plain(k0);
    stage.y=y0.is_same(p0)?stage.p:plain(y0);stage.d=plain(d0);
    auto d=stage.d;
    if(need_independent_y)stage.gy=at::empty(stage.g.sizes(),stage.g.options());
    if(need_prior)stage.gp=at::empty(stage.p.sizes(),stage.p.options());
    stage.intermediate_gy=stage.gy.defined()?stage.gy:stage.gp;
    // Keep the complex allocation: host real(gd) must have stride two for
    // precisely the same B reduction and subsequent regularizer reduction.
    stage.gd=at::empty(stage.g.sizes(),stage.g.options());
    auto stream=c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(d.scalar_type(),"full_scale1_batch_prepare",[&]{
        scale1_batch_prepare<scalar_t><<<(stage.g.numel()+255)/256,256,0,stream>>>(
            stage.g.data_ptr<Z<scalar_t>>(),stage.p.data_ptr<Z<scalar_t>>(),stage.k.data_ptr<Z<scalar_t>>(),stage.y.data_ptr<Z<scalar_t>>(),
            d.data_ptr<scalar_t>(),stage.intermediate_gy.data_ptr<Z<scalar_t>>(),stage.gd.data_ptr<Z<scalar_t>>(),
            stage.g.numel(),stage.k.numel());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stage;
}

Tensor full_scale1_batch_kernel_cuda(const Scale1BatchAdjoint& stage,Tensor power0,bool shared) {
    TORCH_CHECK(stage.g.size(0)==2||stage.g.size(0)==4,"batch reduction supports B2/B4 only");
    auto power=plain(power0),out=at::empty(stage.k.sizes(),stage.k.options());
    auto stream=c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(power.scalar_type(),"full_scale1_batch_kernel",[&]{
        auto launch=[&](auto tag) {
            constexpr int Batch=decltype(tag)::value;
            scale1_batch_kernel<scalar_t,Batch><<<(stage.k.numel()+255)/256,256,0,stream>>>(
                stage.g.data_ptr<Z<scalar_t>>(),stage.p.data_ptr<Z<scalar_t>>(),stage.k.data_ptr<Z<scalar_t>>(),
                stage.y.data_ptr<Z<scalar_t>>(),stage.d.data_ptr<scalar_t>(),stage.intermediate_gy.data_ptr<Z<scalar_t>>(),
                optional_data<Z<scalar_t>>(stage.gp),power.data_ptr<scalar_t>(),out.data_ptr<Z<scalar_t>>(),
                stage.k.numel(),shared);
        };
        if(stage.g.size(0)==2)launch(std::integral_constant<int,2>{});
        else launch(std::integral_constant<int,4>{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
