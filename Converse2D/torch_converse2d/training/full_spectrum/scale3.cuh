#include "../../common/fp32_dispatch.h"
// Included within converse2d::full_training, after the generic CUDA helpers.
// For contiguous B,C,3,H,3,W with W>1 and a 32-bit ATen iterator, each
// output's nine aliases stay within one reduction thread. Reduce.cuh uses
// four accumulators: (0,4,8), (1,5), (2,6), (3,7), then combines them from
// left to right. A sequential sum of all nine values changes FP32 rounding.

template<class T>__device__ Z<T> scale3_combine(const Z<T>*values) {
    return add(add(add(values[0],values[1]),values[2]),values[3]);
}

template<class T>__device__ T scale3_combine_real(const T*values) {
    return add_rn(add_rn(add_rn(values[0],values[1]),values[2]),values[3]);
}

template<class T>__global__ void scale3_forward(
    const Z<T>*y,const Z<T>*p,const Z<T>*k,const T*l,
    Z<T>*out,Z<T>*q,T*d,I n,I C,I H,I W,I KB,I KC,
    T prediction_factor,T power_factor) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const I hw=H*W,bc=i/hw,batch=bc/C,c=bc%C,h=(i/W)%H,w=i%W;
    const I kbc=kc(bc,C,KB,KC),di=((KB==1?0:batch)*C+c)*hw+i%hw;
    Z<T>filters[9],priors[9];
    Z<T>prediction_sums[4]={Z<T>(0,0),Z<T>(0,0),Z<T>(0,0),Z<T>(0,0)};
    T power_sums[4]={T(0),T(0),T(0),T(0)};
    #pragma unroll
    for(int j=0;j<9;++j) {
        const I offset=(h+(j/3)*H)*(3*W)+w+(j%3)*W;
        filters[j]=k[kbc*(9*hw)+offset];
        priors[j]=p[bc*(9*hw)+offset];
        prediction_sums[j%4]=add(prediction_sums[j%4],product(filters[j],priors[j]));
        power_sums[j%4]=add_rn(power_sums[j%4],norm(filters[j]));
    }
    const Z<T>prediction_sum=scale3_combine(prediction_sums);
    const Z<T>prediction(mul_rn(prediction_sum.real(),prediction_factor),
                          mul_rn(prediction_sum.imag(),prediction_factor));
    const T km=mul_rn(scale3_combine_real(power_sums),power_factor);
    const T denominator=add_rn(km,l[c]);
    const Z<T>correction=add(y[i],-prediction)/Z<T>(denominator,0);
    q[i]=correction;
    // D has shape (KB,C,H,W), also when the kernel broadcasts channels.
    if(KB!=1||batch==0)d[di]=denominator;
    #pragma unroll
    for(int j=0;j<9;++j) {
        const I offset=(h+(j/3)*H)*(3*W)+w+(j%3)*W;
        out[bc*(9*hw)+offset]=add(priors[j],product(cj(filters[j]),correction));
    }
}

template<class T>__global__ void scale3_adjoint(
    const Z<T>*g,const Z<T>*p,const Z<T>*k,const Z<T>*q,const T*d,
    Z<T>*gy,Z<T>*gp,Z<T>*direct,Z<T>*prediction,Z<T>*gd,
    I n,I C,I H,I W,I KB,I KC,T inverse_aliases) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const I hw=H*W,bc=i/hw,h=(i/W)%H,w=i%W,kbc=kc(bc,C,KB,KC);
    const I di=((KB==1?0:bc/C)*C+bc%C)*hw+i%hw;
    Z<T>filters[9],grads[9];
    Z<T>adjoint_sums[4]={Z<T>(0,0),Z<T>(0,0),Z<T>(0,0),Z<T>(0,0)};
    #pragma unroll
    for(int j=0;j<9;++j) {
        const I offset=(h+(j/3)*H)*(3*W)+w+(j%3)*W;
        filters[j]=k[kbc*(9*hw)+offset];
        grads[j]=g[bc*(9*hw)+offset];
        adjoint_sums[j%4]=add(adjoint_sums[j%4],product(grads[j],filters[j]));
    }
    const Z<T>t=scale3_combine(adjoint_sums),den(d[di],0);
    Z<T>gyi;
    if(gy||gp||prediction)gyi=t/den;
    if(gy)gy[i]=gyi;
    // Keep complex storage so real(gd) has the original stride-2 layout at
    // every subsequent batch/channel and regularizer reduction boundary.
    if(gd)gd[i]=product(-t,cj(q[i]/den));
    if(gp||direct||prediction) {
        // This is ATen's complex CPU-scalar division by nine. Do not replace
        // it with componentwise real scaling or the forward mean's factor.
        const Z<T>gm=product(-gyi,Z<T>(inverse_aliases,T(0)));
        #pragma unroll
        for(int j=0;j<9;++j) {
            const I offset=(h+(j/3)*H)*(3*W)+w+(j%3)*W,index=bc*(9*hw)+offset;
            if(direct)direct[index]=product(grads[j],cj(q[i]));
            if(gp)gp[index]=add(grads[j],product(gm,cj(filters[j])));
            if(prediction)prediction[index]=product(gm,cj(p[index]));
        }
    }
}

std::vector<Tensor> full_scale3_forward_cuda(Tensor y0,Tensor p0,Tensor k0,Tensor l0) {
    auto y=plain(y0),p=plain(p0),k=plain(k0),l=plain(l0);
    auto out=at::empty(p.sizes(),p.options()),q=at::empty(y.sizes(),y.options());
    auto d=at::empty({k.size(0),y.size(1),y.size(2),y.size(3)},l.options());
    // Match MeanOps' host-side float(num_output_elements) / numel exactly.
    // Counts above 2**24 can make this differ from float(1/9), and the
    // prediction and power iterators have different counts with broadcasting.
    const float prediction_factor=static_cast<float>(y.numel())/p.numel();
    const float power_factor=static_cast<float>(k.numel()/9)/k.numel();
    auto stream=c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(l.scalar_type(),"full_scale3_forward",[&]{
        scale3_forward<scalar_t><<<(y.numel()+255)/256,256,0,stream>>>(
            y.data_ptr<Z<scalar_t>>(),p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),l.data_ptr<scalar_t>(),
            out.data_ptr<Z<scalar_t>>(),q.data_ptr<Z<scalar_t>>(),d.data_ptr<scalar_t>(),
            y.numel(),y.size(1),y.size(2),y.size(3),k.size(0),k.size(1),prediction_factor,power_factor);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,q,d};
}

std::vector<Tensor> full_scale3_adjoint_cuda(Tensor g0,Tensor p0,Tensor k0,Tensor q0,Tensor d0,bool need_y,bool need_p,bool need_k,bool need_l) {
    auto g=plain(g0),p=need_k?plain(p0):Tensor(),k=plain(k0),q=(need_k||need_l)?plain(q0):Tensor(),d=plain(d0);
    auto gy=need_y?at::empty(q0.sizes(),q0.options()):Tensor(),gp=need_p?at::empty(p0.sizes(),p0.options()):Tensor();
    auto direct=need_k?at::empty(g.sizes(),g.options()):Tensor(),prediction=need_k?at::empty(p0.sizes(),p0.options()):Tensor();
    auto gd=(need_k||need_l)?at::empty(q0.sizes(),q0.options()):Tensor();
    const float inverse_aliases=1.0f/9.0f;
    auto stream=c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(d.scalar_type(),"full_scale3_adjoint",[&]{
        scale3_adjoint<scalar_t><<<(q0.numel()+255)/256,256,0,stream>>>(
            g.data_ptr<Z<scalar_t>>(),optional_data<Z<scalar_t>>(p),k.data_ptr<Z<scalar_t>>(),optional_data<Z<scalar_t>>(q),d.data_ptr<scalar_t>(),
            optional_data<Z<scalar_t>>(gy),optional_data<Z<scalar_t>>(gp),optional_data<Z<scalar_t>>(direct),optional_data<Z<scalar_t>>(prediction),optional_data<Z<scalar_t>>(gd),
            q0.numel(),q0.size(1),q0.size(2),q0.size(3),k.size(0),k.size(1),inverse_aliases);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gy,gp,direct,prediction,gd};
}
