#include "../../common/fp32_dispatch.h"
// Included within converse2d::full_training, after the generic CUDA helpers.
// Preserve the generic path's explicit FP32/FP64 rounding boundaries. s1 has
// no alias reductions, so only its pointwise stages are fused here.
template<class T>__global__ void scale1_forward(
    const Z<T>*y,const Z<T>*p,const Z<T>*k,const T*l,
    Z<T>*out,T*d,I n,I C,I HW,I KB,I KC) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const I bc=i/HW,b=bc/C,c=bc%C,offset=i%HW;
    const I ki=kc(bc,C,KB,KC)*HW+offset;
    const I di=((KB==1?0:b)*C+c)*HW+offset;
    const Z<T>ki_value=k[ki],pi_value=p[i];
    const Z<T>pm=product(ki_value,pi_value);
    const T denominator=add_rn(norm(ki_value),l[c]);
    const Z<T>qi_value=add(y[i],-pm)/Z<T>(denominator,0);
    out[i]=add(pi_value,product(cj(ki_value),qi_value));
    // D is (KB,C,H,W), including broadcast KC==1. Give each element exactly
    // one writer even when all batches compute the same denominator.
    if(KB!=1||b==0)d[di]=denominator;
}

template<class T,bool FuseKernel>__global__ void scale1_adjoint(
    const Z<T>*g,const Z<T>*p,const Z<T>*k,const Z<T>*y,const T*d,
    Z<T>*gy,Z<T>*gp,Z<T>*direct,Z<T>*prediction,Z<T>*gd,Z<T>*gk,
    I n,I C,I HW,I KB,I KC,bool shared) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const I bc=i/HW,offset=i%HW;
    const I ki=kc(bc,C,KB,KC)*HW+offset;
    const I di=((KB==1?0:bc/C)*C+bc%C)*HW+offset;
    const Z<T>gi=g[i],ki_value=k[ki],den(d[di],0);
    const Z<T>t=product(gi,ki_value);
    Z<T>gyi,gm;
    if(gy||gp||direct||FuseKernel){gyi=t/den;gm=-gyi;}
    // Shared inputs return gp as their combined gradient; keep gyi only in a
    // register and omit the unused independent-y output allocation/store.
    if(gy)gy[i]=gyi;
    // Keep complex storage: at::real(gd) must retain its stride-2 layout for
    // precisely the same broadcast and lambda reduction behavior as before.
    Z<T>qi,gd_value;
    if(gd||direct||FuseKernel)qi=scale1_recompute_q(y[i],p[i],ki_value,d[di]);
    if(gd||FuseKernel){gd_value=product(-t,cj(qi/den));if(gd)gd[i]=gd_value;}
    if(gp)gp[i]=add(shared?add(gi,-gm):gi,product(gm,cj(ki_value)));
    if constexpr(FuseKernel) {
        // KB==B && KC==C: sum_to has no reduced dimensions. Match the
        // separate adj_kernel's conj, two mul_rn and nested-add boundaries.
        const T v=gd_value.real();
        const Z<T>direct_value=product(gi,cj(qi));
        const Z<T>prediction_value=product(gm,cj(p[i]));
        const Z<T>power_value(
            mul_rn(v,mul_rn(T(2),ki_value.real())),
            mul_rn(v,mul_rn(T(2),ki_value.imag())));
        gk[i]=add(add(add(cj(direct_value),prediction_value),
            Z<T>(0,power_value.imag())),Z<T>(power_value.real(),0));
    } else {
        // Do not conjugate direct before the host-side broadcast reduction.
        if(direct)direct[i]=product(gi,cj(qi));
        if(prediction)prediction[i]=product(gm,cj(p[i]));
    }
}

std::vector<Tensor> full_scale1_forward_cuda(Tensor y0,Tensor p0,Tensor k0,Tensor l0) {
    // Tensor identity is tested before resolving lazy flags/layout. The same
    // read-only spectrum then needs at most one materialization per call.
    auto y=plain(y0),p=y0.is_same(p0)?y:plain(p0),k=plain(k0),l=plain(l0);
    auto out=at::empty(p.sizes(),p.options());
    auto d=at::empty({k.size(0),y.size(1),y.size(2),y.size(3)},l.options());
    auto stream=c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(l.scalar_type(),"full_scale1_forward",[&]{
        scale1_forward<scalar_t><<<(y.numel()+255)/256,256,0,stream>>>(
            y.data_ptr<Z<scalar_t>>(),p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),l.data_ptr<scalar_t>(),
            out.data_ptr<Z<scalar_t>>(),d.data_ptr<scalar_t>(),
            y.numel(),y.size(1),y.size(2)*y.size(3),k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,Tensor(),d};
}

std::vector<Tensor> full_scale1_adjoint_cuda(Tensor g0,Tensor p0,Tensor k0,Tensor y0,Tensor d0,bool shared,bool need_y,bool need_p,bool need_k,bool need_l) {
    auto g=plain(g0),p=(need_k||need_l)?plain(p0):Tensor(),k=plain(k0),
        y=(need_k||need_l)?(y0.is_same(p0)?p:plain(y0)):Tensor(),d=plain(d0);
    const bool fuse_kernel=need_k&&k.size(0)==g.size(0)&&k.size(1)==g.size(1);
    auto gy=need_y?at::empty(g.sizes(),g.options()):Tensor(),gp=need_p?at::empty(p0.sizes(),p0.options()):Tensor();
    auto direct=need_k&&!fuse_kernel?at::empty(g.sizes(),g.options()):Tensor();
    auto prediction=need_k&&!fuse_kernel?at::empty(p0.sizes(),p0.options()):Tensor();
    auto gd=(need_l||(need_k&&!fuse_kernel))?at::empty(g.sizes(),g.options()):Tensor();
    auto gk=fuse_kernel?at::empty(k.sizes(),k.options()):Tensor();
    auto stream=c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(d.scalar_type(),"full_scale1_adjoint",[&]{
        auto launch=[&](auto tag) {
            constexpr bool FuseKernel=decltype(tag)::value;
            scale1_adjoint<scalar_t,FuseKernel><<<(g.numel()+255)/256,256,0,stream>>>(
                g.data_ptr<Z<scalar_t>>(),optional_data<Z<scalar_t>>(p),k.data_ptr<Z<scalar_t>>(),optional_data<Z<scalar_t>>(y),d.data_ptr<scalar_t>(),
                optional_data<Z<scalar_t>>(gy),optional_data<Z<scalar_t>>(gp),
                optional_data<Z<scalar_t>>(direct),optional_data<Z<scalar_t>>(prediction),
                optional_data<Z<scalar_t>>(gd),optional_data<Z<scalar_t>>(gk),
                g.numel(),g.size(1),g.size(2)*g.size(3),k.size(0),k.size(1),shared);
        };
        if(fuse_kernel)launch(std::true_type{});else launch(std::false_type{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gy,gp,direct,prediction,gd,gk};
}
