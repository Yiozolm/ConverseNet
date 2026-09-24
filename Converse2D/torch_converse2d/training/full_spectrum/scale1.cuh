// Included within converse2d::full_training, after the generic CUDA helpers.
// Preserve the generic path's explicit FP32/FP64 rounding boundaries. s1 has
// no alias reductions, so only its pointwise stages are fused here.
template<class T>__global__ void scale1_forward(
    const Z<T>*y,const Z<T>*p,const Z<T>*k,const T*l,
    Z<T>*out,Z<T>*q,T*d,I n,I C,I HW,I KB,I KC) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const I bc=i/HW,b=bc/C,c=bc%C,offset=i%HW;
    const I ki=kc(bc,C,KB,KC)*HW+offset;
    const I di=((KB==1?0:b)*C+c)*HW+offset;
    const Z<T>ki_value=k[ki],pi_value=p[i];
    const Z<T>pm=product(ki_value,pi_value);
    const T denominator=add_rn(norm(ki_value),l[c]);
    const Z<T>qi_value=add(y[i],-pm)/Z<T>(denominator,0);
    q[i]=qi_value;
    out[i]=add(pi_value,product(cj(ki_value),qi_value));
    // D is (KB,C,H,W), including broadcast KC==1. Give each element exactly
    // one writer even when all batches compute the same denominator.
    if(KB!=1||b==0)d[di]=denominator;
}

template<class T,bool FuseKernel>__global__ void scale1_adjoint(
    const Z<T>*g,const Z<T>*p,const Z<T>*k,const Z<T>*q,const T*d,
    Z<T>*gy,Z<T>*gp,Z<T>*direct,Z<T>*prediction,Z<T>*gd,Z<T>*gk,
    I n,I C,I HW,I KB,I KC,bool shared) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const I bc=i/HW,offset=i%HW;
    const I ki=kc(bc,C,KB,KC)*HW+offset;
    const I di=((KB==1?0:bc/C)*C+bc%C)*HW+offset;
    const Z<T>gi=g[i],ki_value=k[ki],qi=q[i],den(d[di],0);
    const Z<T>t=product(gi,ki_value);
    const Z<T>gyi=t/den;
    const Z<T>gm=-gyi;
    // Shared inputs return gp as their combined gradient; keep gyi only in a
    // register and omit the unused independent-y output allocation/store.
    if(!shared)gy[i]=gyi;
    // Keep complex storage: at::real(gd) must retain its stride-2 layout for
    // precisely the same broadcast and lambda reduction behavior as before.
    const Z<T>gd_value=product(-t,cj(qi/den));
    gd[i]=gd_value;
    gp[i]=add(shared?add(gi,-gm):gi,product(gm,cj(ki_value)));
    const Z<T>direct_value=product(gi,cj(qi));
    const Z<T>prediction_value=product(gm,cj(p[i]));
    if constexpr(FuseKernel) {
        // KB==B && KC==C: sum_to has no reduced dimensions. Match the
        // separate adj_kernel's conj, two mul_rn and nested-add boundaries.
        const T v=gd_value.real();
        const Z<T>power_value(
            mul_rn(v,mul_rn(T(2),ki_value.real())),
            mul_rn(v,mul_rn(T(2),ki_value.imag())));
        gk[i]=add(add(add(cj(direct_value),prediction_value),
            Z<T>(0,power_value.imag())),Z<T>(power_value.real(),0));
    } else {
        // Do not conjugate direct before the host-side broadcast reduction.
        direct[i]=direct_value;
        prediction[i]=prediction_value;
    }
}

std::vector<Tensor> full_scale1_forward_cuda(Tensor y0,Tensor p0,Tensor k0,Tensor l0) {
    // Tensor identity is tested before resolving lazy flags/layout. The same
    // read-only spectrum then needs at most one materialization per call.
    auto y=plain(y0),p=y0.is_same(p0)?y:plain(p0),k=plain(k0),l=plain(l0);
    auto out=at::empty(p.sizes(),p.options()),q=at::empty(y.sizes(),y.options());
    auto d=at::empty({k.size(0),y.size(1),y.size(2),y.size(3)},l.options());
    auto stream=c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"full_scale1_forward",[&]{
        scale1_forward<scalar_t><<<(y.numel()+255)/256,256,0,stream>>>(
            y.data_ptr<Z<scalar_t>>(),p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),l.data_ptr<scalar_t>(),
            out.data_ptr<Z<scalar_t>>(),q.data_ptr<Z<scalar_t>>(),d.data_ptr<scalar_t>(),
            y.numel(),y.size(1),y.size(2)*y.size(3),k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,q,d};
}

std::vector<Tensor> full_scale1_adjoint_cuda(Tensor g0,Tensor p0,Tensor k0,Tensor q0,Tensor d0,bool shared) {
    auto g=plain(g0),p=plain(p0),k=plain(k0),q=plain(q0),d=plain(d0);
    const bool fuse_kernel=k.size(0)==g.size(0)&&k.size(1)==g.size(1);
    auto gy=shared?Tensor():at::empty(g.sizes(),g.options()),gp=at::empty(p.sizes(),p.options());
    auto direct=fuse_kernel?Tensor():at::empty(g.sizes(),g.options());
    auto prediction=fuse_kernel?Tensor():at::empty(p.sizes(),p.options());
    auto gd=at::empty(g.sizes(),g.options());
    auto gk=fuse_kernel?at::empty(k.sizes(),k.options()):Tensor();
    auto stream=c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"full_scale1_adjoint",[&]{
        auto launch=[&](auto tag) {
            constexpr bool FuseKernel=decltype(tag)::value;
            scale1_adjoint<scalar_t,FuseKernel><<<(g.numel()+255)/256,256,0,stream>>>(
                g.data_ptr<Z<scalar_t>>(),p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),q.data_ptr<Z<scalar_t>>(),d.data_ptr<scalar_t>(),
                shared?nullptr:gy.data_ptr<Z<scalar_t>>(),gp.data_ptr<Z<scalar_t>>(),
                FuseKernel?nullptr:direct.data_ptr<Z<scalar_t>>(),FuseKernel?nullptr:prediction.data_ptr<Z<scalar_t>>(),
                gd.data_ptr<Z<scalar_t>>(),FuseKernel?gk.data_ptr<Z<scalar_t>>():nullptr,
                g.numel(),g.size(1),g.size(2)*g.size(3),k.size(0),k.size(1),shared);
        };
        if(fuse_kernel)launch(std::true_type{});else launch(std::false_type{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gy,gp,direct,prediction,gd,gk};
}
