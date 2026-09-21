// Isolated experiment derived from production converse2d_training.cu.
// No HR nearest-prior or prior-VJP buffer, floating atomics or fast math.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <vector>

namespace {
template<class T> using Z = c10::complex<T>;
using I = int64_t;
template<class T> __device__ Z<T> cj(Z<T> z) { return {z.real(),-z.imag()}; }
template<class T> __device__ T norm2(Z<T> z);
template<> __device__ float norm2(Z<float> z) {
    return __fadd_rn(__fmul_rn(z.real(),z.real()),__fmul_rn(z.imag(),z.imag()));
}
template<> __device__ double norm2(Z<double> z) {
    return __dadd_rn(__dmul_rn(z.real(),z.real()),__dmul_rn(z.imag(),z.imag()));
}
template<class T> __device__ Z<T> read(const Z<T>* p,I bc,I h,I w,I H,I W) {
    const bool mirror=w>W/2;
    if(mirror) { h=(H-h)%H; w=W-w; }
    auto z=p[(bc*H+h)*(W/2+1)+w];
    return mirror?cj(z):z;
}
__device__ I filter_channel(I bc,I C,I KB,I KC) {
    return (KB==1?0:bc/C)*KC+(KC==1?0:bc%C);
}

// Reflect missing HR columns BEFORE evaluating the stored nearest prior.
// This matches full_spectrum(N(y)) for arbitrary complex boundary values.
template<class T> __device__ Z<T> prior_read(const Z<T>* y,const Z<T>* ph,const Z<T>* pw,
    I bc,I h,I w,I H,I W,I s) {
    const I hs=H*s,ws=W*s;
    const bool mirror=w>ws/2;
    if(mirror) { h=(hs-h)%hs; w=ws-w; }
    const auto value=read(y,bc,h%H,w%W,H,W)*(ph[h]*pw[w]);
    return mirror?cj(value):value;
}

template<class T>
__global__ void nearest_solve_alias(const Z<T>* y,const Z<T>* k,const T* lambda,
    const Z<T>* ph,const Z<T>* pw,Z<T>* q,T* d,I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I kc=filter_channel(bc,C,KB,KC);
    Z<T> prediction(0,0);
    T power=0;
    for(I a=0;a<s;++a) for(I b=0;b<s;++b) {
        const I hh=h+a*H,ww=w+b*W;
        auto fk=read(k,kc,hh,ww,H*s,W*s);
        prediction+=fk*prior_read(y,ph,pw,bc,hh,ww,H,W,s);
        power+=norm2(fk);
    }
    const T count=T(s)*T(s);
    d[i]=power/count+lambda[bc%C];
    q[i]=(y[i]-prediction/count)/d[i];
}

template<class T>
__global__ void nearest_solve_output(const Z<T>* y,const Z<T>* k,const Z<T>* q,
    const Z<T>* ph,const Z<T>* pw,Z<T>* out,I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I hs=H*s,sw=W*s/2+1,w=i%sw,h=(i/sw)%hs,bc=i/(hs*sw);
    const I kc=filter_channel(bc,C,KB,KC);
    out[i]=prior_read(y,ph,pw,bc,h,w,H,W,s)+cj(k[(kc*hs+h)*sw+w])*read(q,bc,h%H,w%W,H,W);
}

// Unchanged production expansion VJP, including LR DC/Nyquist handling.
template<class T>
__global__ void nearest_adjoint_q(const Z<T>* g,const Z<T>* k,const Z<T>* q,const T* d,
    Z<T>* r,T* gd,I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I hs=H*s,ws=W*s,hsw=ws/2+1,kc=filter_channel(bc,C,KB,KC);
    Z<T> sum(0,0);
    for(I a=0;a<s;++a) for(I b=0;b<s;++b) {
        I hh=h+a*H,ww=w+b*W;
        if(ww<=ws/2) sum+=k[(kc*hs+hh)*hsw+ww]*g[(bc*hs+hh)*hsw+ww];
        if(w>0 && 2*w!=W) {
            hh=(H-h)%H+a*H; ww=W-w+b*W;
            if(ww<=ws/2) sum+=cj(k[(kc*hs+hh)*hsw+ww]*g[(bc*hs+hh)*hsw+ww]);
        }
    }
    const auto value=sum/d[i];
    r[i]=value;
    if(gd) gd[i]=-(cj(value)*q[i]).real();
}

template<class T> __device__ void alias_adjoint(const Z<T>* r,const T* gd,I bc,I h,I w,
    I H,I W,I s,Z<T>& value,T& power) {
    const I hs=H*s,ws=W*s,sw=W/2+1;
    value=Z<T>(0,0); power=0;
    I lh=h%H,lw=w%W;
    if(lw<=W/2) {
        const I j=(bc*H+lh)*sw+lw;
        value+=r[j];
        if(gd) power+=gd[j];
    }
    if(w>0 && 2*w!=ws) {
        lh=((hs-h)%hs)%H; lw=(ws-w)%W;
        if(lw<=W/2) {
            const I j=(bc*H+lh)*sw+lw;
            value+=cj(r[j]);
            if(gd) power+=gd[j];
        }
    }
    const T count=T(s)*T(s);
    value/=count; power/=count;
}

template<class T> __device__ Z<T> prior_gradient(const Z<T>* g,const Z<T>* k,const Z<T>* r,
    I bc,I kc,I h,I w,I H,I W,I s) {
    Z<T> value; T ignored;
    alias_adjoint(r,static_cast<const T*>(nullptr),bc,h,w,H,W,s,value,ignored);
    const I plane=H*s*(W*s/2+1),frequency=h*(W*s/2+1)+w;
    return g[bc*plane+frequency]-cj(k[kc*plane+frequency])*value;
}

// J_N^H gp: one LR stored frequency per thread, no scatter/atomic.
// A*y contributes conj(A)*gp; A*conj(y) contributes A*conj(gp).
// LR boundary columns have only a direct branch. Fixed a,b reduction order.
template<class T>
__global__ void nearest_adjoint_y(const Z<T>* g,const Z<T>* k,const Z<T>* r,
    const Z<T>* ph,const Z<T>* pw,Z<T>* gy,I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I ws=W*s,kc=filter_channel(bc,C,KB,KC);
    Z<T> sum(0,0);
    for(I a=0;a<s;++a) for(I b=0;b<s;++b) {
        I hh=h+a*H,ww=w+b*W;
        if(ww<=ws/2) {
            const auto gp=prior_gradient(g,k,r,bc,kc,hh,ww,H,W,s);
            sum+=cj(ph[hh]*pw[ww])*gp;
        }
        if(w>0 && 2*w!=W) {
            hh=(H-h)%H+a*H; ww=W-w+b*W;
            if(ww<=ws/2) {
                const auto gp=prior_gradient(g,k,r,bc,kc,hh,ww,H,W,s);
                sum+=(ph[hh]*pw[ww])*cj(gp);
            }
        }
    }
    gy[i]=r[i]+sum;
}

// Preserve production b then c order for all four filter broadcast cases.
template<class T>
__global__ void nearest_adjoint_filter(const Z<T>* g,const Z<T>* y,const Z<T>* k,
    const Z<T>* q,const Z<T>* r,const T* gd,const Z<T>* ph,const Z<T>* pw,
    Z<T>* gk,I n,I B,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I hs=H*s,sw=W*s/2+1,w=i%sw,h=(i/sw)%hs,kbc=i/(hs*sw);
    const I first_b=KB==1?0:kbc/KC,last_b=KB==1?B:first_b+1;
    const I first_c=KC==1?0:kbc%KC,last_c=KC==1?C:first_c+1;
    Z<T> sum(0,0);
    for(I b=first_b;b<last_b;++b) for(I c=first_c;c<last_c;++c) {
        const I bc=b*C+c,j=(bc*hs+h)*sw+w;
        Z<T> value; T power;
        alias_adjoint(r,gd,bc,h,w,H,W,s,value,power);
        const auto prior=prior_read(y,ph,pw,bc,h,w,H,W,s);
        sum+=cj(g[j])*read(q,bc,h%H,w%W,H,W)-cj(prior)*value+k[i]*(T(2)*power);
    }
    gk[i]=sum;
}

at::Tensor plain(const at::Tensor& t) { return t.resolve_conj().resolve_neg().contiguous(); }
} // namespace

std::vector<at::Tensor> nearest_forward_cuda(const at::Tensor& fy,const at::Tensor& fb,
    const at::Tensor& lambda,const at::Tensor& phase_h,const at::Tensor& phase_w,I H,I W,I s) {
    auto y=plain(fy),k=plain(fb),l=plain(lambda),ph=plain(phase_h),pw=plain(phase_w);
    auto q=at::empty(y.sizes(),y.options()),d=at::empty(y.sizes(),l.options());
    auto out=at::empty({y.size(0),y.size(1),H*s,W*s/2+1},y.options());
    auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
    AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"nearest_training_forward",[&] {
        using T=scalar_t;
        nearest_solve_alias<T><<<(q.numel()+255)/256,256,0,stream>>>(y.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
            l.data_ptr<T>(),ph.data_ptr<Z<T>>(),pw.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
            q.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
        nearest_solve_output<T><<<(out.numel()+255)/256,256,0,stream>>>(y.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
            q.data_ptr<Z<T>>(),ph.data_ptr<Z<T>>(),pw.data_ptr<Z<T>>(),out.data_ptr<Z<T>>(),
            out.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,q,d};
}

std::vector<at::Tensor> nearest_backward_cuda(const at::Tensor& grad,const at::Tensor& fy,
    const at::Tensor& fb,const at::Tensor& q0,const at::Tensor& d0,const at::Tensor& phase_h,
    const at::Tensor& phase_w,I H,I W,I s,bool need_y,bool need_k,bool need_l) {
    if(!(need_y||need_k||need_l)) return {at::Tensor(),at::Tensor(),at::Tensor()};
    auto g=plain(grad),y=plain(fy),k=plain(fb),q=plain(q0),d=plain(d0),ph=plain(phase_h),pw=plain(phase_w);
    auto r=at::empty(q.sizes(),q.options());
    auto gd=need_k||need_l?at::empty(d.sizes(),d.options()):at::Tensor();
    auto gy=need_y?at::empty(y.sizes(),y.options()):at::Tensor();
    auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();
    auto stream=c10::cuda::getCurrentCUDAStream(g.get_device());
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"nearest_training_backward",[&] {
        using T=scalar_t;
        nearest_adjoint_q<T><<<(r.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
            q.data_ptr<Z<T>>(),d.data_ptr<T>(),r.data_ptr<Z<T>>(),gd.defined()?gd.data_ptr<T>():nullptr,
            r.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
        if(need_y) nearest_adjoint_y<T><<<(gy.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
            r.data_ptr<Z<T>>(),ph.data_ptr<Z<T>>(),pw.data_ptr<Z<T>>(),gy.data_ptr<Z<T>>(),
            gy.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
        if(need_k) nearest_adjoint_filter<T><<<(gk.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),y.data_ptr<Z<T>>(),
            k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),r.data_ptr<Z<T>>(),gd.data_ptr<T>(),ph.data_ptr<Z<T>>(),pw.data_ptr<Z<T>>(),
            gk.data_ptr<Z<T>>(),gk.numel(),y.size(0),y.size(1),H,W,s,k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gl=need_l?gd.sum({0,2,3},true):at::Tensor();
    return {gy,gk,gl};
}
