// Trainable half-spectrum solve and its real/complex VJP. No full spectra,
// repeat buffers, floating atomics or low-precision reductions are used.
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

template<class T>
__global__ void solve_alias(const Z<T>* y,const Z<T>* prior,const Z<T>* k,const T* lambda,
                           Z<T>* q,T* d,I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I kc=filter_channel(bc,C,KB,KC);
    Z<T> prediction(0,0);
    T power=0;
    for(I a=0;a<s;++a) for(I b=0;b<s;++b) {
        auto fk=read(k,kc,h+a*H,w+b*W,H*s,W*s);
        prediction+=fk*read(prior,bc,h+a*H,w+b*W,H*s,W*s);
        power+=norm2(fk);
    }
    const T count=T(s)*T(s);
    d[i]=power/count+lambda[bc%C];
    q[i]=(y[i]-prediction/count)/d[i];
}

template<class T>
__global__ void solve_output(const Z<T>* prior,const Z<T>* k,const Z<T>* q,Z<T>* out,
                            I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I hs=H*s,sw=W*s/2+1,w=i%sw,h=(i/sw)%hs,bc=i/(hs*sw);
    const I kc=filter_channel(bc,C,KB,KC);
    out[i]=prior[i]+cj(k[(kc*hs+h)*sw+w])*read(q,bc,h%H,w%W,H,W);
}

// Adjoint of expansion q -> read(q,h%H,w%W). Gather both the direct aliases
// and aliases reading conj(q); DC/even Nyquist have only the direct branch.
template<class T>
__global__ void adjoint_q(const Z<T>* g,const Z<T>* k,const Z<T>* q,const T* d,
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

// Adjoint of the alias reduction, evaluated once for a stored HR frequency.
// A missing HR column reads the conjugate of its stored counterpart. Boundary
// LR columns can receive BOTH contributions, including different row indices.
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

template<class T>
__global__ void adjoint_inputs(const Z<T>* g,const Z<T>* prior,const Z<T>* k,
    const Z<T>* q,const Z<T>* r,const T* gd,Z<T>* gp,Z<T>* gk,
    I n,I C,I H,I W,I s,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I hs=H*s,sw=W*s/2+1,w=i%sw,h=(i/sw)%hs,bc=i/(hs*sw);
    const I kc=filter_channel(bc,C,KB,KC),ki=(kc*hs+h)*sw+w;
    Z<T> value; T power;
    alias_adjoint(r,gd,bc,h,w,H,W,s,value,power);
    if(gp) gp[i]=g[i]-cj(k[ki])*value;
    // Only used when k has no broadcasting, so every thread owns its output.
    if(gk) gk[i]=cj(g[i])*read(q,bc,h%H,w%W,H,W)-cj(prior[i])*value+k[ki]*(T(2)*power);
}

template<class T>
__global__ void adjoint_filter(const Z<T>* g,const Z<T>* prior,const Z<T>* k,
    const Z<T>* q,const Z<T>* r,const T* gd,Z<T>* gk,I n,I B,I C,I H,I W,I s,I KB,I KC) {
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
        sum+=cj(g[j])*read(q,bc,h%H,w%W,H,W)-cj(prior[j])*value+k[i]*(T(2)*power);
    }
    gk[i]=sum;
}

at::Tensor plain(const at::Tensor& t) { return t.resolve_conj().resolve_neg().contiguous(); }
} // namespace

std::vector<at::Tensor> converse_training_forward_cuda(const at::Tensor& fy,const at::Tensor& fx0,
    const at::Tensor& fb,const at::Tensor& lambda,int64_t H,int64_t W,int64_t s) {
    auto y=plain(fy),p=plain(fx0),k=plain(fb),l=plain(lambda);
    auto q=at::empty(y.sizes(),y.options()),d=at::empty(y.sizes(),l.options());
    auto out=at::empty(p.sizes(),p.options());
    auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
    AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"converse_training_forward",[&] {
        using T=scalar_t;
        solve_alias<T><<<(q.numel()+255)/256,256,0,stream>>>(y.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),
            k.data_ptr<Z<T>>(),l.data_ptr<T>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
            q.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
        solve_output<T><<<(out.numel()+255)/256,256,0,stream>>>(p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
            q.data_ptr<Z<T>>(),out.data_ptr<Z<T>>(),out.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,q,d};
}

std::vector<at::Tensor> converse_training_backward_cuda(const at::Tensor& grad,const at::Tensor& fx0,
    const at::Tensor& fb,const at::Tensor& q0,const at::Tensor& d0,
    int64_t H,int64_t W,int64_t s,bool need_y,bool need_p,bool need_k,bool need_l) {
    auto g=plain(grad),p=plain(fx0),k=plain(fb),q=plain(q0),d=plain(d0);
    auto r=at::empty(q.sizes(),q.options());
    auto gd=need_k||need_l?at::empty(d.sizes(),d.options()):at::Tensor();
    auto gp=need_p?at::empty(p.sizes(),p.options()):at::Tensor();
    auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();
    const bool no_broadcast=k.size(0)==p.size(0)&&k.size(1)==p.size(1);
    auto stream=c10::cuda::getCurrentCUDAStream(g.get_device());
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"converse_training_backward",[&] {
        using T=scalar_t;
        adjoint_q<T><<<(r.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
            q.data_ptr<Z<T>>(),d.data_ptr<T>(),r.data_ptr<Z<T>>(),gd.defined()?gd.data_ptr<T>():nullptr,
            r.numel(),p.size(1),H,W,s,k.size(0),k.size(1));
        if(need_p || (need_k && no_broadcast))
            adjoint_inputs<T><<<(p.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),
                k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),r.data_ptr<Z<T>>(),gd.defined()?gd.data_ptr<T>():nullptr,
                need_p?gp.data_ptr<Z<T>>():nullptr,need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr,
                p.numel(),p.size(1),H,W,s,k.size(0),k.size(1));
        if(need_k && !no_broadcast)
            adjoint_filter<T><<<(k.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),
                k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),r.data_ptr<Z<T>>(),gd.data_ptr<T>(),gk.data_ptr<Z<T>>(),
                k.numel(),p.size(0),p.size(1),H,W,s,k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gl=need_l?gd.sum({0,2,3},true):at::Tensor();
    return {need_y?r:at::Tensor(),gp,gk,gl};
}
