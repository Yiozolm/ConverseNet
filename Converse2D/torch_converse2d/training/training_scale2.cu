// Experimental s=2 training; not in the default build or dispatch.
// Caller must validate s==2, positive dimensions, and int32 element indexing:
// p.numel() <= INT32_MAX-256, H <= INT32_MAX/2, W <= INT32_MAX/2.
// See docs/training_scale2.md for precision gates and benchmark scope.
#include <cstdint>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
namespace converse2d::scale2 {

template<class T> using Z = c10::complex<T>;
using I = int32_t;
template<class T> __device__ Z<T> cj(Z<T> z) { return {z.real(),-z.imag()}; }
template<class T> __device__ T norm2(Z<T> z);
template<> __device__ inline float norm2(Z<float> z) {
    return __fadd_rn(__fmul_rn(z.real(),z.real()),__fmul_rn(z.imag(),z.imag()));
}
template<> __device__ inline double norm2(Z<double> z) {
    return __dadd_rn(__dmul_rn(z.real(),z.real()),__dmul_rn(z.imag(),z.imag()));
}
template<class T> __device__ Z<T> read(const Z<T>* p,I bc,I h,I w,I H,I W) {
    const bool mirror=w>W/2;
    if(mirror) { h=(H-h)%H; w=W-w; }
    auto z=p[(bc*H+h)*(W/2+1)+w];
    return mirror?cj(z):z;
}
__device__ inline I filter_channel(I bc,I C,I KB,I KC) {
    return (KB==1?0:bc/C)*KC+(KC==1?0:bc%C);
}

namespace {

// Adjoint of expansion q -> read(q,h%H,w%W). Gather both the direct aliases
// and aliases reading conj(q); DC/even Nyquist have only the direct branch.

// Adjoint of the alias reduction, evaluated once for a stored HR frequency.
// A missing HR column reads the conjugate of its stored counterpart. Boundary
// LR columns can receive BOTH contributions, including different row indices.
template<class T> __device__ void alias_adjoint(const Z<T>* r,const T* gd,I bc,I h,I w,
                                              I H,I W,I runtime_scale,Z<T>& value,T& power) {
    constexpr I s = 2;
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
__global__ void adjoint_filter(const Z<T>* g,const Z<T>* prior,const Z<T>* k,
    const Z<T>* q,const Z<T>* r,const T* gd,Z<T>* gk,I n,I B,I C,I H,I W,I runtime_scale,I KB,I KC) {
    constexpr I s = 2;
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

}
namespace {
// Each LR half-spectrum frequency owns its four HR aliases. An even LR
// Nyquist column owns only the two direct aliases; the reflected two belong
// to the reflected row's LR frequency. This works for arbitrary complex
// half spectra, including unconstrained DC/Nyquist rows.
template<class T>
__global__ void fused_forward_s2(const Z<T>* y,const Z<T>* p,const Z<T>* k,const T* l,
    Z<T>* out,Z<T>* q,T* d,I n,I C,I H,I W,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I hs=2*H,ws=2*W,stored=W+1,kc=filter_channel(bc,C,KB,KC);
    Z<T> filters[4],priors[4];
    I offsets[4]; bool mirrors[4];
    Z<T> prediction(0,0); T power=0;
    #pragma unroll
    for(I a=0;a<2;++a) {
        #pragma unroll
        for(I b=0;b<2;++b) {
            const I j=2*a+b;
            I hh=h+a*H,ww=w+b*W;
            mirrors[j]=ww>W;
            if(mirrors[j]) { hh=(hs-hh)%hs; ww=ws-ww; }
            offsets[j]=hh*stored+ww;
            filters[j]=k[kc*hs*stored+offsets[j]];
            priors[j]=p[bc*hs*stored+offsets[j]];
            const auto fk=mirrors[j]?cj(filters[j]):filters[j];
            const auto prior=mirrors[j]?cj(priors[j]):priors[j];
            prediction+=fk*prior;
            power+=norm2(fk);
        }
    }
    const T denominator=power/T(4)+l[bc%C];
    const auto correction=(y[i]-prediction/T(4))/denominator;
    d[i]=denominator; q[i]=correction;
    #pragma unroll
    for(I j=0;j<4;++j) {
        if(2*w==W && mirrors[j]) continue;
        const auto value=mirrors[j]?cj(correction):correction;
        out[bc*hs*stored+offsets[j]]=priors[j]+cj(filters[j])*value;
    }
}

template<class T>
__global__ void fused_backward_s2(const Z<T>* g,const Z<T>* p,const Z<T>* k,
    const Z<T>* q,const T* d,Z<T>* r,T* gd,Z<T>* gp,Z<T>* gk,
    I n,I C,I H,I W,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I hs=2*H,ws=2*W,stored=W+1,kc=filter_channel(bc,C,KB,KC);
    const bool nyquist=2*w==W;
    Z<T> filters[4],grads[4]; I offsets[4]; bool mirrors[4];
    #pragma unroll
    for(I a=0;a<2;++a) {
        #pragma unroll
        for(I b=0;b<2;++b) {
            const I j=2*a+b;
            I hh=h+a*H,ww=w+b*W;
            mirrors[j]=ww>W;
            if(mirrors[j]) { hh=(hs-hh)%hs; ww=ws-ww; }
            offsets[j]=hh*stored+ww;
            filters[j]=k[kc*hs*stored+offsets[j]];
            grads[j]=g[bc*hs*stored+offsets[j]];
        }
    }
    // Preserve adjoint_q's direct/reflected accumulation order exactly.
    Z<T> sum(0,0);
    #pragma unroll
    for(I a=0;a<2;++a) {
        #pragma unroll
        for(I b=0;b<2;++b) {
            if(w+b*W<=W) sum+=filters[2*a+b]*grads[2*a+b];
            if(w>0 && !nyquist && b==0) {
                const I j=2*(h==0?a:1-a)+1;
                sum+=cj(filters[j]*grads[j]);
            }
        }
    }
    const auto value=sum/d[i];
    const auto correction=q[i];
    const T denominator_grad=-(cj(value)*correction).real();
    r[i]=value;
    if(gd) gd[i]=denominator_grad;
    Z<T> neighbor(0,0); T neighbor_power=0;
    if(nyquist && (gp||gk)) {
        // The only cross-row dependency is this LR Nyquist column. Its
        // reflected row's two products are already in our registers.
        Z<T> other_sum(0,0);
        #pragma unroll
        for(I a=0;a<2;++a) {
            const I j=2*(h==0?a:1-a)+1;
            other_sum+=filters[j]*grads[j];
        }
        const I other=(bc*H+(H-h)%H)*sw+w;
        neighbor=other_sum/d[other];
        neighbor_power=-(cj(neighbor)*q[other]).real();
    }
    if(!(gp||gk)) return;
    #pragma unroll
    for(I j=0;j<4;++j) {
        if(nyquist && mirrors[j]) continue;
        Z<T> alias_value(0,0); T alias_power=0;
        alias_value+=mirrors[j]?cj(value):value;
        alias_power+=denominator_grad;
        if(nyquist) { alias_value+=cj(neighbor); alias_power+=neighbor_power; }
        alias_value/=T(4); alias_power/=T(4);
        const I index=bc*hs*stored+offsets[j];
        if(gp) gp[index]=grads[j]-cj(filters[j])*alias_value;
        if(gk) {
            const auto c=mirrors[j]?cj(correction):correction;
            gk[index]=cj(grads[j])*c-cj(p[index])*alias_value+filters[j]*(T(2)*alias_power);
        }
    }
}

}
void launch_training_scale2_forward(const at::Tensor& y,const at::Tensor& p,const at::Tensor& k,const at::Tensor& l,at::Tensor& out,at::Tensor& q,at::Tensor& d,int64_t H,int64_t W,int64_t s,cudaStream_t stream) {
AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"converse_training_forward",[&] {
using T=scalar_t;

    fused_forward_s2<T><<<(q.numel()+255)/256,256,0,stream>>>(
        y.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),l.data_ptr<T>(),
        out.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
        q.numel(),y.size(1),H,W,k.size(0),k.size(1));

});
}
void launch_training_scale2_backward(const at::Tensor& g,const at::Tensor& p,const at::Tensor& k,const at::Tensor& q,const at::Tensor& d,at::Tensor& r,at::Tensor& gd,at::Tensor& gp,at::Tensor& gk,int64_t H,int64_t W,int64_t s,bool need_p,bool need_k,bool no_broadcast,bool reduce_filter,cudaStream_t stream) {
AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"converse_training_backward",[&] {
using T=scalar_t;

    fused_backward_s2<T><<<(q.numel()+255)/256,256,0,stream>>>(
        g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
        r.data_ptr<Z<T>>(),gd.defined()?gd.data_ptr<T>():nullptr,
        need_p?gp.data_ptr<Z<T>>():nullptr,need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr,
        q.numel(),p.size(1),H,W,k.size(0),k.size(1));
            if(reduce_filter)
                adjoint_filter<T><<<(k.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),
                    k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),r.data_ptr<Z<T>>(),gd.data_ptr<T>(),gk.data_ptr<Z<T>>(),
                    k.numel(),p.size(0),p.size(1),H,W,s,k.size(0),k.size(1));
        
});
}

}
