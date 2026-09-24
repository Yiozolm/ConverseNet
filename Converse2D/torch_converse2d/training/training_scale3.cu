// Experimental s=3 training. Not enabled by the default build/dispatch.
// Caller must guard s==3, p.numel()<=INT32_MAX-256, H/W<=INT32_MAX/3.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cstdint>
namespace converse2d::scale3 {

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
template<class T> __device__ void alias_adjoint(const Z<T>* r,const T* gd,I bc,I h,I w,
                                              I H,I W,I runtime_scale,Z<T>& value,T& power) {
    constexpr I s=3;
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
    constexpr I s=3;
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
// One LR half-spectrum frequency owns nine virtual HR aliases. Interior LR
// columns own all nine physical outputs. DC/even-Nyquist columns own only
// unreflected aliases; reflected ones belong to the opposite LR row.
template<class T>
__global__ void fused_forward_s3(const Z<T>* y,const Z<T>* p,const Z<T>* k,const T* l,
    Z<T>* out,Z<T>* q,T* d,I n,I C,I H,I W,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I hs=3*H,ws=3*W,stored=ws/2+1,kc=filter_channel(bc,C,KB,KC);
    const bool boundary=w==0||2*w==W;
    Z<T> filters[9],priors[9];I offsets[9];bool mirrors[9];
    Z<T> prediction(0,0);T power=0;
    #pragma unroll
    for(I a=0;a<3;++a) {
        #pragma unroll
        for(I b=0;b<3;++b) {
            const I j=3*a+b;
            I hh=h+a*H,ww=w+b*W;
            mirrors[j]=ww>ws/2;
            if(mirrors[j]) {hh=(hs-hh)%hs;ww=ws-ww;}
            offsets[j]=hh*stored+ww;
            filters[j]=k[kc*hs*stored+offsets[j]];
            priors[j]=p[bc*hs*stored+offsets[j]];
            const auto fk=mirrors[j]?cj(filters[j]):filters[j];
            const auto prior=mirrors[j]?cj(priors[j]):priors[j];
            prediction+=fk*prior;
            power+=norm2(fk);
        }
    }
    const T denominator=power/T(9)+l[bc%C];
    const auto correction=(y[i]-prediction/T(9))/denominator;
    d[i]=denominator;q[i]=correction;
    #pragma unroll
    for(I j=0;j<9;++j) {
        if(boundary&&mirrors[j]) continue;
        out[bc*hs*stored+offsets[j]]=priors[j]+cj(filters[j])*(mirrors[j]?cj(correction):correction);
    }
}

template<class T>
__global__ void fused_backward_s3(const Z<T>* g,const Z<T>* p,const Z<T>* k,
    const Z<T>* q,const T* d,Z<T>* r,T* gd,Z<T>* gp,Z<T>* gk,
    I n,I C,I H,I W,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I sw=W/2+1,w=i%sw,h=(i/sw)%H,bc=i/(H*sw);
    const I hs=3*H,ws=3*W,stored=ws/2+1,kc=filter_channel(bc,C,KB,KC);
    const bool boundary=w==0||2*w==W;
    Z<T> filters[9],grads[9];I offsets[9];bool mirrors[9];
    #pragma unroll
    for(I a=0;a<3;++a) {
        #pragma unroll
        for(I b=0;b<3;++b) {
            const I j=3*a+b;
            I hh=h+a*H,ww=w+b*W;
            mirrors[j]=ww>ws/2;
            if(mirrors[j]) {hh=(hs-hh)%hs;ww=ws-ww;}
            offsets[j]=hh*stored+ww;
            filters[j]=k[kc*hs*stored+offsets[j]];
            grads[j]=g[bc*hs*stored+offsets[j]];
        }
    }
    // Same interleaved direct/reflected sum as the generic adjoint_q.
    Z<T> sum(0,0);
    #pragma unroll
    for(I a=0;a<3;++a) {
        #pragma unroll
        for(I b=0;b<3;++b) {
            if(w+b*W<=ws/2) sum+=filters[3*a+b]*grads[3*a+b];
            if(!boundary&&W-w+b*W<=ws/2) {
                const I ar=h==0?(3-a)%3:2-a;
                const I j=3*ar+(2-b);
                sum+=cj(filters[j]*grads[j]);
            }
        }
    }
    const auto value=sum/d[i],correction=q[i];
    const T power=-(cj(value)*correction).real();
    r[i]=value;if(gd) gd[i]=power;
    if(!(gp||gk)) return;
    Z<T> neighbor(0,0);T neighbor_power=0;
    if(boundary) {
        const I other_h=(H-h)%H;
        if(other_h==h) {neighbor=value;neighbor_power=power;}
        else {
            // Unlike s2, s3 DC aliases include an internal HR column. Its
            // adjoint needs the opposite row, including HR-edge values not
            // present in this thread's cache. Recompute only at LR boundaries.
            Z<T> other_sum(0,0);
            #pragma unroll
            for(I a=0;a<3;++a) {
                #pragma unroll
                for(I b=0;b<3;++b) {
                    const I hh=other_h+a*H,ww=w+b*W;
                    if(ww<=ws/2) other_sum+=k[(kc*hs+hh)*stored+ww]*g[(bc*hs+hh)*stored+ww];
                }
            }
            const I other=(bc*H+other_h)*sw+w;
            neighbor=other_sum/d[other];
            neighbor_power=-(cj(neighbor)*q[other]).real();
        }
    }
    #pragma unroll
    for(I j=0;j<9;++j) {
        if(boundary&&mirrors[j]) continue;
        Z<T> alias_value(0,0);T alias_power=0;
        alias_value+=mirrors[j]?cj(value):value;
        alias_power+=power;
        const I column=offsets[j]%stored;
        if(boundary&&column>0&&2*column!=ws) {
            alias_value+=cj(neighbor);alias_power+=neighbor_power;
        }
        alias_value/=T(9);alias_power/=T(9);
        const I index=bc*hs*stored+offsets[j];
        if(gp) gp[index]=grads[j]-cj(filters[j])*alias_value;
        if(gk) {
            const auto c=mirrors[j]?cj(correction):correction;
            gk[index]=cj(grads[j])*c-cj(p[index])*alias_value+filters[j]*(T(2)*alias_power);
        }
    }
}

}
void launch_training_scale3_forward(const at::Tensor& y,const at::Tensor& p,const at::Tensor& k,const at::Tensor& l,at::Tensor& out,at::Tensor& q,at::Tensor& d,int64_t H,int64_t W,int64_t s,cudaStream_t stream) {
    AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"converse_training_scale3_forward",[&] {
        using T=scalar_t;
        fused_forward_s3<T><<<(q.numel()+255)/256,256,0,stream>>>(
            y.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),l.data_ptr<T>(),
            out.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),q.numel(),y.size(1),H,W,k.size(0),k.size(1));
    });
}
void launch_training_scale3_backward(const at::Tensor& g,const at::Tensor& p,const at::Tensor& k,const at::Tensor& q,const at::Tensor& d,at::Tensor& r,at::Tensor& gd,at::Tensor& gp,at::Tensor& gk,int64_t H,int64_t W,int64_t s,bool need_p,bool need_k,bool no_broadcast,bool reduce_filter,cudaStream_t stream) {
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"converse_training_scale3_backward",[&] {
        using T=scalar_t;
        fused_backward_s3<T><<<(q.numel()+255)/256,256,0,stream>>>(
            g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
            r.data_ptr<Z<T>>(),gd.defined()?gd.data_ptr<T>():nullptr,
            need_p?gp.data_ptr<Z<T>>():nullptr,need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr,
            q.numel(),p.size(1),H,W,k.size(0),k.size(1));
        if(reduce_filter) adjoint_filter<T><<<(k.numel()+255)/256,256,0,stream>>>(
            g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),r.data_ptr<Z<T>>(),
            gd.data_ptr<T>(),gk.data_ptr<Z<T>>(),k.numel(),p.size(0),p.size(1),H,W,s,k.size(0),k.size(1));
    });
}
}
