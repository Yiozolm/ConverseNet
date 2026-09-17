// Experimental scale-one specialization. Generic scales use the original source.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <vector>

std::vector<at::Tensor> converse_training_forward_cuda(
    const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t);
std::vector<at::Tensor> converse_training_backward_cuda(
    const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
    int64_t,int64_t,int64_t,bool,bool,bool,bool);

namespace {
template<class T> using Z=c10::complex<T>;
using I=int64_t;
template<class T> __device__ Z<T> cj(Z<T> z) { return {z.real(),-z.imag()}; }
template<class T> __device__ T norm2(Z<T> z);
template<> __device__ float norm2(Z<float> z) {
    return __fadd_rn(__fmul_rn(z.real(),z.real()),__fmul_rn(z.imag(),z.imag()));
}
template<> __device__ double norm2(Z<double> z) {
    return __dadd_rn(__dmul_rn(z.real(),z.real()),__dmul_rn(z.imag(),z.imag()));
}
__device__ I filter_channel(I bc,I C,I KB,I KC) {
    return (KB==1?0:bc/C)*KC+(KC==1?0:bc%C);
}

template<class T>
__global__ void forward_scale1(const Z<T>* y,const Z<T>* p,const Z<T>* k,const T* lambda,
    Z<T>* out,Z<T>* q,T* d,I n,I C,I plane,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I bc=i/plane,ki=filter_channel(bc,C,KB,KC)*plane+i%plane;
    const auto filter=k[ki],prior=p[i];
    Z<T> prediction(0,0);
    prediction+=filter*prior;
    T power=0;
    power+=norm2(filter);
    const T denominator=power/T(1)+lambda[bc%C];
    const auto residual=(y[i]-prediction/T(1))/denominator;
    d[i]=denominator;
    q[i]=residual;
    out[i]=prior+cj(filter)*residual;
}

// At s1 the mirrored branch is absent for stored interior frequencies;
// DC/Nyquist skip it explicitly in the generic implementation.
template<class T>
__global__ void backward_scale1(const Z<T>* g,const Z<T>* p,const Z<T>* k,
    const Z<T>* q,const T* d,Z<T>* r,T* gd,Z<T>* gp,Z<T>* gk,
    I n,I C,I plane,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I bc=i/plane,ki=filter_channel(bc,C,KB,KC)*plane+i%plane;
    const auto filter=k[ki],grad=g[i];
    Z<T> sum(0,0);
    sum+=filter*grad;
    const auto value=sum/d[i];
    if(r) r[i]=value;
    T denominator_grad=0;
    if(gd||gk) denominator_grad=-(cj(value)*q[i]).real();
    if(gd) gd[i]=denominator_grad;
    Z<T> alias_value(0,0);
    alias_value+=value;
    alias_value/=T(1);
    T alias_power=0;
    alias_power+=denominator_grad;
    alias_power/=T(1);
    if(gp) gp[i]=grad-cj(filter)*alias_value;
    if(gk) gk[i]=cj(grad)*q[i]-cj(p[i])*alias_value+filter*(T(2)*alias_power);
}

template<class T>
__global__ void filter_scale1(const Z<T>* g,const Z<T>* p,const Z<T>* k,
    const Z<T>* q,const Z<T>* r,const T* gd,Z<T>* gk,
    I n,I B,I C,I plane,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I kbc=i/plane,frequency=i%plane;
    const I first_b=KB==1?0:kbc/KC,last_b=KB==1?B:first_b+1;
    const I first_c=KC==1?0:kbc%KC,last_c=KC==1?C:first_c+1;
    Z<T> sum(0,0);
    for(I b=first_b;b<last_b;++b) for(I c=first_c;c<last_c;++c) {
        const I j=(b*C+c)*plane+frequency;
        Z<T> value(0,0);
        value+=r[j];
        value/=T(1);
        T power=0;
        power+=gd[j];
        power/=T(1);
        sum+=cj(g[j])*q[j]-cj(p[j])*value+k[i]*(T(2)*power);
    }
    gk[i]=sum;
}
at::Tensor plain(const at::Tensor& t) { return t.resolve_conj().resolve_neg().contiguous(); }
}

std::vector<at::Tensor> training_speed_forward_cuda(const at::Tensor& fy,const at::Tensor& fx0,
    const at::Tensor& fb,const at::Tensor& lambda,int64_t H,int64_t W,int64_t s) {
    if(s!=1) return converse_training_forward_cuda(fy,fx0,fb,lambda,H,W,s);
    auto y=plain(fy),p=plain(fx0),k=plain(fb),l=plain(lambda);
    auto q=at::empty(y.sizes(),y.options()),d=at::empty(y.sizes(),l.options());
    auto out=at::empty(p.sizes(),p.options());
    const I n=q.numel(),plane=H*(W/2+1);
    const auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
    AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"training_speed_forward_scale1",[&] {
        using T=scalar_t;
        if(n>0) forward_scale1<T><<<(n+255)/256,256,0,stream>>>(y.data_ptr<Z<T>>(),
            p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),l.data_ptr<T>(),out.data_ptr<Z<T>>(),
            q.data_ptr<Z<T>>(),d.data_ptr<T>(),n,y.size(1),plane,k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,q,d};
}

std::vector<at::Tensor> training_speed_backward_cuda(const at::Tensor& grad,const at::Tensor& fx0,
    const at::Tensor& fb,const at::Tensor& q0,const at::Tensor& d0,int64_t H,int64_t W,int64_t s,
    bool need_y,bool need_p,bool need_k,bool need_l) {
    if(s!=1) return converse_training_backward_cuda(grad,fx0,fb,q0,d0,H,W,s,need_y,need_p,need_k,need_l);
    if(!(need_y||need_p||need_k||need_l)) return {at::Tensor(),at::Tensor(),at::Tensor(),at::Tensor()};
    auto g=plain(grad),p=plain(fx0),k=plain(fb),q=plain(q0),d=plain(d0);
    const bool no_broadcast=k.size(0)==p.size(0)&&k.size(1)==p.size(1);
    const bool reduce_filter=need_k&&!no_broadcast;
    auto r=need_y||reduce_filter?at::empty(q.sizes(),q.options()):at::Tensor();
    auto gd=need_l||reduce_filter?at::empty(d.sizes(),d.options()):at::Tensor();
    auto gp=need_p?at::empty(p.sizes(),p.options()):at::Tensor();
    auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();
    const I n=q.numel(),plane=H*(W/2+1);
    const auto stream=c10::cuda::getCurrentCUDAStream(g.get_device());
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"training_speed_backward_scale1",[&] {
        using T=scalar_t;
        if(n>0) backward_scale1<T><<<(n+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),
            p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
            r.defined()?r.data_ptr<Z<T>>():nullptr,gd.defined()?gd.data_ptr<T>():nullptr,
            need_p?gp.data_ptr<Z<T>>():nullptr,need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr,
            n,p.size(1),plane,k.size(0),k.size(1));
        if(reduce_filter&&gk.numel()>0) filter_scale1<T><<<(gk.numel()+255)/256,256,0,stream>>>(
            g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),
            r.data_ptr<Z<T>>(),gd.data_ptr<T>(),gk.data_ptr<Z<T>>(),gk.numel(),p.size(0),
            p.size(1),plane,k.size(0),k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gl=need_l?gd.sum({0,2,3},true):at::Tensor();
    return {need_y?r:at::Tensor(),gp,gk,gl};
}
