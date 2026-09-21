#include "launchers.cuh"
#include "detail/math.cuh"
using namespace converse2d::training_detail;
namespace {
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

// At s=1 the mirrored alias branch cannot reach another stored frequency;
// DC/Nyquist also skip it. Preserve the generic arithmetic order, including
// zero-initialized accumulations, rather than reassociating the complex VJP.
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

}
void launch_training_scale1_forward(const at::Tensor& y,const at::Tensor& p,const at::Tensor& k,const at::Tensor& l,at::Tensor& out,at::Tensor& q,at::Tensor& d,int64_t H,int64_t W,int64_t s,cudaStream_t stream) {
AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"converse_training_forward",[&] {
using T=scalar_t;

            const I n=q.numel(),plane=H*(W/2+1);
            forward_scale1<T><<<(n+255)/256,256,0,stream>>>(y.data_ptr<Z<T>>(),
                p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),l.data_ptr<T>(),out.data_ptr<Z<T>>(),
                q.data_ptr<Z<T>>(),d.data_ptr<T>(),n,y.size(1),plane,k.size(0),k.size(1));
        
});
}
void launch_training_scale1_backward(const at::Tensor& g,const at::Tensor& p,const at::Tensor& k,const at::Tensor& q,const at::Tensor& d,at::Tensor& r,at::Tensor& gd,at::Tensor& gp,at::Tensor& gk,int64_t H,int64_t W,int64_t s,bool need_p,bool need_k,bool no_broadcast,bool reduce_filter,cudaStream_t stream) {
AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"converse_training_backward",[&] {
using T=scalar_t;

            const I n=q.numel(),plane=H*(W/2+1);
            backward_scale1<T><<<(n+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),
                p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
                r.defined()?r.data_ptr<Z<T>>():nullptr,gd.defined()?gd.data_ptr<T>():nullptr,
                need_p?gp.data_ptr<Z<T>>():nullptr,need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr,
                n,p.size(1),plane,k.size(0),k.size(1));
            if(reduce_filter) filter_scale1<T><<<(gk.numel()+255)/256,256,0,stream>>>(
                g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),
                r.data_ptr<Z<T>>(),gd.data_ptr<T>(),gk.data_ptr<Z<T>>(),gk.numel(),p.size(0),
                p.size(1),plane,k.size(0),k.size(1));
        
});
}
