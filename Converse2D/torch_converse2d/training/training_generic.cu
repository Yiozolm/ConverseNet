#include "launchers.cuh"
#include "detail/math.cuh"
using namespace converse2d::training_detail;
namespace {
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

}
void launch_training_generic_forward(const at::Tensor& y,const at::Tensor& p,const at::Tensor& k,const at::Tensor& l,at::Tensor& out,at::Tensor& q,at::Tensor& d,int64_t H,int64_t W,int64_t s,cudaStream_t stream) {
AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"converse_training_forward",[&] {
using T=scalar_t;

            solve_alias<T><<<(q.numel()+255)/256,256,0,stream>>>(y.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),
                k.data_ptr<Z<T>>(),l.data_ptr<T>(),q.data_ptr<Z<T>>(),d.data_ptr<T>(),
                q.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
            solve_output<T><<<(out.numel()+255)/256,256,0,stream>>>(p.data_ptr<Z<T>>(),k.data_ptr<Z<T>>(),
                q.data_ptr<Z<T>>(),out.data_ptr<Z<T>>(),out.numel(),y.size(1),H,W,s,k.size(0),k.size(1));
        
});
}
void launch_training_generic_backward(const at::Tensor& g,const at::Tensor& p,const at::Tensor& k,const at::Tensor& q,const at::Tensor& d,at::Tensor& r,at::Tensor& gd,at::Tensor& gp,at::Tensor& gk,int64_t H,int64_t W,int64_t s,bool need_p,bool need_k,bool no_broadcast,bool reduce_filter,cudaStream_t stream) {
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
            if(reduce_filter)
                adjoint_filter<T><<<(k.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<T>>(),p.data_ptr<Z<T>>(),
                    k.data_ptr<Z<T>>(),q.data_ptr<Z<T>>(),r.data_ptr<Z<T>>(),gd.data_ptr<T>(),gk.data_ptr<Z<T>>(),
                    k.numel(),p.size(0),p.size(1),H,W,s,k.size(0),k.size(1));
        
});
}
