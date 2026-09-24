#include "../../common/fp32_dispatch.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <vector>
#include <type_traits>
namespace converse2d::full_training {
using at::Tensor;using I=int64_t;
template<class T>using Z=c10::complex<T>;
template<class T>__device__ Z<T> cj(Z<T> z){return {z.real(),-z.imag()};}
template<class T>__device__ T add_rn(T a,T b);
template<>__device__ float add_rn(float a,float b){return __fadd_rn(a,b);}
template<class T>__device__ T mul_rn(T a,T b);
template<>__device__ float mul_rn(float a,float b){return __fmul_rn(a,b);}
template<class T>__device__ Z<T> add(Z<T>a,Z<T>b){return {add_rn(a.real(),b.real()),add_rn(a.imag(),b.imag())};}
template<class T>__device__ T fma_rn(T a,T b,T c);
template<>__device__ float fma_rn(float a,float b,float c){return __fmaf_rn(a,b,c);}
// Match the two FMA boundaries of the measured ATen complex multiply. Keeping
// these explicit prevents negation/conjugation from changing which term fuses.
template<class T>__device__ Z<T> product(Z<T>a,Z<T>b){
 return {fma_rn(a.real(),b.real(),-mul_rn(a.imag(),b.imag())),
         fma_rn(a.imag(),b.real(),mul_rn(a.real(),b.imag()))};
}
template<class T>__device__ T norm(Z<T> z);
template<>__device__ float norm(Z<float>z){return __fadd_rn(__fmul_rn(z.real(),z.real()),__fmul_rn(z.imag(),z.imag()));}
__device__ I kc(I bc,I C,I KB,I KC){return (KB==1?0:bc/C)*KC+(KC==1?0:bc%C);}
Tensor plain(Tensor t){return t.resolve_conj().resolve_neg().contiguous();}
template<class T>__global__ void prepare(const Z<T>*p,const Z<T>*k,Z<T>*pm,T*pw,I n,I kn,I C,I HW,I KB,I KC){
 I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 I ki=kc(i/HW,C,KB,KC)*HW+i%HW;pm[i]=product(k[ki],p[i]);if(i<kn)pw[i]=norm(k[i]);
}
template<class T>__global__ void output(const Z<T>*y,const Z<T>*pm,const Z<T>*p,const Z<T>*k,const T*d,Z<T>*out,Z<T>*q,I n,I C,I H,I W,I s,I KB,I KC){
 I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 I hw=H*W,bc=i/hw,h=(i/W)%H,w=i%W,kbc=kc(bc,C,KB,KC),di=((KB==1?0:bc/C)*C+bc%C)*hw+h*W+w;
 auto v=add(y[i],-pm[i])/Z<T>(d[di],0);q[i]=v;
 for(I a=0;a<s;++a)for(I b=0;b<s;++b){I pos=(h+a*H)*(W*s)+w+b*W,j=bc*hw*s*s+pos,ki=kbc*hw*s*s+pos;out[j]=add(p[j],product(cj(k[ki]),v));}
}
template<class T>__global__ void adj_output(const Z<T>*g,const Z<T>*k,const Z<T>*q,Z<T>*t,Z<T>*direct,I n,I C,I H,I W,I s,I KB,I KC){
 I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 I hs=H*s,ws=W*s,bc=i/(hs*ws),h=(i/ws)%hs,w=i%ws,ki=kc(bc,C,KB,KC)*hs*ws+h*ws+w,qi=(bc*H+h%H)*W+w%W;
 t[i]=product(g[i],k[ki]);direct[i]=product(g[i],cj(q[qi]));
}
template<class T>__global__ void adj_div(const Z<T>*t,const Z<T>*q,const T*d,Z<T>*gy,Z<T>*gd,I n,I HW,I C,I KB){
 I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 I bc=i/HW,di=((KB==1?0:bc/C)*C+bc%C)*HW+i%HW;Z<T>den(d[di],0);
 gy[i]=t[i]/den;gd[i]=product(-t[i],cj(q[i]/den));
}
template<class T>__global__ void adj_prediction(const Z<T>*g,const Z<T>*p,const Z<T>*k,const Z<T>*gm,Z<T>*gp,Z<T>*gk,I n,I C,I H,I W,I s,I KB,I KC,bool shared){
 I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 I hs=H*s,ws=W*s,bc=i/(hs*ws),h=(i/ws)%hs,w=i%ws,ki=kc(bc,C,KB,KC)*hs*ws+h*ws+w,qi=(bc*H+h%H)*W+w%W;
 gp[i]=add(shared?add(g[i],-gm[qi]):g[i],product(gm[qi],cj(k[ki])));gk[i]=product(gm[qi],cj(p[i]));
}
template<class T>__global__ void adj_kernel(const Z<T>*k,const Z<T>*a,const Z<T>*b,const T*power,Z<T>*out,I n,I H,I W,I s){
 I i=I(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 I hs=H*s,ws=W*s,bc=i/(hs*ws),h=(i/ws)%hs,w=i%ws;T v=power[(bc*H+h%H)*W+w%W];
 Z<T>c(mul_rn(v,mul_rn(T(2),k[i].real())),mul_rn(v,mul_rn(T(2),k[i].imag())));
 out[i]=add(add(add(a[i],b[i]),Z<T>(0,c.imag())),Z<T>(c.real(),0));
}
std::vector<Tensor> full_prepare_cuda(Tensor p0,Tensor k0){
 auto p=plain(p0),k=plain(k0),pm=at::empty(p.sizes(),p.options()),pw=at::empty(k.sizes(),k.options().dtype(at::toRealValueType(k.scalar_type())));
 auto stream=c10::cuda::getCurrentCUDAStream();CONVERSE_DISPATCH_FP32(pw.scalar_type(),"full_prepare",[&]{prepare<scalar_t><<<(p.numel()+255)/256,256,0,stream>>>(p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),pm.data_ptr<Z<scalar_t>>(),pw.data_ptr<scalar_t>(),p.numel(),k.numel(),p.size(1),p.size(2)*p.size(3),k.size(0),k.size(1));});C10_CUDA_KERNEL_LAUNCH_CHECK();return {pm,pw};
}
std::vector<Tensor> full_output_cuda(Tensor y0,Tensor pm0,Tensor p0,Tensor k0,Tensor d0,I s){
 auto y=plain(y0),pm=plain(pm0),p=plain(p0),k=plain(k0),d=plain(d0),out=at::empty(p.sizes(),p.options()),q=at::empty(y.sizes(),y.options());auto stream=c10::cuda::getCurrentCUDAStream();
 CONVERSE_DISPATCH_FP32(d.scalar_type(),"full_output",[&]{output<scalar_t><<<(y.numel()+255)/256,256,0,stream>>>(y.data_ptr<Z<scalar_t>>(),pm.data_ptr<Z<scalar_t>>(),p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),d.data_ptr<scalar_t>(),out.data_ptr<Z<scalar_t>>(),q.data_ptr<Z<scalar_t>>(),y.numel(),y.size(1),y.size(2),y.size(3),s,k.size(0),k.size(1));});C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,q};
}
std::vector<Tensor> full_adjoint_output_cuda(Tensor g0,Tensor k0,Tensor q0,I s){
 auto g=plain(g0),k=plain(k0),q=plain(q0),t=at::empty(g.sizes(),g.options()),d=at::empty_like(t);auto stream=c10::cuda::getCurrentCUDAStream();
 CONVERSE_DISPATCH_FP32(at::toRealValueType(g.scalar_type()),"full_adj_output",[&]{adj_output<scalar_t><<<(g.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),q.data_ptr<Z<scalar_t>>(),t.data_ptr<Z<scalar_t>>(),d.data_ptr<Z<scalar_t>>(),g.numel(),g.size(1),q.size(2),q.size(3),s,k.size(0),k.size(1));});C10_CUDA_KERNEL_LAUNCH_CHECK();return {t,d};
}
std::vector<Tensor> full_adjoint_div_cuda(Tensor t0,Tensor q0,Tensor d0){
 auto t=plain(t0),q=plain(q0),d=plain(d0),gy=at::empty(t.sizes(),t.options()),gd=at::empty_like(gy);auto stream=c10::cuda::getCurrentCUDAStream();
 CONVERSE_DISPATCH_FP32(d.scalar_type(),"full_adj_div",[&]{adj_div<scalar_t><<<(t.numel()+255)/256,256,0,stream>>>(t.data_ptr<Z<scalar_t>>(),q.data_ptr<Z<scalar_t>>(),d.data_ptr<scalar_t>(),gy.data_ptr<Z<scalar_t>>(),gd.data_ptr<Z<scalar_t>>(),t.numel(),t.size(2)*t.size(3),t.size(1),d.size(0));});C10_CUDA_KERNEL_LAUNCH_CHECK();return {gy,gd};
}
std::vector<Tensor> full_adjoint_prediction_cuda(Tensor g0,Tensor p0,Tensor k0,Tensor gm0,I s,bool shared){
 auto g=plain(g0),p=plain(p0),k=plain(k0),gm=plain(gm0),gp=at::empty(p.sizes(),p.options()),gk=at::empty_like(gp);auto stream=c10::cuda::getCurrentCUDAStream();
 CONVERSE_DISPATCH_FP32(at::toRealValueType(g.scalar_type()),"full_adj_pred",[&]{adj_prediction<scalar_t><<<(p.numel()+255)/256,256,0,stream>>>(g.data_ptr<Z<scalar_t>>(),p.data_ptr<Z<scalar_t>>(),k.data_ptr<Z<scalar_t>>(),gm.data_ptr<Z<scalar_t>>(),gp.data_ptr<Z<scalar_t>>(),gk.data_ptr<Z<scalar_t>>(),p.numel(),p.size(1),gm.size(2),gm.size(3),s,k.size(0),k.size(1),shared);});C10_CUDA_KERNEL_LAUNCH_CHECK();return {gp,gk};
}
Tensor full_adjoint_kernel_cuda(Tensor k0,Tensor a0,Tensor b0,Tensor power0,I s){
 auto k=plain(k0),a=plain(a0),b=plain(b0),power=plain(power0),out=at::empty(k.sizes(),k.options());auto stream=c10::cuda::getCurrentCUDAStream();
 CONVERSE_DISPATCH_FP32(power.scalar_type(),"full_adj_kernel",[&]{adj_kernel<scalar_t><<<(k.numel()+255)/256,256,0,stream>>>(k.data_ptr<Z<scalar_t>>(),a.data_ptr<Z<scalar_t>>(),b.data_ptr<Z<scalar_t>>(),power.data_ptr<scalar_t>(),out.data_ptr<Z<scalar_t>>(),k.numel(),power.size(2),power.size(3),s);});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

#include "scale1.cuh"
#include "scale2.cuh"

} // namespace converse2d::full_training
