// Isolated s=1, SAME observation/prior transfer. FFTs retain external autograd.
// No floating atomics: fixed batch/channel loops and fixed lambda trees.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <vector>

namespace {
using I = int64_t;
template<class T> using Z = c10::complex<T>;
template<class T> __device__ Z<T> cj(Z<T> value) { return {value.real(),-value.imag()}; }
template<class T> __device__ Z<double> wide(Z<T> value) { return {double(value.real()),double(value.imag())}; }
template<class T> __device__ Z<T> narrow(Z<double> value) { return {T(value.real()),T(value.imag())}; }
template<class T> __device__ T norm2(Z<T> value);
template<> __device__ float norm2(Z<float> z) {
    return __fadd_rn(__fmul_rn(z.real(),z.real()),__fmul_rn(z.imag(),z.imag()));
}
template<> __device__ double norm2(Z<double> z) {
    return __dadd_rn(__dmul_rn(z.real(),z.real()),__dmul_rn(z.imag(),z.imag()));
}
constexpr int THREADS = 256;

template<class T>
__global__ void shared_s1_prepare(const Z<T>* k,const T* lambda,Z<T>* transfer,T* denominator,
    I count,I C,I KC,I plane) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const I row=i/plane,c=row%C,b=row/C,frequency=i%plane;
    const auto value=k[(b*KC+(KC==1?0:c))*plane+frequency];
    const T regularizer=lambda[c],d=norm2(value)+regularizer;
    denominator[i]=d;
    transfer[i]=(cj(value)+Z<T>(regularizer,0))/d;
}

template<class T>
__global__ void shared_s1_apply(const Z<T>* y,const Z<T>* transfer,Z<T>* output,
    I count,I C,I KB,I plane) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const I row=i/plane,c=row%C,b=row/C;
    const I j=((KB==1?0:b)*C+c)*plane+i%plane;
    output[i]=transfer[j]*y[i];
}

template<class T>
__global__ void shared_s1_grad_y(const Z<T>* g,const Z<T>* transfer,Z<T>* gy,
    I count,I C,I KB,I plane) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const I row=i/plane,c=row%C,b=row/C;
    const I j=((KB==1?0:b)*C+c)*plane+i%plane;
    gy[i]=g[i]*cj(transfer[j]);
}

// One block per (kernel batch, output channel, frequency tile). Products
// conj(y)*g use the execution dtype; their B reduction accumulates in FP64.
// Only KC=1 needs a temporary expanded kernel gradient, stored in FP64 so
// the following C reduction does not add an intermediate FP32 rounding.
template<class T>
__global__ void shared_s1_transfer_vjp(const Z<T>* y,const Z<T>* g,const Z<T>* k,
    const Z<T>* transfer,const T* denominator,Z<T>* gk,Z<double>* expanded_gk,
    double* lambda_partials,I B,I C,I KB,I KC,I plane,I tiles) {
    const I tile=I(blockIdx.x)%tiles,row=I(blockIdx.x)/tiles;
    const I b=row/C,c=row%C,frequency=tile*THREADS+threadIdx.x;
    double lambda_value=0;
    if(frequency<plane) {
        const I i=row*plane+frequency;
        Z<double> t(0,0);
        const I first=KB==1?0:b,last=KB==1?B:b+1;
        for(I batch=first;batch<last;++batch) {
            const I j=(batch*C+c)*plane+frequency;
            t+=wide(cj(y[j])*g[j]);
        }
        const auto h=wide(transfer[i]);
        const double d=double(denominator[i]);
        const double cross=(cj(t)*h).real();
        if(gk||expanded_gk) {
            const I ki=(b*KC+(KC==1?0:c))*plane+frequency;
            const auto value=cj(t)/d-wide(k[ki])*(2.0*cross/d);
            if(expanded_gk) expanded_gk[i]=value;
            else gk[ki]=narrow<T>(value);
        }
        if(lambda_partials) lambda_value=(t.real()-cross)/d;
    }
    if(lambda_partials) {
        __shared__ double sum[THREADS];
        sum[threadIdx.x]=lambda_value;
        __syncthreads();
        for(int offset=THREADS/2;offset>0;offset/=2) {
            if(threadIdx.x<offset) sum[threadIdx.x]+=sum[threadIdx.x+offset];
            __syncthreads();
        }
        if(threadIdx.x==0) lambda_partials[row*tiles+tile]=sum[0];
    }
}

template<class T>
__global__ void shared_s1_reduce_channels(const Z<double>* expanded,Z<T>* gk,
    I count,I C,I plane) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const I b=i/plane,frequency=i%plane;
    Z<double> sum(0,0);
    for(I c=0;c<C;++c) sum+=expanded[(b*C+c)*plane+frequency];
    gk[i]=narrow<T>(sum);
}

template<class T>
__global__ void shared_s1_reduce_lambda(const double* partials,T* gl,I C,I KB,I tiles) {
    const I c=blockIdx.x;
    double value=0;
    for(I j=threadIdx.x;j<KB*tiles;j+=THREADS) {
        const I b=j/tiles,tile=j%tiles;
        value+=partials[(b*C+c)*tiles+tile];
    }
    __shared__ double sum[THREADS];
    sum[threadIdx.x]=value;
    __syncthreads();
    for(int offset=THREADS/2;offset>0;offset/=2) {
        if(threadIdx.x<offset) sum[threadIdx.x]+=sum[threadIdx.x+offset];
        __syncthreads();
    }
    if(threadIdx.x==0) gl[c]=T(sum[0]);
}

at::Tensor plain(const at::Tensor& tensor) { return tensor.resolve_conj().resolve_neg().contiguous(); }
} // namespace

std::vector<at::Tensor> converse_training_shared_s1_forward_cuda(const at::Tensor& fy,const at::Tensor& fk,const at::Tensor& lambda) {
    auto y=plain(fy),k=plain(fk),l=plain(lambda);
    const I C=y.size(1),KB=k.size(0),KC=k.size(1),plane=y.size(2)*y.size(3);
    auto h=at::empty({KB,C,y.size(2),y.size(3)},y.options());
    auto d=at::empty(h.sizes(),l.options());
    auto out=at::empty(y.sizes(),y.options());
    auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
    AT_DISPATCH_FLOATING_TYPES(l.scalar_type(),"shared_s1_forward",[&] {
        using T=scalar_t;
        shared_s1_prepare<T><<<(h.numel()+THREADS-1)/THREADS,THREADS,0,stream>>>(
            k.data_ptr<Z<T>>(),l.data_ptr<T>(),h.data_ptr<Z<T>>(),d.data_ptr<T>(),h.numel(),C,KC,plane);
        shared_s1_apply<T><<<(y.numel()+THREADS-1)/THREADS,THREADS,0,stream>>>(
            y.data_ptr<Z<T>>(),h.data_ptr<Z<T>>(),out.data_ptr<Z<T>>(),y.numel(),C,KB,plane);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,h,d};
}

std::vector<at::Tensor> converse_training_shared_s1_backward_cuda(const at::Tensor& grad,const at::Tensor& fy,const at::Tensor& fk,
    const at::Tensor& transfer,const at::Tensor& denominator,bool need_y,bool need_k,bool need_l) {
    if(!(need_y||need_k||need_l)) return {at::Tensor(),at::Tensor(),at::Tensor()};
    auto g=plain(grad),y=(need_k||need_l)?plain(fy):fy,k=need_k?plain(fk):fk;
    auto h=plain(transfer),d=plain(denominator);
    const I B=y.size(0),C=y.size(1),KB=k.size(0),KC=k.size(1),plane=y.size(2)*y.size(3);
    const I tiles=(plane+THREADS-1)/THREADS;
    const bool reduce_channels=need_k&&KC==1&&C>1;
    auto gy=need_y?at::empty(y.sizes(),y.options()):at::Tensor();
    auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();
    auto gl=need_l?at::empty({1,C,1,1},d.options()):at::Tensor();
    auto expanded=reduce_channels?at::empty(h.sizes(),h.options().dtype(at::kComplexDouble)):at::Tensor();
    auto partials=need_l?at::empty({KB,C,tiles},d.options().dtype(at::kDouble)):at::Tensor();
    auto stream=c10::cuda::getCurrentCUDAStream(g.get_device());
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(),"shared_s1_backward",[&] {
        using T=scalar_t;
        if(need_y) shared_s1_grad_y<T><<<(gy.numel()+THREADS-1)/THREADS,THREADS,0,stream>>>(
            g.data_ptr<Z<T>>(),h.data_ptr<Z<T>>(),gy.data_ptr<Z<T>>(),gy.numel(),C,KB,plane);
        if(need_k||need_l) shared_s1_transfer_vjp<T><<<KB*C*tiles,THREADS,0,stream>>>(
            y.data_ptr<Z<T>>(),g.data_ptr<Z<T>>(),need_k?k.data_ptr<Z<T>>():nullptr,h.data_ptr<Z<T>>(),d.data_ptr<T>(),
            need_k&&!reduce_channels?gk.data_ptr<Z<T>>():nullptr,
            reduce_channels?expanded.data_ptr<Z<double>>():nullptr,need_l?partials.data_ptr<double>():nullptr,
            B,C,KB,KC,plane,tiles);
        if(reduce_channels) shared_s1_reduce_channels<T><<<(gk.numel()+THREADS-1)/THREADS,THREADS,0,stream>>>(
            expanded.data_ptr<Z<double>>(),gk.data_ptr<Z<T>>(),gk.numel(),C,plane);
        if(need_l) shared_s1_reduce_lambda<T><<<C,THREADS,0,stream>>>(partials.data_ptr<double>(),gl.data_ptr<T>(),C,KB,tiles);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gy,gk,gl};
}
