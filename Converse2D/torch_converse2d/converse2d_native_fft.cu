// Native cuFFT FP16/BF16 transforms with GPU-resident power-of-two scaling.
// Complex low storage is represented as [..., 2] real elements (including BF16).
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cufftXt.h>
#include <cmath>
#include <list>
#include <memory>
#include <mutex>

namespace {
using I=int64_t;
struct Key {
    int device; I stream,h,w,batch; at::ScalarType dtype; bool inverse;
    bool operator==(const Key& k) const {
        return device==k.device&&stream==k.stream&&h==k.h&&w==k.w&&batch==k.batch&&dtype==k.dtype&&inverse==k.inverse;
    }
};
void check(cufftResult r) { TORCH_CHECK(r==CUFFT_SUCCESS,"native low FFT: cuFFT error ",int(r)); }
struct Plan {
    Key key; cufftHandle handle=0; size_t bytes=0; bool supported=false;
    explicit Plan(Key k):key(k) {
        check(cufftCreate(&handle));
        try {
            check(cufftSetAutoAllocation(handle,0));
            long long sizes[2]={k.h,k.w};
            const auto real=k.dtype==at::kHalf?CUDA_R_16F:CUDA_R_16BF;
            const auto complex=k.dtype==at::kHalf?CUDA_C_16F:CUDA_C_16BF;
            auto status=cufftXtMakePlanMany(handle,2,sizes,nullptr,1,1,k.inverse?complex:real,
                nullptr,1,1,k.inverse?real:complex,k.batch,&bytes,complex);
            if(status==CUFFT_NOT_SUPPORTED||status==CUFFT_INVALID_SIZE) return;
            check(status); supported=true;
        } catch(...) { cufftDestroy(handle); handle=0; throw; }
    }
    ~Plan() noexcept {
        int previous;
        if(!handle||cudaGetDevice(&previous)!=cudaSuccess) return;
        if(previous!=key.device&&cudaSetDevice(key.device)!=cudaSuccess) return;
        cufftDestroy(handle);
        if(previous!=key.device) cudaSetDevice(previous);
    }
};
std::list<std::shared_ptr<Plan>> plans;
std::mutex plan_mutex;
std::shared_ptr<Plan> plan_for(Key key) {
    for(auto it=plans.begin();it!=plans.end();++it) if((*it)->key==key) {
        auto p=*it; plans.splice(plans.begin(),plans,it); return p;
    }
    auto p=std::make_shared<Plan>(key);
    plans.push_front(p);
    if(plans.size()>32) plans.pop_back();
    return p;
}

__device__ float block_max(float value,float* scratch) {
    scratch[threadIdx.x]=value; __syncthreads();
    for(int n=blockDim.x/2;n>0;n/=2) {
        if(threadIdx.x<n) scratch[threadIdx.x]=fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+n]);
        __syncthreads();
    }
    return scratch[0];
}
__device__ float block_sum(float value,float* scratch) {
    __syncthreads(); scratch[threadIdx.x]=value; __syncthreads();
    for(int n=blockDim.x/2;n>0;n/=2) {
        if(threadIdx.x<n) scratch[threadIdx.x]+=scratch[threadIdx.x+n];
        __syncthreads();
    }
    return scratch[0];
}

template<class T,bool INVERSE>
__global__ void pack_scaled(const float* source,T* packed,int* exponent,I h,I w,int log_n,bool adjoint) {
    const I bc=blockIdx.x,sw=w/2+1,n=INVERSE?h*sw*2:h*w;
    __shared__ float scratch[256];
    float maximum=0;
    for(I j=threadIdx.x;j<n;j+=blockDim.x) maximum=fmaxf(maximum,fabsf(source[bc*n+j]));
    maximum=block_max(maximum,scratch);
    int e=0;
    if(maximum>0&&isfinite(maximum)) frexpf(maximum,&e);
    // Bound the transform's worst-case L1 growth below 4096, with extra
    // headroom for C2R's complex components. Integer exponents avoid overflow
    // of a floating scale factor for tiny/huge FP32 gradients.
    int shift=maximum>0&&isfinite(maximum)?12-log_n-e:0;
    if constexpr(INVERSE) {
        // Use a normalized Hermitian L1 bound for the inverse. N*max would
        // excessively shrink non-DC frequencies in images with a strong DC
        // component. Normalize before summing to avoid FP32 sum overflow.
        float sum=0;
        for(I j=threadIdx.x;j<n;j+=blockDim.x) {
            const I col=(j/2)%sw;
            const float weight=!adjoint&&col>0&&2*col!=w?2.f:1.f;
            sum+=scalbnf(fabsf(source[bc*n+j]),-e)*weight;
        }
        sum=block_sum(sum,scratch);
        int sum_exp=0;
        if(sum>0&&isfinite(sum)) { frexpf(sum,&sum_exp); shift=12-e-sum_exp; }
        else shift=0;
    }
    if(threadIdx.x==0) exponent[bc]=shift;
    if constexpr(INVERSE) {
        for(I j=threadIdx.x;j<h*sw;j+=blockDim.x) {
            const I row=j/sw,col=j%sw,base=bc*n+2*j;
            float re=source[base],im=source[base+1];
            // cuFFT C2R requires Hermitian boundary columns. Projection also
            // defines the correct real-IFFT behavior for arbitrary gradients.
            if(col==0||2*col==w) {
                const I other=bc*n+2*(((h-row)%h)*sw+col);
                re=0.5f*re+0.5f*source[other];
                im=0.5f*im-0.5f*source[other+1];
            } else if(adjoint) { re*=0.5f; im*=0.5f; }
            packed[base]=T(scalbnf(re,shift));
            packed[base+1]=T(scalbnf(im,shift));
        }
    } else {
        for(I j=threadIdx.x;j<n;j+=blockDim.x) packed[bc*n+j]=T(scalbnf(source[bc*n+j],shift));
    }
}

template<class T,bool INVERSE>
__global__ void unpack_scaled(const T* packed,const int* exponent,float* out,I total,I h,I w,int log_n,bool adjoint) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=total) return;
    const I size=INVERSE?h*w:h*(w/2+1)*2,bc=i/size;
    int shift=-exponent[bc];
    if constexpr(INVERSE) { if(!adjoint) shift-=log_n; }
    else if(adjoint) {
        shift-=log_n;
        const I col=(i/2)%(w/2+1);
        if(col>0&&2*col!=w) ++shift;
    }
    out[i]=scalbnf(float(packed[i]),shift);
}

at::Tensor execute(const at::Tensor& source,I h,I w,at::ScalarType dtype,bool inverse,bool adjoint) {
    c10::cuda::CUDAGuard guard(source.device());
    auto stream=c10::cuda::getCurrentCUDAStream(source.get_device());
    // The wrapper excludes capture before reaching this cache. Plans/workspaces
    // are eager-only and never lend evictable pointers to a CUDA Graph.
    std::unique_lock<std::mutex> lock(plan_mutex);
    auto plan=plan_for({source.get_device(),stream.id(),h,w,source.size(0)*source.size(1),dtype,inverse});
    if(!plan->supported) return at::Tensor();
    auto input=source.resolve_conj().resolve_neg().contiguous();
    const at::DimVector real_shape{source.size(0),source.size(1),h,w};
    const at::DimVector complex_shape{source.size(0),source.size(1),h,w/2+1,2};
    auto low_options=source.options().dtype(dtype);
    auto packed=at::empty(inverse?complex_shape:real_shape,low_options);
    auto transformed=at::empty(inverse?real_shape:complex_shape,low_options);
    auto exponent=at::empty({plan->key.batch},source.options().dtype(at::kInt));
    TORCH_CHECK(plan->bytes<=size_t(INT64_MAX),"native FFT workspace overflow");
    auto workspace=at::empty({int64_t(plan->bytes)},source.options().dtype(at::kByte));
    int log_n=0;
    for(I n=h*w;n>1;n>>=1) ++log_n;
    const float* raw=inverse?reinterpret_cast<const float*>(input.data_ptr<c10::complex<float>>()):input.data_ptr<float>();
    AT_DISPATCH_SWITCH(dtype,"native_fft_pack",
        AT_DISPATCH_CASE(at::kHalf,[&] {
            if(inverse) pack_scaled<scalar_t,true><<<plan->key.batch,256,0,stream>>>(raw,packed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),h,w,log_n,adjoint);
            else pack_scaled<scalar_t,false><<<plan->key.batch,256,0,stream>>>(raw,packed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),h,w,log_n,adjoint);
        })
        AT_DISPATCH_CASE(at::kBFloat16,[&] {
            if(inverse) pack_scaled<scalar_t,true><<<plan->key.batch,256,0,stream>>>(raw,packed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),h,w,log_n,adjoint);
            else pack_scaled<scalar_t,false><<<plan->key.batch,256,0,stream>>>(raw,packed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),h,w,log_n,adjoint);
        })
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    check(cufftSetStream(plan->handle,stream));
    check(cufftSetWorkArea(plan->handle,workspace.data_ptr()));
    check(cufftXtExec(plan->handle,packed.data_ptr(),transformed.data_ptr(),inverse?CUFFT_INVERSE:CUFFT_FORWARD));
    lock.unlock();
    packed=at::Tensor(); workspace=at::Tensor();
    auto output=at::empty(inverse?real_shape:at::DimVector{source.size(0),source.size(1),h,w/2+1},
                          source.options().dtype(inverse?at::kFloat:at::kComplexFloat));
    float* dest=inverse?output.data_ptr<float>():reinterpret_cast<float*>(output.data_ptr<c10::complex<float>>());
    const I total=transformed.numel();
    AT_DISPATCH_SWITCH(dtype,"native_fft_unpack",
        AT_DISPATCH_CASE(at::kHalf,[&] {
            if(inverse) unpack_scaled<scalar_t,true><<<(total+255)/256,256,0,stream>>>(transformed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),dest,total,h,w,log_n,adjoint);
            else unpack_scaled<scalar_t,false><<<(total+255)/256,256,0,stream>>>(transformed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),dest,total,h,w,log_n,adjoint);
        })
        AT_DISPATCH_CASE(at::kBFloat16,[&] {
            if(inverse) unpack_scaled<scalar_t,true><<<(total+255)/256,256,0,stream>>>(transformed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),dest,total,h,w,log_n,adjoint);
            else unpack_scaled<scalar_t,false><<<(total+255)/256,256,0,stream>>>(transformed.data_ptr<scalar_t>(),exponent.data_ptr<int>(),dest,total,h,w,log_n,adjoint);
        })
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
} // namespace

at::Tensor converse_native_fft_cuda(const at::Tensor& input,int64_t h,int64_t w,at::ScalarType dtype,bool inverse,bool adjoint) {
    return execute(input,h,w,dtype,inverse,adjoint);
}
void converse_native_fft_clear_cache() {
    std::lock_guard<std::mutex> lock(plan_mutex);
    plans.clear();
}
