#pragma once
#include <atomic>
#include <torch/csrc/autograd/custom_function.h>

#ifdef CONVERSE2D_WITH_CUDA
at::Tensor converse_native_fft_cuda(const at::Tensor&,int64_t,int64_t,at::ScalarType,bool,bool);
void converse_native_fft_clear_cache();
#endif

namespace converse2d::native_fft {
static thread_local bool active=false;
static std::atomic<int64_t> counts[7]{};
static bool set_enabled(bool enabled) { const bool previous=active; active=enabled; return previous; }
static bool enabled() { return active; }
static void reset_stats() { for(auto& value:counts) value.store(0); }
static c10::Dict<std::string,int64_t> stats() {
    c10::Dict<std::string,int64_t> out;
    const char* names[]={"rfft","irfft","rfft_adjoint","irfft_adjoint","fallback_shape_device","fallback_capture","fallback_plan"};
    for(int i=0;i<7;++i) out.insert(names[i],counts[i].load());
    return out;
}
static bool eligible(const Tensor& x,int64_t h,int64_t w,at::ScalarType dtype) {
#ifdef CONVERSE2D_WITH_CUDA
    if(x.is_cuda()&&x.dim()==4&&x.numel()>0&&h>=2&&w>=2&&!(h&(h-1))&&!(w&(w-1))&&
       h<=INT32_MAX/w&&x.size(0)<=INT32_MAX/x.size(1)/(h*w)&&
       (dtype==at::kHalf||dtype==at::kBFloat16)) {
        c10::cuda::CUDAGuard guard(x.device());
        auto prop=at::cuda::getDeviceProperties(x.get_device());
        if(prop->major*10+prop->minor>=(dtype==at::kHalf?53:80)) {
            if(c10::cuda::currentStreamCaptureStatusMayInitCtx()==c10::cuda::CaptureStatus::None) return true;
            ++counts[5]; return false;
        }
    }
#endif
    ++counts[4]; return false;
}

// Exact FP32 analytical adjoints for higher derivatives and fallback paths.
static Tensor rfft_adjoint_aten(const Tensor& g,int64_t h,int64_t w) {
    auto padded=at::constant_pad_nd(g,{0,w-(w/2+1)},0);
    return at::real(at::fft_ifft2(padded,at::IntArrayRef({h,w}),{-2,-1},"forward"));
}
static Tensor irfft_adjoint_aten(const Tensor& g,int64_t h,int64_t w) {
    auto f=at::fft_rfft2(g);
    auto weights=at::ones({w/2+1},g.options());
    weights.slice(0,1,(w+1)/2).mul_(2);
    return f*weights/(double(h)*double(w));
}

#ifdef CONVERSE2D_WITH_CUDA
class Rfft : public torch::autograd::Function<Rfft> {
public:
    static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor input,int64_t precision) {
        const auto h=input.size(2),w=input.size(3);
        const auto dtype=precision==0?at::kHalf:at::kBFloat16;
        auto out=converse_native_fft_cuda(input,h,w,dtype,false,false);
        ctx->saved_data["h"]=h; ctx->saved_data["w"]=w; ctx->saved_data["precision"]=precision;
        ctx->saved_data["used"]=out.defined();
        if(out.defined()) { ++counts[0]; return out; }
        ++counts[6]; return at::fft_rfft2(input);
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
        const auto h=ctx->saved_data["h"].toInt(),w=ctx->saved_data["w"].toInt();
        auto dtype=ctx->saved_data["precision"].toInt()==0?at::kHalf:at::kBFloat16;
        if(!at::GradMode::is_enabled()&&ctx->saved_data["used"].toBool()&&eligible(g[0],h,w,dtype)) {
            auto out=converse_native_fft_cuda(g[0],h,w,dtype,true,true);
            if(out.defined()) { ++counts[2]; return {out,Tensor()}; }
            ++counts[6];
        }
        return {rfft_adjoint_aten(g[0],h,w),Tensor()};
    }
};
class Irfft : public torch::autograd::Function<Irfft> {
public:
    static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor input,int64_t h,int64_t w,int64_t precision) {
        const auto dtype=precision==0?at::kHalf:at::kBFloat16;
        auto out=converse_native_fft_cuda(input,h,w,dtype,true,false);
        ctx->saved_data["h"]=h; ctx->saved_data["w"]=w; ctx->saved_data["precision"]=precision;
        ctx->saved_data["used"]=out.defined();
        if(out.defined()) { ++counts[1]; return out; }
        ++counts[6]; return at::fft_irfft2(input,at::IntArrayRef({h,w}));
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
        const auto h=ctx->saved_data["h"].toInt(),w=ctx->saved_data["w"].toInt();
        auto dtype=ctx->saved_data["precision"].toInt()==0?at::kHalf:at::kBFloat16;
        if(!at::GradMode::is_enabled()&&ctx->saved_data["used"].toBool()&&eligible(g[0],h,w,dtype)) {
            auto out=converse_native_fft_cuda(g[0],h,w,dtype,false,true);
            if(out.defined()) { ++counts[3]; return {out,Tensor(),Tensor(),Tensor()}; }
            ++counts[6];
        }
        return {irfft_adjoint_aten(g[0],h,w),Tensor(),Tensor(),Tensor()};
    }
};
#endif

static Tensor rfft(Tensor input,int64_t precision) {
    TORCH_CHECK(input.dim()==4&&input.numel()>0&&input.scalar_type()==at::kFloat,"native RFFT expects nonempty FP32 B,C,H,W input");
    TORCH_CHECK(precision==0||precision==1,"precision must be 0 (FP16) or 1 (BF16)");
#ifdef CONVERSE2D_WITH_CUDA
    if(active&&eligible(input,input.size(2),input.size(3),precision==0?at::kHalf:at::kBFloat16))
        return Rfft::apply(input,precision);
#endif
    return at::fft_rfft2(input);
}
static Tensor irfft(Tensor input,int64_t h,int64_t w,int64_t precision) {
    TORCH_CHECK(input.dim()==4&&input.numel()>0&&h>0&&w>0&&input.size(2)==h&&input.size(3)==w/2+1&&
                input.scalar_type()==at::kComplexFloat,"native IRFFT expects a matching nonempty complex64 half spectrum");
    TORCH_CHECK(precision==0||precision==1,"precision must be 0 (FP16) or 1 (BF16)");
#ifdef CONVERSE2D_WITH_CUDA
    if(active&&eligible(input,h,w,precision==0?at::kHalf:at::kBFloat16)) return Irfft::apply(input,h,w,precision);
#endif
    return at::fft_irfft2(input,at::IntArrayRef({h,w}));
}
static Tensor real_fft(const Tensor& input,at::ScalarType dtype) {
    return active?rfft(input,dtype==at::kHalf?0:1):at::fft_rfft2(input);
}
static Tensor real_ifft(const Tensor& input,int64_t h,int64_t w,at::ScalarType dtype) {
    return active?irfft(input,h,w,dtype==at::kHalf?0:1):at::fft_irfft2(input,at::IntArrayRef({h,w}));
}
#ifdef CONVERSE2D_WITH_CUDA
static Tensor inference_inverse(Tensor& input,int64_t h,int64_t w,at::ScalarType dtype,Tensor destination=Tensor()) {
    if(active&&eligible(input,h,w,dtype)) {
        auto result=Irfft::apply(input,h,w,dtype==at::kHalf?int64_t(0):int64_t(1));
        if(destination.defined()) { destination.copy_(result); return destination; }
        return result.to(dtype);
    }
    return converse2d::c2r::inverse(input,h,w,destination,dtype);
}
#endif
} // namespace converse2d::native_fft
