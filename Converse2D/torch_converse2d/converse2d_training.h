#pragma once

// Included after the shared alias helpers. FP32 and low-precision CUDA public
// training paths select this backend. FP64 support in the internal spectral
// entry exists for independent complex gradcheck/gradgradcheck validation.
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>

#ifdef CONVERSE2D_WITH_CUDA
std::vector<at::Tensor> converse_training_forward_cuda(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t);
std::vector<at::Tensor> converse_training_backward_cuda(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,bool,bool,bool,bool);
#endif

namespace converse2d::training {
static Tensor reference(const Tensor& y,const Tensor& p,const Tensor& k,const Tensor& lambda,
                        int64_t H,int64_t W,int64_t s) {
    auto power=at::real(k).square()+at::imag(k).square();
    auto denom=alias_mean(s>1?full_spectrum(power,W*s):power,s);
    auto prediction=k*p;
    if(s>1) prediction=full_spectrum(prediction,W*s);
    prediction=alias_mean(prediction,s);
    if(s>1) {
        prediction=prediction.slice(-1,0,W/2+1);
        denom=denom.slice(-1,0,W/2+1);
    }
    auto q=(y-prediction)/(denom+lambda);
    if(s>1) q=full_spectrum(q,W).repeat({1,1,s,s}).slice(-1,0,W*s/2+1);
    return p+k.conj()*q;
}

#ifdef CONVERSE2D_WITH_CUDA
class SpectralSolve : public torch::autograd::Function<SpectralSolve> {
public:
    static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor y,Tensor p,Tensor k,Tensor lambda,
                          int64_t H,int64_t W,int64_t s) {
        auto result=converse_training_forward_cuda(y,p,k,lambda,H,W,s);
        ctx->save_for_backward({y,p,k,lambda,result[1],result[2]});
        ctx->saved_data["H"]=H; ctx->saved_data["W"]=W; ctx->saved_data["s"]=s;
        return result[0];
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grads) {
        auto saved=ctx->get_saved_variables();
        const auto H=ctx->saved_data["H"].toInt(),W=ctx->saved_data["W"].toInt(),s=ctx->saved_data["s"].toInt();
        c10::cuda::CUDAGuard guard(saved[0].device());
        torch::autograd::variable_list result(7);
        if(at::GradMode::is_enabled()) {
            // Rebuild every dependency for higher derivatives. Separate views
            // are essential when y and p are the SAME FFT tensor (scale=1):
            // each formal argument needs its own VJP before autograd adds them.
            std::vector<Tensor> proxy,requested;
            for(size_t i=0;i<4;++i) {
                proxy.push_back(saved[i].view_as(saved[i]));
                if(ctx->needs_input_grad(i)) requested.push_back(proxy.back());
            }
            auto out=reference(proxy[0],proxy[1],proxy[2],proxy[3],H,W,s);
            auto values=torch::autograd::grad({out},requested,{grads[0]},true,true,true);
            size_t j=0;
            for(size_t i=0;i<4;++i) if(ctx->needs_input_grad(i)) result[i]=values[j++];
        } else {
            auto values=converse_training_backward_cuda(grads[0],saved[1],saved[2],saved[4],saved[5],H,W,s,
                ctx->needs_input_grad(0),ctx->needs_input_grad(1),ctx->needs_input_grad(2),ctx->needs_input_grad(3));
            for(size_t i=0;i<4;++i) result[i]=values[i];
        }
        return result;
    }
};
#endif

// Internal entry is validated as rigorously as the public spatial operator.
static Tensor spectral(Tensor y,Tensor p,Tensor k,Tensor lambda,int64_t H,int64_t W,int64_t s) {
    TORCH_CHECK(H>0&&W>0&&s>0&&H<=INT64_MAX/s&&W<=INT64_MAX/s,"invalid spectral dimensions");
    TORCH_CHECK(y.dim()==4&&y.numel()>0&&y.size(2)==H&&y.size(3)==W/2+1,"invalid y spectrum shape");
    const auto B=y.size(0),C=y.size(1);
    TORCH_CHECK(p.sizes()==at::IntArrayRef({B,C,H*s,W*s/2+1}),"invalid prior spectrum shape");
    TORCH_CHECK(k.dim()==4&&(k.size(0)==1||k.size(0)==B)&&(k.size(1)==1||k.size(1)==C)&&
                k.size(2)==H*s&&k.size(3)==W*s/2+1,"invalid kernel spectrum shape");
    TORCH_CHECK(lambda.sizes()==at::IntArrayRef({1,C,1,1}),"invalid lambda shape");
    TORCH_CHECK(y.scalar_type()==at::kComplexFloat||y.scalar_type()==at::kComplexDouble,"invalid spectrum dtype");
    TORCH_CHECK(p.scalar_type()==y.scalar_type()&&k.scalar_type()==y.scalar_type()&&
                lambda.scalar_type()==(y.scalar_type()==at::kComplexFloat?at::kFloat:at::kDouble),"spectral dtype mismatch");
    TORCH_CHECK(p.device()==y.device()&&k.device()==y.device()&&lambda.device()==y.device(),"spectral device mismatch");
#ifdef CONVERSE2D_WITH_CUDA
    if(y.is_cuda()) {
        c10::cuda::CUDAGuard guard(y.device());
        return SpectralSolve::apply(y,p,k,lambda,H,W,s);
    }
#endif
    return reference(y,p,k,lambda,H,W,s);
}

static Tensor forward(Tensor x,Tensor x0,Tensor weight,Tensor bias,int64_t scale,double eps,bool nearest) {
    const auto output_dtype=x.scalar_type();
    const bool same=nearest?scale==1:x.is_same(x0);
    x=x.to(at::kFloat).contiguous();
    if(nearest) x0=scale==1?x:at::upsample_nearest2d(x,at::IntArrayRef({x.size(2)*scale,x.size(3)*scale}),double(scale),double(scale));
    else x0=same?x:x0.to(at::kFloat).contiguous();
    weight=weight.to(at::kFloat); bias=bias.to(at::kFloat);
    const auto H=x.size(2),W=x.size(3),kh=weight.size(2),kw=weight.size(3);
    auto psf=at::constant_pad_nd(weight,{0,W*scale-kw,0,H*scale-kh},0);
    auto k=at::fft_rfft2(at::roll(psf,{-(kh/2),-(kw/2)},{-2,-1}));
    // The native FFT policy is exclusively for low-precision activations.
    // FP32 must keep its FFT precision even inside a native_fft() scope.
    const bool low=output_dtype==at::kHalf||output_dtype==at::kBFloat16;
    auto y=low?native_fft::real_fft(x,output_dtype):at::fft_rfft2(x);
    auto p=same?y:(low?native_fft::real_fft(x0,output_dtype):at::fft_rfft2(x0));
    auto lambda=at::sigmoid(bias-9.0)+eps;
    auto out=spectral(y,p,k,lambda,H,W,scale);
    return (low?native_fft::real_ifft(out,H*scale,W*scale,output_dtype):
                at::fft_irfft2(out,at::IntArrayRef({H*scale,W*scale}))).to(output_dtype);
}
} // namespace converse2d::training
