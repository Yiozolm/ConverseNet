// Independent namespace substituted by loader.py; no production registration.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <climits>

using at::Tensor;
std::vector<Tensor> nearest_forward_cuda(const Tensor&,const Tensor&,const Tensor&,
    const Tensor&,const Tensor&,int64_t,int64_t,int64_t);
std::vector<Tensor> nearest_backward_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,int64_t,bool,bool,bool);

namespace nearest_experiment {
// Copied from production, retaining its Hermitian extension convention for
// arbitrary (not necessarily real-FFT-derived) complex half-spectrum inputs.
static Tensor alias_mean(const Tensor& a,int64_t s) {
    if(s==1) return a;
    const auto h=a.size(-2)/s,w=a.size(-1)/s;
    return a.reshape({a.size(0),a.size(1),s,h,s,w}).mean({2,4});
}
static Tensor full_spectrum(const Tensor& half,int64_t width) {
    const int64_t end=(width+1)/2;
    auto tail=half.slice(-1,1,end).flip({-2,-1}).roll({1},{-2});
    if(half.is_complex()) tail=tail.conj();
    return at::cat({half,tail},-1);
}
static Tensor nearest_prior(const Tensor& y,const Tensor& ph,const Tensor& pw,
    int64_t H,int64_t W,int64_t s) {
    const auto ws=W*s;
    // Higher-order fallback only: repeat's reduction avoids index_select's
    // non-reentrant CUDA scatter VJP. Preserve canonical LR Hermitian reads
    // for arbitrary complex boundaries, without changing first-order CUDA.
    auto lifted=full_spectrum(y,W).repeat({1,1,s,s}).slice(-1,0,ws/2+1);
    return lifted*(ph.unsqueeze(1)*pw.slice(0,0,ws/2+1).unsqueeze(0));
}
static Tensor reference(const Tensor& y,const Tensor& k,const Tensor& lambda,
    const Tensor& ph,const Tensor& pw,int64_t H,int64_t W,int64_t s) {
    auto p=nearest_prior(y,ph,pw,H,W,s);
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

class NearestSolve : public torch::autograd::Function<NearestSolve> {
public:
    static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor y,Tensor k,Tensor lambda,
        Tensor ph,Tensor pw,int64_t H,int64_t W,int64_t s) {
        auto values=nearest_forward_cuda(y,k,lambda,ph,pw,H,W,s);
        ctx->save_for_backward({y,k,lambda,ph,pw,values[1],values[2]});
        ctx->saved_data["H"]=H; ctx->saved_data["W"]=W; ctx->saved_data["s"]=s;
        return values[0];
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grads) {
        auto saved=ctx->get_saved_variables();
        const auto H=ctx->saved_data["H"].toInt(),W=ctx->saved_data["W"].toInt(),s=ctx->saved_data["s"].toInt();
        c10::cuda::CUDAGuard guard(saved[0].device());
        torch::autograd::variable_list result(8);
        if(at::GradMode::is_enabled()) {
            std::vector<Tensor> proxy,requested;
            for(size_t i=0;i<3;++i) {
                proxy.push_back(saved[i].view_as(saved[i]));
                if(ctx->needs_input_grad(i)) requested.push_back(proxy.back());
            }
            auto out=reference(proxy[0],proxy[1],proxy[2],saved[3],saved[4],H,W,s);
            auto values=torch::autograd::grad({out},requested,{grads[0]},true,true,true);
            size_t j=0;
            for(size_t i=0;i<3;++i) if(ctx->needs_input_grad(i)) result[i]=values[j++];
        } else {
            auto values=nearest_backward_cuda(grads[0],saved[0],saved[1],saved[5],saved[6],saved[3],saved[4],H,W,s,
                ctx->needs_input_grad(0),ctx->needs_input_grad(1),ctx->needs_input_grad(2));
            for(size_t i=0;i<3;++i) result[i]=values[i];
        }
        return result;
    }
};

static Tensor spectral(Tensor y,Tensor k,Tensor lambda,Tensor ph,Tensor pw,int64_t H,int64_t W,int64_t s) {
    TORCH_CHECK((s==1||s==3)&&H>0&&W>0&&H<=INT64_MAX/s&&W<=INT64_MAX/s,"only valid scale 1 or 3 dimensions supported");
    TORCH_CHECK(y.dim()==4&&y.numel()>0&&y.size(2)==H&&y.size(3)==W/2+1,"invalid y spectrum shape");
    const auto B=y.size(0),C=y.size(1);
    TORCH_CHECK(k.dim()==4&&(k.size(0)==1||k.size(0)==B)&&(k.size(1)==1||k.size(1)==C)&&
        k.size(2)==H*s&&k.size(3)==W*s/2+1,"invalid kernel spectrum shape");
    TORCH_CHECK(lambda.sizes()==at::IntArrayRef({1,C,1,1}),"invalid lambda shape");
    TORCH_CHECK(ph.sizes()==at::IntArrayRef({H*s})&&pw.sizes()==at::IntArrayRef({W*s}),"invalid phase shape");
    TORCH_CHECK(!ph.requires_grad()&&!pw.requires_grad(),"phase must not require gradients");
    TORCH_CHECK(y.scalar_type()==at::kComplexFloat||y.scalar_type()==at::kComplexDouble,"invalid spectrum dtype");
    TORCH_CHECK(k.scalar_type()==y.scalar_type()&&ph.scalar_type()==y.scalar_type()&&pw.scalar_type()==y.scalar_type()&&
        lambda.scalar_type()==(y.scalar_type()==at::kComplexFloat?at::kFloat:at::kDouble),"spectral dtype mismatch");
    TORCH_CHECK(k.device()==y.device()&&lambda.device()==y.device()&&ph.device()==y.device()&&pw.device()==y.device(),
        "spectral device mismatch");
    TORCH_CHECK(y.is_cuda(),"isolated nearest candidate requires CUDA; use Python reference for CPU");
    c10::cuda::CUDAGuard guard(y.device());
    return NearestSolve::apply(y,k,lambda,ph,pw,H,W,s);
}
} // namespace nearest_experiment

TORCH_LIBRARY(nearest_experiment,m) {
    m.def("_training_nearest_spectral(Tensor y, Tensor k, Tensor regularizer, Tensor phaseH, Tensor phaseW, int H, int W, int s) -> Tensor",
        &nearest_experiment::spectral);
}
