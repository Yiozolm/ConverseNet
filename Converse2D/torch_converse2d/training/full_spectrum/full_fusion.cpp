#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <ATen/ExpandUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
namespace converse2d::full_training {
using at::Tensor;
std::vector<Tensor> full_prepare_cuda(Tensor p,Tensor k);
std::vector<Tensor> full_output_cuda(Tensor y,Tensor pm,Tensor p,Tensor k,Tensor d,int64_t s);
std::vector<Tensor> full_adjoint_output_cuda(Tensor g,Tensor k,Tensor q,int64_t s);
std::vector<Tensor> full_adjoint_div_cuda(Tensor t,Tensor q,Tensor d);
std::vector<Tensor> full_adjoint_prediction_cuda(Tensor g,Tensor p,Tensor k,Tensor gm,int64_t s,bool shared);
Tensor full_adjoint_kernel_cuda(Tensor k,Tensor direct,Tensor prediction,Tensor power,int64_t s);
std::vector<Tensor> full_scale1_forward_cuda(Tensor y,Tensor p,Tensor k,Tensor l);
std::vector<Tensor> full_scale1_adjoint_cuda(Tensor g,Tensor p,Tensor k,Tensor q,Tensor d,bool shared);
std::vector<Tensor> full_scale2_forward_cuda(Tensor y,Tensor p,Tensor k,Tensor l);
std::vector<Tensor> full_scale2_adjoint_cuda(Tensor g,Tensor p,Tensor k,Tensor q,Tensor d);

bool scale2_fusion_eligible(const Tensor& y,const Tensor& p,int64_t s) {
    // A non-reduced contiguous W dimension keeps all four aliases within one
    // ATen reduction thread. Very large iterators can split reductions.
    return s==2 && y.size(3)>1 && p.numel()<=INT32_MAX/p.element_size();
}

Tensor aliases(Tensor t,int64_t s,bool mean) {
    if(s==1)return t;
    auto v=t.reshape({t.size(0),t.size(1),s,t.size(2)/s,s,t.size(3)/s});
    return mean?v.mean({2,4}):v.sum({2,4});
}
Tensor reference(Tensor y,Tensor p,Tensor k,Tensor l,int64_t s) {
    auto power=at::real(k).square()+at::imag(k).square();
    auto q=(y-aliases(k*p,s,true))/(aliases(power,s,true)+l);
    return p+k.conj()*(s==1?q:q.repeat({1,1,s,s}));
}

class FullSolve:public torch::autograd::Function<FullSolve> {
public:
 static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor y,Tensor p,Tensor k,Tensor l,int64_t s) {
    if(s==1) {
        auto out=full_scale1_forward_cuda(y,p,k,l);
        ctx->save_for_backward({y,p,k,l,out[1],out[2]});ctx->saved_data["s"]=s;
        ctx->saved_data["shared"]=y.is_same(p);
        return out[0];
    }
    if(scale2_fusion_eligible(y,p,s)) {
        auto out=full_scale2_forward_cuda(y,p,k,l);
        ctx->save_for_backward({y,p,k,l,out[1],out[2]});ctx->saved_data["s"]=s;
        ctx->saved_data["shared"]=false;
        return out[0];
    }
    auto prepared=full_prepare_cuda(p,k);
    auto pm=aliases(prepared[0],s,true),km=aliases(prepared[1],s,true);
    auto d=km+l;
    auto out=full_output_cuda(y,pm,p,k,d,s);
    ctx->save_for_backward({y,p,k,l,out[1],d});ctx->saved_data["s"]=s;
    ctx->saved_data["shared"]=s==1&&y.is_same(p);
    return out[0];
 }
 static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list incoming) {
    auto a=ctx->get_saved_variables();const auto s=ctx->saved_data["s"].toInt();const bool shared=ctx->saved_data["shared"].toBool();
    c10::cuda::CUDAGuard guard(a[0].device());
    torch::autograd::variable_list result(5);
    if(at::GradMode::is_enabled()) {
        if(shared){
            auto x=a[0].view_as(a[0]),k=a[2].view_as(a[2]),l=a[3].view_as(a[3]);
            std::vector<Tensor> proxy={x,k,l},req;std::vector<int> slot={0,2,3};
            for(int j=0;j<3;++j)if(ctx->needs_input_grad(slot[j]))req.push_back(proxy[j]);
            auto values=torch::autograd::grad({reference(x,x,k,l,s)},req,{incoming[0]},true,true,true);
            int j=0;for(int i:slot)if(ctx->needs_input_grad(i))result[i]=values[j++];
            return result;
        }
        std::vector<Tensor> proxy,req;
        for(int i=0;i<4;++i){proxy.push_back(a[i].view_as(a[i]));if(ctx->needs_input_grad(i))req.push_back(proxy.back());}
        auto z=reference(proxy[0],proxy[1],proxy[2],proxy[3],s);
        auto grads=torch::autograd::grad({z},req,{incoming[0]},true,true,true);
        int j=0;for(int i=0;i<4;++i)if(ctx->needs_input_grad(i))result[i]=grads[j++];
        return result;
    }
    if(s==1) {
        // Keep every broadcast reduction and its input layout unchanged. Only
        // pointwise work moves across the former kernel launch boundaries.
        auto pointwise=full_scale1_adjoint_cuda(incoming[0],a[1],a[2],a[4],a[5],shared);
        auto gd=at::sum_to(at::real(pointwise[4]),a[5].sizes());
        auto gl=at::sum_to(gd,a[3].sizes());
        auto gk=pointwise[5];
        if(!gk.defined()) {
            auto direct=at::sum_to(pointwise[2],a[2].sizes()).conj();
            auto power=at::sum_to(gd,at::IntArrayRef({a[2].size(0),a[2].size(1),a[0].size(2),a[0].size(3)}));
            auto gkp=at::sum_to(pointwise[3],a[2].sizes());
            gk=full_adjoint_kernel_cuda(a[2],direct,gkp,power,s);
        }
        std::vector<Tensor> grads={pointwise[0],pointwise[1],gk,gl};
        for(int i=0;i<4;++i)if(ctx->needs_input_grad(i))result[i]=grads[i];
        if(shared){result[0]=ctx->needs_input_grad(0)?pointwise[1]:Tensor();result[1]=Tensor();}
        return result;
    }
    if(scale2_fusion_eligible(a[0],a[1],s)) {
        auto pointwise=full_scale2_adjoint_cuda(incoming[0],a[1],a[2],a[4],a[5]);
        auto direct=at::sum_to(pointwise[2],a[2].sizes()).conj();
        auto gd=at::sum_to(at::real(pointwise[4]),a[5].sizes());
        auto gl=at::sum_to(gd,a[3].sizes());
        auto power=at::sum_to(gd,at::IntArrayRef({a[2].size(0),a[2].size(1),a[0].size(2),a[0].size(3)}));
        power=power/(s*s);
        auto gkp=at::sum_to(pointwise[3],a[2].sizes());
        auto gk=full_adjoint_kernel_cuda(a[2],direct,gkp,power,s);
        std::vector<Tensor> grads={pointwise[0],pointwise[1],gk,gl};
        for(int i=0;i<4;++i)if(ctx->needs_input_grad(i))result[i]=grads[i];
        return result;
    }
    auto prep=full_adjoint_output_cuda(incoming[0],a[2],a[4],s);
    auto t=aliases(prep[0],s,false);
    auto direct=at::sum_to(prep[1],a[2].sizes()).conj();
    auto div=full_adjoint_div_cuda(t,a[4],a[5]);
    auto gy=div[0];
    // Preserve the real-view stride and each original broadcast reduction.
    auto gd=at::sum_to(at::real(div[1]),a[5].sizes());
    auto gl=at::sum_to(gd,a[3].sizes());
    auto gpmean=-gy;
    auto power=at::sum_to(gd,at::IntArrayRef({a[2].size(0),a[2].size(1),a[0].size(2),a[0].size(3)}));
    if(s>1){gpmean=gpmean/(s*s);power=power/(s*s);}
    auto pred=full_adjoint_prediction_cuda(incoming[0],a[1],a[2],gpmean,s,shared);
    auto gkp=at::sum_to(pred[1],a[2].sizes());
    auto gk=full_adjoint_kernel_cuda(a[2],direct,gkp,power,s);
    std::vector<Tensor> grads={gy,pred[0],gk,gl};
    for(int i=0;i<4;++i)if(ctx->needs_input_grad(i))result[i]=grads[i];
    if(shared){result[0]=ctx->needs_input_grad(0)?pred[0]:Tensor();result[1]=Tensor();}
    return result;
 }
};

Tensor full_spectral(Tensor y,Tensor p,Tensor k,Tensor l,int64_t s) {
 TORCH_CHECK(y.is_cuda()&&y.dim()==4&&y.numel()>0,"expected nonempty CUDA spectrum");
 TORCH_CHECK(s>0&&s<=INT64_MAX/s&&y.size(2)<=INT64_MAX/s&&y.size(3)<=INT64_MAX/s,"invalid scale");
 TORCH_CHECK(p.dim()==4&&p.size(0)==y.size(0)&&p.size(1)==y.size(1)&&p.size(2)==y.size(2)*s&&p.size(3)==y.size(3)*s,"invalid prior dimensions");
 TORCH_CHECK(k.dim()==4&&(k.size(0)==1||k.size(0)==p.size(0))&&(k.size(1)==1||k.size(1)==p.size(1))&&k.size(2)==p.size(2)&&k.size(3)==p.size(3),"invalid kernel dimensions");
 TORCH_CHECK(l.sizes()==at::IntArrayRef({1,y.size(1),1,1}),"invalid regularizer dimensions");
 TORCH_CHECK(y.scalar_type()==at::kComplexFloat&&p.scalar_type()==y.scalar_type()&&k.scalar_type()==y.scalar_type()&&l.scalar_type()==at::kFloat,"invalid dtype");
 TORCH_CHECK(p.device()==y.device()&&k.device()==y.device()&&l.device()==y.device(),"device mismatch");
 c10::cuda::CUDAGuard guard(y.device());
 return FullSolve::apply(y,p,k,l,s);
}
} // namespace converse2d::full_training
