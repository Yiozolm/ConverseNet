// Content-addressed experimental namespace substituted by loader.py.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>

using at::Tensor;
std::vector<Tensor> converse_training_shared_s1_forward_cuda(const Tensor&,const Tensor&,const Tensor&);
std::vector<Tensor> converse_training_shared_s1_backward_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    const Tensor&,bool,bool,bool);

namespace converse2d::shared_s1 {
// s=1 with SAME observation and prior only. FFT/IFFT, kernel preparation
// and lambda parameterization remain outside this spectral custom Function.
static Tensor reference(const Tensor& y,const Tensor& k,const Tensor& lambda) {
    auto d=at::real(k).square()+at::imag(k).square()+lambda;
    return ((k.conj()+lambda)/d)*y;
}

class SharedTransfer : public torch::autograd::Function<SharedTransfer> {
public:
    static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor y,Tensor k,Tensor lambda) {
        auto values=converse_training_shared_s1_forward_cuda(y,k,lambda);
        ctx->save_for_backward({y,k,lambda,values[1],values[2]});
        return values[0];
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list gradients) {
        auto saved=ctx->get_saved_variables();
        c10::cuda::CUDAGuard guard(saved[0].device());
        torch::autograd::variable_list result(3);
        if(at::GradMode::is_enabled()) {
            std::vector<Tensor> proxy,requested;
            for(size_t i=0;i<3;++i) {
                proxy.push_back(saved[i].view_as(saved[i]));
                if(ctx->needs_input_grad(i)) requested.push_back(proxy.back());
            }
            auto output=reference(proxy[0],proxy[1],proxy[2]);
            auto values=torch::autograd::grad({output},requested,{gradients[0]},true,true,true);
            size_t j=0;
            for(size_t i=0;i<3;++i) if(ctx->needs_input_grad(i)) result[i]=values[j++];
        } else {
            auto values=converse_training_shared_s1_backward_cuda(gradients[0],saved[0],saved[1],saved[3],saved[4],
                ctx->needs_input_grad(0),ctx->needs_input_grad(1),ctx->needs_input_grad(2));
            for(size_t i=0;i<3;++i) result[i]=values[i];
        }
        return result;
    }
};

static Tensor transfer(Tensor y,Tensor k,Tensor lambda) {
    TORCH_CHECK(y.dim()==4&&y.numel()>0,"y must be a nonempty BCHW-half spectrum");
    const auto B=y.size(0),C=y.size(1);
    TORCH_CHECK(k.dim()==4&&(k.size(0)==1||k.size(0)==B)&&(k.size(1)==1||k.size(1)==C)&&
        k.size(2)==y.size(2)&&k.size(3)==y.size(3),"invalid kernel spectrum shape");
    TORCH_CHECK(lambda.sizes()==at::IntArrayRef({1,C,1,1}),"lambda must have shape (1,C,1,1)");
    TORCH_CHECK(y.scalar_type()==at::kComplexFloat||y.scalar_type()==at::kComplexDouble,"expected complex64 or complex128");
    TORCH_CHECK(k.scalar_type()==y.scalar_type()&&
        lambda.scalar_type()==(y.scalar_type()==at::kComplexFloat?at::kFloat:at::kDouble),"spectral dtype mismatch");
    TORCH_CHECK(y.device()==k.device()&&y.device()==lambda.device(),"spectral device mismatch");
    TORCH_CHECK(y.is_cuda(),"isolated shared-s1 candidate requires CUDA");
    c10::cuda::CUDAGuard guard(y.device());
    return SharedTransfer::apply(y,k,lambda);
}
} // namespace converse2d::shared_s1

TORCH_LIBRARY_FRAGMENT(converse2d,m) {
    m.def("_training_shared_s1(Tensor y, Tensor k, Tensor regularizer) -> Tensor",&converse2d::shared_s1::transfer);
}
