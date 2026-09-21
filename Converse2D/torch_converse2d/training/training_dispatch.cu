#include "training.h"
#include "launchers.cuh"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <vector>

namespace {
using I=int64_t;
// The scale-one candidate only benefits the measured larger spatial grids.
// Keep the same gate in forward and backward; small grids retain the generic
// kernels. The candidate also supports the internal FP64 spectral checks.
bool use_scale1(I H,I W,I s) { return s==1 && H*W>=65536; }

at::Tensor plain(const at::Tensor& t) { return t.resolve_conj().resolve_neg().contiguous(); }
}
std::vector<at::Tensor> converse_training_forward_cuda(const at::Tensor& fy,const at::Tensor& fx0,
    const at::Tensor& fb,const at::Tensor& lambda,int64_t H,int64_t W,int64_t s) {
    auto y=plain(fy),p=plain(fx0),k=plain(fb),l=plain(lambda);
    auto q=at::empty(y.sizes(),y.options()),d=at::empty(y.sizes(),l.options());
    auto out=at::empty(p.sizes(),p.options());
    auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
    if(use_scale1(H,W,s)) launch_training_scale1_forward(y,p,k,l,out,q,d,H,W,s,stream);
    else launch_training_generic_forward(y,p,k,l,out,q,d,H,W,s,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,q,d};
}

std::vector<at::Tensor> converse_training_backward_cuda(const at::Tensor& grad,const at::Tensor& fx0,
    const at::Tensor& fb,const at::Tensor& q0,const at::Tensor& d0,
    int64_t H,int64_t W,int64_t s,bool need_y,bool need_p,bool need_k,bool need_l) {
    if(!(need_y||need_p||need_k||need_l)) return {at::Tensor(),at::Tensor(),at::Tensor(),at::Tensor()};
    auto g=plain(grad),p=plain(fx0),k=plain(fb),q=plain(q0),d=plain(d0);
    const bool scale1=use_scale1(H,W,s);
    const bool no_broadcast=k.size(0)==p.size(0)&&k.size(1)==p.size(1);
    const bool reduce_filter=need_k&&!no_broadcast;
    // The fused s=1 kernel can keep r/gd in registers unless they are returned
    // or needed by a separate broadcast-filter reduction.
    auto r=!scale1||need_y||reduce_filter?at::empty(q.sizes(),q.options()):at::Tensor();
    auto gd=(scale1?(need_l||reduce_filter):(need_k||need_l))?
        at::empty(d.sizes(),d.options()):at::Tensor();
    auto gp=need_p?at::empty(p.sizes(),p.options()):at::Tensor();
    auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();
    auto stream=c10::cuda::getCurrentCUDAStream(g.get_device());
    if(scale1) launch_training_scale1_backward(g,p,k,q,d,r,gd,gp,gk,H,W,s,need_p,need_k,no_broadcast,reduce_filter,stream);
    else launch_training_generic_backward(g,p,k,q,d,r,gd,gp,gk,H,W,s,need_p,need_k,no_broadcast,reduce_filter,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gl=need_l?gd.sum({0,2,3},true):at::Tensor();
    return {need_y?r:at::Tensor(),gp,gk,gl};
}
