#pragma once
#include <ATen/ATen.h>
#include <vector>
at::Tensor prepare_training_kernel(const at::Tensor&, int64_t, int64_t);
at::Tensor training_spectrum_cast(const at::Tensor&);
at::Tensor training_spectrum(const at::Tensor&, const at::Tensor&, int64_t, int64_t);
void begin_training_cache();
void begin_training_cache_for(std::vector<at::Tensor>);
std::vector<int64_t> end_training_cache();
namespace converse2d::training {
at::Tensor spectral(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t);
}
#ifdef CONVERSE2D_WITH_CUDA
std::vector<at::Tensor> converse_training_forward_cuda(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t);
std::vector<at::Tensor> converse_training_backward_cuda(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,bool,bool,bool,bool);
#endif
