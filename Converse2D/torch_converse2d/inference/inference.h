#pragma once
#include <ATen/ATen.h>
#include <utility>
std::pair<at::Tensor,at::Tensor> spectrum(const at::Tensor&,const at::Tensor&,
    int64_t,int64_t,int64_t,bool);
#ifdef CONVERSE2D_WITH_CUDA
at::Tensor converse_spectral_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,bool);
at::Tensor converse_psf_cuda(const at::Tensor&,int64_t,int64_t);
#endif
