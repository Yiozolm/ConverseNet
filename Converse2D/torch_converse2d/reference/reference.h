#pragma once
#include <ATen/ATen.h>
namespace converse2d::reference {
at::Tensor spectral(const at::Tensor& fy,const at::Tensor& fx0,const at::Tensor& fb,
    const at::Tensor& invw,const at::Tensor& lambda,int64_t W,int64_t Ws,int64_t scale,bool real_fft);
}
