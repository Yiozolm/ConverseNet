#pragma once
#include <ATen/ATen.h>
namespace converse2d::full_training {
at::Tensor full_spectral(at::Tensor y, at::Tensor p, at::Tensor k, at::Tensor regularizer, int64_t scale);
at::Tensor spatial(at::Tensor x, at::Tensor prior, at::Tensor weight, at::Tensor bias, int64_t scale, double eps);
at::Tensor circular_pad_complex(at::Tensor x,int64_t padding);
at::Tensor circular_s1(at::Tensor x,at::Tensor weight,at::Tensor bias,int64_t padding,double eps);
at::Tensor real_crop(at::Tensor spectrum,int64_t padding);
}
