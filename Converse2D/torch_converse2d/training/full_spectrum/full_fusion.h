#pragma once
#include <ATen/ATen.h>
namespace converse2d::full_training {
at::Tensor full_spectral(at::Tensor y, at::Tensor p, at::Tensor k, at::Tensor regularizer, int64_t scale);
at::Tensor spatial(at::Tensor x, at::Tensor prior, at::Tensor weight, at::Tensor bias, int64_t scale, double eps);
}
