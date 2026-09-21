#pragma once
#include <ATen/ATen.h>
#include <string>
at::Tensor converse2d_forward(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                             int64_t, double, const std::string&);
