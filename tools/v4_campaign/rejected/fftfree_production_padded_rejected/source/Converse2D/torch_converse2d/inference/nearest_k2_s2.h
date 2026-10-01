#pragma once
#include <ATen/ATen.h>

#ifdef CONVERSE2D_WITH_CUDA
at::Tensor converse_nearest_k2_s2_cuda(at::Tensor x, at::Tensor weight,
                                     at::Tensor bias, double eps);
#endif
