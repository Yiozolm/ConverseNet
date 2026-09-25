#pragma once
#include <ATen/ATen.h>

namespace converse2d::full_training {
// This per-backward state owns the materialized inputs across both launches.
// intermediate_gy aliases a required output: gy if present, otherwise gp.
// No differentiable spectrum or preparation is reused across calls.
struct Scale1BatchAdjoint {
    at::Tensor gy,gp,gd,intermediate_gy;
    at::Tensor g,p,k,q;
};

Scale1BatchAdjoint full_scale1_batch_prepare_cuda(
    at::Tensor g,at::Tensor p,at::Tensor k,at::Tensor q,at::Tensor d,
    bool need_independent_y,bool need_prior);
at::Tensor full_scale1_batch_kernel_cuda(const Scale1BatchAdjoint& stage,
                                         at::Tensor power,bool shared);
}
