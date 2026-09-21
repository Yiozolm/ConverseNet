#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
void launch_training_scale1_forward(const at::Tensor& y,const at::Tensor& p,const at::Tensor& k,const at::Tensor& l,at::Tensor& out,at::Tensor& q,at::Tensor& d,int64_t H,int64_t W,int64_t s,cudaStream_t stream);
void launch_training_scale1_backward(const at::Tensor& g,const at::Tensor& p,const at::Tensor& k,const at::Tensor& q,const at::Tensor& d,at::Tensor& r,at::Tensor& gd,at::Tensor& gp,at::Tensor& gk,int64_t H,int64_t W,int64_t s,bool need_p,bool need_k,bool no_broadcast,bool reduce_filter,cudaStream_t stream);
void launch_training_generic_forward(const at::Tensor& y,const at::Tensor& p,const at::Tensor& k,const at::Tensor& l,at::Tensor& out,at::Tensor& q,at::Tensor& d,int64_t H,int64_t W,int64_t s,cudaStream_t stream);
void launch_training_generic_backward(const at::Tensor& g,const at::Tensor& p,const at::Tensor& k,const at::Tensor& q,const at::Tensor& d,at::Tensor& r,at::Tensor& gd,at::Tensor& gp,at::Tensor& gk,int64_t H,int64_t W,int64_t s,bool need_p,bool need_k,bool no_broadcast,bool reduce_filter,cudaStream_t stream);
