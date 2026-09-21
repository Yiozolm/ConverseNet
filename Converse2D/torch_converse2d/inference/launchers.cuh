#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
void launch_inference_scale1(const at::Tensor& y,const at::Tensor& prior,const at::Tensor& kernel,const at::Tensor& denom,const at::Tensor& lambda,at::Tensor& q,at::Tensor& out,int64_t H,int64_t W,int64_t s,bool half,cudaStream_t stream);
void launch_inference_scale2(const at::Tensor& y,const at::Tensor& prior,const at::Tensor& kernel,const at::Tensor& denom,const at::Tensor& lambda,at::Tensor& q,at::Tensor& out,int64_t H,int64_t W,int64_t s,bool half,cudaStream_t stream);
void launch_inference_scale3(const at::Tensor& y,const at::Tensor& prior,const at::Tensor& kernel,const at::Tensor& denom,const at::Tensor& lambda,at::Tensor& q,at::Tensor& out,int64_t H,int64_t W,int64_t s,bool half,cudaStream_t stream);
void launch_inference_generic(const at::Tensor& y,const at::Tensor& prior,const at::Tensor& kernel,const at::Tensor& denom,const at::Tensor& lambda,at::Tensor& q,at::Tensor& out,int64_t H,int64_t W,int64_t s,bool half,cudaStream_t stream);
