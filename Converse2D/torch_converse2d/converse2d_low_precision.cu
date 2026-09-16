// FP16/BF16 storage kernels. Arithmetic before the final cast stays FP32.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

template <typename T>
__global__ void normalize_cast(const float* input, T* output, int64_t n, float factor) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = T(input[i] * factor);
}

at::Tensor converse_normalize_cast_cuda(const at::Tensor& input, double factor, at::ScalarType dtype,
                                       at::Tensor output) {
    TORCH_INTERNAL_ASSERT(input.is_cuda() && input.is_contiguous() && input.scalar_type() == at::kFloat);
    if (!output.defined()) output = at::empty(input.sizes(), input.options().dtype(dtype));
    TORCH_INTERNAL_ASSERT(output.sizes() == input.sizes() && output.is_contiguous() &&
                         output.device() == input.device() && output.scalar_type() == dtype);
    auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    AT_DISPATCH_SWITCH(dtype, "converse_normalize_cast",
        AT_DISPATCH_CASE(at::kHalf, [&] {
            normalize_cast<scalar_t><<<(input.numel()+255)/256,256,0,stream>>>(
                input.data_ptr<float>(), output.data_ptr<scalar_t>(), input.numel(), float(factor));
        })
        AT_DISPATCH_CASE(at::kBFloat16, [&] {
            normalize_cast<scalar_t><<<(input.numel()+255)/256,256,0,stream>>>(
                input.data_ptr<float>(), output.data_ptr<scalar_t>(), input.numel(), float(factor));
        })
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

