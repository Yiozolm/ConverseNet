#include "pointwise_wgrad.h"
#include <climits>
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

namespace converse2d::peripheral {
namespace {
bool plain_cuda(const at::Tensor& t) {
    return t.is_cuda() && t.layout() == c10::kStrided && t.is_contiguous() && !t.is_neg() && !t.is_conj() &&
           t.scalar_type() == at::kFloat;
}
} // namespace

at::Tensor pointwise_wgrad(const at::Tensor& grad_output, const at::Tensor& input) {
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat && input.scalar_type() == at::kFloat,
                "pointwise weight gradient accepts FP32 only");
    TORCH_CHECK(grad_output.dim() == 4 && input.dim() == 4 && grad_output.size(0) == input.size(0) &&
                    grad_output.size(2) == input.size(2) && grad_output.size(3) == input.size(3),
                "expected NCHW grad_output and input with equal batch and spatial sizes");
    TORCH_CHECK(grad_output.device() == input.device(), "device mismatch");
#ifdef CONVERSE2D_WITH_CUDA
    TORCH_CHECK(plain_cuda(grad_output) && plain_cuda(input),
                "pointwise weight gradient requires contiguous CUDA FP32 tensors without lazy value flags");
    TORCH_CHECK(input.size(1) >= 1 && grad_output.size(1) >= 1 && input.size(2) * input.size(3) >= 1 &&
                    input.size(0) >= 1 && input.size(2) * input.size(3) <= INT_MAX &&
                    input.size(1) * grad_output.size(1) <= INT_MAX / 2 && input.numel() <= INT_MAX &&
                    grad_output.numel() <= INT_MAX,
                "pointwise weight gradient geometry out of range");
    c10::cuda::CUDAGuard guard(input.device());
    return pointwise_wgrad_cuda(grad_output, input);
#else
    TORCH_CHECK(false, "pointwise weight gradient requires the CUDA build");
#endif
}

} // namespace converse2d::peripheral
