#include <torch/extension.h>

at::Tensor nearest_k2_compensated_fused_lambda_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &nearest_k2_compensated_fused_lambda_cuda,
        "Compensated FP32 k2/s2 solve with fused sigmoid regularizer; explicit nearest prior");
}
