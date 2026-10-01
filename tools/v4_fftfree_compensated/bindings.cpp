#include <torch/extension.h>

at::Tensor nearest_k2_compensated_cuda(at::Tensor x, at::Tensor weight, at::Tensor regularizer);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &nearest_k2_compensated_cuda,
        "Compensated FP32 k2/s2 solve; explicit nearest prior and per-channel regularizer");
}
