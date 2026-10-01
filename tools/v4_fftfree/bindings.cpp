#include <torch/extension.h>

at::Tensor nearest_k2_cuda(at::Tensor x, at::Tensor weight, at::Tensor denominator);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &nearest_k2_cuda,
        "Inference-only FP32 k2/s2 solve; this explicit API defines its prior as nearest(x)");
}
