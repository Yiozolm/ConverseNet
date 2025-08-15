#include <torch/extension.h>

torch::Tensor converse2d_cuda_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int scale,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &converse2d_cuda_forward, "Converse2D forward (CUDA)");
}