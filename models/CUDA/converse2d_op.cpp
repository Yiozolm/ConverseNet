#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> converse2d_cuda_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int scale, float eps
);

std::vector<torch::Tensor> converse2d_cuda_backward(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int scale,
    const std::vector<torch::Tensor>& saved_tensors
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &converse2d_cuda_forward, "Converse2D forward (CUDA)");
    m.def("backward", &converse2d_cuda_backward, "Converse2D backward (CUDA)");
}