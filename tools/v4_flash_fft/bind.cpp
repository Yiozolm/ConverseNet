#include <torch/extension.h>

at::Tensor flash_s1_forward(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, int64_t pad, int64_t variant);

std::vector<at::Tensor> flash_s1_backward(const at::Tensor& x, const at::Tensor& g, const at::Tensor& k,
                                          const at::Tensor& l, int64_t pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_s1_forward);
    m.def("backward", &flash_s1_backward);
}
