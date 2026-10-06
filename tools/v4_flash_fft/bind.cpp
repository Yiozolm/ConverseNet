#include <torch/extension.h>

at::Tensor flash_s1_forward(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, int64_t pad, int64_t variant);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &flash_s1_forward); }
