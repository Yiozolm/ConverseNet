#include <torch/extension.h>

at::Tensor flash_s1_forward(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, int64_t pad, int64_t variant);

std::vector<at::Tensor> flash_s1_backward(const at::Tensor& x, const at::Tensor& g, const at::Tensor& k,
                                          const at::Tensor& l, int64_t pad);

bool flash_supported(int64_t h0, int64_t w0, int64_t scale, int64_t pad, int64_t device);
int64_t flash_smem_capacity(int64_t device);
at::Tensor flash_scaled_forward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& k, const at::Tensor& l,
                                int64_t scale);
std::vector<at::Tensor> flash_scaled_backward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& g,
                                              const at::Tensor& k, const at::Tensor& l, int64_t scale);
at::Tensor flash_debug_fft2(const at::Tensor& x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_s1_forward);
    m.def("backward", &flash_s1_backward);
    m.def("supported", &flash_supported);
    m.def("smem_capacity", &flash_smem_capacity);
    m.def("scaled_forward", &flash_scaled_forward);
    m.def("scaled_backward", &flash_scaled_backward);
    m.def("debug_fft2", &flash_debug_fft2);
}
