#include <torch/extension.h>

at::Tensor flash_s1_forward(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, int64_t pad, int64_t variant,
                            int64_t mode);

std::vector<at::Tensor> flash_s1_backward(const at::Tensor& x, const at::Tensor& g, const at::Tensor& k,
                                          const at::Tensor& l, int64_t pad, int64_t mode);

bool flash_supported(int64_t h0, int64_t w0, int64_t scale, int64_t pad, int64_t device);
int64_t flash_smem_capacity(int64_t device);
at::Tensor flash_scaled_forward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& k, const at::Tensor& l,
                                int64_t scale);
std::vector<at::Tensor> flash_scaled_backward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& g,
                                              const at::Tensor& k, const at::Tensor& l, int64_t scale);
at::Tensor flash_debug_fft2(const at::Tensor& x);
bool flash_half_supported(int64_t h0, int64_t w0, int64_t scale, int64_t pad, int64_t device);
at::Tensor flash_half_forward(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, int64_t pad, int64_t mode);
at::Tensor flash_half_scaled_forward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& k,
                                     const at::Tensor& l, int64_t scale);
int64_t flash_half_occupancy(int64_t side, int64_t device);
at::Tensor flash_debug_rfft2(const at::Tensor& x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // mode: 0 circular, 1 replicate, 2 reflect, 3 zeros (s1 padding of the plane).
    m.def("forward", &flash_s1_forward, py::arg("x"), py::arg("k"), py::arg("l"), py::arg("pad"), py::arg("variant"),
          py::arg("mode") = 0);
    m.def("backward", &flash_s1_backward, py::arg("x"), py::arg("g"), py::arg("k"), py::arg("l"), py::arg("pad"),
          py::arg("mode") = 0);
    m.def("supported", &flash_supported);
    m.def("smem_capacity", &flash_smem_capacity);
    m.def("scaled_forward", &flash_scaled_forward);
    m.def("scaled_backward", &flash_scaled_backward);
    m.def("debug_fft2", &flash_debug_fft2);
    // Half-spectrum inference forward (no_grad): k is the rfft2 of the PSF.
    m.def("half_supported", &flash_half_supported);
    m.def("half_forward", &flash_half_forward, py::arg("x"), py::arg("k"), py::arg("l"), py::arg("pad"),
          py::arg("mode") = 0);
    m.def("half_scaled_forward", &flash_half_scaled_forward);
    m.def("half_occupancy", &flash_half_occupancy);
    m.def("debug_rfft2", &flash_debug_rfft2);
}
