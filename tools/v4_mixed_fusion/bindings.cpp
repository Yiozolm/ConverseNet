#include <torch/extension.h>
#include <string>

#ifndef MIXED_FUSION_BLOCK_THREADS
#error "The checked loader must set MIXED_FUSION_BLOCK_THREADS"
#endif

at::Tensor mixed_pad_cast_cuda(const at::Tensor& x, int64_t padding, const std::string& mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("pad_cast", &mixed_pad_cast_cuda,
        "Inference-only CUDA FP16/BF16 padding directly into one contiguous FP32 allocation",
        pybind11::arg("x"), pybind11::arg("padding"), pybind11::arg("mode"));
    module.attr("block_threads") = MIXED_FUSION_BLOCK_THREADS;
}
