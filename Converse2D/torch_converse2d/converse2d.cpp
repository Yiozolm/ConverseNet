#include <torch/extension.h>
#include "operator.h"
#include "training/training.h"
#include "training/full_spectrum/full_fusion.h"
#include "inference/cache.h"
TORCH_LIBRARY(converse2d, m) {
    m.def("forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int scale, float eps=1e-5, str variant='v7') -> Tensor");
    m.def("_training_spectral(Tensor y, Tensor p, Tensor k, Tensor regularizer, int H, int W, int scale) -> Tensor");
#ifdef CONVERSE2D_WITH_CUDA
    m.def("_training_full_spectral(Tensor y, Tensor p, Tensor k, Tensor regularizer, int scale) -> Tensor");
#endif
    m.def("_begin_training_cache() -> ()");
    m.def("_begin_training_cache_for(Tensor[] weights) -> ()");
    m.def("_end_training_cache() -> int[]");
    m.def("clear_cache() -> ()");
    m.def("supports_cuda_graphs() -> bool");
    m.def("begin_graph_cache() -> ()");
    m.def("end_graph_cache() -> Tensor[]");
}
TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m) {
    m.impl("forward", TORCH_FN(converse2d_forward));
    m.impl("_training_spectral", TORCH_FN(converse2d::training::spectral));
#ifdef CONVERSE2D_WITH_CUDA
    m.impl("_training_full_spectral", TORCH_FN(converse2d::full_training::full_spectral));
#endif
    m.impl("_begin_training_cache", TORCH_FN(begin_training_cache));
    m.impl("_begin_training_cache_for", TORCH_FN(begin_training_cache_for));
    m.impl("_end_training_cache", TORCH_FN(end_training_cache));
    m.impl("clear_cache", TORCH_FN(clear_fb_cache));
    m.impl("supports_cuda_graphs", TORCH_FN(supports_cuda_graphs));
    m.impl("begin_graph_cache", TORCH_FN(begin_graph_cache));
    m.impl("end_graph_cache", TORCH_FN(end_graph_cache));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
