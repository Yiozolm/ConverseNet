#include "inference/cache.h"
#include "operator.h"
#include "peripheral/alpha_residual.h"
#include "peripheral/layernorm.h"
#include "training/full_spectrum/full_fusion.h"
#include <torch/extension.h>
TORCH_LIBRARY(converse2d, m) {
    m.def("forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int scale, "
          "float eps=1e-5, str variant='v7') -> Tensor");
    m.def("_nearest_k2_s2(Tensor x, Tensor weight, Tensor bias, float eps=1e-5, "
          "str variant='v7') -> Tensor");
    m.def("_alpha_residual(Tensor alpha, Tensor branch, Tensor residual) -> "
          "Tensor");
    m.def("_channel_affine(Tensor scale, Tensor input, Tensor bias) -> Tensor");
    m.def("_channel_layernorm(Tensor input, Tensor weight, Tensor bias, float "
          "eps=1e-5) -> Tensor");
#ifdef CONVERSE2D_WITH_CUDA
    m.def("_training_full_spectral(Tensor y, Tensor p, Tensor k, Tensor "
          "regularizer, int scale) -> Tensor");
    m.def("_training_pad_complex(Tensor x, int padding) -> Tensor");
    m.def("_training_real_crop(Tensor(a) spectrum, int padding) -> Tensor(a)");
    m.def("_training_circular_s1(Tensor x, Tensor weight, Tensor bias, int padding, float eps=1e-5) -> Tensor");
#endif
    m.def("clear_cache() -> ()");
    m.def("supports_cuda_graphs() -> bool");
    m.def("begin_graph_cache() -> ()");
    m.def("end_graph_cache() -> Tensor[]");
}
TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m) {
    m.impl("forward", TORCH_FN(converse2d_forward));
    m.impl("_nearest_k2_s2", TORCH_FN(converse2d_nearest_k2_s2));
    m.impl("_alpha_residual", TORCH_FN(converse2d::peripheral::alpha_residual));
    m.impl("_channel_affine", TORCH_FN(converse2d::peripheral::channel_affine));
    m.impl("_channel_layernorm",
           TORCH_FN(converse2d::peripheral::channel_layernorm));
#ifdef CONVERSE2D_WITH_CUDA
    m.impl("_training_full_spectral",
           TORCH_FN(converse2d::full_training::full_spectral));
m.impl("_training_pad_complex", TORCH_FN(converse2d::full_training::circular_pad_complex));
    m.impl("_training_real_crop", TORCH_FN(converse2d::full_training::real_crop));
    m.impl("_training_circular_s1", TORCH_FN(converse2d::full_training::circular_s1));
#endif
    m.impl("clear_cache", TORCH_FN(clear_fb_cache));
    m.impl("supports_cuda_graphs", TORCH_FN(supports_cuda_graphs));
    m.impl("begin_graph_cache", TORCH_FN(begin_graph_cache));
    m.impl("end_graph_cache", TORCH_FN(end_graph_cache));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
