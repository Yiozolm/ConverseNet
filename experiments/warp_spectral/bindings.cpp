// Keep host registration out of nvcc's Windows front end.
#include <ATen/ATen.h>
#include <torch/library.h>
namespace warp_spectral_experiment {
at::Tensor run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
               const at::Tensor&, int64_t, int64_t, int64_t);
at::Tensor trace(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                 const at::Tensor&, int64_t, int64_t, int64_t);
}
TORCH_LIBRARY(warp_spectral, m) {
    m.def("run(Tensor fy, Tensor prior, Tensor fb, Tensor lambda, int H, int W, int mode) -> Tensor");
    m.def("trace(Tensor fy, Tensor prior, Tensor fb, Tensor lambda, int H, int W, int iterations) -> Tensor");
}
TORCH_LIBRARY_IMPL(warp_spectral, CUDA, m) {
    m.impl("run", &warp_spectral_experiment::run);
    m.impl("trace", &warp_spectral_experiment::trace);
}
