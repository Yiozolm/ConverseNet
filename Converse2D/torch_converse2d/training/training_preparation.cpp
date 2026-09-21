#include "training.h"
#include <ATen/record_function.h>
#include <ATen/ATen.h>
using at::Tensor;
Tensor prepare_training_kernel(const Tensor& weight, int64_t h, int64_t w) {
    RECORD_FUNCTION("converse2d::prepare_training_kernel", std::vector<c10::IValue>());
    const auto kh = weight.size(2), kw = weight.size(3);
    const auto filters = weight.size(0) * weight.size(1);
    const auto area = h * w;
    // Small spatial grids can still have expensive FP64 adjoints when every
    // batch/channel has its own kernel. Use total transformed work as well.
    const bool many_filters = area >= (1048576 - 1) / filters + 1;
    Tensor fb;
    if ((area >= 16384 || many_filters) && kh <= h / 4) {
        auto rows = at::constant_pad_nd(weight.to(at::kDouble), {0, w - kw}, 0);
        rows = at::roll(rows, {-(kw / 2)}, {-1});
        auto horizontal = at::fft_rfft(rows, c10::nullopt, -1);
        auto columns = at::constant_pad_nd(horizontal, {0, 0, 0, h - kh}, 0);
        columns = at::roll(columns, {-(kh / 2)}, {-2});
        fb = at::fft_fft(columns, c10::nullopt, -2);
    } else {
        auto psf = at::constant_pad_nd(weight.to(at::kDouble), {0, w - kw, 0, h - kh}, 0);
        fb = at::fft_rfft2(at::roll(psf, {-(kh / 2), -(kw / 2)}, {-2, -1}));
    }
    return fb;
}

Tensor training_spectrum_cast(const Tensor& fb) {
    // Each use has its own cast node: shared preparation gradients accumulate
    // in FP64 before the FFT adjoint, rather than summing complex64 VJPs first.
    // Layout conversion is fused with the cast so the solve needs no copies.
    return fb.to(fb.options().dtype(at::kComplexFloat), false, false, at::MemoryFormat::Contiguous);
}
