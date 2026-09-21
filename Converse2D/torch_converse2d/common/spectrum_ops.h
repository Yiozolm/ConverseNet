#pragma once
#include <ATen/ATen.h>
namespace converse2d::detail {
using at::Tensor;
inline Tensor alias_mean(const Tensor& a, int64_t s) {
    if (s == 1) return a;
    const auto h = a.size(-2) / s, w = a.size(-1) / s;
    return a.reshape({a.size(0), a.size(1), s, h, s, w}).mean({2, 4});
}

// F[-h,-w] = conj(F[h,w]); reflect BOTH dimensions.
inline Tensor full_spectrum(const Tensor& half, int64_t width) {
    const int64_t end = (width + 1) / 2;
    auto tail = half.slice(-1, 1, end).flip({-2, -1}).roll({1}, {-2});
    if (half.is_complex()) tail = tail.conj();
    return at::cat({half, tail}, -1);
}


}
