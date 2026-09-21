#pragma once
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>

namespace converse2d::inference_detail {
// Match the separate square + add in the ATen power spectrum. In particular,
// do not silently contract these three operations into a fused multiply-add.
template <typename T> __device__ T squared_norm(c10::complex<T> z);
template <> __device__ inline float squared_norm(c10::complex<float> z) {
    return __fadd_rn(__fmul_rn(z.real(), z.real()), __fmul_rn(z.imag(), z.imag()));
}
template <> __device__ inline double squared_norm(c10::complex<double> z) {
    return __dadd_rn(__dmul_rn(z.real(), z.real()), __dmul_rn(z.imag(), z.imag()));
}

template <typename T>
__device__ c10::complex<T> conjugate(c10::complex<T> z) {
    return {z.real(), -z.imag()};
}

template <typename T, bool HALF, typename I>
__device__ c10::complex<T> read_frequency(const c10::complex<T>* data,
    I channel, I h, I w, I height, I width) {
    const I stored_w = HALF ? width / 2 + 1 : width;
    bool mirror = HALF && w > width / 2;
    if (mirror) { h = (height - h) % height; w = width - w; }
    auto value = data[(channel * height + h) * stored_w + w];
    return mirror ? conjugate(value) : value;
}

}
