// LTO callbacks are looked up by their unmangled source name; do not use extern "C".
// cuFFT LTO callbacks, compiled to an LTO-IR fatbin by loader.py.
// Loads replace the separate real->complex promote / circular-pad kernels;
// the store replaces ATen's separate 1/N normalization pass after the IFFT.
#include <cufftXt.h>
#include <c10/util/complex.h>

struct PadInfo {
    long long h, w, pad;
};

// (x, +0): the bytes ATen's promote and the production pad kernel write.
__device__ cufftComplex load_real(void *data, unsigned long long offset,
                                             void *, void *) {
    return make_cuComplex(static_cast<const float *>(data)[offset], 0.0f);
}

// Same index map as circular_pad_complex_forward_kernel.
__device__ cufftComplex load_circular(void *data, unsigned long long offset,
                                                 void *info, void *) {
    const PadInfo p = *static_cast<const PadInfo *>(info);
    const long long hp = p.h + 2 * p.pad, wp = p.w + 2 * p.pad;
    const long long i = static_cast<long long>(offset);
    const long long bc = i / (hp * wp);
    long long row = (i / wp) % hp - p.pad, col = i % wp - p.pad;
    if (row < 0) row += p.h;
    else if (row >= p.h) row -= p.h;
    if (col < 0) col += p.w;
    else if (col >= p.w) col -= p.w;
    return make_cuComplex(static_cast<const float *>(data)[(bc * p.h + row) * p.w + col], 0.0f);
}

// ATen: self.mul_(1.0 / n) runs MulFunctor(scalar, x) on complex<float>,
// i.e. complex<float>(float(1/n), 0) * x. Keep that exact expression.
__device__ void store_scaled(void *data, unsigned long long offset,
                                        cufftComplex element, void *info, void *) {
    const float scale = *static_cast<const float *>(info);
    const c10::complex<float> r =
        c10::complex<float>(scale, 0.0f) * c10::complex<float>(element.x, element.y);
    static_cast<cufftComplex *>(data)[offset] = make_cuComplex(r.real(), r.imag());
}
