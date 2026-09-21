#pragma once
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <vector>

namespace converse2d::training_detail {
template<class T> using Z = c10::complex<T>;
using I = int64_t;
template<class T> __device__ Z<T> cj(Z<T> z) { return {z.real(),-z.imag()}; }
template<class T> __device__ T norm2(Z<T> z);
template<> __device__ inline float norm2(Z<float> z) {
    return __fadd_rn(__fmul_rn(z.real(),z.real()),__fmul_rn(z.imag(),z.imag()));
}
template<> __device__ inline double norm2(Z<double> z) {
    return __dadd_rn(__dmul_rn(z.real(),z.real()),__dmul_rn(z.imag(),z.imag()));
}
template<class T> __device__ Z<T> read(const Z<T>* p,I bc,I h,I w,I H,I W) {
    const bool mirror=w>W/2;
    if(mirror) { h=(H-h)%H; w=W-w; }
    auto z=p[(bc*H+h)*(W/2+1)+w];
    return mirror?cj(z):z;
}
__device__ inline I filter_channel(I bc,I C,I KB,I KC) {
    return (KB==1?0:bc/C)*KC+(KC==1?0:bc%C);
}

}
