#pragma once
#include "math.cuh"
namespace converse2d::inference_detail {
template <typename T, bool HALF, int SCALE, typename I, bool INLINE_POWER>
__global__ void alias_correction(const c10::complex<T>* fy, const c10::complex<T>* fx0,
    const c10::complex<T>* fb, const T* invw, const T* lambda, c10::complex<T>* q,
    I total, I C, I H, I W, I dynamic_scale, I KB, I KC) {
    const I s = SCALE ? SCALE : dynamic_scale;
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I stored_w = HALF ? W / 2 + 1 : W;
    const I w = i % stored_w, h = (i / stored_w) % H;
    const I bc = i / (H * stored_w), c = bc % C;
    const I kc = (KB == 1 ? 0 : bc / C) * KC + (KC == 1 ? 0 : c);
    c10::complex<T> sum(0, 0);
    T power_sum = 0;
    #pragma unroll
    for (I di = 0; di < s; ++di) {
        #pragma unroll
        for (I dj = 0; dj < s; ++dj) {
            const auto hh = h + di * H, ww = w + dj * W;
            const auto filter = read_frequency<T, HALF>(fb, kc, hh, ww, H*s, W*s);
            sum += filter * read_frequency<T, HALF>(fx0, bc, hh, ww, H*s, W*s);
            if constexpr (INLINE_POWER) power_sum += squared_norm(filter);
        }
    }
    // No float literal: double inputs retain double precision, including s=3.
    const T power = INLINE_POWER ? power_sum / (T(s)*T(s)) : invw[(kc*H+h)*stored_w+w];
    q[i] = (fy[i] - sum / (T(s)*T(s))) / (power + lambda[c]);
}

template <typename T, bool HALF, typename I>
__global__ void apply_correction(const c10::complex<T>* fx0,
    const c10::complex<T>* fb, const c10::complex<T>* q, c10::complex<T>* out,
    I total, I C, I H, I W, I s, I KB, I KC) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I Hs = H*s, Ws = W*s, stored_w = HALF ? Ws/2+1 : Ws;
    const I w = i % stored_w, h = (i / stored_w) % Hs;
    const I bc = i / (Hs * stored_w), c = bc % C;
    const I kc = (KB == 1 ? 0 : bc / C) * KC + (KC == 1 ? 0 : c);
    auto correction = read_frequency<T, HALF>(q, bc, h % H, w % W, H, W);
    out[i] = fx0[i] + conjugate(fb[(kc*Hs+h)*stored_w+w]) * correction;
}

template <int SCALE, typename T, bool HALF, typename I, bool INLINE_POWER = false>
void launch_scaled(const at::Tensor& y, const at::Tensor& prior, const at::Tensor& kernel,
                   const at::Tensor& denom, const at::Tensor& lambda,
                   at::Tensor& q, at::Tensor& out, int64_t h, int64_t w, int64_t scale,
                   cudaStream_t stream) {
    using z = c10::complex<T>;
    constexpr int threads = 256;
    const I nq = I(q.numel()), n = I(prior.numel()), C = I(prior.size(1));
    const I H = I(h), W = I(w), s = I(scale);
    const I KB = I(kernel.size(0)), KC = I(kernel.size(1));
    const auto blocks_q = (q.numel()+threads-1)/threads;
    const T* power = INLINE_POWER ? nullptr : denom.data_ptr<T>();
    alias_correction<T,HALF,SCALE,I,INLINE_POWER><<<blocks_q, threads, 0, stream>>>(
        y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), power,
        lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s, KB, KC);
    apply_correction<T,HALF,I><<<(prior.numel()+threads-1)/threads, threads, 0, stream>>>(
        prior.data_ptr<z>(), kernel.data_ptr<z>(), q.data_ptr<z>(), out.data_ptr<z>(), n, C, H, W, s, KB, KC);
}

}
