// Fused inference kernels. Training uses differentiable ATen in converse2d.cpp.
// A distinct basename avoids .obj collisions in Windows setuptools builds.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>

template <typename T, typename I>
__global__ void prepare_psf(const T* weight, T* otf, I total, I H, I W, I kh, I kw) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I w = (i % W + kw / 2) % W;
    const I h = ((i / W) % H + kh / 2) % H;
    otf[i] = h < kh && w < kw ? weight[(i / (H*W) * kh + h) * kw + w] : T(0);
}

at::Tensor converse_psf_cuda(const at::Tensor& weight, int64_t h, int64_t w) {
    auto source = weight.contiguous();
    auto out = at::empty({weight.size(0), weight.size(1), h, w}, weight.options());
    const auto n = out.numel(), kh = weight.size(2), kw = weight.size(3);
    auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "converse_psf", [&] {
        if (n <= INT_MAX - 256 && h <= INT_MAX / 2 && w <= INT_MAX / 2) {
            prepare_psf<scalar_t,int><<<(n+255)/256,256,0,stream>>>(
                source.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),int(n),int(h),int(w),int(kh),int(kw));
        } else {
            prepare_psf<scalar_t,int64_t><<<(n+255)/256,256,0,stream>>>(
                source.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),n,h,w,kh,kw);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Match the separate square + add in the ATen power spectrum. In particular,
// do not silently contract these three operations into a fused multiply-add.
template <typename T> __device__ T squared_norm(c10::complex<T> z);
template <> __device__ float squared_norm(c10::complex<float> z) {
    return __fadd_rn(__fmul_rn(z.real(), z.real()), __fmul_rn(z.imag(), z.imag()));
}
template <> __device__ double squared_norm(c10::complex<double> z) {
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

template <typename T>
__global__ void correction_scale_one(const c10::complex<T>* fy,
    const c10::complex<T>* fx0, const c10::complex<T>* fb, const T* invw,
    const T* lambda, c10::complex<T>* out, int64_t total, int64_t pixels, int64_t channels,
    int64_t kernel_batches, int64_t kernel_channels) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t bc = i / pixels, c = bc % channels;
    const int64_t kc = (kernel_batches == 1 ? 0 : bc / channels) * kernel_channels +
                       (kernel_channels == 1 ? 0 : c);
    const int64_t p = kc * pixels + i % pixels;
    const auto filter = fb[p];
    const T power = invw ? invw[p] : squared_norm(filter);
    out[i] = fx0[i] + conjugate(filter) * ((fy[i] - filter * fx0[i]) / (power + lambda[c]));
}

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

template <typename T, bool HALF, typename I, bool INLINE_POWER = false>
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
    // Specialize common scales to remove loop control and repeated address work.
    if (s == 2) {
        alias_correction<T,HALF,2,I,INLINE_POWER><<<blocks_q, threads, 0, stream>>>(
            y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), power,
            lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s, KB, KC);
    } else if (s == 3) {
        alias_correction<T,HALF,3,I,INLINE_POWER><<<blocks_q, threads, 0, stream>>>(
            y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), power,
            lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s, KB, KC);
    } else {
        alias_correction<T,HALF,0,I,INLINE_POWER><<<blocks_q, threads, 0, stream>>>(
            y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), power,
            lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s, KB, KC);
    }
    apply_correction<T,HALF,I><<<(prior.numel()+threads-1)/threads, threads, 0, stream>>>(
        prior.data_ptr<z>(), kernel.data_ptr<z>(), q.data_ptr<z>(), out.data_ptr<z>(), n, C, H, W, s, KB, KC);
}

at::Tensor converse_spectral_cuda(const at::Tensor& fy, const at::Tensor& fx0,
    const at::Tensor& fb, const at::Tensor& invw, const at::Tensor& lambda,
    int64_t H, int64_t W, int64_t s, bool half) {
    // FFT output stride is not assumed: PyTorch may return transposed FFT storage.
    auto y = fy.contiguous(), prior = fx0.contiguous(), kernel = fb.contiguous();
    auto denom = invw.defined() ? invw.contiguous() : at::Tensor();
    auto out = at::empty(prior.sizes(), prior.options());
    auto stream = c10::cuda::getCurrentCUDAStream(prior.get_device());
    constexpr int threads = 256;
    const int64_t n = prior.numel(), C = prior.size(1);
    auto q = s == 1 ? at::Tensor() : at::empty(y.sizes(), y.options());
    AT_DISPATCH_FLOATING_TYPES(lambda.scalar_type(), "converse_spectral", [&] {
        using z = c10::complex<scalar_t>;
        if (s == 1) {
            correction_scale_one<scalar_t><<<(n+threads-1)/threads, threads, 0, stream>>>(
                y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(),
                denom.defined() ? denom.data_ptr<scalar_t>() : nullptr,
                lambda.data_ptr<scalar_t>(), out.data_ptr<z>(), n, n/(prior.size(0)*C), C,
                kernel.size(0), kernel.size(1));
        } else {
            const bool small = n <= INT_MAX - threads && H*s <= INT_MAX && W*s <= INT_MAX;
            if (half) {
                if (!denom.defined()) {
                    if (small) launch_scaled<scalar_t,true,int,true>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                    else launch_scaled<scalar_t,true,int64_t,true>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                } else {
                    if (small) launch_scaled<scalar_t,true,int>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                    else launch_scaled<scalar_t,true,int64_t>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                }
            } else {
                if (small) launch_scaled<scalar_t,false,int>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                else launch_scaled<scalar_t,false,int64_t>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
