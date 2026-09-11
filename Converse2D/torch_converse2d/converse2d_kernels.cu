// Fused inference kernels. Training uses differentiable ATen in converse2d.cpp.
// A distinct basename avoids .obj collisions in Windows setuptools builds.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>

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
    const T* lambda, c10::complex<T>* out, int64_t total, int64_t pixels, int64_t channels) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t c = (i / pixels) % channels, p = i % (channels * pixels);
    out[i] = fx0[i] + conjugate(fb[p]) * ((fy[i] - fb[p] * fx0[i]) / (invw[p] + lambda[c]));
}

template <typename T, bool HALF, int SCALE, typename I>
__global__ void alias_correction(const c10::complex<T>* fy, const c10::complex<T>* fx0,
    const c10::complex<T>* fb, const T* invw, const T* lambda, c10::complex<T>* q,
    I total, I C, I H, I W, I dynamic_scale) {
    const I s = SCALE ? SCALE : dynamic_scale;
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I stored_w = HALF ? W / 2 + 1 : W;
    const I w = i % stored_w, h = (i / stored_w) % H;
    const I bc = i / (H * stored_w), c = bc % C;
    c10::complex<T> sum(0, 0);
    #pragma unroll
    for (I di = 0; di < s; ++di) {
        #pragma unroll
        for (I dj = 0; dj < s; ++dj) {
            const auto hh = h + di * H, ww = w + dj * W;
            sum += read_frequency<T, HALF>(fb, c, hh, ww, H*s, W*s) *
                   read_frequency<T, HALF>(fx0, bc, hh, ww, H*s, W*s);
        }
    }
    // No float literal: double inputs retain double precision, including s=3.
    q[i] = (fy[i] - sum / (T(s)*T(s))) / (invw[(c*H+h)*stored_w+w] + lambda[c]);
}

template <typename T, bool HALF, typename I>
__global__ void apply_correction(const c10::complex<T>* fx0,
    const c10::complex<T>* fb, const c10::complex<T>* q, c10::complex<T>* out,
    I total, I C, I H, I W, I s) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I Hs = H*s, Ws = W*s, stored_w = HALF ? Ws/2+1 : Ws;
    const I w = i % stored_w, h = (i / stored_w) % Hs;
    const I bc = i / (Hs * stored_w), c = bc % C;
    auto correction = read_frequency<T, HALF>(q, bc, h % H, w % W, H, W);
    out[i] = fx0[i] + conjugate(fb[(c*Hs+h)*stored_w+w]) * correction;
}

template <typename T, bool HALF, typename I>
void launch_scaled(const at::Tensor& y, const at::Tensor& prior, const at::Tensor& kernel,
                   const at::Tensor& denom, const at::Tensor& lambda,
                   at::Tensor& q, at::Tensor& out, int64_t h, int64_t w, int64_t scale,
                   cudaStream_t stream) {
    using z = c10::complex<T>;
    constexpr int threads = 256;
    const I nq = I(q.numel()), n = I(prior.numel()), C = I(prior.size(1));
    const I H = I(h), W = I(w), s = I(scale);
    const auto blocks_q = (q.numel()+threads-1)/threads;
    // Specialize common scales to remove loop control and repeated address work.
    if (s == 2) {
        alias_correction<T,HALF,2,I><<<blocks_q, threads, 0, stream>>>(
            y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), denom.data_ptr<T>(),
            lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s);
    } else if (s == 3) {
        alias_correction<T,HALF,3,I><<<blocks_q, threads, 0, stream>>>(
            y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), denom.data_ptr<T>(),
            lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s);
    } else {
        alias_correction<T,HALF,0,I><<<blocks_q, threads, 0, stream>>>(
            y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), denom.data_ptr<T>(),
            lambda.data_ptr<T>(), q.data_ptr<z>(), nq, C, H, W, s);
    }
    apply_correction<T,HALF,I><<<(prior.numel()+threads-1)/threads, threads, 0, stream>>>(
        prior.data_ptr<z>(), kernel.data_ptr<z>(), q.data_ptr<z>(), out.data_ptr<z>(), n, C, H, W, s);
}

at::Tensor converse_spectral_cuda(const at::Tensor& fy, const at::Tensor& fx0,
    const at::Tensor& fb, const at::Tensor& invw, const at::Tensor& lambda,
    int64_t H, int64_t W, int64_t s, bool half) {
    // FFT output stride is not assumed: PyTorch may return transposed FFT storage.
    auto y = fy.contiguous(), prior = fx0.contiguous(), kernel = fb.contiguous();
    auto denom = invw.contiguous();
    auto out = at::empty(prior.sizes(), prior.options());
    auto stream = c10::cuda::getCurrentCUDAStream(prior.get_device());
    constexpr int threads = 256;
    const int64_t n = prior.numel(), C = prior.size(1);
    auto q = s == 1 ? at::Tensor() : at::empty(y.sizes(), y.options());
    AT_DISPATCH_FLOATING_TYPES(lambda.scalar_type(), "converse_spectral", [&] {
        using z = c10::complex<scalar_t>;
        if (s == 1) {
            correction_scale_one<scalar_t><<<(n+threads-1)/threads, threads, 0, stream>>>(
                y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(), denom.data_ptr<scalar_t>(),
                lambda.data_ptr<scalar_t>(), out.data_ptr<z>(), n, n/(prior.size(0)*C), C);
        } else {
            const bool small = n <= INT_MAX - threads && H*s <= INT_MAX && W*s <= INT_MAX;
            if (half) {
                if (small) launch_scaled<scalar_t,true,int>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                else launch_scaled<scalar_t,true,int64_t>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
            } else {
                if (small) launch_scaled<scalar_t,false,int>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                else launch_scaled<scalar_t,false,int64_t>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
