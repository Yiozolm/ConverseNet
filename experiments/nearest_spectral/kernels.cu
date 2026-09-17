// Include the same headers first so the Windows SDK's RPC `small` macro cannot
// replace the production kernel launcher's local variable of that name.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>
#ifdef small
#undef small
#endif
#include <c10/cuda/CUDAGuard.h>
#include "../../Converse2D/torch_converse2d/converse2d_kernels.cu"

namespace {

// The FFT of zero insertion followed by nearest-neighbour replication is
// Y[h % H, w % W] * Phi_Hs[h] * Phi_Ws[w], where
// Phi_N[k] = sum_{a=0}^{s-1} exp(-2*pi*i*k*a/N).
// Calculate one canonical frequency for each conjugate pair. Double trig and
// exact DC, alias zeros and Nyquist avoid cancellation at known special bins.
template <typename T>
__global__ void nearest_phase_kernel(c10::complex<T>* phase,
    int64_t hs, int64_t ws, int64_t scale) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= hs + ws) return;
    const int64_t n = i < hs ? hs : ws;
    const int64_t k = i < hs ? i : i - hs;
    if (k == 0) {
        phase[i] = c10::complex<T>(T(scale), T(0));
        return;
    }
    if (k % (n / scale) == 0) {
        phase[i] = c10::complex<T>(T(0), T(0));
        return;
    }
    if (n % 2 == 0 && k == n / 2) {
        phase[i] = c10::complex<T>(T(scale % 2), T(0));
        return;
    }
    const bool reflected = k > n / 2;
    const int64_t canonical = reflected ? n - k : k;
    constexpr double tau = 6.283185307179586476925286766559005768;
    const double angle = -tau * double(canonical) / double(n);
    double real = 1.0, imag = 0.0;
    for (int64_t a = 1; a < scale; ++a) {
        double sine, cosine;
        sincos(angle * double(a), &sine, &cosine);
        real += cosine;
        imag += sine;
    }
    phase[i] = c10::complex<T>(T(real), T(reflected ? -imag : imag));
}

template <typename T, int SCALE, typename I, bool INLINE_POWER>
__global__ void nearest_alias_kernel(const c10::complex<T>* fy,
    const c10::complex<T>* fb, const T* invw, const T* lambda,
    const c10::complex<T>* phase, c10::complex<T>* q,
    I total, I C, I H, I W, I dynamic_scale, I KB, I KC) {
    const I s = SCALE ? SCALE : dynamic_scale;
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I stored_w = W / 2 + 1, Hs = H * s, Ws = W * s;
    const I w = i % stored_w, h = (i / stored_w) % H;
    const I bc = i / (H * stored_w), c = bc % C;
    const I kc = (KB == 1 ? 0 : bc / C) * KC + (KC == 1 ? 0 : c);
    const auto observation = fy[i];
    c10::complex<T> sum(0, 0);
    T power_sum = 0;
    #pragma unroll
    for (I di = 0; di < s; ++di) {
        const I hh = h + di * H;
        const auto row_phase = phase[hh];
        #pragma unroll
        for (I dj = 0; dj < s; ++dj) {
            const I ww = w + dj * W;
            const auto filter = read_frequency<T, true>(fb, kc, hh, ww, Hs, Ws);
            const auto prior = observation * (row_phase * phase[Hs + ww]);
            sum += filter * prior;
            if constexpr (INLINE_POWER) power_sum += squared_norm(filter);
        }
    }
    const T scale_squared = T(s) * T(s);
    const T power = INLINE_POWER ? power_sum / scale_squared
                                : invw[(kc * H + h) * stored_w + w];
    q[i] = (observation - sum / scale_squared) / (power + lambda[c]);
}

template <typename T, typename I>
__global__ void nearest_apply_kernel(const c10::complex<T>* fy,
    const c10::complex<T>* fb, const c10::complex<T>* q,
    const c10::complex<T>* phase, c10::complex<T>* out,
    I total, I C, I H, I W, I s, I KB, I KC) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const I Hs = H * s, Ws = W * s, stored_w = Ws / 2 + 1;
    const I w = i % stored_w, h = (i / stored_w) % Hs;
    const I bc = i / (Hs * stored_w), c = bc % C;
    const I kc = (KB == 1 ? 0 : bc / C) * KC + (KC == 1 ? 0 : c);
    const I lh = h % H, lw = w % W;
    // Both coordinates must be reflected when lw falls outside the stored
    // low-resolution half spectrum. read_frequency performs that reflection.
    const auto observation = read_frequency<T, true>(fy, bc, lh, lw, H, W);
    const auto prior = observation * (phase[h] * phase[Hs + w]);
    const auto correction = read_frequency<T, true>(q, bc, lh, lw, H, W);
    out[i] = prior + conjugate(fb[(kc * Hs + h) * stored_w + w]) * correction;
}

template <typename T, typename I, bool INLINE_POWER>
void launch_nearest_spectral(const at::Tensor& y, const at::Tensor& kernel,
    const at::Tensor& denom, const at::Tensor& lambda, const at::Tensor& phase,
    at::Tensor& q, at::Tensor& out, int64_t h, int64_t w, int64_t scale,
    cudaStream_t stream) {
    using z = c10::complex<T>;
    constexpr int threads = 256;
    const I nq = I(q.numel()), n = I(out.numel()), C = I(y.size(1));
    const I H = I(h), W = I(w), s = I(scale);
    const I KB = I(kernel.size(0)), KC = I(kernel.size(1));
    const auto blocks_q = (q.numel() + threads - 1) / threads;
    const T* power = INLINE_POWER ? nullptr : denom.data_ptr<T>();
    // Common scales have fully unrolled alias loops; arbitrary integer scales
    // use the same algebra with a dynamic loop.
#define LAUNCH_NEAREST_ALIAS(S) \
    nearest_alias_kernel<T, S, I, INLINE_POWER><<<blocks_q, threads, 0, stream>>>( \
        y.data_ptr<z>(), kernel.data_ptr<z>(), power, lambda.data_ptr<T>(), \
        phase.data_ptr<z>(), q.data_ptr<z>(), nq, C, H, W, s, KB, KC)
    if (s == 2) {
        LAUNCH_NEAREST_ALIAS(2);
    } else if (s == 3) {
        LAUNCH_NEAREST_ALIAS(3);
    } else if (s == 4) {
        LAUNCH_NEAREST_ALIAS(4);
    } else if (s == 5) {
        LAUNCH_NEAREST_ALIAS(5);
    } else {
        LAUNCH_NEAREST_ALIAS(0);
    }
#undef LAUNCH_NEAREST_ALIAS
    nearest_apply_kernel<T, I><<<(out.numel() + threads - 1) / threads,
        threads, 0, stream>>>(y.data_ptr<z>(), kernel.data_ptr<z>(), q.data_ptr<z>(),
        phase.data_ptr<z>(), out.data_ptr<z>(), n, C, H, W, s, KB, KC);
}

} // namespace

at::Tensor nearest_phase_cuda(const at::Tensor& like,
    int64_t hs, int64_t ws, int64_t scale) {
    TORCH_CHECK(like.is_cuda(), "nearest phase requires CUDA input");
    TORCH_CHECK(like.scalar_type() == at::kFloat || like.scalar_type() == at::kDouble,
        "nearest phase requires float or double compute dtype");
    TORCH_CHECK(scale > 0 && hs > 0 && ws > 0 && hs % scale == 0 && ws % scale == 0,
        "nearest phase dimensions must be positive multiples of scale");
    TORCH_CHECK(hs <= INT64_MAX - ws, "nearest phase dimensions overflow int64");
    const c10::cuda::CUDAGuard guard(like.device());
    const auto complex_type = like.scalar_type() == at::kDouble
        ? at::kComplexDouble : at::kComplexFloat;
    auto phase = at::empty({hs + ws}, like.options().dtype(complex_type));
    auto stream = c10::cuda::getCurrentCUDAStream(like.get_device());
    AT_DISPATCH_FLOATING_TYPES(like.scalar_type(), "nearest_phase", [&] {
        nearest_phase_kernel<scalar_t><<<(phase.numel() + 255) / 256, 256, 0, stream>>>(
            phase.data_ptr<c10::complex<scalar_t>>(), hs, ws, scale);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return phase;
}

at::Tensor nearest_spectral_cuda(const at::Tensor& fy, const at::Tensor& fb,
    const at::Tensor& invw, const at::Tensor& lambda, const at::Tensor& phase,
    int64_t H, int64_t W, int64_t scale) {
    TORCH_CHECK(fy.is_cuda() && fb.is_cuda() && lambda.is_cuda() && phase.is_cuda(),
        "nearest spectral tensors must be CUDA tensors");
    TORCH_CHECK(fy.device() == fb.device() && fy.device() == lambda.device()
        && fy.device() == phase.device(), "nearest spectral tensors must share a device");
    TORCH_CHECK(H > 0 && W > 0 && scale > 0
        && H <= INT64_MAX / scale && W <= INT64_MAX / scale,
        "nearest spectral dimensions must be positive and fit int64");
    const int64_t hs = H * scale, ws = W * scale;
    TORCH_CHECK(hs <= INT64_MAX - ws, "nearest phase dimensions overflow int64");
    TORCH_CHECK(fy.dim() == 4 && fy.size(2) == H && fy.size(3) == W / 2 + 1,
        "nearest spectral observation must be [B,C,H,W/2+1]");
    const int64_t B = fy.size(0), C = fy.size(1);
    TORCH_CHECK(B > 0 && C > 0 && fb.dim() == 4 && fb.size(2) == hs
        && fb.size(3) == ws / 2 + 1 && (fb.size(0) == 1 || fb.size(0) == B)
        && (fb.size(1) == 1 || fb.size(1) == C), "nearest spectral filter shape mismatch");
    TORCH_CHECK((lambda.scalar_type() == at::kFloat || lambda.scalar_type() == at::kDouble)
        && lambda.numel() == C, "nearest spectral lambda must have C float/double values");
    const auto complex_type = lambda.scalar_type() == at::kDouble
        ? at::kComplexDouble : at::kComplexFloat;
    TORCH_CHECK(fy.scalar_type() == complex_type && fb.scalar_type() == complex_type
        && phase.scalar_type() == complex_type && phase.dim() == 1
        && phase.numel() == hs + ws, "nearest spectral complex dtype or phase shape mismatch");
    if (invw.defined()) {
        TORCH_CHECK(invw.device() == fy.device() && invw.scalar_type() == lambda.scalar_type()
            && invw.dim() == 4 && invw.size(0) == fb.size(0) && invw.size(1) == fb.size(1)
            && invw.size(2) == H && invw.size(3) == W / 2 + 1,
            "nearest spectral power shape or dtype mismatch");
    }
    const c10::cuda::CUDAGuard guard(fy.device());
    // cuFFT output can have transposed strides, so do not assume contiguous
    // storage at the experiment's public spectral boundary.
    auto y = fy.contiguous(), kernel = fb.contiguous(), coefficients = lambda.contiguous();
    auto phases = phase.contiguous();
    auto denom = invw.defined() ? invw.contiguous() : at::Tensor();
    auto q = at::empty(y.sizes(), y.options());
    auto out = at::empty({B, C, hs, ws / 2 + 1}, y.options());
    auto stream = c10::cuda::getCurrentCUDAStream(fy.get_device());
    constexpr int threads = 256;
    // Include the full spatial plane even though only its half spectrum is
    // stored, and cover broadcast filter/phase indexing as well as output.
    const bool use_int32 = out.numel() <= INT_MAX - threads
        && y.numel() <= INT_MAX - threads && kernel.numel() <= INT_MAX - threads
        && phase.numel() <= INT_MAX && hs <= INT_MAX && ws <= INT_MAX
        && hs <= INT_MAX / ws;
    AT_DISPATCH_FLOATING_TYPES(lambda.scalar_type(), "nearest_spectral", [&] {
        if (denom.defined()) {
            if (use_int32) launch_nearest_spectral<scalar_t, int, false>(
                y, kernel, denom, coefficients, phases, q, out, H, W, scale, stream);
            else launch_nearest_spectral<scalar_t, int64_t, false>(
                y, kernel, denom, coefficients, phases, q, out, H, W, scale, stream);
        } else {
            if (use_int32) launch_nearest_spectral<scalar_t, int, true>(
                y, kernel, denom, coefficients, phases, q, out, H, W, scale, stream);
            else launch_nearest_spectral<scalar_t, int64_t, true>(
                y, kernel, denom, coefficients, phases, q, out, H, W, scale, stream);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
