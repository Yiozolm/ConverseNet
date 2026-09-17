// Isolated scale-2 inference experiment; never selected by the production op.
// Compile with torch.utils.cpp_extension.load(..., is_python_module=false).
#include <c10/cuda/CUDAGuard.h>
#include "../../Converse2D/torch_converse2d/converse2d_kernels.cu"
#include "specialized.cuh"

namespace warp_spectral_experiment {
using Z = c10::complex<float>;

__device__ int filter_channel(int bc, int C, int KB, int KC) {
    return (KB == 1 ? 0 : bc / C) * KC + (KC == 1 ? 0 : bc % C);
}

// An interior LR half-spectrum bin owns four distinct stored HR bins. The
// two missing HR aliases are written through their conjugate representatives.
// LR boundary columns are excluded: their stored representatives overlap.
__global__ void thread_fused(const Z* y, const Z* prior, const Z* kernel,
    const float* lambda, Z* out, int count, int C, int H, int W, int KB, int KC) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int columns = (W - 1) / 2;
    const int w = i % columns + 1, h = (i / columns) % H;
    const int bc = i / (H * columns), kc = filter_channel(bc, C, KB, KC);
    Z filters[4], priors[4];
    Z prediction(0, 0);
    float power = 0;
    #pragma unroll
    for (int member = 0; member < 4; ++member) {
        const int hh = h + (member / 2) * H, ww = w + (member % 2) * W;
        filters[member] = read_frequency<float, true>(kernel, kc, hh, ww, H * 2, W * 2);
        priors[member] = read_frequency<float, true>(prior, bc, hh, ww, H * 2, W * 2);
        prediction += filters[member] * priors[member];
        power += squared_norm(filters[member]);
    }
    const Z q = (y[(bc * H + h) * (W / 2 + 1) + w] - prediction / 4.0f) /
                (power / 4.0f + lambda[bc % C]);
    #pragma unroll
    for (int member = 0; member < 4; ++member) {
        int hh = h + (member / 2) * H, ww = w + (member % 2) * W;
        Z value = priors[member] + conjugate(filters[member]) * q;
        if (ww > W) {
            hh = (H * 2 - hh) % (H * 2);
            ww = W * 2 - ww;
            value = conjugate(value);
        }
        out[(bc * H * 2 + hh) * (W + 1) + ww] = value;
    }
}

// lane = alias_member * 8 + adjacent_frequency. Each alias segment accesses
// eight consecutive complex values, and all four segments exchange partials.
// This is warp cooperation, with identical work per warp, not specialization.
__global__ void warp_fused(const Z* y, const Z* prior, const Z* kernel,
    const float* lambda, Z* out, int count, int C, int H, int W, int KB, int KC) {
    const int global_thread = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31, adjacent = lane & 7, member = lane >> 3;
    const int i = (global_thread >> 5) * 8 + adjacent;
    const bool valid = i < count;
    const int columns = (W - 1) / 2;
    const int w = i % columns + 1, h = (i / columns) % H;
    const int bc = i / (H * columns), kc = filter_channel(bc, C, KB, KC);
    int hh = h + (member / 2) * H, ww = w + (member % 2) * W;
    Z filter(0, 0), p(0, 0);
    if (valid) {
        filter = read_frequency<float, true>(kernel, kc, hh, ww, H * 2, W * 2);
        p = read_frequency<float, true>(prior, bc, hh, ww, H * 2, W * 2);
    }
    const Z product = filter * p;
    const float partial_power = squared_norm(filter);
    Z prediction(0, 0);
    float power = 0;
    // Preserve baseline left-to-right sum order instead of a different tree.
    // Tail lanes participate with zeros: no early return before these shuffles.
    #pragma unroll
    for (int source_member = 0; source_member < 4; ++source_member) {
        const int source_lane = source_member * 8 + adjacent;
        const float real = __shfl_sync(0xffffffffu, product.real(), source_lane);
        const float imag = __shfl_sync(0xffffffffu, product.imag(), source_lane);
        prediction += Z(real, imag);
        power += __shfl_sync(0xffffffffu, partial_power, source_lane);
    }
    Z q(0, 0);
    if (valid && member == 0) {
        q = (y[(bc * H + h) * (W / 2 + 1) + w] - prediction / 4.0f) /
            (power / 4.0f + lambda[bc % C]);
    }
    const float qreal = __shfl_sync(0xffffffffu, q.real(), adjacent);
    const float qimag = __shfl_sync(0xffffffffu, q.imag(), adjacent);
    if (!valid) return;
    Z value = p + conjugate(filter) * Z(qreal, qimag);
    if (ww > W) {
        hh = (H * 2 - hh) % (H * 2);
        ww = W * 2 - ww;
        value = conjugate(value);
    }
    out[(bc * H * 2 + hh) * (W + 1) + ww] = value;
}

// One owner per stored HR boundary element. Recompute its LR q locally so
// the experiment needs neither q storage nor a cross-block synchronization.
// Includes w=0,W and, for even W, w=W/2. This also handles W=1 or W=2.
__global__ void boundary_fused(const Z* y, const Z* prior, const Z* kernel,
    const float* lambda, Z* out, int count, int C, int H, int W, int KB, int KC) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int boundary_columns = W % 2 == 0 ? 3 : 2;
    const int column = i % boundary_columns;
    const int w = W % 2 == 0 ? column * (W / 2) : column * W;
    const int h = (i / boundary_columns) % (H * 2);
    const int bc = i / (boundary_columns * H * 2);
    const int kc = filter_channel(bc, C, KB, KC);
    const int lh = h % H, lw = w % W;
    Z prediction(0, 0);
    float power = 0;
    #pragma unroll
    for (int member = 0; member < 4; ++member) {
        const int hh = lh + (member / 2) * H, ww = lw + (member % 2) * W;
        const Z filter = read_frequency<float, true>(kernel, kc, hh, ww, H * 2, W * 2);
        prediction += filter * read_frequency<float, true>(prior, bc, hh, ww, H * 2, W * 2);
        power += squared_norm(filter);
    }
    const Z q = (y[(bc * H + lh) * (W / 2 + 1) + lw] - prediction / 4.0f) /
                (power / 4.0f + lambda[bc % C]);
    const int index = (bc * H * 2 + h) * (W + 1) + w;
    const int filter_index = (kc * H * 2 + h) * (W + 1) + w;
    out[index] = prior[index] + conjugate(kernel[filter_index]) * q;
}

void validate_inputs(const at::Tensor& y, const at::Tensor& prior, const at::Tensor& kernel,
    const at::Tensor& lambda, int64_t H, int64_t W) {
    TORCH_CHECK(H > 0 && W > 0 && H <= INT_MAX / 2 && W <= INT_MAX / 2,
                "positive H/W must fit scale-2 32-bit indexing");
    TORCH_CHECK(y.is_cuda(), "CUDA tensors required");
    for (const auto* tensor : {&y, &prior, &kernel, &lambda}) {
        TORCH_CHECK(tensor->device() == y.device(), "all tensors must use the same CUDA device");
        TORCH_CHECK(tensor->is_contiguous() && !tensor->is_conj() && !tensor->is_neg(),
                    "plain contiguous storage required; resolve conjugate/negative views first");
        TORCH_CHECK(!tensor->requires_grad(), "isolated inference experiment does not implement autograd");
    }
    TORCH_CHECK(y.scalar_type() == at::kComplexFloat && prior.scalar_type() == at::kComplexFloat &&
                kernel.scalar_type() == at::kComplexFloat && lambda.scalar_type() == at::kFloat,
                "complex64 spectra and float32 lambda required");
    TORCH_CHECK(y.dim() == 4 && prior.dim() == 4 && kernel.dim() == 4,
                "spectra must have B,C,H,W dimensions");
    const int64_t B = y.size(0), C = y.size(1);
    TORCH_CHECK(B > 0 && C > 0, "positive batch and channel dimensions required");
    TORCH_CHECK(y.size(2) == H && y.size(3) == W / 2 + 1, "incorrect LR half-spectrum shape");
    TORCH_CHECK(prior.size(0) == B && prior.size(1) == C && prior.size(2) == H * 2 &&
                prior.size(3) == W + 1, "incorrect HR prior half-spectrum shape");
    TORCH_CHECK((kernel.size(0) == 1 || kernel.size(0) == B) &&
                (kernel.size(1) == 1 || kernel.size(1) == C) &&
                kernel.size(2) == H * 2 && kernel.size(3) == W + 1,
                "kernel must broadcast over B/C and match HR half-spectrum shape");
    TORCH_CHECK(lambda.numel() == C, "lambda must contain C values");
    TORCH_CHECK(prior.numel() <= INT_MAX - 256, "experiment requires 32-bit indexing");
}

at::Tensor run(const at::Tensor& y, const at::Tensor& prior, const at::Tensor& kernel,
    const at::Tensor& lambda, int64_t H, int64_t W, int64_t mode) {
    TORCH_CHECK(mode >= 0 && mode <= 3,
                "mode must be 0 (baseline), 1 (thread), 2 (warp), or 3 (specialized)");
    validate_inputs(y, prior, kernel, lambda, H, W);
    c10::cuda::CUDAGuard device_guard(y.device());
    if (mode == 0) return converse_spectral_cuda(y, prior, kernel, at::Tensor(), lambda, H, W, 2, true);
    const int64_t B = y.size(0), C = y.size(1);
    auto out = at::empty_like(prior);
    auto stream = c10::cuda::getCurrentCUDAStream(y.get_device());
    const int count = int(B * C * H * ((W - 1) / 2));
    const int boundaries = int(B * C * H * 2 * (W % 2 == 0 ? 3 : 2));
    const int channels = int(C), height = int(H), width = int(W);
    const int KB = int(kernel.size(0)), KC = int(kernel.size(1));
    if (count > 0 && mode == 1) {
        thread_fused<<<(count + 255) / 256, 256, 0, stream>>>(y.data_ptr<Z>(), prior.data_ptr<Z>(),
            kernel.data_ptr<Z>(), lambda.data_ptr<float>(), out.data_ptr<Z>(),
            count, channels, height, width, KB, KC);
    } else if (count > 0 && mode == 2) {
        // Four warps per block, eight LR frequencies per warp.
        warp_fused<<<(count + 31) / 32, 128, 0, stream>>>(y.data_ptr<Z>(), prior.data_ptr<Z>(),
            kernel.data_ptr<Z>(), lambda.data_ptr<float>(), out.data_ptr<Z>(),
            count, channels, height, width, KB, KC);
    } else if (count > 0) {
        constexpr int iterations = 8;
        ws_interior<false><<<(count + 32 * iterations - 1) / (32 * iterations), 64, 0, stream>>>(
            y.data_ptr<Z>(), prior.data_ptr<Z>(), kernel.data_ptr<Z>(), lambda.data_ptr<float>(),
            out.data_ptr<Z>(), count, channels, height, width, KB, KC, iterations, nullptr);
    }
    boundary_fused<<<(boundaries + 255) / 256, 256, 0, stream>>>(y.data_ptr<Z>(), prior.data_ptr<Z>(),
        kernel.data_ptr<Z>(), lambda.data_ptr<float>(), out.data_ptr<Z>(),
        boundaries, channels, height, width, KB, KC);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Profile one CTA only: every timestamp uses the same SM clock. The returned
// int32 tensor stores raw uint32 clock bits as [role, event, iteration].
at::Tensor trace(const at::Tensor& y, const at::Tensor& prior, const at::Tensor& kernel,
    const at::Tensor& lambda, int64_t H, int64_t W, int64_t iterations) {
    validate_inputs(y, prior, kernel, lambda, H, W);
    TORCH_CHECK(iterations >= 1 && iterations <= 64, "trace iterations must be 1..64");
    const int64_t count = y.size(0) * y.size(1) * H * ((W - 1) / 2);
    TORCH_CHECK(count >= 32 && count <= 32 * iterations,
                "trace requires 32 <= interior groups <= 32 * iterations for one CTA");
    c10::cuda::CUDAGuard device_guard(y.device());
    auto out = at::empty_like(prior);
    auto stamps = at::zeros({2, 8, 64}, lambda.options().dtype(at::kInt));
    auto stream = c10::cuda::getCurrentCUDAStream(y.get_device());
    ws_interior<true><<<1, 64, 0, stream>>>(y.data_ptr<Z>(), prior.data_ptr<Z>(), kernel.data_ptr<Z>(),
        lambda.data_ptr<float>(), out.data_ptr<Z>(), int(count), int(y.size(1)), int(H), int(W),
        int(kernel.size(0)), int(kernel.size(1)), int(iterations),
        reinterpret_cast<unsigned int*>(stamps.data_ptr<int32_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stamps;
}
} // namespace warp_spectral_experiment
