#pragma once
#include <climits>

// Included inside converse2d::full_training by full_fusion.cu, after its
// ATen, CUDAGuard, CUDAStream and CUDAException headers. These kernels keep
// ATen's circular-pad CopySlices accumulation order without atomics.
template<class Index> __global__ void circular_pad_complex_forward_kernel(
    const float* source, Z<float>* output, Index n, Index h, Index w, Index pad) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Index hp = h + 2 * pad, wp = w + 2 * pad;
    const Index bc = i / (hp * wp);
    Index row = (i / wp) % hp - pad;
    Index col = i % wp - pad;
    if (row < 0) row += h;
    else if (row >= h) row -= h;
    if (col < 0) col += w;
    else if (col >= w) col -= w;
    output[i] = Z<float>(source[(bc * h + row) * w + col], 0.0f);
}

template<class Index> __device__ __forceinline__ float circular_pad_vertical_fold(
    const Z<float>* gradient, Index base, Index row, Index col,
    Index h, Index wp, Index pad) {
    const float center = gradient[base + (pad + row) * wp + col].real();
    const float bottom = row < pad
        ? gradient[base + (pad + h + row) * wp + col].real() : 0.0f;
    const float after_bottom = __fadd_rn(center, bottom);
    // The top source first crosses the preceding bottom VJP's zero-add.
    // Do not drop these additions: they also define the signed-zero result.
    const float top = row >= h - pad
        ? __fadd_rn(gradient[base + (row - (h - pad)) * wp + col].real(), 0.0f)
        : 0.0f;
    return __fadd_rn(after_bottom, top);
}

template<class Index> __global__ void circular_pad_complex_backward_kernel(
    const Z<float>* gradient, float* output, Index n, Index h, Index w, Index pad) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Index hp = h + 2 * pad, wp = w + 2 * pad;
    const Index bc = i / (h * w), row = (i / w) % h, col = i % w;
    const Index base = bc * hp * wp;
    const float center = circular_pad_vertical_fold(gradient, base, row, pad + col, h, wp, pad);
    const float right = col < pad
        ? circular_pad_vertical_fold(gradient, base, row, pad + w + col, h, wp, pad)
        : 0.0f;
    const float after_right = __fadd_rn(center, right);
    const float left = col >= w - pad
        ? __fadd_rn(circular_pad_vertical_fold(
            gradient, base, row, col - (w - pad), h, wp, pad), 0.0f)
        : 0.0f;
    output[i] = __fadd_rn(after_right, left);
}

Tensor circular_pad_complex_cuda(const Tensor& input, I pad) {
    TORCH_CHECK(input.is_cuda() && input.layout() == c10::kStrided &&
                input.scalar_type() == at::kFloat,
                "circular_pad_complex requires a CUDA FP32 strided input");
    TORCH_CHECK(input.dim() == 4 && input.numel() > 0 && input.is_contiguous() &&
                !input.is_neg() && !input.is_conj(),
                "circular_pad_complex requires a nonempty contiguous NCHW input without lazy flags");
    const I h = input.size(2), w = input.size(3);
    TORCH_CHECK(pad > 0 && pad <= h && pad <= w,
                "circular_pad_complex requires 1 <= padding <= min(H,W)");
    TORCH_CHECK(pad <= (INT64_MAX - h) / 2 && pad <= (INT64_MAX - w) / 2,
                "circular_pad_complex output size overflow");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty({input.size(0), input.size(1), h + 2 * pad, w + 2 * pad},
                            input.options().dtype(at::kComplexFloat));
    const I n = output.numel(), blocks = (n - 1) / 256 + 1;
    TORCH_CHECK(blocks <= INT_MAX, "circular_pad_complex launch exceeds CUDA grid limit");
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    if (n <= INT_MAX - 256) {
        circular_pad_complex_forward_kernel<int><<<blocks, 256, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<Z<float>>(),
            int(n), int(h), int(w), int(pad));
    } else {
        circular_pad_complex_forward_kernel<I><<<blocks, 256, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<Z<float>>(), n, h, w, pad);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor circular_pad_complex_backward_cuda(const Tensor& incoming, I h, I w, I pad) {
    TORCH_CHECK(incoming.is_cuda() && incoming.layout() == c10::kStrided &&
                incoming.scalar_type() == at::kComplexFloat,
                "circular_pad_complex backward requires a CUDA complex64 strided gradient");
    TORCH_CHECK(h > 0 && w > 0 && pad > 0 && pad <= h && pad <= w &&
                pad <= (INT64_MAX - h) / 2 && pad <= (INT64_MAX - w) / 2,
                "circular_pad_complex backward received invalid dimensions or padding");
    TORCH_CHECK(incoming.dim() == 4 && incoming.numel() > 0 &&
                incoming.size(2) == h + 2 * pad && incoming.size(3) == w + 2 * pad,
                "circular_pad_complex backward gradient has the wrong shape");
    c10::cuda::CUDAGuard guard(incoming.device());
    auto gradient = incoming.resolve_conj().resolve_neg().contiguous();
    auto output = at::empty({gradient.size(0), gradient.size(1), h, w},
                            gradient.options().dtype(at::kFloat));
    const I n = output.numel(), blocks = (n - 1) / 256 + 1;
    TORCH_CHECK(blocks <= INT_MAX, "circular_pad_complex backward launch exceeds CUDA grid limit");
    const auto stream = c10::cuda::getCurrentCUDAStream(gradient.get_device());
    if (gradient.numel() <= INT_MAX - 256) {
        circular_pad_complex_backward_kernel<int><<<blocks, 256, 0, stream>>>(
            gradient.data_ptr<Z<float>>(), output.data_ptr<float>(),
            int(n), int(h), int(w), int(pad));
    } else {
        circular_pad_complex_backward_kernel<I><<<blocks, 256, 0, stream>>>(
            gradient.data_ptr<Z<float>>(), output.data_ptr<float>(), n, h, w, pad);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
