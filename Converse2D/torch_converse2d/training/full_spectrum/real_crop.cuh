#pragma once
#include <climits>

// Included inside converse2d::full_training by full_fusion.cu. Each output
// element has one writer: crop gradients are copied without floating-point
// arithmetic, and both components outside the crop are positive zero.
template<class Index> __global__ void real_crop_backward_kernel(
    const float* gradient, Z<float>* output,
    Index n, Index c, Index h, Index w, Index pad,
    Index stride_b, Index stride_c, Index stride_h, Index stride_w) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Index bc = i / (h * w);
    const Index row = (i / w) % h, col = i % w;
    float real = 0.0f;
    if (row >= pad && row < h - pad && col >= pad && col < w - pad) {
        const Index offset = (bc / c) * stride_b + (bc % c) * stride_c
            + (row - pad) * stride_h + (col - pad) * stride_w;
        real = gradient[offset];
    }
    output[i] = Z<float>(real, 0.0f);
}

Tensor real_crop_cuda_backward(const Tensor& incoming, I b, I c, I h, I w, I pad) {
    TORCH_CHECK(incoming.is_cuda() && incoming.layout() == c10::kStrided &&
                incoming.scalar_type() == at::kFloat,
                "real_crop backward requires a CUDA FP32 strided gradient");
    // Subtraction-based bounds avoid overflowing 2 * pad before validation.
    TORCH_CHECK(b >= 0 && c >= 0 && h > 0 && w > 0 && pad >= 0 &&
                pad <= (h - 1) / 2 && pad <= (w - 1) / 2,
                "real_crop backward received invalid dimensions or padding");
    TORCH_CHECK(incoming.dim() == 4 && incoming.size(0) == b && incoming.size(1) == c &&
                incoming.size(2) == h - 2 * pad && incoming.size(3) == w - 2 * pad,
                "real_crop backward gradient has the wrong shape");
    I n = 1;
    for (const I size : {b, c, h, w}) {
        TORCH_CHECK(size == 0 || n <= INT64_MAX / size,
                    "real_crop backward output size overflow");
        n *= size;
    }
    TORCH_CHECK(n <= INT64_MAX / I(sizeof(Z<float>)),
                "real_crop backward output storage size overflow");
    const I blocks = n == 0 ? 0 : (n - 1) / 256 + 1;
    TORCH_CHECK(blocks <= INT_MAX, "real_crop backward launch exceeds CUDA grid limit");
    c10::cuda::CUDAGuard guard(incoming.device());
    // Lazy flags must be resolved before raw pointer access. Ordinary strided,
    // transposed and expanded gradients are read directly, without a copy.
    auto gradient = incoming.resolve_conj().resolve_neg();
    auto output = at::empty({b, c, h, w}, gradient.options().dtype(at::kComplexFloat));
    if (n == 0) return output;
    I strides[4], max_offset = 0;
    for (int dim = 0; dim < 4; ++dim) {
        // A singleton dimension never contributes an offset; its arbitrary
        // stride must not prevent otherwise valid 32-bit indexing.
        strides[dim] = gradient.size(dim) > 1 ? gradient.stride(dim) : 0;
        TORCH_CHECK(strides[dim] >= 0,
                    "real_crop backward does not support negative storage strides");
        const I extent = gradient.size(dim) - 1;
        TORCH_CHECK(extent == 0 || strides[dim] <= (INT64_MAX - max_offset) / extent,
                    "real_crop backward gradient offset overflow");
        max_offset += extent * strides[dim];
    }
    const auto stream = c10::cuda::getCurrentCUDAStream(gradient.get_device());
    if (n <= INT_MAX - 256 && max_offset <= INT_MAX) {
        real_crop_backward_kernel<int><<<blocks, 256, 0, stream>>>(
            gradient.data_ptr<float>(), output.data_ptr<Z<float>>(),
            int(n), int(c), int(h), int(w), int(pad),
            int(strides[0]), int(strides[1]), int(strides[2]), int(strides[3]));
    } else {
        real_crop_backward_kernel<I><<<blocks, 256, 0, stream>>>(
            gradient.data_ptr<float>(), output.data_ptr<Z<float>>(), n, c, h, w, pad,
            strides[0], strides[1], strides[2], strides[3]);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
