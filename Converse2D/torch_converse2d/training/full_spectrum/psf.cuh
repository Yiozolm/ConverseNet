#pragma once
#include <climits>

// Included within converse2d::full_training by full_fusion.cu. Both kernels
// only move FP32 values; no reduction, arithmetic reassociation or atomics.
template<class Index> __global__ void psf_pad_roll_kernel(
    const float* source, float* output, Index n, Index C, Index H, Index W,
    Index kh, Index kw, I s0, I s1, I s2, I s3) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Index bc = i / (H * W);
    const Index h = ((i / W) % H + kh / 2) % H;
    const Index w = (i % W + kw / 2) % W;
    output[i] = h < kh && w < kw
        ? source[I(bc / C) * s0 + I(bc % C) * s1 + I(h) * s2 + I(w) * s3]
        : 0.0f;
}

template<class Index> __global__ void psf_pad_roll_backward_kernel(
    const float* gradient, float* output, Index n, Index C, Index H, Index W,
    Index kh, Index kw, I s0, I s1, I s2, I s3) {
    const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Index bc = i / (kh * kw);
    Index h = (i / kw) % kh - kh / 2;
    Index w = i % kw - kw / 2;
    if (h < 0) h += H;
    if (w < 0) w += W;
    output[i] = gradient[I(bc / C) * s0 + I(bc % C) * s1 + I(h) * s2 + I(w) * s3];
}

Tensor psf_pad_roll_cuda(const Tensor& weight, I h, I w) {
    auto source = weight.resolve_conj().resolve_neg();
    auto output = at::empty({weight.size(0), weight.size(1), h, w}, weight.options());
    const I n = output.numel(), c = weight.size(1), kh = weight.size(2), kw = weight.size(3);
    const auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());
    const auto blocks = (n + 255) / 256;
    TORCH_CHECK(blocks <= INT_MAX, "PSF launch exceeds CUDA grid limit");
    if (n <= INT_MAX - 256 && h <= INT_MAX / 2 && w <= INT_MAX / 2) {
        psf_pad_roll_kernel<int><<<blocks, 256, 0, stream>>>(
            source.data_ptr<float>(), output.data_ptr<float>(), int(n), int(c), int(h), int(w), int(kh), int(kw),
            source.stride(0), source.stride(1), source.stride(2), source.stride(3));
    } else {
        psf_pad_roll_kernel<I><<<blocks, 256, 0, stream>>>(
            source.data_ptr<float>(), output.data_ptr<float>(), n, c, h, w, kh, kw,
            source.stride(0), source.stride(1), source.stride(2), source.stride(3));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor psf_pad_roll_backward_cuda(const Tensor& incoming, I kh, I kw) {
    auto gradient = incoming.resolve_conj().resolve_neg();
    auto output = at::empty({gradient.size(0), gradient.size(1), kh, kw}, gradient.options());
    const I n = output.numel(), c = gradient.size(1), h = gradient.size(2), w = gradient.size(3);
    const auto stream = c10::cuda::getCurrentCUDAStream(gradient.get_device());
    const auto blocks = (n + 255) / 256;
    TORCH_CHECK(blocks <= INT_MAX, "PSF backward launch exceeds CUDA grid limit");
    if (n <= INT_MAX - 256 && h <= INT_MAX && w <= INT_MAX) {
        psf_pad_roll_backward_kernel<int><<<blocks, 256, 0, stream>>>(
            gradient.data_ptr<float>(), output.data_ptr<float>(), int(n), int(c), int(h), int(w), int(kh), int(kw),
            gradient.stride(0), gradient.stride(1), gradient.stride(2), gradient.stride(3));
    } else {
        psf_pad_roll_backward_kernel<I><<<blocks, 256, 0, stream>>>(
            gradient.data_ptr<float>(), output.data_ptr<float>(), n, c, h, w, kh, kw,
            gradient.stride(0), gradient.stride(1), gradient.stride(2), gradient.stride(3));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
