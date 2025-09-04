#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
struct acc_type_map
{
    using type = float;
};
template <>
struct acc_type_map<double>
{
    using type = double;
};
template <>
struct acc_type_map<float>
{
    using type = float;
};
template <>
struct acc_type_map<at::Half>
{
    using type = float;
};
template <>
struct acc_type_map<at::BFloat16>
{
    using type = float;
};
template <>
struct acc_type_map<c10::complex<double>>
{
    using type = c10::complex<double>;
};
template <>
struct acc_type_map<c10::complex<float>>
{
    using type = c10::complex<float>;
};
template <>
struct acc_type_map<c10::complex<at::Half>>
{
    using type = c10::complex<float>;
};

template <typename T>
using acc_scalar_t = typename acc_type_map<T>::type;

template <typename scalar_t>
__global__ void weighted_block_mean_kernel(
    const scalar_t *__restrict__ in,
    scalar_t *__restrict__ out,
    int B, int C, int Ho, int Wo_r, int s, int Hs, int Ws_r,
    int Ws_full, long long total_out)
{
    using acc_t = acc_scalar_t<scalar_t>;

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out)
        return;

    const int wo = static_cast<int>(idx % Wo_r);
    const int ho = static_cast<int>((idx / Wo_r) % Ho);
    const int c = static_cast<int>((idx / (1LL * Wo_r * Ho)) % C);
    const int b = static_cast<int>(idx / (1LL * Wo_r * Ho * C));

    const int hi0 = ho * s;
    const int wi0 = wo * s;
    const long long base_in = ((long long)b * C + c) * Hs * Ws_r;

    acc_t acc_val = acc_t(0);
    float acc_w = 0.0f;

    for (int di = 0; di < s; ++di)
    {
        const int hi = hi0 + di;
        if (hi >= Hs)
            continue;
        const long long row_off = base_in + (long long)hi * Ws_r;
        for (int dj = 0; dj < s; ++dj)
        {
            const int wi = wi0 + dj;
            if (wi >= Ws_r)
                continue;

            float w = 2.0f;
            if (wi == 0)
                w = 1.0f;
            if ((Ws_full % 2 == 0) && wi == (Ws_r - 1))
                w = 1.0f;

            acc_val += static_cast<acc_t>(in[row_off + wi]) * static_cast<acc_t>(w);
            acc_w += w;
        }
    }

    const long long out_off = ((long long)b * C + c) * Ho * Wo_r + (long long)ho * Wo_r + wo;
    out[out_off] = (acc_w > 1e-8f)
                       ? static_cast<scalar_t>(acc_val / static_cast<acc_t>(acc_w))
                       : static_cast<scalar_t>(0);
}

template <typename scalar_t>
__global__ void weighted_block_mean_grad_kernel(
    const scalar_t *__restrict__ grad_out,
    scalar_t *__restrict__ grad_in,
    int B, int C, int Ho, int Wo_r, int s, int Hs, int Ws_r,
    int Ws_full, long long total_in)
{
    using acc_t = acc_scalar_t<scalar_t>;

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in)
        return;

    const int wi = static_cast<int>(idx % Ws_r);
    const int hi = static_cast<int>((idx / Ws_r) % Hs);
    const int c = static_cast<int>((idx / (1LL * Ws_r * Hs)) % C);
    const int b = static_cast<int>(idx / (1LL * Ws_r * Hs * C));

    const int ho = hi / s;
    const int wo = wi / s;

    float w = 2.0f;
    if (wi == 0)
        w = 1.0f;
    if ((Ws_full % 2 == 0) && wi == (Ws_r - 1))
        w = 1.0f;

    float denom = 0.0f;
    for (int di = 0; di < s; ++di)
    {
        const int hi2 = ho * s + di;
        if (hi2 >= Hs)
            continue;
        for (int dj = 0; dj < s; ++dj)
        {
            const int wi2 = wo * s + dj;
            if (wi2 >= Ws_r)
                continue;
            float w2 = 2.0f;
            if (wi2 == 0)
                w2 = 1.0f;
            if ((Ws_full % 2 == 0) && wi2 == (Ws_r - 1))
                w2 = 1.0f;
            denom += w2;
        }
    }
    if (denom < 1e-8f)
        denom = 1.0f;

    const long long go_off = ((long long)b * C + c) * Ho * Wo_r + (long long)ho * Wo_r + wo;
    const long long gi_off = ((long long)b * C + c) * Hs * Ws_r + (long long)hi * Ws_r + wi;

    const acc_t go_val = static_cast<acc_t>(grad_out[go_off]);
    const acc_t scale = static_cast<acc_t>(w / denom);
    grad_in[gi_off] = static_cast<scalar_t>(go_val * scale);
}

void weighted_block_mean_kernel_launcher(
    const at::Tensor &in, at::Tensor &out,
    int B, int C, int Ho, int Wo_r, int s, int Hs, int Ws_r,
    int Ws_full, long long total_out)
{
    const int threads = 256;
    const int blocks = (int)((total_out + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kHalf, at::kBFloat16, in.scalar_type(),
        "weighted_block_mean_fwd", [&]
        { weighted_block_mean_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
              B, C, Ho, Wo_r, s, Hs, Ws_r, Ws_full, total_out); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void weighted_block_mean_grad_kernel_launcher(
    const at::Tensor &grad_out, at::Tensor &grad_in,
    int B, int C, int Ho, int Wo_r, int s, int Hs, int Ws_r,
    int Ws_full, long long total_in)
{
    const int threads = 256;
    const int blocks = (int)((total_in + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad_out.scalar_type(),
        "weighted_block_mean_bwd", [&]
        { weighted_block_mean_grad_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              grad_out.data_ptr<scalar_t>(), grad_in.data_ptr<scalar_t>(),
              B, C, Ho, Wo_r, s, Hs, Ws_r, Ws_full, total_in); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
