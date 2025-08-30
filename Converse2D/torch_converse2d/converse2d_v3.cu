#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

using at::Tensor;

template <typename scalar_t>
__global__ void block_mean_kernel(
    const scalar_t *__restrict__ input,
    scalar_t *__restrict__ output,
    int64_t B, int64_t C, int64_t H, int64_t W, int64_t s,
    int64_t Hs, int64_t Ws, int64_t n_out // B*C*H*W
)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_out)
        return;

    int64_t w = tid % W;
    int64_t t1 = tid / W;
    int64_t h = t1 % H;
    int64_t t2 = t1 / H;
    int64_t c = t2 % C;
    int64_t b = t2 / C;

    const int64_t hs0 = h * s;
    const int64_t ws0 = w * s;

    const int64_t in_bc_off = ((b * C + c) * Hs);
    scalar_t sum = scalar_t(0);

    for (int64_t di = 0; di < s; ++di)
    {
        const int64_t hs = hs0 + di;
        const int64_t row_off = (in_bc_off + hs) * Ws;
        for (int64_t dj = 0; dj < s; ++dj)
        {
            const int64_t ws = ws0 + dj;
            sum += input[row_off + ws];
        }
    }

    using value_t = typename c10::scalar_value_type<scalar_t>::type;
    const value_t denom = static_cast<value_t>(s * s);
    output[tid] = sum / denom;
}

Tensor block_mean_cuda(const Tensor &input, int64_t s)
{
    TORCH_CHECK(input.is_cuda(), "block_mean_cuda: input must be CUDA");
    TORCH_CHECK(input.dim() == 4, "block_mean_cuda: input must be (B,C,Hs,Ws)");
    TORCH_CHECK(s > 0, "block_mean_cuda: s must be > 0");
    const int64_t B = input.size(0);
    const int64_t C = input.size(1);
    const int64_t Hs = input.size(2);
    const int64_t Ws = input.size(3);
    TORCH_CHECK(Hs % s == 0 && Ws % s == 0, "Hs and Ws must be divisible by s");

    const int64_t H = Hs / s;
    const int64_t W = Ws / s;

    Tensor output = at::empty({B, C, H, W}, input.options());
    const int64_t n_out = B * C * H * W;

    const int threads = 256;
    const int blocks = static_cast<int>((n_out + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kComplexFloat, at::kComplexDouble,
                                    input.scalar_type(), "block_mean_kernel", [&]
                                    { block_mean_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                          input.data_ptr<scalar_t>(),
                                          output.data_ptr<scalar_t>(),
                                          B, C, H, W, s, Hs, Ws, n_out); });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

TORCH_LIBRARY_FRAGMENT(converse2d_v3, m) {}
