#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/complex.h>

// ======================
// block mean (forward):
//   in : (B,C,Hs,Ws)
//   out: (B,C,Ho,Wo), Ho=Hs/s, Wo=Ws/s
// ======================

template <typename T>
struct AccT
{
    using type = T;
};
template <>
struct AccT<at::Half>
{
    using type = float;
};
template <>
struct AccT<at::BFloat16>
{
    using type = float;
};
template <typename scalar_t>
__global__ void block_mean_kernel(
    const scalar_t *__restrict__ in, // (B,C,Hs,Ws)
    scalar_t *__restrict__ out,      // (B,C,Ho,Wo)
    int B, int C, int Ho, int Wo, int s, int Hs, int Ws,
    long long total_out)
{
    using acc_t = typename AccT<scalar_t>::type;

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out)
        return;

    int wo = static_cast<int>(idx % Wo);
    int ho = static_cast<int>((idx / Wo) % Ho);
    int c = static_cast<int>((idx / (1LL * Wo * Ho)) % C);
    int b = static_cast<int>(idx / (1LL * Wo * Ho * C));

    const int hi0 = ho * s;
    const int wi0 = wo * s;

    const long long base_in = ((long long)b * C + c) * Hs * Ws;

    acc_t acc = acc_t(0);
    for (int di = 0; di < s; ++di)
    {
        const int hi = hi0 + di;
        const long long row_off = base_in + (long long)hi * Ws + wi0;
#pragma unroll
        for (int dj = 0; dj < s; ++dj)
        {
            acc += static_cast<acc_t>(in[row_off + dj]);
        }
    }
    const float inv_area = 1.0f / (s * s);
    acc = acc * static_cast<acc_t>(inv_area);

    const long long out_off = ((long long)b * C + c) * Ho * Wo + (long long)ho * Wo + wo;
    out[out_off] = static_cast<scalar_t>(acc);
}

struct BlockMeanFunctionV3 : public torch::autograd::Function<BlockMeanFunctionV3>
{
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &input, int64_t s)
    {
        TORCH_CHECK(input.is_cuda() && input.dim() == 4, "block_mean_cuda: input must be (B,C,Hs,Ws) CUDA");
        TORCH_CHECK(s >= 1, "block_mean_cuda: s must be >= 1");

        auto x = input.contiguous();
        const int B = (int)x.size(0);
        const int C = (int)x.size(1);
        const int Hs = (int)x.size(2);
        const int Ws = (int)x.size(3);

        TORCH_CHECK(Hs % s == 0 && Ws % s == 0, "block_mean_cuda: H,W must be divisible by s");
        const int Ho = Hs / (int)s;
        const int Wo = Ws / (int)s;

        auto out = at::empty({B, C, Ho, Wo}, x.options());

        // launch forward kernel
        {
            const long long total_out = 1LL * B * C * Ho * Wo;
            const int threads = 256;
            const int blocks = (int)((total_out + threads - 1) / threads);
            auto stream = at::cuda::getCurrentCUDAStream();

            AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                        x.scalar_type(), "block_mean_v3_fwd", [&]
                                                        { block_mean_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                                              x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                                                              B, C, Ho, Wo, (int)s, Hs, Ws, total_out); });
        }

        // save for backward
        ctx->saved_data["B"] = (int64_t)B;
        ctx->saved_data["C"] = (int64_t)C;
        ctx->saved_data["Hs"] = (int64_t)Hs;
        ctx->saved_data["Ws"] = (int64_t)Ws;
        ctx->saved_data["s"] = (int64_t)s;
        return out;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::tensor_list grad_outputs)
    {
        auto go = grad_outputs[0]; // (B,C,Ho,Wo)
        const int B = (int)ctx->saved_data["B"].toInt();
        const int C = (int)ctx->saved_data["C"].toInt();
        const int Hs = (int)ctx->saved_data["Hs"].toInt();
        const int Ws = (int)ctx->saved_data["Ws"].toInt();
        const int s = (int)ctx->saved_data["s"].toInt();

        const int Ho = Hs / s;
        const int Wo = Ws / s;

        // gi = expand( go / (s*s), dims=[B,C,Ho,1,Wo,1] ) -> reshape(B,C,Hs,Ws)
        auto go_scaled = go / static_cast<double>(s * s);
        auto gi = go_scaled.view({B, C, Ho, 1, Wo, 1})
                      .expand({B, C, Ho, s, Wo, s})
                      .reshape({B, C, Hs, Ws})
                      .contiguous();

        return {gi, torch::Tensor()}; // no grad for s
    }
};

at::Tensor block_mean_cuda(const at::Tensor &input, int64_t s)
{
    return BlockMeanFunctionV3::apply(input, s);
}
