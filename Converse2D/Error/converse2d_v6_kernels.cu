#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/complex.h>
#include <ATen/ATen.h>
#include <string>
#include <ATen/Dispatch.h>
#include <thrust/complex.h>

// ======================================================================
//  S-FOLD UPSAMPLE  (zero-insertion upsample)
//    forward: out[b,c,h*s, w*s] = x[b,c,h,w]; others = 0
//    backward: grad_x[b,c,h,w]  = grad_out[b,c,h*s, w*s]
//  dtypes: float/double/half/bfloat16
// ======================================================================

using namespace at;
using namespace at::indexing;

template <typename scalar_t>
__global__ void sfold_upsample_kernel(
    const scalar_t *__restrict__ x,
    scalar_t *__restrict__ out,
    int B, int C, int H, int W, int s,
    int Hs, int Ws, long long total_in)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in)
        return;

    int w = static_cast<int>(idx % W);
    int h = static_cast<int>((idx / W) % H);
    int c = static_cast<int>((idx / (1LL * W * H)) % C);
    int b = static_cast<int>(idx / (1LL * W * H * C));

    long long in_off = ((long long)b * C + c) * H * W + (long long)h * W + w;
    long long out_off = ((long long)b * C + c) * Hs * Ws + (long long)(h * s) * Ws + (w * s);

    out[out_off] = x[in_off];
}

template <typename scalar_t>
__global__ void sfold_downsample_grad_kernel( // backward of zero-insertion upsample
    const scalar_t *__restrict__ grad_out,    // (B,C,Hs,Ws)
    scalar_t *__restrict__ grad_in,           // (B,C,H,W)
    int B, int C, int H, int W, int s, int Hs, int Ws, long long total_in)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in)
        return;

    int w = static_cast<int>(idx % W);
    int h = static_cast<int>((idx / W) % H);
    int c = static_cast<int>((idx / (1LL * W * H)) % C);
    int b = static_cast<int>(idx / (1LL * W * H * C));

    long long in_off = ((long long)b * C + c) * H * W + (long long)h * W + w;
    long long out_off = ((long long)b * C + c) * Hs * Ws + (long long)(h * s) * Ws + (w * s);

    grad_in[in_off] = grad_out[out_off];
}

struct SFoldFunction : public torch::autograd::Function<SFoldFunction>
{
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &x, int64_t scale)
    {
        TORCH_CHECK(x.is_cuda() && x.dim() == 4, "sfold: x must be (B,C,H,W) CUDA");
        TORCH_CHECK(scale >= 1, "sfold: scale must be >= 1");
        if (scale == 1)
        {
            ctx->saved_data["s"] = (int64_t)1;
            return x;
        }

        auto x_ = x.contiguous();
        const int B = (int)x_.size(0), C = (int)x_.size(1), H = (int)x_.size(2), W = (int)x_.size(3);
        const int s = (int)scale, Hs = H * s, Ws = W * s;

        auto out = at::zeros({B, C, Hs, Ws}, x_.options());

        const long long total = 1LL * B * C * H * W;
        const int threads = 256, blocks = (int)((total + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_.scalar_type(), "sfold_fwd", [&]
                                        { sfold_upsample_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                              x_.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                                              B, C, H, W, s, Hs, Ws, total); });

        // save for backward
        ctx->saved_data["B"] = (int64_t)B;
        ctx->saved_data["C"] = (int64_t)C;
        ctx->saved_data["H"] = (int64_t)H;
        ctx->saved_data["W"] = (int64_t)W;
        ctx->saved_data["s"] = (int64_t)s;
        return out;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::tensor_list grad_outputs)
    {
        auto go = grad_outputs[0]; // (B,C,Hs,Ws)
        const int B = (int)ctx->saved_data["B"].toInt();
        const int C = (int)ctx->saved_data["C"].toInt();
        const int H = (int)ctx->saved_data["H"].toInt();
        const int W = (int)ctx->saved_data["W"].toInt();
        const int s = (int)ctx->saved_data["s"].toInt();
        const int Hs = H * s, Ws = W * s;

        at::Tensor gx;
        if (s == 1)
        {
            gx = go; // identity
        }
        else
        {
            gx = go.index({Slice(), Slice(), Slice(0, Hs, s), Slice(0, Ws, s)}).contiguous();
        }
        return {gx, torch::Tensor()}; // no grad for scale
    }
};

// exposed symbol for v4.cpp
at::Tensor sfold_upsample_cuda_launcher(const at::Tensor &x, int64_t scale)
{
    return SFoldFunction::apply(x, scale);
}

// ======================================================================
//  BLOCK MEAN  over non-overlapping s×s tiles
//    forward: out[b,c,ho,wo] = mean_{i,j in s×s} in[b,c, ho*s+i, wo*s+j]
//    backward: grad_in[b,c,hi,wi] = grad_out[b,c,hi/s, wi/s] / (s*s)
//  dtypes: float/double/half/bfloat16 + complex64/complex128
// ======================================================================

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

template <typename scalar_t>
__global__ void block_mean_grad_kernel(
    const scalar_t *__restrict__ grad_out, // (B,C,Ho,Wo)
    scalar_t *__restrict__ grad_in,        // (B,C,Hs,Ws)
    int B, int C, int Ho, int Wo, int s, int Hs, int Ws,
    long long total_in)
{
    using acc_t = typename AccT<scalar_t>::type;

    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in)
        return;

    int wi = static_cast<int>(idx % Ws);
    int hi = static_cast<int>((idx / Ws) % Hs);
    int c = static_cast<int>((idx / (1LL * Ws * Hs)) % C);
    int b = static_cast<int>(idx / (1LL * Ws * Hs * C));

    const int ho = hi / s;
    const int wo = wi / s;

    const long long out_off = ((long long)b * C + c) * Ho * Wo + (long long)ho * Wo + wo;
    acc_t g = static_cast<acc_t>(grad_out[out_off]) * static_cast<acc_t>(1.0f / (s * s));

    const long long in_off = ((long long)b * C + c) * Hs * Ws + (long long)hi * Ws + wi;
    grad_in[in_off] = static_cast<scalar_t>(g);
}

struct BlockMeanFunction : public torch::autograd::Function<BlockMeanFunction>
{
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &input, int64_t s)
    {
        TORCH_CHECK(input.is_cuda() && input.dim() == 4, "block_mean: input must be (B,C,Hs,Ws) CUDA");
        TORCH_CHECK(s >= 1, "block_mean: s must be >= 1");

        auto x = input.contiguous();
        const int B = (int)x.size(0), C = (int)x.size(1), Hs = (int)x.size(2), Ws = (int)x.size(3);
        TORCH_CHECK(Hs % s == 0 && Ws % s == 0, "block_mean: H,W must be divisible by s");
        const int Ho = Hs / (int)s, Wo = Ws / (int)s;

        auto out = at::empty({B, C, Ho, Wo}, x.options());

        const long long total_out = 1LL * B * C * Ho * Wo;
        const int threads = 256, blocks = (int)((total_out + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "block_mean_fwd", [&]
                                                    { block_mean_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                                          x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                                                          B, C, Ho, Wo, (int)s, Hs, Ws, total_out); });

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
        const int Ho = Hs / s, Wo = Ws / s;

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
    return BlockMeanFunction::apply(input, s);
}

template <typename real_t>
__global__ void fb_postprocess_kernel(
    const thrust::complex<real_t>* __restrict__ FB,
    thrust::complex<real_t>* __restrict__ FBC,
    real_t* __restrict__ F2B,
    int64_t N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    thrust::complex<real_t> val = FB[idx];
    real_t  re = val.real();
    real_t  im = val.imag();
    F2B[idx] = re * re + im * im;
    FBC[idx] = thrust::complex<real_t>(re, -im);
}

std::tuple<at::Tensor, at::Tensor> fb_postprocess_cuda(const at::Tensor& FB) {
    TORCH_CHECK(FB.is_cuda(),   "FB must be CUDA tensor");
    TORCH_CHECK(FB.is_complex(),"FB must be complex");

    auto FBc = FB.contiguous();
    const auto N = FBc.numel();

    at::Tensor FBC = at::empty_like(FBc);
    at::Tensor F2B = at::empty(FBc.sizes(),
        FBc.scalar_type() == at::kComplexFloat  ? FBc.options().dtype(at::kFloat)
                                                : FBc.options().dtype(at::kDouble));

    constexpr int threads = 256;
    const int blocks = (static_cast<int>(N) + threads - 1) / threads;

    AT_DISPATCH_COMPLEX_TYPES(FB.scalar_type(), "fb_postprocess_cuda", [&] {
        using real_t = typename scalar_t::value_type;
        fb_postprocess_kernel<real_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const thrust::complex<real_t>*>(FBc.data_ptr<scalar_t>()),
            reinterpret_cast<thrust::complex<real_t>*>(FBC.data_ptr<scalar_t>()),
            F2B.data_ptr<real_t>(),
            N
        );
    });

    return {FBC, F2B};
}