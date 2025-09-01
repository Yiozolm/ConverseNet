#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>   // getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>      // CUDAGuard (moved from at::cuda)
#include <type_traits>

using at::Tensor;

template <typename T> struct acc_type_map { using type = float; };
template <> struct acc_type_map<double>   { using type = double; };
template <> struct acc_type_map<float>    { using type = float;  };
template <> struct acc_type_map<at::Half> { using type = float;  };
template <> struct acc_type_map<at::BFloat16> { using type = float; };
template <typename T> using acc_t = typename acc_type_map<T>::type;

// ======================= block_mean forward =======================
template <typename scalar_t>
__global__ void block_mean_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int B, int C, int Hin, int Win, int s)
{
    const int Hout = Hin / s;
    const int Wout = Win / s;
    const int Nout = B * C * Hout * Wout;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nout) return;

    int wout = idx % Wout;
    int t    = idx / Wout;
    int hout = t % Hout;
    t       /= Hout;
    int c    = t % C;
    int b    = t / C;

    const int h0 = hout * s;
    const int w0 = wout * s;

    const int64_t base_in  = (((int64_t)b * C + c) * Hin + h0) * Win + w0;
    const int64_t base_out = (((int64_t)b * C + c) * Hout + hout) * Wout + wout;

    acc_t<scalar_t> sum = static_cast<acc_t<scalar_t>>(0);
    for (int dh = 0; dh < s; ++dh) {
        int64_t row = base_in + (int64_t)dh * Win;
        for (int dw = 0; dw < s; ++dw) {
            sum += static_cast<acc_t<scalar_t>>(x[row + dw]);
        }
    }
    const acc_t<scalar_t> denom = static_cast<acc_t<scalar_t>>(s) * static_cast<acc_t<scalar_t>>(s);
    y[base_out] = static_cast<scalar_t>(sum / denom);
}

Tensor block_mean_cuda(const Tensor &input, int64_t s)
{
    TORCH_CHECK(input.is_cuda(), "block_mean_cuda: input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "block_mean_cuda: expect (B,C,H,W)");
    TORCH_CHECK(s >= 1, "block_mean_cuda: s must be >= 1");
    if (s == 1) return input;

    auto x = input.contiguous();
    const int B = x.size(0);
    const int C = x.size(1);
    const int Hin = x.size(2);
    const int Win = x.size(3);
    TORCH_CHECK(Hin % s == 0 && Win % s == 0, "block_mean_cuda: H/W must be divisible by s");

    const int Hout = Hin / s;
    const int Wout = Win / s;

    auto y = at::empty({B, C, Hout, Wout}, x.options());

    const int threads = 256;
    const int blocks  = (B * C * Hout * Wout + threads - 1) / threads;

    c10::cuda::CUDAGuard guard(x.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "block_mean_forward", [&] {
        block_mean_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, Hin, Win, (int)s);
    });
    return y;
}

// ======================= block_mean backward =======================
// grad_x[b,c,i*s+p,j*s+q] = grad_y[b,c,i,j] / (s*s)
template <typename scalar_t>
__global__ void block_mean_backward_kernel(
    const scalar_t* __restrict__ gy,
    scalar_t* __restrict__ gx,
    int B, int C, int Hin, int Win, int s)
{
    const int Hout = Hin / s;
    const int Wout = Win / s;
    const int Nin  = B * C * Hin * Win;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nin) return;

    int win = idx % Win;
    int t   = idx / Win;
    int hin = t % Hin;
    t      /= Hin;
    int c   = t % C;
    int b   = t / C;

    const int hout = hin / s;
    const int wout = win / s;

    const int64_t index_out = (((int64_t)b * C + c) * Hout + hout) * Wout + wout;
    const acc_t<scalar_t> denom = static_cast<acc_t<scalar_t>>(s) * static_cast<acc_t<scalar_t>>(s);
    gx[idx] = static_cast<scalar_t>( static_cast<acc_t<scalar_t>>(gy[index_out]) / denom );
}

Tensor block_mean_cuda_backward(const Tensor &grad_out, int64_t s)
{
    TORCH_CHECK(grad_out.is_cuda(), "block_mean_cuda_backward: grad_out must be CUDA tensor");
    TORCH_CHECK(grad_out.dim() == 4, "block_mean_cuda_backward: expect (B,C,Hout,Wout)");
    TORCH_CHECK(s >= 1, "block_mean_cuda_backward: s must be >= 1");

    auto gy = grad_out.contiguous();
    const int B = gy.size(0);
    const int C = gy.size(1);
    const int Hout = gy.size(2);
    const int Wout = gy.size(3);

    const int Hin = Hout * s;
    const int Win = Wout * s;

    auto gx = at::empty({B, C, Hin, Win}, gy.options());

    const int threads = 256;
    const int blocks  = (B * C * Hin * Win + threads - 1) / threads;

    c10::cuda::CUDAGuard guard(gy.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gy.scalar_type(), "block_mean_backward", [&] {
        block_mean_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            gy.data_ptr<scalar_t>(), gx.data_ptr<scalar_t>(), B, C, Hin, Win, (int)s);
    });
    return gx;
}

// ======================= s-fold zero insertion forward =======================
template <typename scalar_t>
__global__ void sfold_upsample_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int B, int C, int H, int W, int s, int Hs, int Ws)
{
    const int Nin = B * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nin) return;

    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t    /= H;
    int c = t % C;
    int b = t / C;

    const int hs = h * s;
    const int ws = w * s;

    const int64_t out_index = (((int64_t)b * C + c) * Hs + hs) * Ws + ws;
    const scalar_t v = x[idx];
    y[out_index] = v; // 其它位置保持 0
}

Tensor sfold_upsample_cuda_launcher(const Tensor &x, int64_t s)
{
    TORCH_CHECK(x.is_cuda(), "sfold_upsample_cuda: x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "sfold_upsample_cuda: expect (B,C,H,W)");
    TORCH_CHECK(s >= 1, "sfold_upsample_cuda: s must be >= 1");
    if (s == 1) return x;

    auto xx = x.contiguous();
    const int B = xx.size(0);
    const int C = xx.size(1);
    const int H = xx.size(2);
    const int W = xx.size(3);
    const int Hs = H * s;
    const int Ws = W * s;

    auto y = at::zeros({B, C, Hs, Ws}, xx.options());

    const int threads = 256;
    const int blocks  = (B * C * H * W + threads - 1) / threads;

    c10::cuda::CUDAGuard guard(xx.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xx.scalar_type(), "sfold_upsample_forward", [&] {
        sfold_upsample_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            xx.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), B, C, H, W, (int)s, Hs, Ws);
    });
    return y;
}

// ======================= s-fold zero insertion backward =======================
// grad_x[b,c,h,w] = grad_y[b,c,h*s, w*s]
template <typename scalar_t>
__global__ void sfold_upsample_backward_kernel(
    const scalar_t* __restrict__ gy,
    scalar_t* __restrict__ gx,
    int B, int C, int H, int W, int s, int Hs, int Ws)
{
    const int Nin = B * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nin) return;

    int w = idx % W;
    int t = idx / W;
    int h = t % H;
    t    /= H;
    int c = t % C;
    int b = t / C;

    const int hs = h * s;
    const int ws = w * s;

    const int64_t in_index = (((int64_t)b * C + c) * Hs + hs) * Ws + ws;
    gx[idx] = gy[in_index];
}

Tensor sfold_upsample_cuda_backward(const Tensor &grad_out, int64_t s)
{
    TORCH_CHECK(grad_out.is_cuda(), "sfold_upsample_cuda_backward: grad_out must be CUDA tensor");
    TORCH_CHECK(grad_out.dim() == 4, "sfold_upsample_cuda_backward: expect (B,C,Hs,Ws)");
    TORCH_CHECK(s >= 1, "sfold_upsample_cuda_backward: s must be >= 1");
    if (s == 1) return grad_out;

    auto gy = grad_out.contiguous();
    const int B  = gy.size(0);
    const int C  = gy.size(1);
    const int Hs = gy.size(2);
    const int Ws = gy.size(3);
    TORCH_CHECK(Hs % s == 0 && Ws % s == 0, "sfold_upsample_cuda_backward: Hs/Ws must be divisible by s");
    const int H  = Hs / s;
    const int W  = Ws / s;

    auto gx = at::empty({B, C, H, W}, gy.options());

    const int threads = 256;
    const int blocks  = (B * C * H * W + threads - 1) / threads;

    c10::cuda::CUDAGuard guard(gy.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gy.scalar_type(), "sfold_upsample_backward", [&] {
        sfold_upsample_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            gy.data_ptr<scalar_t>(), gx.data_ptr<scalar_t>(), B, C, H, W, (int)s, Hs, Ws);
    });
    return gx;
}
