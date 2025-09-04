#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/fft_rfftn.h>
#include <ATen/ops/fft_irfftn.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/real.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/upsample_nearest2d.h>

#include <unordered_map>
#include <list>
#include <mutex>
#include <iostream>

using at::Tensor;
using at::indexing::Slice;

void weighted_block_mean_kernel_launcher(
    const at::Tensor &in, at::Tensor &out,
    int B, int C, int Ho, int Wo_r, int s, int Hs, int Ws_r,
    int Ws_full, long long total_out);

void weighted_block_mean_grad_kernel_launcher(
    const at::Tensor &grad_out, at::Tensor &grad_in,
    int B, int C, int Ho, int Wo_r, int s, int Hs, int Ws_r,
    int Ws_full, long long total_in);

struct WeightedBlockMeanFunction : public torch::autograd::Function<WeightedBlockMeanFunction>
{
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &input, int64_t Ws_full, int64_t s)
    {
        TORCH_CHECK(input.is_cuda() && input.dim() == 4,
                    "weighted_block_mean: input must be (B,C,Hs,Ws_r) CUDA Tensor");
        TORCH_CHECK(s >= 1, "weighted_block_mean: s must be >= 1");

        if (s == 1)
        {
            ctx->saved_data["s"] = s;
            return input;
        }

        auto x = input.contiguous();
        const int B = (int)x.size(0);
        const int C = (int)x.size(1);
        const int Hs = (int)x.size(2);
        const int Ws_r = (int)x.size(3);

        TORCH_CHECK(Hs % s == 0, "weighted_block_mean: H must be divisible by s");
        const int Ho = Hs / (int)s;
        const int Wo_r = (Ws_r + (int)s - 1) / (int)s; // ceil_div for safety

        auto out = at::empty({B, C, Ho, Wo_r}, x.options());
        const long long total_out = out.numel();

        weighted_block_mean_kernel_launcher(
            x, out, B, C, Ho, Wo_r, (int)s, Hs, Ws_r, (int)Ws_full, total_out);

        ctx->save_for_backward({x});
        ctx->saved_data["s"] = s;
        ctx->saved_data["Ws_full"] = Ws_full;
        return out;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto go = grad_outputs[0].contiguous();
        auto s = ctx->saved_data["s"].toInt();

        if (s == 1)
        {
            return {go, torch::Tensor(), torch::Tensor()};
        }

        auto Ws_full = ctx->saved_data["Ws_full"].toInt();
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];

        const int B = (int)input.size(0);
        const int C = (int)input.size(1);
        const int Hs = (int)input.size(2);
        const int Ws_r = (int)input.size(3);
        const int Ho = (int)go.size(2);
        const int Wo_r = (int)go.size(3);

        auto grad_in = at::empty_like(input);
        const long long total_in = input.numel();

        weighted_block_mean_grad_kernel_launcher(
            go, grad_in, B, C, Ho, Wo_r, (int)s, Hs, Ws_r, (int)Ws_full, total_in);

        return {grad_in, torch::Tensor(), torch::Tensor()};
    }
};

static inline Tensor sfold_upsample_zero_insertion(const Tensor &x, int64_t s)
{
    TORCH_CHECK(x.dim() == 4, "sfold_upsample expects (B,C,H,W)");
    if (s == 1)
        return x;
    const auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto y = at::zeros({B, C, H * s, W * s}, x.options());
    y.index_put_({Slice(), Slice(), Slice(0, c10::nullopt, s), Slice(0, c10::nullopt, s)}, x);
    return y;
}

at::Tensor weighted_block_mean_cuda(const at::Tensor &input, int64_t Ws_full, int64_t s)
{
    return WeightedBlockMeanFunction::apply(input, Ws_full, s);
}

struct FBKey
{
    int64_t device_id;
    at::ScalarType dtype;
    int64_t channels;
    int64_t H, W;
    void *ptr;
    bool operator==(const FBKey &other) const
    {
        return device_id == other.device_id && dtype == other.dtype &&
               channels == other.channels && H == other.H && W == other.W &&
               ptr == other.ptr;
    }
};
namespace std
{
    template <>
    struct hash<FBKey>
    {
        size_t operator()(const FBKey &k) const
        {
            return ((hash<int64_t>()(k.device_id) ^ hash<int64_t>()(k.channels)) << 1) ^
                   ((hash<int64_t>()(k.H) ^ hash<int64_t>()(k.W)) << 1) ^
                   ((hash<void *>()(k.ptr)) ^ hash<int>()(static_cast<int>(k.dtype)));
        }
    };
}

constexpr size_t FB_CACHE_MAX_SIZE = 64;
static std::unordered_map<FBKey, std::pair<at::Tensor, at::Tensor>> fb_cache;
static std::list<FBKey> fb_cache_lru;
static std::mutex fb_cache_mutex;

static inline std::pair<Tensor, Tensor> p2o_cached_rfft(const Tensor &psf, int64_t H, int64_t W)
{
    const bool training_with_grad = at::GradMode::is_enabled() && psf.requires_grad();

    auto C = psf.size(1);
    FBKey key{psf.device().index(), psf.scalar_type(), C, H, W, psf.data_ptr()};

    if (!training_with_grad)
    {
        std::lock_guard<std::mutex> lock(fb_cache_mutex);
        auto it = fb_cache.find(key);
        if (it != fb_cache.end())
        {
            fb_cache_lru.remove(key);
            fb_cache_lru.push_front(key);
            return it->second;
        }
    }

    Tensor otf = at::zeros({1, C, H, W}, psf.options());
    int64_t kh = psf.size(2), kw = psf.size(3);
    otf.index_put_({0, Slice(), Slice(0, kh), Slice(0, kw)}, psf);
    otf = at::roll(otf, {-kh / 2, -kw / 2}, {-2, -1});

    Tensor FB = at::fft_rfftn(otf, c10::nullopt, {-2, -1}, c10::nullopt);
    Tensor F2B = at::abs(FB).pow(2);

    if (!training_with_grad)
    {
        std::lock_guard<std::mutex> lock(fb_cache_mutex);
        fb_cache[key] = {FB, F2B};
        fb_cache_lru.push_front(key);
        if (fb_cache_lru.size() > FB_CACHE_MAX_SIZE)
        {
            fb_cache.erase(fb_cache_lru.back());
            fb_cache_lru.pop_back();
        }
    }
    return {FB, F2B};
}

Tensor converse2d_forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int64_t scale, double eps)
{
    TORCH_CHECK(x.dim() == 4, "x must be (B,C,H,W)");
    TORCH_CHECK(x0.dim() == 4, "x0 must be (B,C,Hs,Ws)");
    TORCH_CHECK(weight.dim() == 4 && weight.size(0) == 1, "weight must be (1,C,kh,kw)");
    TORCH_CHECK(bias.dim() == 4 && bias.size(0) == 1 && bias.size(2) == 1 && bias.size(3) == 1, "bias must be (1,C,1,1)");
    TORCH_CHECK(x.device() == x0.device() && x.device() == weight.device() && x.device() == bias.device(), "tensors on same device");

    x = x.contiguous();
    x0 = x0.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t Hs = H * scale;
    const int64_t Ws = W * scale;

    Tensor lambda_ = at::sigmoid(bias - 9.0) + eps;
    Tensor STy = sfold_upsample_zero_insertion(x, scale);

    auto [FB, F2B] = p2o_cached_rfft(weight, Hs, Ws);
    Tensor FBC = at::conj_physical(FB);
    Tensor F_STy = at::fft_rfftn(STy, c10::nullopt, {-2, -1}, c10::nullopt);
    Tensor F_lx0 = at::fft_rfftn(lambda_ * x0, c10::nullopt, {-2, -1}, c10::nullopt);

    Tensor FBFy = FBC * F_STy;
    Tensor FR = FBFy + F_lx0;

    Tensor x1 = FB * FR;
    Tensor FBR = weighted_block_mean_cuda(x1, Ws, scale);
    Tensor invW = weighted_block_mean_cuda(F2B, Ws, scale);

    Tensor invW_plus = invW + lambda_;
    Tensor invWBR = FBR / invW_plus;

    const int64_t Ws_rfft = Ws / 2 + 1;
    const int64_t Ho = invWBR.size(2);
    const int64_t Wo_r = invWBR.size(3);

    Tensor invWBR_exp = invWBR
                            .view({B, C, Ho, 1, Wo_r, 1})
                            .expand({B, C, Ho, (int64_t)scale, Wo_r, (int64_t)scale})
                            .reshape({B, C, Ho * (int64_t)scale, Wo_r * (int64_t)scale});

    if (invWBR_exp.size(-2) != Hs || invWBR_exp.size(-1) != Ws_rfft)
    {
        using at::indexing::Slice;
        invWBR_exp = invWBR_exp.index({Slice(), Slice(),
                                       Slice(0, Hs), Slice(0, Ws_rfft)});
    }

    Tensor FCBinvWBR = FBC * invWBR_exp;
    Tensor FX = (FR - FCBinvWBR) / lambda_;
    Tensor out = at::fft_irfftn(FX, {Hs, Ws}, {-2, -1}, c10::nullopt);
    return out;
}

void clear_fb_cache()
{
    std::lock_guard<std::mutex> lock(fb_cache_mutex);
    fb_cache.clear();
    fb_cache_lru.clear();
}

TORCH_LIBRARY(converse2d, m)
{
    m.def("forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int scale, float eps=1e-5) -> Tensor");
    m.def("clear_cache() -> ()");
}
TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m)
{
    m.impl("forward", TORCH_FN(converse2d_forward));
    m.impl("clear_cache", TORCH_FN(clear_fb_cache));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
