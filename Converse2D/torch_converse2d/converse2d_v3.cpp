#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/fft_fftn.h>
#include <ATen/ops/fft_ifftn.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/real.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/conj_physical.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <list>
#include <mutex>

using at::Tensor;

// ---------- FB Cache ----------
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
} // namespace std

constexpr size_t FB_CACHE_MAX_SIZE = 64;
static std::unordered_map<FBKey, std::pair<at::Tensor, at::Tensor>> fb_cache;
static std::list<FBKey> fb_cache_lru;
static std::mutex fb_cache_mutex;

__global__ void block_mean_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    int B, int C, int H, int W, int s)
{
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h = blockIdx.x / W;
    int w = blockIdx.x % W;

    if (h >= H || w >= W)
        return;

    int Hs = H * s;
    int Ws = W * s;

    float sum = 0.0f;
    for (int i = 0; i < s; ++i)
    {
        for (int j = 0; j < s; ++j)
        {
            int hs = h * s + i;
            int ws = w * s + j;
            int idx = ((b * C + c) * Hs + hs) * Ws + ws;
            sum += input[idx];
        }
    }

    int out_idx = ((b * C + c) * H + h) * W + w;
    output[out_idx] = sum / (s * s);
}

Tensor block_mean_cuda(const Tensor &input, int64_t s)
{
    TORCH_CHECK(input.dim() == 4, "input must be (B,C,Hs,Ws)");
    TORCH_CHECK(input.device().is_cuda(), "input must be CUDA tensor");

    const int64_t B = input.size(0);
    const int64_t C = input.size(1);
    const int64_t Hs = input.size(2);
    const int64_t Ws = input.size(3);
    TORCH_CHECK(Hs % s == 0 && Ws % s == 0, "Hs and Ws must be divisible by s");

    const int64_t H = Hs / s;
    const int64_t W = Ws / s;

    Tensor output = at::empty({B, C, H, W}, input.options());

    const int threads = 1;
    const dim3 blocks(H * W, C, B);

    block_mean_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W, s);

    return output;
}

static inline std::pair<Tensor, Tensor> p2o_cached(const Tensor &psf, int64_t H, int64_t W)
{
    auto C = psf.size(1);
    FBKey key{psf.device().index(), psf.scalar_type(), C, H, W, psf.data_ptr()};

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
    otf.index_put_({0, at::indexing::Slice(), at::indexing::Slice(0, kh), at::indexing::Slice(0, kw)}, psf);
    otf = at::roll(otf, {-kh / 2, -kw / 2}, {-2, -1});
    Tensor FB = at::fft_fftn(otf, c10::nullopt, {-2, -1}, c10::nullopt);
    Tensor F2B = at::abs(FB).pow(2);

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

// ---------- Utility ----------
static inline Tensor sfold_upsample_zero_insertion(const Tensor &x, int64_t s)
{
    if (s == 1)
        return x;
    auto sizes = x.sizes().vec();
    sizes[sizes.size() - 2] *= s;
    sizes[sizes.size() - 1] *= s;
    Tensor z = at::zeros(sizes, x.options());
    z.index_put_({at::indexing::Slice(), at::indexing::Slice(),
                  at::indexing::Slice(0, z.size(-2), s),
                  at::indexing::Slice(0, z.size(-1), s)},
                 x);
    return z;
}

static inline Tensor splits_mean_then_mean(const Tensor &a, int64_t s)
{
    const auto &sizes = a.sizes();
    const int64_t L = a.dim();
    const int64_t W = sizes[L - 2];
    const int64_t H = sizes[L - 1];
    const int64_t W_s = W / s;
    const int64_t H_s = H / s;

    std::vector<int64_t> view_shape;
    view_shape.reserve(L + 2);
    for (int64_t i = 0; i < L - 2; ++i)
        view_shape.push_back(sizes[i]);
    view_shape.push_back(s);
    view_shape.push_back(W_s);
    view_shape.push_back(s);
    view_shape.push_back(H_s);
    Tensor v = a.view(view_shape);

    std::vector<int64_t> perm;
    perm.reserve(view_shape.size());
    for (int64_t i = 0; i < L - 2; ++i)
        perm.push_back(i);
    perm.push_back(L - 2 + 1);
    perm.push_back(L - 2 + 3);
    perm.push_back(L - 2 + 0);
    perm.push_back(L - 2 + 2);
    Tensor p = v.permute(perm).contiguous();

    std::vector<int64_t> merge_shape;
    merge_shape.reserve(L + 1);
    for (int64_t i = 0; i < L - 2; ++i)
        merge_shape.push_back(p.size(i));
    merge_shape.push_back(W_s);
    merge_shape.push_back(H_s);
    merge_shape.push_back(s * s);
    Tensor r = p.view(merge_shape);

    return r.mean(-1, /*keepdim=*/false);
}

// ---------- Forward ----------
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

    auto [FB, F2B] = p2o_cached(weight, Hs, Ws);
    Tensor FBC = at::conj_physical(FB);

    Tensor F_STy = at::fft_fftn(STy, c10::nullopt, {-2, -1}, c10::nullopt);
    Tensor FBFy = FBC * F_STy;
    Tensor FR = FBFy + at::fft_fftn(lambda_ * x0, c10::nullopt, {-2, -1}, c10::nullopt);

    Tensor x1 = FB * FR;
    // Tensor FBR = splits_mean_then_mean(x1, scale);
    // Tensor invW = splits_mean_then_mean(F2B, scale);
    Tensor FBR = block_mean_cuda(x1, scale);
    Tensor invW = block_mean_cuda(F2B, scale);

    Tensor invW_plus = invW + lambda_;
    Tensor invWBR = FBR / invW_plus;

    // ---- 关键替换：避免 repeat 物化，使用广播展开（零 stride） ----
    // 形状： (B,C,H,1,W,1) -> expand 为 (B,C,H,scale,W,scale) -> reshape 为 (B,C,Hs,Ws)
    Tensor invWBR_exp = invWBR.view({B, C, H, 1, W, 1})
                            .expand({B, C, H, scale, W, scale})
                            .reshape({B, C, Hs, Ws});
    Tensor FCBinvWBR = FBC * invWBR_exp;

    Tensor FX = (FR - FCBinvWBR) / lambda_;
    Tensor out_c = at::fft_ifftn(FX, c10::nullopt, {-2, -1}, c10::nullopt);
    Tensor out = at::real(out_c);
    return out;
}

// ---------- Clear Cache ----------
void clear_fb_cache()
{
    std::lock_guard<std::mutex> lock(fb_cache_mutex);
    fb_cache.clear();
    fb_cache_lru.clear();
}

// ---------- Registration ----------
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
s