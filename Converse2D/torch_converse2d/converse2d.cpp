#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <list>
#include <mutex>

#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
at::Tensor converse_spectral_cuda(const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, bool);
at::Tensor converse_psf_cuda(const at::Tensor&, int64_t, int64_t);
#endif

using at::Tensor;

// v2 is ATen/full FFT; v3-v6 share fused full FFT inference;
// v7 uses Hermitian half spectra. All labels share the corrected algebra.
static Tensor alias_mean(const Tensor& a, int64_t s) {
    if (s == 1) return a;
    const auto h = a.size(-2) / s, w = a.size(-1) / s;
    return a.reshape({a.size(0), a.size(1), s, h, s, w}).mean({2, 4});
}

// F[-h,-w] = conj(F[h,w]); reflect BOTH dimensions.
static Tensor full_spectrum(const Tensor& half, int64_t width) {
    const int64_t end = (width + 1) / 2;
    auto tail = half.slice(-1, 1, end).flip({-2, -1}).roll({1}, {-2});
    if (half.is_complex()) tail = tail.conj();
    return at::cat({half, tail}, -1);
}

struct CacheEntry {
    // Holding the source prevents TensorImpl address reuse by a new weight.
    Tensor source, fb, invw;
    const void* data;
    uint32_t version;
    int64_t h, w, scale, stream;
    bool real_fft, inference;
    size_t bytes;
};
static std::list<CacheEntry> cache;
static std::mutex cache_mutex;
static size_t cache_bytes = 0;
// A graph runner can own a separate warmup cache. End returns all tensors to
// the caller, which retains them until pending replays finish. This scope never
// shares entries with the evictable eager cache or with another runner/thread.
static thread_local bool graph_cache_active = false;
static thread_local std::list<CacheEntry> graph_cache;
constexpr size_t CACHE_BYTES_LIMIT = 256 * 1024 * 1024;
constexpr size_t CACHE_ENTRIES_LIMIT = 64;

static std::pair<Tensor, Tensor> spectrum(const Tensor& source, const Tensor& weight,
                                         int64_t h, int64_t w, int64_t s, bool real_fft) {
    bool cacheable = !at::GradMode::is_enabled() && !source.is_inference() && source.is_leaf();
    int64_t stream = 0;
#ifdef CONVERSE2D_WITH_CUDA
    if (weight.is_cuda()) {
        stream = c10::cuda::getCurrentCUDAStream(weight.get_device()).id();
        // Captured nodes must own their spectra through the graph memory pool.
        // Never read an evictable eager-cache tensor or publish a graph-private
        // allocation into the global cache, including when warmup hit the cache.
        if (cacheable && !graph_cache_active && c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
                c10::cuda::CaptureStatus::None) cacheable = false;
    }
#else
    if (weight.is_cuda()) cacheable = false;
#endif
    const bool inference = c10::InferenceMode::is_enabled();
    const uint32_t version = cacheable ? source._version() : 0;
    auto& entries = graph_cache_active ? graph_cache : cache;
    if (cacheable) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->source.is_same(source)) {
                if (it->version != version || it->data != source.const_data_ptr()) {
                    if (!graph_cache_active) cache_bytes -= it->bytes;
                    it = entries.erase(it);
                    continue;
                }
                if (it->h == h && it->w == w && it->scale == s && it->stream == stream &&
                    it->real_fft == real_fft && it->inference == inference &&
                    it->fb.device() == weight.device()) {
                    auto result = std::make_pair(it->fb, it->invw);
                    entries.splice(entries.begin(), entries, it);
                    return result;
                }
            }
            ++it;
        }
    }
    const auto kh = weight.size(2), kw = weight.size(3);
    Tensor otf;
#ifdef CONVERSE2D_WITH_CUDA
    const bool fused_prepare = weight.is_cuda() && !at::GradMode::is_enabled() && real_fft;
    if (fused_prepare) otf = converse_psf_cuda(weight, h, w);
    else
#endif
    {
        otf = at::constant_pad_nd(weight, {0, w - kw, 0, h - kh}, 0);
        otf = at::roll(otf, {-(kh / 2), -(kw / 2)}, {-2, -1});
    }
    auto fb = real_fft ? at::fft_rfft2(otf) : at::fft_fft2(otf);
    Tensor invw;
#ifdef CONVERSE2D_WITH_CUDA
    // Uncached dynamic kernels form the denominator in the same loop that
    // already reads FB. Cached fixed kernels still prepare it only once.
    if (!(fused_prepare && !cacheable))
#endif
    {
        // ATen keeps both derivative paths during training.
        auto power = at::real(fb).square() + at::imag(fb).square();
        invw = alias_mean(real_fft && s > 1 ? full_spectrum(power, w) : power, s);
        if (real_fft && s > 1) invw = invw.slice(-1, 0, w / s / 2 + 1).contiguous();
    }
    if (cacheable) {
        const size_t bytes = fb.nbytes() + invw.nbytes() + source.nbytes();
        if (graph_cache_active || bytes <= CACHE_BYTES_LIMIT) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            entries.push_front({source, fb, invw, source.const_data_ptr(), version, h, w, s, stream, real_fft, inference, bytes});
            if (!graph_cache_active) {
                cache_bytes += bytes;
                while (cache.size() > CACHE_ENTRIES_LIMIT || cache_bytes > CACHE_BYTES_LIMIT) {
                    cache_bytes -= cache.back().bytes;
                    cache.pop_back();
                }
            }
        }
    }
    return {fb, invw};
}

Tensor converse2d_forward(Tensor x, Tensor x0, Tensor weight, Tensor bias,
                          int64_t scale, double eps, const std::string& variant) {
    TORCH_CHECK(variant == "v2" || variant == "v3" || variant == "v4" ||
                variant == "v5" || variant == "v6" || variant == "v7", "unknown Converse2D variant");
    TORCH_CHECK(scale >= 1 && std::isfinite(eps) && eps > 0, "scale >= 1 and finite eps > 0 required");
    TORCH_CHECK(x.dim() == 4 && x.numel() > 0, "x must be nonempty (B,C,H,W)");
    auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(H <= INT64_MAX / scale && W <= INT64_MAX / scale, "output size overflow");
    const int64_t Hs = H * scale, Ws = W * scale;
    TORCH_CHECK(x0.sizes() == at::IntArrayRef({B,C,Hs,Ws}), "x0 must be (B,C,H*scale,W*scale)");
    TORCH_CHECK(weight.dim() == 4 && (weight.size(0) == 1 || weight.size(0) == B) &&
                (weight.size(1) == 1 || weight.size(1) == C) &&
                weight.size(2) > 0 && weight.size(3) > 0 && weight.size(2) <= Hs && weight.size(3) <= Ws,
                "weight must be (1|B,1|C,kh,kw) and kernel must fit output");
    TORCH_CHECK(bias.sizes() == at::IntArrayRef({1,C,1,1}), "bias must be (1,C,1,1)");
    TORCH_CHECK(x.is_floating_point() && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble ||
                x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16), "unsupported input dtype");
    for (const auto& t : {x0, weight, bias}) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                    "all tensors must have the same device and dtype");
    }
#ifdef CONVERSE2D_WITH_CUDA
    c10::cuda::OptionalCUDAGuard device_guard;
    if (x.is_cuda()) device_guard.set_index(x.get_device());
#endif
    const auto output_dtype = x.scalar_type();
    const auto compute_dtype = output_dtype == at::kDouble ? at::kDouble : at::kFloat;
    const bool same_prior = x.is_same(x0);
    auto source = weight;
    x = x.to(compute_dtype).contiguous();
    x0 = same_prior ? x : x0.to(compute_dtype).contiguous();
    weight = weight.to(compute_dtype);
    bias = bias.to(compute_dtype).contiguous();
    const bool real_fft = variant == "v7";
    auto lambda = at::sigmoid(bias - 9.0) + eps;
    auto spectra = spectrum(source, weight, Hs, Ws, scale, real_fft);
    auto fb = spectra.first, invw = spectra.second;
    auto fy = real_fft ? at::fft_rfft2(x) : at::fft_fft2(x);
    auto fx0 = same_prior ? fy : (real_fft ? at::fft_rfft2(x0) : at::fft_fft2(x0));
    Tensor fx;
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !at::GradMode::is_enabled() && variant != "v2") {
        fx = converse_spectral_cuda(fy, fx0, fb, invw, lambda, H, W, scale, real_fft);
    } else
#endif
    {
        auto prediction = fb * fx0;
        if (real_fft && scale > 1) prediction = full_spectrum(prediction, Ws);
        prediction = alias_mean(prediction, scale);
        if (real_fft && scale > 1) prediction = prediction.slice(-1, 0, W / 2 + 1);
        auto correction = (fy - prediction) / (invw + lambda);
        if (scale > 1) {
            if (real_fft) correction = full_spectrum(correction, W);
            correction = correction.repeat({1,1,scale,scale});
            if (real_fft) correction = correction.slice(-1, 0, Ws / 2 + 1);
        }
        fx = fx0 + fb.conj() * correction;
    }
    auto out = real_fft ? at::fft_irfft2(fx, at::IntArrayRef({Hs,Ws})) : at::real(at::fft_ifft2(fx));
    return out.to(output_dtype);
}

void clear_fb_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.clear();
    cache_bytes = 0;
}

bool supports_cuda_graphs() { return true; }

void begin_graph_cache() {
    TORCH_CHECK(!graph_cache_active, "graph cache scopes cannot be nested");
    graph_cache.clear();
    graph_cache_active = true;
}

std::vector<Tensor> end_graph_cache() {
    TORCH_CHECK(graph_cache_active, "no graph cache scope is active");
    std::vector<Tensor> owned;
    for (const auto& entry : graph_cache) {
        owned.push_back(entry.source);
        owned.push_back(entry.fb);
        owned.push_back(entry.invw);
    }
    graph_cache.clear();
    graph_cache_active = false;
    return owned;
}

TORCH_LIBRARY(converse2d, m) {
    m.def("forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int scale, float eps=1e-5, str variant='v7') -> Tensor");
    m.def("clear_cache() -> ()");
    m.def("supports_cuda_graphs() -> bool");
    m.def("begin_graph_cache() -> ()");
    m.def("end_graph_cache() -> Tensor[]");
}
TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m) {
    m.impl("forward", TORCH_FN(converse2d_forward));
    m.impl("clear_cache", TORCH_FN(clear_fb_cache));
    m.impl("supports_cuda_graphs", TORCH_FN(supports_cuda_graphs));
    m.impl("begin_graph_cache", TORCH_FN(begin_graph_cache));
    m.impl("end_graph_cache", TORCH_FN(end_graph_cache));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
