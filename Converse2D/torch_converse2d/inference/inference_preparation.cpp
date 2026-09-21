#include <ATen/core/grad_mode.h>
#include "inference.h"
#include "cache_state.h"
#include "../common/spectrum_ops.h"
#include <c10/core/InferenceMode.h>
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <ATen/ATen.h>
using at::Tensor;
using namespace converse2d::detail;
using namespace converse2d::inference_state;
std::pair<Tensor, Tensor> spectrum(const Tensor& source, const Tensor& weight,
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
