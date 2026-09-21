#include "training.h"
#include <torch/csrc/autograd/function.h>
#include <list>
#ifdef CONVERSE2D_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <ATen/ATen.h>
using at::Tensor;
struct TrainingCacheEntry {
    Tensor source, fb;
    std::shared_ptr<torch::autograd::Node> grad_fn;
    const void* data;
    int64_t version;
    int64_t h, w, stream;
    bool requires_grad;
};
struct TrainingCacheScope {
    std::list<TrainingCacheEntry> entries;
    std::vector<Tensor> allowed;
    bool restrict_weights = false;
    int64_t hits = 0, misses = 0;
};
// Explicit, nested scopes belong to one model forward, never to an optimizer
// step or the inference cache. Holding fb preserves its differentiable graph.
static thread_local std::list<TrainingCacheScope> training_scopes;

Tensor training_spectrum(const Tensor& source, const Tensor& weight, int64_t h, int64_t w) {
    if (training_scopes.empty() || source.is_inference())
        return training_spectrum_cast(prepare_training_kernel(weight, h, w));
    auto& scope = training_scopes.back();
    // Models may declare only weights that are reused. Do not retain large
    // per-example dynamic spectra that are consumed once in the forward.
    if (scope.restrict_weights) {
        bool allowed = false;
        for (const auto& candidate : scope.allowed) if (candidate.is_same(source)) { allowed = true; break; }
        if (!allowed) return training_spectrum_cast(prepare_training_kernel(weight, h, w));
    }
    int64_t stream = 0;
#ifdef CONVERSE2D_WITH_CUDA
    if (weight.is_cuda()) stream = c10::cuda::getCurrentCUDAStream(weight.get_device()).id();
#endif
    const auto version = source._version();
    const auto data = source.const_data_ptr();
    const bool requires_grad = source.requires_grad();
    const auto grad_fn = source.grad_fn();
    for (auto it = scope.entries.begin(); it != scope.entries.end();) {
        if (it->source.is_same(source)) {
            if (it->version != version || it->data != data || it->requires_grad != requires_grad ||
                it->grad_fn != grad_fn) {
                it = scope.entries.erase(it);
                continue;
            }
            if (it->h == h && it->w == w && it->stream == stream && it->fb.device() == weight.device()) {
                ++scope.hits;
                return training_spectrum_cast(it->fb);
            }
        }
        ++it;
    }
    auto fb = prepare_training_kernel(weight, h, w);
    scope.entries.push_back({source, fb, grad_fn, data, version, h, w, stream, requires_grad});
    ++scope.misses;
    return training_spectrum_cast(fb);
}

void begin_training_cache() { training_scopes.emplace_back(); }

void begin_training_cache_for(std::vector<Tensor> weights) {
    training_scopes.emplace_back();
    training_scopes.back().allowed = std::move(weights);
    training_scopes.back().restrict_weights = true;
}

std::vector<int64_t> end_training_cache() {
    TORCH_CHECK(!training_scopes.empty(), "no training cache scope is active");
    auto result = std::vector<int64_t>{training_scopes.back().hits, training_scopes.back().misses};
    training_scopes.pop_back();
    return result;
}
