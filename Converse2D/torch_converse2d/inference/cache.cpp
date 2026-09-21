#include "cache.h"
#include "cache_state.h"
namespace converse2d::inference_state {
std::list<CacheEntry> cache;
std::mutex cache_mutex;
size_t cache_bytes = 0;
// A graph runner can own a separate warmup cache. End returns all tensors to
// the caller, which retains them until pending replays finish. This scope never
// shares entries with the evictable eager cache or with another runner/thread.
thread_local bool graph_cache_active = false;
thread_local std::list<CacheEntry> graph_cache;

}
using at::Tensor;
using namespace converse2d::inference_state;
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
