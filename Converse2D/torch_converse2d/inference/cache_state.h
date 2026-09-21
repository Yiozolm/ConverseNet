#pragma once
#include <ATen/ATen.h>
#include <list>
#include <mutex>
namespace converse2d::inference_state {
using at::Tensor;
struct CacheEntry {
    // Holding the source prevents TensorImpl address reuse by a new weight.
    Tensor source, fb, invw;
    const void* data;
    uint32_t version;
    int64_t h, w, scale, stream;
    bool real_fft, inference;
    size_t bytes;
};

extern std::list<CacheEntry> cache;
extern std::mutex cache_mutex;
extern size_t cache_bytes;
extern thread_local bool graph_cache_active;
extern thread_local std::list<CacheEntry> graph_cache;
constexpr size_t CACHE_BYTES_LIMIT = 256 * 1024 * 1024;
constexpr size_t CACHE_ENTRIES_LIMIT = 64;
}
