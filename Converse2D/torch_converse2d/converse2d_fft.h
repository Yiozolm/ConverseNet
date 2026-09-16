#pragma once

// Internal CUDA inference only. The caller gives up the corrected spectrum:
// cuFFT C2R may overwrite it, even with an out-of-place output.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <cufftXt.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

at::Tensor converse_normalize_cast_cuda(const at::Tensor&, double, at::ScalarType, at::Tensor);

namespace converse2d::c2r {

inline void check(cufftResult status) {
    TORCH_CHECK(status == CUFFT_SUCCESS, "Converse2D C2R: cuFFT error ", int(status));
}

struct Key {
    int device;
    int64_t stream, h, w, batch;
    at::ScalarType dtype;

    bool operator==(const Key& other) const {
        return device == other.device && stream == other.stream && h == other.h &&
               w == other.w && batch == other.batch && dtype == other.dtype;
    }
};

struct Plan {
    Key key;
    cufftHandle handle = 0;
    size_t workspace_bytes = 0;

    explicit Plan(Key k) : key(k) {
        check(cufftCreate(&handle));
        try {
            check(cufftSetAutoAllocation(handle, 0));
            long long sizes[2] = {key.h, key.w};
            const bool fp32 = key.dtype == at::kComplexFloat;
            // Use the same contiguous 2D Xt layout and arithmetic types as ATen.
            check(cufftXtMakePlanMany(handle, 2, sizes, nullptr, 1, 1,
                fp32 ? CUDA_C_32F : CUDA_C_64F, nullptr, 1, 1,
                fp32 ? CUDA_R_32F : CUDA_R_64F, key.batch, &workspace_bytes,
                fp32 ? CUDA_C_32F : CUDA_C_64F));
        } catch (...) {
            cufftDestroy(handle);
            handle = 0;
            throw;
        }
    }

    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    ~Plan() noexcept {
        // Cache eviction or graph destruction can happen on another device.
        // At interpreter shutdown the CUDA runtime may already be unavailable.
        int previous;
        if (!handle || cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != key.device && cudaSetDevice(key.device) != cudaSuccess) return;
        cufftDestroy(handle);
        if (previous != key.device) cudaSetDevice(previous);
    }
};

constexpr size_t PLAN_CACHE_LIMIT = 32;
static std::list<std::shared_ptr<Plan>> plans;
static std::mutex plan_mutex;
static thread_local bool graph_scope = false;
static thread_local std::vector<std::shared_ptr<Plan>> graph_plans;

inline void clear_cache() {
    std::lock_guard<std::mutex> lock(plan_mutex);
    plans.clear(); // Graph owners retain their plans independently.
}

inline void begin_graph_scope() {
    TORCH_CHECK(!graph_scope, "C2R graph cache scopes cannot be nested");
    graph_plans.clear();
    graph_scope = true;
}

inline std::vector<at::Tensor> end_graph_scope() {
    TORCH_CHECK(graph_scope, "no C2R graph cache scope is active");
    // Reuse the runner's Tensor[] ownership API with opaque CPU lifetime tokens.
    // The runner waits for replays and destroys the graph before releasing them.
    std::vector<at::Tensor> owners;
    for (const auto& plan : graph_plans) {
        auto token = std::make_shared<uint8_t>(0);
        owners.push_back(at::from_blob(token.get(), {1}, [plan, token](void*) {},
            at::TensorOptions().dtype(at::kByte).device(at::kCPU)));
    }
    graph_plans.clear();
    graph_scope = false;
    return owners;
}

inline at::Tensor inverse(at::Tensor& frequency, int64_t h, int64_t w,
                          at::Tensor destination = at::Tensor(),
                          at::ScalarType output_dtype = at::ScalarType::Undefined) {
    TORCH_INTERNAL_ASSERT(!at::GradMode::is_enabled());
    TORCH_INTERNAL_ASSERT(frequency.is_cuda() && frequency.is_contiguous() &&
        frequency.dim() == 4 && frequency.size(2) == h && frequency.size(3) == w / 2 + 1 &&
        (frequency.scalar_type() == at::kComplexFloat || frequency.scalar_type() == at::kComplexDouble));
    const auto dtype = frequency.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    if (output_dtype == at::ScalarType::Undefined) output_dtype = dtype;
    const bool low_output = output_dtype == at::kHalf || output_dtype == at::kBFloat16;
    TORCH_INTERNAL_ASSERT(output_dtype == dtype || (dtype == at::kFloat && low_output));
    const at::DimVector shape{frequency.size(0), frequency.size(1), h, w};
    if (destination.defined()) {
        TORCH_INTERNAL_ASSERT(destination.sizes() == shape && destination.is_contiguous() &&
            destination.device() == frequency.device() && destination.scalar_type() == output_dtype &&
            !destination.is_alias_of(frequency));
    }
    c10::cuda::CUDAGuard guard(frequency.device());
    auto fallback = [&]() {
        auto result = at::fft_irfft2(frequency, at::IntArrayRef({h, w}));
        if (!destination.defined()) return result.to(output_dtype);
        destination.copy_(result);
        return destination;
    };
    const auto stream = c10::cuda::getCurrentCUDAStream(frequency.get_device());
    const bool capturing = c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None;
    // A direct capture cannot export plan ownership. Never borrow an evictable
    // eager plan for it, including when the shape was already warmed up.
    if (capturing && !graph_scope) return fallback();

    const Key key{frequency.get_device(), stream.id(), h, w,
                  frequency.size(0) * frequency.size(1), frequency.scalar_type()};
    std::unique_lock<std::mutex> lock(plan_mutex);
    std::shared_ptr<Plan> plan;
    // A large graph's warmup can exceed the eager LRU capacity. Its own retained
    // plans must remain available for capture without replanning.
    if (graph_scope) {
        for (const auto& retained : graph_plans) {
            if (retained->key == key) { plan = retained; break; }
        }
    }
    if (!plan) {
        auto it = std::find_if(plans.begin(), plans.end(), [&](const auto& p) { return p->key == key; });
        if (it != plans.end()) {
            plan = *it;
            plans.splice(plans.begin(), plans, it);
        } else {
            // Planning can allocate/load modules; it must occur outside capture.
            if (capturing) { lock.unlock(); return fallback(); }
            plan = std::make_shared<Plan>(key);
            plans.push_front(plan);
            if (plans.size() > PLAN_CACHE_LIMIT) plans.pop_back();
        }
        if (graph_scope) graph_plans.push_back(plan);
    }

    auto output = destination.defined() && !low_output ? destination :
        at::empty(shape, frequency.options().dtype(dtype));
    // An odd-sized batch slice can be aligned only to the real element type.
    // Use a temporary for that exceptional layout rather than handing cuFFT a
    // pointer lacking complex-type alignment.
    const bool copy_output = reinterpret_cast<uintptr_t>(output.data_ptr()) % frequency.element_size() != 0;
    auto target = copy_output ? at::empty(shape, output.options()) : output;
    TORCH_CHECK(plan->workspace_bytes <= size_t(INT64_MAX), "C2R workspace size overflow");
    // Per-call workspace belongs to the current stream's allocator, or the
    // graph-private pool during capture. It is never reused through the plan.
    auto workspace = at::empty({int64_t(plan->workspace_bytes)}, frequency.options().dtype(at::kByte));
    check(cufftSetStream(plan->handle, stream));
    check(cufftSetWorkArea(plan->handle, workspace.data_ptr()));
    check(cufftXtExec(plan->handle, frequency.data_ptr(), target.data_ptr(), CUFFT_INVERSE));
    lock.unlock();
    // Preserve FP32 post-IFFT scaling, but fuse it with the final low-precision
    // store. This removes an FP32 write/read and one pointwise kernel launch.
    if (low_output) {
        // cuFFT is queued on the current allocator stream. Release disposable
        // buffers before allocating the low output, so the fused store does not
        // extend their peak lifetime compared with casting after inverse().
        workspace = at::Tensor();
        frequency = at::Tensor();
        return converse_normalize_cast_cuda(target, 1.0 / (double(h) * double(w)), output_dtype, destination);
    }
    // Ordinary ATen multiplication AFTER C2R: preserve the normalization order.
    target.mul_(1.0 / (double(h) * double(w)));
    if (copy_output) output.copy_(target);
    return output;
}

} // namespace converse2d::c2r
