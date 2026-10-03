#ifdef CONVERSE2D_WITH_CUDA
#include "fft_callbacks.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <nvrtc.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace converse2d::full_training::fft_callbacks {
namespace {
// Compiled to LTO-IR by NVRTC for the current device and JIT-linked by cuFFT.
// cuFFT resolves callbacks by source name; extern "C" names are not found.
constexpr const char* kSource = R"CUDA(
__device__ float2 fwd_load_real(void* data, unsigned long long offset, void*, void*) {
    float2 r;
    r.x = static_cast<const float*>(data)[offset];
    r.y = 0.0f;
    return r;
}

// Load callbacks run once per element inside the first FFT pass, so their index
// math is on the critical path: 32-bit quotients (plans require batch*h*w < 2^32)
// and each remainder from its quotient, instead of 64-bit div/mod.

// circular_pad_complex_forward_kernel's index map; info = {h, w, pad}.
__device__ float2 fwd_load_circular(void* data, unsigned long long offset, void* info, void*) {
    const long long* p = static_cast<const long long*>(info);
    const unsigned int h = p[0], w = p[1], pad = p[2], hp = h + 2 * pad, wp = w + 2 * pad;
    const unsigned int i = static_cast<unsigned int>(offset), q = i / wp, bc = q / hp;
    int row = static_cast<int>(q - bc * hp) - static_cast<int>(pad);
    int col = static_cast<int>(i - q * wp) - static_cast<int>(pad);
    if (row < 0) row += h;
    else if (row >= static_cast<int>(h)) row -= h;
    if (col < 0) col += w;
    else if (col >= static_cast<int>(w)) col -= w;
    float2 r;
    r.x = static_cast<const float*>(data)[(static_cast<unsigned long long>(bc) * h + row) * w + col];
    r.y = 0.0f;
    return r;
}

// real_crop_backward_kernel's embedding; info = {h, w, pad, c, s0, s1, s2, s3}
// with (h, w) the full transform size and s* the gradient strides.
__device__ float2 vjp_crop_embed(void* data, unsigned long long offset, void* info, void*) {
    const long long* p = static_cast<const long long*>(info);
    const unsigned int h = p[0], w = p[1], pad = p[2], c = p[3];
    const unsigned int i = static_cast<unsigned int>(offset), q = i / w, bc = q / h;
    const unsigned int row = q - bc * h, col = i - q * w;
    float real = 0.0f;
    if (row >= pad && row < h - pad && col >= pad && col < w - pad) {
        const unsigned int b = bc / c;
        real = static_cast<const float*>(data)[b * p[4] + (bc - b * c) * p[5] +
                                               (row - pad) * p[6] + (col - pad) * p[7]];
    }
    float2 r;
    r.x = real;
    r.y = 0.0f;
    return r;
}

// ATen's normalization runs MulFunctor(scalar, z) on complex<float> with the
// runtime scalar (float(1/N), 0): (a + bi)(c + di) = (a*c - b*d) + (a*d + b*c)i.
// info = {a, b}; b is read at run time, as in ATen, and the form is kept.
__device__ void inv_store_scaled(void* data, unsigned long long offset, float2 element, void* info, void*) {
    const float a = static_cast<const float*>(info)[0], b = static_cast<const float*>(info)[1];
    float2 r;
    r.x = a * element.x - b * element.y;
    r.y = a * element.y + b * element.x;
    static_cast<float2*>(data)[offset] = r;
}
)CUDA";

constexpr int64_t kMinSide = 16;  // smaller planes stay on ATen (see tools/v4_cufft_callbacks)
enum Kind : int64_t { LoadReal = 1, LoadCircular = 2, StoreScaled = 3, CropEmbedScaled = 4 };

struct Plan {
    bool ok = false;
    cufftHandle handle = 0;
    size_t workspace = 0;
};

std::mutex mutex;
std::map<std::vector<int64_t>, Plan> plans;
std::map<int, std::string> lto_ir;  // per device architecture; empty = unavailable

// CONVERSE2D_FFT_CALLBACKS: unset or "1" enables every site, "0" none, and a
// comma list of real,circular,inverse,crop_embed enables only those sites.
bool enabled(Kind kind) {
    static const std::vector<bool> sites = [] {
        const char* flag = std::getenv("CONVERSE2D_FFT_CALLBACKS");
        const std::string value = flag ? flag : "1";
        std::vector<bool> on(5, value == "1");
        if (value == "0" || value == "1")
            return on;
        const char* names[] = {nullptr, "real", "circular", "inverse", "crop_embed"};
        std::string list = "," + value + ",";
        for (int k = LoadReal; k <= CropEmbedScaled; ++k)
            on[k] = list.find(std::string(",") + names[k] + ",") != std::string::npos;
        return on;
    }();
    return sites[kind];
}

const std::string& compiled(int device) {
    const auto* props = at::cuda::getDeviceProperties(device);
    const int arch = props->major * 10 + props->minor;
    auto found = lto_ir.find(arch);
    if (found != lto_ir.end())
        return found->second;
    std::string ir;
    nvrtcProgram program;
    if (nvrtcCreateProgram(&program, kSource, "converse2d_fft_callbacks.cu", 0, nullptr, nullptr) ==
        NVRTC_SUCCESS) {
        const std::string arch_flag = "--gpu-architecture=compute_" + std::to_string(arch);
        const char* options[] = {arch_flag.c_str(), "-dlto", "--relocatable-device-code=true"};
        size_t size = 0;
        if (nvrtcCompileProgram(program, 3, options) == NVRTC_SUCCESS &&
            nvrtcGetLTOIRSize(program, &size) == NVRTC_SUCCESS && size > 0) {
            ir.resize(size);
            if (nvrtcGetLTOIR(program, ir.data()) != NVRTC_SUCCESS)
                ir.clear();
        }
        nvrtcDestroyProgram(&program);
    }
    return lto_ir.emplace(arch, std::move(ir)).first->second;
}

// Persistent device copy for callerInfo; plans live for the process.
void* device_info(const void* host, size_t bytes) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    if (cudaMemcpy(ptr, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(ptr);
        return nullptr;
    }
    return ptr;
}

bool attach(cufftHandle handle, const std::string& ir, const char* symbol, cufftXtCallbackType type,
            const void* info, size_t bytes) {
    void* device = nullptr;
    if (info && !(device = device_info(info, bytes)))
        return false;
    void* infos[1] = {device};
    return cufftXtSetJITCallback(handle, symbol, ir.data(), ir.size(), type,
                                 device ? infos : nullptr) == CUFFT_SUCCESS;
}

void execute(Plan& plan, const void* in, void* out, int direction, const at::Tensor& like);

bool same_bits(const at::Tensor& a, const at::Tensor& b) {
    auto bits = [](const at::Tensor& t) {
        return at::view_as_real(t.contiguous()).view(at::kInt);
    };
    return a.sizes() == b.sizes() && at::equal(bits(a), bits(b));
}

// A callback plan may use a different FFT algorithm from ATen's plan. Admit it
// only if one random probe through it reproduces the replaced ATen expression
// bit for bit; otherwise the shape keeps the ATen path. Not a proof for every
// input, but differing algorithms disagree on random data.
bool reproduces_aten(Plan& plan, Kind kind, int device, int64_t batch, int64_t h, int64_t w,
                     const std::vector<int64_t>& info) {
    c10::AutoGradMode no_grad(false);
    auto generator = at::make_generator<at::CPUGeneratorImpl>(0x5eed + h * 131 + w);
    const auto cpu = at::TensorOptions().dtype(at::kFloat);
    const auto cuda = at::TensorOptions().device(at::kCUDA, device);
    at::Tensor input, expected, actual;
    if (kind == LoadReal || kind == LoadCircular) {
        const int64_t pad = kind == LoadCircular ? info[2] : 0;
        input = at::randn({batch, 1, h - 2 * pad, w - 2 * pad}, generator, cpu).to(cuda);
        const auto source = pad ? at::pad(input, {pad, pad, pad, pad}, "circular") : input;
        expected = at::fft_fft2(source.to(at::kComplexFloat));
        actual = at::empty({batch, 1, h, w}, cuda.dtype(at::kComplexFloat));
    } else if (kind == StoreScaled) {
        input = at::randn({batch, 1, h, w}, generator, cpu.dtype(at::kComplexFloat)).to(cuda);
        expected = at::fft_ifft2(input);
        actual = at::empty_like(input);
    } else {
        // Same strides as the plan's callerInfo: storage covering the largest offset.
        const int64_t pad = info[2], c = info[3];
        const std::vector<int64_t> sizes{batch / c, c, h - 2 * pad, w - 2 * pad};
        const std::vector<int64_t> strides{info[4], info[5], info[6], info[7]};
        int64_t extent = 1;
        for (int dim = 0; dim < 4; ++dim)
            extent += (sizes[dim] - 1) * strides[dim];
        input = at::randn({extent}, generator, cpu).to(cuda).as_strided(sizes, strides);
        auto full = at::zeros({batch / c, c, h, w}, cuda.dtype(at::kComplexFloat));
        at::real(full).slice(2, pad, h - pad).slice(3, pad, w - pad).copy_(input);
        expected = at::_fft_c2c(full, {2, 3}, 2, true);
        actual = at::empty({batch / c, c, h, w}, cuda.dtype(at::kComplexFloat));
    }
    execute(plan, input.data_ptr(), actual.data_ptr(),
            kind == StoreScaled ? CUFFT_INVERSE : CUFFT_FORWARD, input);
    return same_bits(actual, expected);
}

// Must hold `mutex`. Returns the cached plan, creating it on first use.
Plan& plan_for(Kind kind, int device, int64_t batch, int64_t h, int64_t w,
               const std::vector<int64_t>& info) {
    std::vector<int64_t> key{kind, device, batch, h, w};
    key.insert(key.end(), info.begin(), info.end());
    auto found = plans.find(key);
    if (found != plans.end())
        return found->second;
    // No allocation is legal during stream capture; leave capture to ATen
    // without caching, so the plan can still be created outside capture.
    static Plan unavailable;
    if (c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None)
        return unavailable;
    Plan plan;
    if (batch * h * w > 0xffffffffLL)  // callback offsets are decomposed in 32 bits
        return plans.emplace(key, plan).first->second;
    const auto& ir = compiled(device);
    if (!ir.empty() && cufftCreate(&plan.handle) == CUFFT_SUCCESS) {
        // ATen: double scale = 1.0 / n, applied by mul_ as complex<float>(float(scale), 0).
        const float scale[2] = {static_cast<float>(1.0 / static_cast<double>(h * w)), 0.0f};
        bool ok = cufftSetAutoAllocation(plan.handle, 0) == CUFFT_SUCCESS;
        if (ok && kind == LoadReal)
            ok = attach(plan.handle, ir, "fwd_load_real", CUFFT_CB_LD_COMPLEX, nullptr, 0);
        if (ok && kind == LoadCircular)
            ok = attach(plan.handle, ir, "fwd_load_circular", CUFFT_CB_LD_COMPLEX, info.data(),
                        info.size() * sizeof(int64_t));
        if (ok && kind == CropEmbedScaled)
            ok = attach(plan.handle, ir, "vjp_crop_embed", CUFFT_CB_LD_COMPLEX, info.data(),
                        info.size() * sizeof(int64_t));
        if (ok && (kind == StoreScaled || kind == CropEmbedScaled))
            ok = attach(plan.handle, ir, "inv_store_scaled", CUFFT_CB_ST_COMPLEX, scale, sizeof(scale));
        // ATen's simple-layout c2c plan: null embeds, unit stride, dense batch.
        long long n[2] = {h, w};
        ok = ok && cufftXtMakePlanMany(plan.handle, 2, n, nullptr, 1, h * w, CUDA_C_32F, nullptr, 1,
                                       h * w, CUDA_C_32F, batch, &plan.workspace,
                                       CUDA_C_32F) == CUFFT_SUCCESS;
        ok = ok && reproduces_aten(plan, kind, device, batch, h, w, info);
        if (!ok) {
            cufftDestroy(plan.handle);
            plan.handle = 0;
        }
        plan.ok = ok;
    }
    return plans.emplace(key, plan).first->second;
}

void execute(Plan& plan, const void* in, void* out, int direction, const at::Tensor& like) {
    auto workspace = plan.workspace
        ? at::empty({static_cast<int64_t>(plan.workspace)}, like.options().dtype(at::kByte))
        : at::Tensor();
    TORCH_CHECK(cufftSetStream(plan.handle, at::cuda::getCurrentCUDAStream()) == CUFFT_SUCCESS &&
                    (!plan.workspace ||
                     cufftSetWorkArea(plan.handle, workspace.data_ptr()) == CUFFT_SUCCESS) &&
                    cufftXtExec(plan.handle, const_cast<void*>(in), out, direction) == CUFFT_SUCCESS,
                "cuFFT callback transform failed");
}

bool plain_cuda(const at::Tensor& t, at::ScalarType dtype, Kind kind) {
    return t.defined() && enabled(kind) && t.is_cuda() && t.dim() == 4 && t.scalar_type() == dtype &&
           !t.is_conj() && !t.is_neg() && t.numel() > 0;
}
} // namespace

bool ready_real(const at::Tensor& x, int64_t pad) {
    if (!plain_cuda(x, at::kFloat, pad ? LoadCircular : LoadReal) || !x.is_contiguous() || pad < 0 || pad > x.size(2) ||
        pad > x.size(3))
        return false;
    const auto h = x.size(2) + 2 * pad, w = x.size(3) + 2 * pad;
    if (h < kMinSide || w < kMinSide)
        return false;
    c10::cuda::CUDAGuard guard(x.device());
    std::lock_guard<std::mutex> lock(mutex);
    const std::vector<int64_t> info = pad ? std::vector<int64_t>{x.size(2), x.size(3), pad}
                                          : std::vector<int64_t>{};
    return plan_for(pad ? LoadCircular : LoadReal, x.get_device(), x.size(0) * x.size(1), h, w, info).ok;
}

at::Tensor fft2_real(const at::Tensor& x, int64_t pad) {
    c10::cuda::CUDAGuard guard(x.device());
    const auto h = x.size(2) + 2 * pad, w = x.size(3) + 2 * pad;
    auto out = at::empty({x.size(0), x.size(1), h, w}, x.options().dtype(at::kComplexFloat));
    std::lock_guard<std::mutex> lock(mutex);
    const std::vector<int64_t> info = pad ? std::vector<int64_t>{x.size(2), x.size(3), pad}
                                          : std::vector<int64_t>{};
    auto& plan = plan_for(pad ? LoadCircular : LoadReal, x.get_device(), x.size(0) * x.size(1), h, w, info);
    TORCH_CHECK(plan.ok, "fft2_real requires ready_real");
    execute(plan, x.data_ptr(), out.data_ptr(), CUFFT_FORWARD, x);
    return out;
}

bool ready_inverse(const at::Tensor& z) {
    if (!plain_cuda(z, at::kComplexFloat, StoreScaled) || !z.is_contiguous() || z.size(2) < kMinSide ||
        z.size(3) < kMinSide)
        return false;
    c10::cuda::CUDAGuard guard(z.device());
    std::lock_guard<std::mutex> lock(mutex);
    return plan_for(StoreScaled, z.get_device(), z.size(0) * z.size(1), z.size(2), z.size(3), {}).ok;
}

at::Tensor ifft2_scaled(const at::Tensor& z) {
    c10::cuda::CUDAGuard guard(z.device());
    auto out = at::empty_like(z, at::MemoryFormat::Contiguous);
    std::lock_guard<std::mutex> lock(mutex);
    auto& plan = plan_for(StoreScaled, z.get_device(), z.size(0) * z.size(1), z.size(2), z.size(3), {});
    TORCH_CHECK(plan.ok, "ifft2_scaled requires ready_inverse");
    execute(plan, z.data_ptr(), out.data_ptr(), CUFFT_INVERSE, z);
    return out;
}

namespace {
std::vector<int64_t> crop_info(const at::Tensor& g, int64_t c, int64_t h, int64_t w, int64_t pad) {
    std::vector<int64_t> info{h, w, pad, c};
    for (int dim = 0; dim < 4; ++dim)
        info.push_back(g.size(dim) > 1 ? g.stride(dim) : 0);  // singleton strides never contribute
    return info;
}
} // namespace

bool ready_crop_embed(const at::Tensor& g, int64_t c, int64_t h, int64_t w, int64_t pad) {
    if (!plain_cuda(g, at::kFloat, CropEmbedScaled) || h < kMinSide || w < kMinSide || g.size(1) != c ||
        g.size(2) != h - 2 * pad || g.size(3) != w - 2 * pad)
        return false;
    for (int dim = 0; dim < 4; ++dim)
        if (g.stride(dim) < 0)
            return false;
    c10::cuda::CUDAGuard guard(g.device());
    std::lock_guard<std::mutex> lock(mutex);
    return plan_for(CropEmbedScaled, g.get_device(), g.size(0) * c, h, w, crop_info(g, c, h, w, pad)).ok;
}

at::Tensor crop_embed_fft2_scaled(const at::Tensor& g, int64_t c, int64_t h, int64_t w, int64_t pad) {
    c10::cuda::CUDAGuard guard(g.device());
    auto out = at::empty({g.size(0), c, h, w}, g.options().dtype(at::kComplexFloat));
    std::lock_guard<std::mutex> lock(mutex);
    auto& plan = plan_for(CropEmbedScaled, g.get_device(), g.size(0) * c, h, w, crop_info(g, c, h, w, pad));
    TORCH_CHECK(plan.ok, "crop_embed_fft2_scaled requires ready_crop_embed");
    execute(plan, g.data_ptr(), out.data_ptr(), CUFFT_FORWARD, g);
    return out;
}
} // namespace converse2d::full_training::fft_callbacks
#endif
