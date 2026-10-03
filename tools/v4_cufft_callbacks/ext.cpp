// Research-only cuFFT plans with LTO callbacks. Not production: no autograd,
// contiguous CUDA inputs only, workspace auto-allocated by cuFFT.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <torch/extension.h>

#include <map>
#include <string>
#include <tuple>

namespace {
std::string fatbin;

void check(cufftResult r, const char *what) {
    TORCH_CHECK(r == CUFFT_SUCCESS, what, " failed with cufftResult ", int(r));
}

struct Plan {
    cufftHandle handle = 0;
    void *info = nullptr;   // device memory passed as load callerInfo
    void *store = nullptr;  // device memory passed as store callerInfo
};
// kind: 0 plain, 1 load_real, 2 load_circular, 3 store_scaled,
// 4 load_crop_embed + store_scaled, 5 store_real
using Key = std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, int64_t>;

void *device_copy(const void *host, size_t bytes) {
    void *ptr = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    C10_CUDA_CHECK(cudaMemcpy(ptr, host, bytes, cudaMemcpyHostToDevice));
    return ptr;
}

void set_callback(Plan &p, const char *symbol, cufftXtCallbackType type, void *info) {
    TORCH_CHECK(!fatbin.empty(), "set_fatbin must be called first");
    void *infos[1] = {info};
    check(cufftXtSetJITCallback(p.handle, symbol, fatbin.data(), fatbin.size(), type,
                                info ? infos : nullptr),
          "cufftXtSetJITCallback");
}
std::map<Key, Plan> plans;

Plan &plan_for(int kind, int64_t batch, int64_t H, int64_t W, int64_t h, int64_t w, int64_t pad) {
    const Key key{kind, at::cuda::current_device(), batch, H, W, pad, h * 100000 + w};
    auto found = plans.find(key);
    if (found != plans.end())
        return found->second;
    Plan p;
    check(cufftCreate(&p.handle), "cufftCreate");
    // ATen: double scale = 1.0 / n, converted to float by mul_.
    const float scale = static_cast<float>(1.0 / static_cast<double>(H * W));
    if (kind == 1)
        set_callback(p, "load_real", CUFFT_CB_LD_COMPLEX, nullptr);
    if (kind == 2) {
        const long long info[3] = {h, w, pad};
        set_callback(p, "load_circular", CUFFT_CB_LD_COMPLEX, p.info = device_copy(info, sizeof(info)));
    }
    if (kind == 4) {
        // Crop info (with strides) is per call; allocate and fill in exec.
        C10_CUDA_CHECK(cudaMalloc(&p.info, 8 * sizeof(long long)));
        set_callback(p, "load_crop_embed", CUFFT_CB_LD_COMPLEX, p.info);
    }
    if (kind == 3 || kind == 4)
        set_callback(p, "store_scaled", CUFFT_CB_ST_COMPLEX, p.store = device_copy(&scale, sizeof(scale)));
    if (kind == 5) {
        C10_CUDA_CHECK(cudaMalloc(&p.store, sizeof(float *)));
        set_callback(p, "store_real", CUFFT_CB_ST_COMPLEX, p.store);
    }
    // Match ATen's simple-layout c2c plan: null embeds, unit stride, dense batch.
    long long n[2] = {H, W};
    size_t workspace = 0;
    check(cufftXtMakePlanMany(p.handle, 2, n, nullptr, 1, H * W, CUDA_C_32F, nullptr, 1, H * W,
                              CUDA_C_32F, batch, &workspace, CUDA_C_32F),
          "cufftXtMakePlanMany");
    return plans.emplace(key, p).first->second;
}

void exec(Plan &p, const void *in, void *out, int direction) {
    check(cufftSetStream(p.handle, at::cuda::getCurrentCUDAStream()), "cufftSetStream");
    check(cufftXtExec(p.handle, const_cast<void *>(in), out, direction), "cufftXtExec");
}

void require(const at::Tensor &t, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.dim() == 4 && t.is_contiguous() && t.scalar_type() == dtype &&
                    !t.is_conj() && !t.is_neg(),
                "contiguous 4D CUDA input of the expected dtype required");
}
}  // namespace

void set_fatbin(py::bytes data) { fatbin = data; }

at::Tensor fft2(at::Tensor z, bool inverse) {
    require(z, at::kComplexFloat);
    c10::cuda::CUDAGuard guard(z.device());
    auto out = at::empty_like(z);
    auto &p = plan_for(0, z.size(0) * z.size(1), z.size(2), z.size(3), 0, 0, 0);
    exec(p, z.data_ptr(), out.data_ptr(), inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    return out;
}

at::Tensor fft2_real(at::Tensor x, int64_t pad) {
    require(x, at::kFloat);
    c10::cuda::CUDAGuard guard(x.device());
    const auto h = x.size(2), w = x.size(3), H = h + 2 * pad, W = w + 2 * pad;
    TORCH_CHECK(pad >= 0 && pad <= h && pad <= w, "invalid circular padding");
    auto out = at::empty({x.size(0), x.size(1), H, W}, x.options().dtype(at::kComplexFloat));
    auto &p = plan_for(pad ? 2 : 1, x.size(0) * x.size(1), H, W, h, w, pad);
    exec(p, x.data_ptr(), out.data_ptr(), CUFFT_FORWARD);
    return out;
}

at::Tensor ifft2_scaled(at::Tensor z) {
    require(z, at::kComplexFloat);
    c10::cuda::CUDAGuard guard(z.device());
    auto out = at::empty_like(z);
    auto &p = plan_for(3, z.size(0) * z.size(1), z.size(2), z.size(3), 0, 0, 0);
    exec(p, z.data_ptr(), out.data_ptr(), CUFFT_INVERSE);
    return out;
}

// Backward of ifft2 -> real -> crop(pad): (1/N) * FFT(embed(g)). Reads the
// gradient through its strides; per-call metadata is copied stream-ordered.
at::Tensor crop_embed_fft2_scaled(at::Tensor g, int64_t pad) {
    TORCH_CHECK(g.is_cuda() && g.dim() == 4 && g.scalar_type() == at::kFloat && !g.is_neg(),
                "4D CUDA FP32 gradient required");
    c10::cuda::CUDAGuard guard(g.device());
    const auto h = g.size(2), w = g.size(3), H = h + 2 * pad, W = w + 2 * pad;
    auto out = at::empty({g.size(0), g.size(1), H, W}, g.options().dtype(at::kComplexFloat));
    auto &p = plan_for(4, g.size(0) * g.size(1), H, W, h, w, pad);
    const long long info[8] = {h, w, pad, g.size(1), g.size(0) > 1 ? g.stride(0) : 0,
                               g.size(1) > 1 ? g.stride(1) : 0, g.size(2) > 1 ? g.stride(2) : 0,
                               g.size(3) > 1 ? g.stride(3) : 0};
    C10_CUDA_CHECK(cudaMemcpyAsync(p.info, info, sizeof(info), cudaMemcpyHostToDevice,
                                   at::cuda::getCurrentCUDAStream()));
    exec(p, g.data_ptr(), out.data_ptr(), CUFFT_FORWARD);
    return out;
}

// Backward of fft2(real x): real(IFFT_unnormalized(gy)), written as FP32.
at::Tensor ifft2_real(at::Tensor gy) {
    require(gy, at::kComplexFloat);
    c10::cuda::CUDAGuard guard(gy.device());
    auto out = at::empty(gy.sizes(), gy.options().dtype(at::kFloat));
    auto scratch = at::empty_like(gy);  // cuFFT's own complex output buffer
    auto &p = plan_for(5, gy.size(0) * gy.size(1), gy.size(2), gy.size(3), 0, 0, 0);
    float *dest = out.data_ptr<float>();
    C10_CUDA_CHECK(cudaMemcpyAsync(p.store, &dest, sizeof(dest), cudaMemcpyHostToDevice,
                                   at::cuda::getCurrentCUDAStream()));
    exec(p, gy.data_ptr(), scratch.data_ptr(), CUFFT_INVERSE);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_fatbin", &set_fatbin);
    m.def("fft2", &fft2);
    m.def("fft2_real", &fft2_real);
    m.def("ifft2_scaled", &ifft2_scaled);
    m.def("crop_embed_fft2_scaled", &crop_embed_fft2_scaled);
    m.def("ifft2_real", &ifft2_real);
}
