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

struct Plan {
    cufftHandle handle = 0;
    void *info = nullptr;  // device memory passed as callerInfo
};
// kind: 0 plain, 1 load_real, 2 load_circular, 3 store_scaled
using Key = std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, int64_t>;
std::map<Key, Plan> plans;

void check(cufftResult r, const char *what) {
    TORCH_CHECK(r == CUFFT_SUCCESS, what, " failed with cufftResult ", int(r));
}

Plan &plan_for(int kind, int64_t batch, int64_t H, int64_t W, int64_t h, int64_t w, int64_t pad) {
    const Key key{kind, at::cuda::current_device(), batch, H, W, pad, h * 100000 + w};
    auto found = plans.find(key);
    if (found != plans.end())
        return found->second;
    Plan p;
    check(cufftCreate(&p.handle), "cufftCreate");
    if (kind != 0) {
        TORCH_CHECK(!fatbin.empty(), "set_fatbin must be called first");
        const char *symbol = kind == 1 ? "load_real" : kind == 2 ? "load_circular" : "store_scaled";
        const auto type = kind == 3 ? CUFFT_CB_ST_COMPLEX : CUFFT_CB_LD_COMPLEX;
        if (kind == 2) {
            const long long info[3] = {h, w, pad};
            C10_CUDA_CHECK(cudaMalloc(&p.info, sizeof(info)));
            C10_CUDA_CHECK(cudaMemcpy(p.info, info, sizeof(info), cudaMemcpyHostToDevice));
        } else if (kind == 3) {
            // ATen: double scale = 1.0 / n, converted to float by mul_.
            const float scale = static_cast<float>(1.0 / static_cast<double>(H * W));
            C10_CUDA_CHECK(cudaMalloc(&p.info, sizeof(scale)));
            C10_CUDA_CHECK(cudaMemcpy(p.info, &scale, sizeof(scale), cudaMemcpyHostToDevice));
        }
        void *infos[1] = {p.info};
        check(cufftXtSetJITCallback(p.handle, symbol, fatbin.data(), fatbin.size(), type,
                                    p.info ? infos : nullptr),
              "cufftXtSetJITCallback");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_fatbin", &set_fatbin);
    m.def("fft2", &fft2);
    m.def("fft2_real", &fft2_real);
    m.def("ifft2_scaled", &ifft2_scaled);
}
