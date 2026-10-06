// Research-only VkFFT (CUDA backend) wrapper for batched 2-D FP32 transforms.
// Plans are NVRTC-compiled by VkFFT at first use and cached per shape/mode.
#define VKFFT_BACKEND 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "vkFFT.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <map>
#include <memory>
#include <tuple>

namespace {

// mode: 0 = C2C, 1 = R2C/C2R.
using Key = std::tuple<int, int64_t, int64_t, int64_t, int, int, int, int>;

struct Plan {
    VkFFTApplication app = {};
    CUdevice device = 0;
    cudaStream_t stream = nullptr;
    pfUINT buffer_size = 0;
    pfUINT input_size = 0;
    bool ready = false;
    ~Plan() {
        if (ready) deleteVkFFT(&app);
    }
};

std::map<Key, std::unique_ptr<Plan>> plans;

// isInputFormatted plans read a separate input buffer on forward and, with
// inverseReturnToInputBuffer, write it on inverse. Otherwise the transform is in place.
Plan& plan(int dev, int64_t batch, int64_t h, int64_t w, int mode, int lut, int normalize, int out_of_place) {
    Key key{dev, batch, h, w, mode, lut, normalize, out_of_place};
    auto& slot = plans[key];
    if (slot) return *slot;
    auto p = std::make_unique<Plan>();
    TORCH_CHECK(cuDeviceGet(&p->device, dev) == CUDA_SUCCESS, "cuDeviceGet failed");
    VkFFTConfiguration c = {};
    c.FFTdim = 2;
    c.size[0] = w;
    c.size[1] = h;
    c.numberBatches = batch;
    c.device = &p->device;
    c.stream = &p->stream;
    c.num_streams = 1;
    c.normalize = normalize;
    c.useLUT = lut;
    const int64_t wc = mode ? w / 2 + 1 : w;
    p->buffer_size = sizeof(float) * 2 * wc * h * batch;
    c.bufferSize = &p->buffer_size;
    if (mode) c.performR2C = 1;
    if (out_of_place) {
        p->input_size = (mode ? sizeof(float) : sizeof(float) * 2) * w * h * batch;
        c.isInputFormatted = 1;
        c.inverseReturnToInputBuffer = 1;
        c.inputBufferSize = &p->input_size;
        c.inputBufferStride[0] = w;
        c.inputBufferStride[1] = w * h;
    }
    auto result = initializeVkFFT(&p->app, c);
    TORCH_CHECK(result == VKFFT_SUCCESS, "initializeVkFFT failed: ", getVkFFTErrorString(result));
    p->ready = true;
    slot = std::move(p);
    return *slot;
}

void run(Plan& p, void* buffer, void* input, int direction) {
    p.stream = at::cuda::getCurrentCUDAStream();
    VkFFTLaunchParams launch = {};
    launch.buffer = &buffer;
    if (input) launch.inputBuffer = &input;
    auto result = VkFFTAppend(&p.app, direction, &launch);
    TORCH_CHECK(result == VKFFT_SUCCESS, "VkFFTAppend failed: ", getVkFFTErrorString(result));
}

void check(const at::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.dim() == 4 && t.is_contiguous(),
                "expected contiguous 4-D CUDA ", dtype);
}

// Forward C2C, out of place; the input is preserved.
at::Tensor fft2(const at::Tensor& z, int64_t lut) {
    check(z, at::kComplexFloat);
    c10::cuda::CUDAGuard guard(z.device());
    auto out = at::empty_like(z);
    auto& p = plan(z.get_device(), z.size(0) * z.size(1), z.size(2), z.size(3), 0, lut, 0, 1);
    run(p, out.data_ptr(), z.data_ptr(), -1);
    return out;
}

// C2C in place on t; direction -1 forward, 1 inverse; normalize applies 1/N on inverse.
void fft2_(at::Tensor& t, int64_t direction, bool normalize, int64_t lut) {
    check(t, at::kComplexFloat);
    c10::cuda::CUDAGuard guard(t.device());
    auto& p = plan(t.get_device(), t.size(0) * t.size(1), t.size(2), t.size(3), 0, lut, normalize, 0);
    run(p, t.data_ptr(), nullptr, static_cast<int>(direction));
}

// Out-of-place inverse: writes `out`, overwrites `z` (VkFFT uses it as its work buffer).
void ifft2_out_destructive(at::Tensor& z, at::Tensor& out, bool normalize, int64_t lut) {
    check(z, at::kComplexFloat);
    check(out, at::kComplexFloat);
    c10::cuda::CUDAGuard guard(z.device());
    auto& p = plan(z.get_device(), z.size(0) * z.size(1), z.size(2), z.size(3), 0, lut, normalize, 1);
    run(p, z.data_ptr(), out.data_ptr(), 1);
}

// Real-input forward, out of place: [B, C, H, W] float -> [B, C, H, W/2+1] complex.
at::Tensor rfft2(const at::Tensor& x, int64_t lut) {
    check(x, at::kFloat);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({x.size(0), x.size(1), x.size(2), x.size(3) / 2 + 1}, x.options().dtype(at::kComplexFloat));
    auto& p = plan(x.get_device(), x.size(0) * x.size(1), x.size(2), x.size(3), 1, lut, 0, 1);
    run(p, out.data_ptr(), x.data_ptr(), -1);
    return out;
}

// Complex-to-real inverse with 1/N, like torch.fft.irfft2(s=(H, W)). Clones the
// half spectrum, which VkFFT overwrites.
at::Tensor irfft2(const at::Tensor& z, int64_t w, int64_t lut) {
    check(z, at::kComplexFloat);
    TORCH_CHECK(z.size(3) == w / 2 + 1, "half-spectrum width mismatch");
    c10::cuda::CUDAGuard guard(z.device());
    auto work = z.clone();
    auto out = at::empty({z.size(0), z.size(1), z.size(2), w}, z.options().dtype(at::kFloat));
    auto& p = plan(z.get_device(), z.size(0) * z.size(1), z.size(2), w, 1, lut, 1, 1);
    run(p, work.data_ptr(), out.data_ptr(), 1);
    return out;
}

void clear() { plans.clear(); }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft2", &fft2);
    m.def("fft2_", &fft2_);
    m.def("ifft2_out_destructive", &ifft2_out_destructive);
    m.def("rfft2", &rfft2);
    m.def("irfft2", &irfft2);
    m.def("clear", &clear);
}
