#include "../common/fp32_dispatch.h"
#include "launchers.cuh"
#include "detail/math.cuh"
using namespace converse2d::inference_detail;
namespace {
template <typename T>
__global__ void correction_scale_one(const c10::complex<T>* fy,
    const c10::complex<T>* fx0, const c10::complex<T>* fb, const T* invw,
    const T* lambda, c10::complex<T>* out, int64_t total, int64_t pixels, int64_t channels,
    int64_t kernel_batches, int64_t kernel_channels) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t bc = i / pixels, c = bc % channels;
    const int64_t kc = (kernel_batches == 1 ? 0 : bc / channels) * kernel_channels +
                       (kernel_channels == 1 ? 0 : c);
    const int64_t p = kc * pixels + i % pixels;
    const auto filter = fb[p];
    const T power = invw ? invw[p] : squared_norm(filter);
    out[i] = fx0[i] + conjugate(filter) * ((fy[i] - filter * fx0[i]) / (power + lambda[c]));
}

}
void launch_inference_scale1(const at::Tensor& y,const at::Tensor& prior,const at::Tensor& kernel,const at::Tensor& denom,const at::Tensor& lambda,at::Tensor& q,at::Tensor& out,int64_t H,int64_t W,int64_t s,bool half,cudaStream_t stream) {
const int64_t n=prior.numel(),C=prior.size(1);
constexpr int threads=256;
CONVERSE_DISPATCH_FP32(lambda.scalar_type(), "converse_spectral", [&] {
using z=c10::complex<scalar_t>;
            correction_scale_one<scalar_t><<<(n+threads-1)/threads, threads, 0, stream>>>(
                y.data_ptr<z>(), prior.data_ptr<z>(), kernel.data_ptr<z>(),
                denom.defined() ? denom.data_ptr<scalar_t>() : nullptr,
                lambda.data_ptr<scalar_t>(), out.data_ptr<z>(), n, n/(prior.size(0)*C), C,
                kernel.size(0), kernel.size(1));
});
}
