#include "nearest_k2_s2.h"
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>

namespace {
using I = int64_t;

// A two-component FP32 approximation. Full TwoSum has no |a| >= |b|
// precondition, which matters for signed kernels and a cancelling numerator.
// Error-free transform claims require finite, non-overflowing intermediates;
// residuals smaller than the FP32 subnormal range cannot be retained.
__device__ __forceinline__ float2 two_sum(float a, float b) {
    const float hi = __fadd_rn(a, b);
    const float recovered_b = __fsub_rn(hi, a);
    const float err_a = __fsub_rn(a, __fsub_rn(hi, recovered_b));
    const float err_b = __fsub_rn(b, recovered_b);
    return make_float2(hi, __fadd_rn(err_a, err_b));
}

__device__ __forceinline__ float2 two_product(float a, float b) {
    const float hi = __fmul_rn(a, b);
    return make_float2(hi, __fmaf_rn(a, b, -hi));
}

__device__ __forceinline__ float2 pair_add(float2 a, float2 b) {
    const float2 high = two_sum(a.x, b.x);
    const float low = __fadd_rn(__fadd_rn(a.y, b.y), high.y);
    return two_sum(high.x, low);
}

__device__ __forceinline__ float2 pair_neg(float2 a) {
    return make_float2(-a.x, -a.y);
}

__device__ __forceinline__ float2 compensated_output_pair(float x, float w, float qhi, float qlo) {
    float2 sum = pair_add(make_float2(x, 0.0f), two_product(w, qhi));
    return pair_add(sum, two_product(w, qlo));
}

__device__ __forceinline__ float compensated_output(float x, float w, float qhi, float qlo) {
    const float2 sum = compensated_output_pair(x, w, qhi, qlo);
    return __fadd_rn(sum.x, sum.y);
}

__device__ __forceinline__ float2 corrected_ratio(float2 numerator, float2 denominator) {
    const float hi = __fdiv_rn(numerator.x, denominator.x);
    float remainder = __fmaf_rn(-hi, denominator.x, numerator.x);
    remainder = __fadd_rn(remainder, numerator.y);
    remainder = __fsub_rn(remainder, __fmul_rn(hi, denominator.y));
    return make_float2(hi, __fdiv_rn(remainder, denominator.x));
}

__global__ void nearest_k2_compensated_fused_lambda(
    const float* x, const float* weight, const float* bias, float eps, float* output,
    I n, I channels, I height, I width, I kb, I kc) {
    const I step = I(blockDim.x) * gridDim.x;
    for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
        const I hw = height * width, bc = i / hw, b = bc / channels, c = bc % channels;
        const I kbc = (kb == 1 ? 0 : b) * kc + (kc == 1 ? 0 : c);
        const float xi = x[i];
        const float weights[4] = {weight[4*kbc], weight[4*kbc+1],
                                  weight[4*kbc+2], weight[4*kbc+3]};
        // Same FP32 parameterization, now local to the solve. expf is the
        // ordinary CUDA math function, never the fast intrinsic __expf.
        // Sigmoid bit identity with ATen is not assumed; the budget gates it.
        const float shift = __fsub_rn(bias[c], 9.0f);
        const float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-shift)));
        const float regularizer = __fadd_rn(sigmoid, eps);
        float2 numerator = make_float2(xi, 0.0f);
        float2 denominator = make_float2(regularizer, 0.0f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            numerator = pair_add(numerator, pair_neg(two_product(weights[j], xi)));
            denominator = pair_add(denominator, two_product(weights[j], weights[j]));
        }
        float values[4];
        if (isfinite(numerator.x) && isfinite(numerator.y)
                && isfinite(denominator.x) && isfinite(denominator.y)) {
            const float2 q = corrected_ratio(numerator, denominator);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                values[j] = compensated_output(xi, weights[j], q.x, q.y);
        } else {
            // Finite large weights can overflow w*w while the FFT baseline
            // still returns a finite result. TwoSum on infinity is invalid.
            // Rescale by a power of two in FP32, factoring x out:
            // a=w/s, t=(1/s-sum(a))/(sum(a*a)+lambda/s/s), out=x*(1+a*t).
            // Power-of-two scaling preserves signed cancellation and the
            // two-component coefficient avoids an extra rounding before x.
            float maximum = 1.0f;
            bool finite_inputs = isfinite(xi) && isfinite(regularizer);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                maximum = fmaxf(maximum, fabsf(weights[j]));
                finite_inputs = finite_inputs && isfinite(weights[j]);
            }
            // Nonfinite inputs retain IEEE propagation and must not enter
            // ilogbf(infinity), whose exponent cannot safely be negated.
            const int exponent = finite_inputs ? ilogbf(maximum) : 0;
            float scaled[4];
            numerator = make_float2(scalbnf(1.0f, -exponent), 0.0f);
            denominator = make_float2(scalbnf(regularizer, -2 * exponent), 0.0f);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scaled[j] = scalbnf(weights[j], -exponent);
                numerator = pair_add(numerator, make_float2(-scaled[j], 0.0f));
                denominator = pair_add(denominator, two_product(scaled[j], scaled[j]));
            }
            const float2 t = corrected_ratio(numerator, denominator);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 coefficient = compensated_output_pair(1.0f, scaled[j], t.x, t.y);
                values[j] = __fmaf_rn(xi, coefficient.x, __fmul_rn(xi, coefficient.y));
            }
        }
        const I h = (i / width) % height, w = i % width;
        const I base = bc * (4*hw) + (2*h) * (2*width) + 2*w;
        // Existing roll(-1,-1) phase convention; each output has one writer.
        output[base+2*width+1] = values[0];
        output[base+2*width] = values[1];
        output[base+1] = values[2];
        output[base] = values[3];
    }
}
}

// Host scalar boundary only: Python eps arrives as double and is converted to
// float before launch, as in ATen's scalar operand conversion for FP32 add.
at::Tensor converse_nearest_k2_s2_cuda(at::Tensor x0, at::Tensor weight0, at::Tensor bias0, double eps) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "FFT-free nearest is inference-only: use no_grad or inference_mode");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
    TORCH_CHECK(x0.is_cuda() && weight0.is_cuda() && bias0.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(x0.scalar_type()==at::kFloat && weight0.scalar_type()==at::kFloat && bias0.scalar_type()==at::kFloat, "FP32 tensors required");
    TORCH_CHECK(x0.device()==weight0.device() && x0.device()==bias0.device(), "device mismatch");
    TORCH_CHECK(x0.dim()==4 && x0.numel()>0, "expected nonempty NCHW input");
    TORCH_CHECK(x0.numel()<=INT64_MAX/4 && x0.size(2)<=INT64_MAX/2 && x0.size(3)<=INT64_MAX/2, "output dimensions overflow");
    TORCH_CHECK(weight0.dim()==4 && weight0.size(2)==2 && weight0.size(3)==2, "only k2/s2 is supported");
    TORCH_CHECK((weight0.size(0)==1 || weight0.size(0)==x0.size(0)) &&
                (weight0.size(1)==1 || weight0.size(1)==x0.size(1)), "invalid kernel broadcast");
    TORCH_CHECK(bias0.sizes()==at::IntArrayRef({1,x0.size(1),1,1}), "expected per-channel bias (1,C,1,1)");
    const c10::cuda::CUDAGuard guard(x0.device());
    auto x = x0.resolve_neg().contiguous(), weight = weight0.resolve_neg().contiguous();
    auto bias = bias0.resolve_neg().contiguous();
    auto output = at::empty({x.size(0),x.size(1),2*x.size(2),2*x.size(3)}, x.options());
    const int blocks = static_cast<int>(std::min<I>((x.numel()+255)/256, 65535));
    nearest_k2_compensated_fused_lambda<<<blocks,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), static_cast<float>(eps), output.data_ptr<float>(),
        x.numel(), x.size(1), x.size(2), x.size(3), weight.size(0), weight.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
