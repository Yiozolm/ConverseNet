#include "pointwise_wgrad.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// FP32 weight gradient of a 1x1 convolution as a split-K SIMT GEMM.
//
//   dW[co][ci] = sum_{b,p} G[b][co][p] * X[b][ci][p]
//
// Both operands are NCHW, so the contraction index k = b*HW + p is the
// contiguous one of each row. A block owns a 128x64 output tile and a chunk of
// kTilesPerSplit*kTileK = 256 contraction terms; tiles stream through a
// three-stage cp.async pipeline. Every thread accumulates 8x4 outputs with
// FMA chains of kChain terms, then folds each chain sum into a (hi, lo)
// compensated accumulator (Knuth TwoSum). The per-split (hi, lo) partials are
// reduced in index order by a second kernel with the same compensation, so
// the result is deterministic and its rounding error is a few FP32 ulps.
// The chunking depends only on K, not on the device, so a given geometry
// rounds identically on every GPU.
namespace converse2d::peripheral {
namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;
constexpr int kStages = 3;
constexpr int kLd = kTileK + 1; // padded shared row: conflict-free column reads
constexpr int kStageFloats = (kTileM + kTileN) * kLd;
constexpr int kTilesPerSplit = 16; // 256 contraction terms per block partial
#ifndef CONVERSE2D_WGRAD_CHAIN
#define CONVERSE2D_WGRAD_CHAIN 8
#endif
constexpr int kChain = CONVERSE2D_WGRAD_CHAIN;
static_assert(kTileK % kChain == 0, "chain must divide the K tile");
static_assert(kThreads == 16 * 16, "thread grid is 16 (m) x 16 (n)");
static_assert(kTileM == 8 * 16 && kTileN == 4 * 16, "each thread owns 8 x 4 outputs");

__device__ __forceinline__ void cp_async4(float* smem, const float* global, bool valid) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = valid ? 4 : 0; // src-size 0 zero-fills without reading
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(global), "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int Pending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending));
}

// Error-free transformation: sum + err == a + b exactly.
__device__ __forceinline__ void two_sum(float a, float b, float& sum, float& err) {
    sum = __fadd_rn(a, b);
    const float z = __fsub_rn(sum, a);
    err = __fadd_rn(__fsub_rn(a, __fsub_rn(sum, z)), __fsub_rn(b, z));
}

// One K tile of A (128 rows) and B (64 rows) into a pipeline stage. Rows past
// the operand, and columns past K, are zero-filled.
__device__ __forceinline__ void load_stage(float* stage, const float* __restrict__ a, int m_count, int m_base,
                                           const float* __restrict__ b, int n_count, int n_base, int hw,
                                           long long k_total, long long k_base, int tid) {
    const int k = tid & 15;
    const int row0 = tid >> 4;
    const long long kg = k_base + k;
    const bool k_valid = kg < k_total;
    const long long batch = k_valid ? kg / hw : 0;
    const long long pixel = k_valid ? kg - batch * hw : 0;
    float* as = stage;
    float* bs = stage + kTileM * kLd;
#pragma unroll
    for (int i = 0; i < kTileM / 16; ++i) {
        const int row = row0 + 16 * i;
        const int m = m_base + row;
        const bool valid = k_valid && m < m_count;
        const float* src = valid ? a + (batch * m_count + m) * static_cast<long long>(hw) + pixel : a;
        cp_async4(as + row * kLd + k, src, valid);
    }
#pragma unroll
    for (int i = 0; i < kTileN / 16; ++i) {
        const int row = row0 + 16 * i;
        const int n = n_base + row;
        const bool valid = k_valid && n < n_count;
        const float* src = valid ? b + (batch * n_count + n) * static_cast<long long>(hw) + pixel : b;
        cp_async4(bs + row * kLd + k, src, valid);
    }
}

__global__ void __launch_bounds__(kThreads, 2)
pointwise_wgrad_partials(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ partials,
                         int m_count, int n_count, int hw, long long k_total, int tiles_per_split) {
    __shared__ __align__(16) float smem[kStages * kStageFloats];
    const int tid = threadIdx.x;
    const int split = blockIdx.x;
    const int n_base = blockIdx.y * kTileN;
    const int m_base = blockIdx.z * kTileM;
    const long long k_begin = static_cast<long long>(split) * tiles_per_split * kTileK;
    const long long k_end = min(k_begin + static_cast<long long>(tiles_per_split) * kTileK, k_total);
    const int tiles = static_cast<int>((k_end - k_begin + kTileK - 1) / kTileK);
    const int tm = tid & 15;
    const int tn = tid >> 4;

    float acc[8][4], hi[8][4], lo[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = hi[i][j] = lo[i][j] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < tiles)
            load_stage(smem + s * kStageFloats, a, m_count, m_base, b, n_count, n_base, hw, k_total,
                       k_begin + static_cast<long long>(s) * kTileK, tid);
        cp_async_commit();
    }

    for (int t = 0; t < tiles; ++t) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        {
            const int next = t + kStages - 1;
            if (next < tiles)
                load_stage(smem + (next % kStages) * kStageFloats, a, m_count, m_base, b, n_count, n_base, hw,
                           k_total, k_begin + static_cast<long long>(next) * kTileK, tid);
            cp_async_commit();
        }
        const float* as = smem + (t % kStages) * kStageFloats;
        const float* bs = as + kTileM * kLd;
#pragma unroll
        for (int k = 0; k < kTileK; ++k) {
            float av[8], bv[4];
#pragma unroll
            for (int i = 0; i < 8; ++i) av[i] = as[(tm + 16 * i) * kLd + k];
#pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = bs[(4 * tn + j) * kLd + k];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
            if ((k + 1) % kChain == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float sum, err;
                        two_sum(hi[i][j], acc[i][j], sum, err);
                        hi[i][j] = sum;
                        lo[i][j] = __fadd_rn(lo[i][j], err);
                        acc[i][j] = 0.f;
                    }
            }
        }
    }

    const size_t plane = static_cast<size_t>(n_count) * m_count;
    float* out_hi = partials + static_cast<size_t>(split) * 2 * plane;
    float* out_lo = out_hi + plane;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n_base + 4 * tn + j;
        if (n >= n_count) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = m_base + tm + 16 * i;
            if (m >= m_count) continue;
            const size_t index = static_cast<size_t>(n) * m_count + m;
            out_hi[index] = hi[i][j];
            out_lo[index] = lo[i][j];
        }
    }
}

__global__ void pointwise_wgrad_reduce(const float* __restrict__ partials, float* __restrict__ out, int splits,
                                       int m_count, int n_count, int out_stride_m, int out_stride_n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m_count * n_count) return;
    const int m = index % m_count;
    const int n = index / m_count;
    const size_t plane = static_cast<size_t>(n_count) * m_count;
    float hi = 0.f, lo = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* base = partials + static_cast<size_t>(s) * 2 * plane + index;
        float sum, err;
        two_sum(hi, base[0], sum, err);
        hi = sum;
        lo = __fadd_rn(lo, __fadd_rn(err, base[plane]));
    }
    out[static_cast<size_t>(m) * out_stride_m + static_cast<size_t>(n) * out_stride_n] = __fadd_rn(hi, lo);
}

} // namespace

at::Tensor pointwise_wgrad_cuda(const at::Tensor& grad_output, const at::Tensor& input) {
    const int64_t batch = input.size(0);
    const int64_t cin = input.size(1);
    const int64_t cout = grad_output.size(1);
    const int64_t hw = input.size(2) * input.size(3);
    const long long k_total = static_cast<long long>(batch) * hw;
    // The operand with more channels fills the 128-row side of the tile.
    const bool swap = cout < cin;
    const at::Tensor& a = swap ? input : grad_output;
    const at::Tensor& b = swap ? grad_output : input;
    const int m_count = static_cast<int>(swap ? cin : cout);
    const int n_count = static_cast<int>(swap ? cout : cin);
    const long long k_tiles = (k_total + kTileK - 1) / kTileK;
    const int64_t splits = (k_tiles + kTilesPerSplit - 1) / kTilesPerSplit;
    auto partials = at::empty({splits, 2, n_count, m_count}, input.options());
    auto out = at::empty({cout, cin, 1, 1}, input.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    const dim3 grid(static_cast<unsigned>(splits), static_cast<unsigned>((n_count + kTileN - 1) / kTileN),
                    static_cast<unsigned>((m_count + kTileM - 1) / kTileM));
    pointwise_wgrad_partials<<<grid, kThreads, 0, stream>>>(
        a.const_data_ptr<float>(), b.const_data_ptr<float>(), partials.data_ptr<float>(), m_count, n_count,
        static_cast<int>(hw), k_total, kTilesPerSplit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int outputs = m_count * n_count;
    pointwise_wgrad_reduce<<<(outputs + 255) / 256, 256, 0, stream>>>(
        partials.const_data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(splits), m_count, n_count,
        swap ? 1 : static_cast<int>(cin), swap ? static_cast<int>(cin) : 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace converse2d::peripheral
