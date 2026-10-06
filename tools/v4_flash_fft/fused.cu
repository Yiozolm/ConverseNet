// Research prototype: SRAM-resident s1 training forward, one block per (b, c) plane.
//   out = real_crop(ifft2(solve(fft2(circular_pad(x)), k, l)), pad)
// The padded complex plane lives in shared memory for the whole chain; global
// memory sees one read of x, one read of k and one write of the output.
// FFTs are mixed-radix (2, 3, 4, 5) Stockham passes, in place through registers,
// with twiddles from a host-rounded table of exp(-2*pi*i*t/n).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <map>
#include <vector>

namespace {

constexpr int kThreads = 512;
constexpr int kMaxElements = 10240;  // 80 KB of float2
constexpr int kMaxStages = 8;

struct Axis {
    int n;
    int stages;
    int radix[kMaxStages];
    const float2* twiddle;  // n roots, device
};

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x));
}
__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
// -i*z for forward, +i*z for inverse.
template <bool Inverse> __device__ __forceinline__ float2 rot(float2 z) {
    return Inverse ? make_float2(-z.y, z.x) : make_float2(z.y, -z.x);
}

template <int R, bool Inverse> __device__ __forceinline__ void dft(float2* v) {
    if constexpr (R == 2) {
        float2 a = v[0], b = v[1];
        v[0] = cadd(a, b);
        v[1] = csub(a, b);
    } else if constexpr (R == 4) {
        float2 s02 = cadd(v[0], v[2]), d02 = csub(v[0], v[2]);
        float2 s13 = cadd(v[1], v[3]), d13 = rot<Inverse>(csub(v[1], v[3]));
        v[0] = cadd(s02, s13);
        v[2] = csub(s02, s13);
        v[1] = cadd(d02, d13);
        v[3] = csub(d02, d13);
    } else if constexpr (R == 3) {
        constexpr float c = -0.5f, s = 0.866025403784438646763723170752936183f;
        float2 sum = cadd(v[1], v[2]), dif = rot<Inverse>(csub(v[1], v[2]));
        float2 t = make_float2(fmaf(c, sum.x, v[0].x), fmaf(c, sum.y, v[0].y));
        v[0] = cadd(v[0], sum);
        v[1] = make_float2(fmaf(s, dif.x, t.x), fmaf(s, dif.y, t.y));
        v[2] = make_float2(fmaf(-s, dif.x, t.x), fmaf(-s, dif.y, t.y));
    } else if constexpr (R == 5) {
        constexpr float c1 = 0.309016994374947424102293417182819059f, c2 = -0.809016994374947424102293417182819059f;
        constexpr float s1 = 0.951056516295153572116439333379382143f, s2 = 0.587785252292473129168705954639072769f;
        float2 a1 = cadd(v[1], v[4]), b1 = rot<Inverse>(csub(v[1], v[4]));
        float2 a2 = cadd(v[2], v[3]), b2 = rot<Inverse>(csub(v[2], v[3]));
        float2 t1 = make_float2(v[0].x + fmaf(c1, a1.x, c2 * a2.x), v[0].y + fmaf(c1, a1.y, c2 * a2.y));
        float2 t2 = make_float2(v[0].x + fmaf(c2, a1.x, c1 * a2.x), v[0].y + fmaf(c2, a1.y, c1 * a2.y));
        float2 u1 = make_float2(fmaf(s1, b1.x, s2 * b2.x), fmaf(s1, b1.y, s2 * b2.y));
        float2 u2 = make_float2(fmaf(s2, b1.x, -s1 * b2.x), fmaf(s2, b1.y, -s1 * b2.y));
        v[0] = cadd(v[0], cadd(a1, a2));
        v[1] = cadd(t1, u1);
        v[4] = csub(t1, u1);
        v[2] = cadd(t2, u2);
        v[3] = csub(t2, u2);
    }
}

// One Stockham stage over `lines` independent transforms of length n.
// Element e of line l is at s[l * line_stride + e * elem_stride]. LineFastest
// maps consecutive threads to consecutive lines (column passes).
template <int R, bool Inverse, bool LineFastest>
__device__ void stage(float2* s, int lines, int line_stride, int elem_stride, const Axis& a, int ns) {
    constexpr int cap = (kMaxElements / R + kThreads - 1) / kThreads;
    const int n = a.n, per_line = n / R, total = lines * per_line, span = n / (ns * R);
    float2 v[cap][R];
#pragma unroll
    for (int q = 0; q < cap; ++q) {
        const int u = threadIdx.x + q * kThreads;
        if (u < total) {
            const int line = LineFastest ? u % lines : u / per_line, j = LineFastest ? u / lines : u % per_line;
            const float2* base = s + line * line_stride;
            const int t = (j % ns) * span;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float2 x = base[(j + r * per_line) * elem_stride];
                if (r) {
                    float2 w = __ldg(a.twiddle + t * r);
                    if (Inverse) w.y = -w.y;
                    x = cmul(x, w);
                }
                v[q][r] = x;
            }
            dft<R, Inverse>(v[q]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < cap; ++q) {
        const int u = threadIdx.x + q * kThreads;
        if (u < total) {
            const int line = LineFastest ? u % lines : u / per_line, j = LineFastest ? u / lines : u % per_line;
            float2* base = s + line * line_stride;
            const int out = (j / ns) * ns * R + j % ns;
#pragma unroll
            for (int r = 0; r < R; ++r)
                base[(out + r * ns) * elem_stride] = v[q][r];
        }
    }
    __syncthreads();
}

template <bool Inverse, bool LineFastest>
__device__ void fft_lines(float2* s, int lines, int line_stride, int elem_stride, const Axis& a) {
    int ns = 1;
    for (int i = 0; i < a.stages; ++i) {
        switch (a.radix[i]) {
        case 2: stage<2, Inverse, LineFastest>(s, lines, line_stride, elem_stride, a, ns); break;
        case 3: stage<3, Inverse, LineFastest>(s, lines, line_stride, elem_stride, a, ns); break;
        case 4: stage<4, Inverse, LineFastest>(s, lines, line_stride, elem_stride, a, ns); break;
        default: stage<5, Inverse, LineFastest>(s, lines, line_stride, elem_stride, a, ns); break;
        }
        ns *= a.radix[i];
    }
}

__device__ __forceinline__ int wrap(int i, int n) { return i < 0 ? i + n : (i >= n ? i - n : i); }

// Same FP32 operation boundaries as scale1_forward_planes with p = y.
__device__ __forceinline__ float2 solve(float2 y, float2 k, float l) {
    const float d = __fadd_rn(__fadd_rn(__fmul_rn(k.x, k.x), __fmul_rn(k.y, k.y)), l);
    const c10::complex<float> pm(__fmaf_rn(k.x, y.x, -__fmul_rn(k.y, y.y)), __fmaf_rn(k.y, y.x, __fmul_rn(k.x, y.y)));
    const c10::complex<float> q = c10::complex<float>(__fadd_rn(y.x, -pm.real()), __fadd_rn(y.y, -pm.imag())) /
                                  c10::complex<float>(d, 0);
    const float2 kq = make_float2(__fmaf_rn(k.x, q.real(), -__fmul_rn(-k.y, q.imag())),
                                  __fmaf_rn(-k.y, q.real(), __fmul_rn(k.x, q.imag())));
    return make_float2(__fadd_rn(y.x, kq.x), __fadd_rn(y.y, kq.y));
}

__global__ void __launch_bounds__(kThreads) fused_s1_forward(const float* __restrict__ x, const float2* __restrict__ k,
                                                             const float* __restrict__ l, float* __restrict__ out,
                                                             int C, int h0, int w0, int pad, Axis rows, Axis cols) {
    extern __shared__ float2 s[];
    const int H = cols.n, W = rows.n, plane = blockIdx.x, c = plane % C;
    const float* xp = x + size_t(plane) * h0 * w0;
    for (int i = threadIdx.x; i < H * W; i += kThreads) {
        const int r = i / W, q = i % W;
        s[i] = make_float2(xp[wrap(r - pad, h0) * w0 + wrap(q - pad, w0)], 0.f);
    }
    __syncthreads();
    fft_lines<false, false>(s, H, W, 1, rows);
    fft_lines<false, true>(s, W, 1, W, cols);
    const float2* kp = k + size_t(c) * H * W;
    const float lc = l[c];
    for (int i = threadIdx.x; i < H * W; i += kThreads)
        s[i] = solve(s[i], __ldg(kp + i), lc);
    __syncthreads();
    fft_lines<true, true>(s, W, 1, W, cols);
    fft_lines<true, false>(s, H, W, 1, rows);
    const float scale = float(1.0 / double(H * W));
    float* op = out + size_t(plane) * h0 * w0;
    for (int i = threadIdx.x; i < h0 * w0; i += kThreads) {
        const int r = i / w0, q = i % w0;
        op[i] = s[(r + pad) * W + q + pad].x * scale;
    }
}


// ---- v2: sizes, radices and strides fixed at compile time ------------------
__host__ __device__ constexpr int radix_at(int n, int i) {
    const int rs[4] = {4, 2, 3, 5};
    int m = n, idx = 0;
    for (int a = 0; a < 4; ++a)
        while (m % rs[a] == 0 && !(rs[a] == 2 && m % 4 == 0)) {
            if (idx == i) return rs[a];
            ++idx;
            m /= rs[a];
        }
    return 0;
}

// Lines are independent, so a stage may run in line chunks: at most about
// kBudget complex values are staged per thread (100x100 and smaller stay one
// chunk; 128x128 takes 2 and 144x144 takes 3), which keeps large planes in
// registers without spilling.
constexpr int kBudget = 20;

template <int R, bool Inverse, bool LineFastest, int N, int NS, int Lines, int LineStride, int ElemStride>
__device__ __forceinline__ void static_stage(float2* s, const float2* __restrict__ tw, int tid) {
    constexpr int per_line = N / R, span = N / (NS * R);
    constexpr int fit = kBudget * kThreads / N > 0 ? kBudget * kThreads / N : 1;
    constexpr int chunks = (Lines + fit - 1) / fit, lpc = (Lines + chunks - 1) / chunks;
    constexpr int total = lpc * per_line, cap = (total + kThreads - 1) / kThreads;
#pragma unroll 1
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int first = chunk * lpc;
        float2 v[cap][R];
#pragma unroll
        for (int q = 0; q < cap; ++q) {
            const int u = tid + q * kThreads;
            const int local = LineFastest ? u % lpc : u / per_line, j = LineFastest ? u / lpc : u % per_line;
            if ((total % kThreads == 0 || u < total) && (Lines % lpc == 0 || first + local < Lines)) {
                const float2* base = s + (first + local) * LineStride;
                const int t = (j % NS) * span;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float2 x = base[(j + r * per_line) * ElemStride];
                    if (r && NS > 1) {
                        float2 w = __ldg(tw + t * r);
                        if (Inverse) w.y = -w.y;
                        x = cmul(x, w);
                    }
                    v[q][r] = x;
                }
                dft<R, Inverse>(v[q]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < cap; ++q) {
            const int u = tid + q * kThreads;
            const int local = LineFastest ? u % lpc : u / per_line, j = LineFastest ? u / lpc : u % per_line;
            if ((total % kThreads == 0 || u < total) && (Lines % lpc == 0 || first + local < Lines)) {
                float2* base = s + (first + local) * LineStride;
                const int out = (j / NS) * NS * R + j % NS;
#pragma unroll
                for (int r = 0; r < R; ++r)
                    base[(out + r * NS) * ElemStride] = v[q][r];
            }
        }
        __syncthreads();
    }
}

template <int N, int I, int NS, bool Inverse, bool LineFastest, int Lines, int LineStride, int ElemStride>
__device__ __forceinline__ void static_fft(float2* s, const float2* __restrict__ tw, int tid) {
    constexpr int R = radix_at(N, I);
    if constexpr (R != 0) {
        static_stage<R, Inverse, LineFastest, N, NS, Lines, LineStride, ElemStride>(s, tw, tid);
        static_fft<N, I + 1, NS * R, Inverse, LineFastest, Lines, LineStride, ElemStride>(s, tw, tid);
    } else {
        static_assert(NS == N, "length must be 2/3/5-smooth");
    }
}

// Pair packs batches (2m, 2m+1) of one channel as x_2m + i*x_2m+1. The solve is
// linear in the spectrum, so the real and imaginary outputs are the two results.
template <int H, int W, bool Pair>
__global__ void __launch_bounds__(kThreads) static_s1_forward(const float* __restrict__ x, const float2* __restrict__ k,
                                                              const float* __restrict__ l, float* __restrict__ out,
                                                              int C, int h0, int w0, int pad,
                                                              const float2* __restrict__ tw_rows,
                                                              const float2* __restrict__ tw_cols) {
    extern __shared__ float2 s[];
    const int c = blockIdx.x % C, b = Pair ? 2 * (blockIdx.x / C) : blockIdx.x / C;
    const size_t plane = size_t(h0) * w0;
    const float* x0 = x + (size_t(b) * C + c) * plane;
    const float* x1 = x0 + size_t(C) * plane;
    for (int i = threadIdx.x; i < H * W; i += kThreads) {
        const int r = i / W, q = i % W;
        const int src = wrap(r - pad, h0) * w0 + wrap(q - pad, w0);
        s[i] = make_float2(x0[src], Pair ? x1[src] : 0.f);
    }
    __syncthreads();
    static_fft<W, 0, 1, false, false, H, W, 1>(s, tw_rows, threadIdx.x);
    static_fft<H, 0, 1, false, true, W, 1, W>(s, tw_cols, threadIdx.x);
    const float2* kp = k + size_t(c) * H * W;
    const float lc = l[c];
    for (int i = threadIdx.x; i < H * W; i += kThreads)
        s[i] = solve(s[i], __ldg(kp + i), lc);
    __syncthreads();
    static_fft<H, 0, 1, true, true, W, 1, W>(s, tw_cols, threadIdx.x);
    static_fft<W, 0, 1, true, false, H, W, 1>(s, tw_rows, threadIdx.x);
    const float scale = float(1.0 / double(H * W));
    float* o0 = out + (size_t(b) * C + c) * plane;
    float* o1 = o0 + size_t(C) * plane;
    for (int i = threadIdx.x; i < h0 * w0; i += kThreads) {
        const int r = i / w0, q = i % w0;
        const float2 z = s[(r + pad) * W + q + pad];
        o0[i] = z.x * scale;
        if (Pair) o1[i] = z.y * scale;
    }
}

template <int H, int W, bool Pair>
void launch_static(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, at::Tensor& out, int pad,
                   const float2* tw_rows, const float2* tw_cols) {
    auto kernel = static_s1_forward<H, W, Pair>;
    const int bytes = H * W * sizeof(float2);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    const int blocks = (Pair ? x.size(0) / 2 : x.size(0)) * x.size(1);
    kernel<<<blocks, kThreads, bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), reinterpret_cast<const float2*>(k.data_ptr()), l.data_ptr<float>(), out.data_ptr<float>(),
        x.size(1), x.size(2), x.size(3), pad, tw_rows, tw_cols);
}

template <bool Pair>
void dispatch_static(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, at::Tensor& out, int pad, int H,
                     int W, const float2* tr, const float2* tc) {
    if (H == 100 && W == 100) return launch_static<100, 100, Pair>(x, k, l, out, pad, tr, tc);
    if (H == 96 && W == 96) return launch_static<96, 96, Pair>(x, k, l, out, pad, tr, tc);
    if (H == 64 && W == 64) return launch_static<64, 64, Pair>(x, k, l, out, pad, tr, tc);
    TORCH_CHECK(false, "no static instantiation for ", H, "x", W);
}

// ---- s1 backward: one block per channel, batches in order ------------------
// Production adjoint (scale1_adjoint, shared prior), with G = FFT(embed(g))/N
// and Y = FFT(circular_pad(x)) recomputed on chip instead of saved:
//   t = G*k, gy = t/d, gm = -gy, q = (Y - k*Y)/d, gd = Re((-t)*conj(q/d))
//   grad_Y = (G + gy) + gm*conj(k)
//   grad_k += conj(G*conj(q)) + gm*conj(Y) + 2*k*gd,   grad_l += gd
// grad_x folds Re(unnormalized IFFT(grad_Y)) back through the circular pad.
__device__ __forceinline__ float2 prod(float2 a, float2 b) {
    return make_float2(__fmaf_rn(a.x, b.x, -__fmul_rn(a.y, b.y)), __fmaf_rn(a.y, b.x, __fmul_rn(a.x, b.y)));
}
__device__ __forceinline__ float2 conj2(float2 a) { return make_float2(a.x, -a.y); }
__device__ __forceinline__ float2 add2(float2 a, float2 b) { return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y)); }
__device__ __forceinline__ float2 div_real(float2 a, float d) {
    const auto r = c10::complex<float>(a.x, a.y) / c10::complex<float>(d, 0);
    return make_float2(r.real(), r.imag());
}

// Block-wide sum of every thread's `value`, added in launch (batch) order into
// *out: the first batch writes, later batches accumulate.
__device__ void store_block_sum(float value, float* partial, float* out, int b) {
    for (int o = 16; o; o >>= 1) value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, o));
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kThreads / 32 ? partial[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) *out = b ? __fadd_rn(*out, v) : v;
    }
}

template <int H, int W>
__global__ void __launch_bounds__(kThreads, 1) static_s1_backward(
    const float* __restrict__ x, const float* __restrict__ g, const float2* __restrict__ k, const float* __restrict__ l,
    float* __restrict__ gx, float2* __restrict__ gk, float* __restrict__ gl, float2* __restrict__ scratch, int b, int C,
    int h0, int w0, int pad, const float2* __restrict__ tw_rows, const float2* __restrict__ tw_cols) {
    extern __shared__ float2 s[];
    __shared__ float partial[kThreads / 32];
    constexpr int N = H * W;
    const int c = blockIdx.x;
    const size_t plane = size_t(h0) * w0;
    const float2* kp = k + size_t(c) * N;
    float2* gkp = gk + size_t(c) * N;
    // Y is parked per channel in global scratch (L2-resident), not registers:
    // holding it across the gradient FFT spilled.
    float2* yp = scratch + size_t(c) * N;
    const float lc = l[c], scale = float(1.0 / double(N));
    float gl_sum = 0.f;
    const float2* tw_r = tw_rows;
    const float2* tw_c = tw_cols;
    const int tid = threadIdx.x;
    {
        const size_t offset = (size_t(b) * C + c) * plane;
        for (int i = threadIdx.x; i < N; i += kThreads) {
            const int r = i / W, q = i % W;
            s[i] = make_float2(x[offset + wrap(r - pad, h0) * w0 + wrap(q - pad, w0)], 0.f);
        }
        __syncthreads();
        static_fft<W, 0, 1, false, false, H, W, 1>(s, tw_r, tid);
        static_fft<H, 0, 1, false, true, W, 1, W>(s, tw_c, tid);
        for (int i = threadIdx.x; i < N; i += kThreads) yp[i] = s[i];
        __syncthreads();
        for (int i = threadIdx.x; i < N; i += kThreads) {
            const int r = i / W - pad, q = i % W - pad;
            const bool inside = r >= 0 && r < h0 && q >= 0 && q < w0;
            s[i] = make_float2(inside ? g[offset + r * w0 + q] : 0.f, 0.f);
        }
        __syncthreads();
        static_fft<W, 0, 1, false, false, H, W, 1>(s, tw_r, tid);
        static_fft<H, 0, 1, false, true, W, 1, W>(s, tw_c, tid);
        for (int i = threadIdx.x; i < N; i += kThreads) {
            {
                const float2 z = s[i], G = make_float2(z.x * scale, z.y * scale), ki = __ldg(kp + i), Y = yp[i];
                const float den = __fadd_rn(__fadd_rn(__fmul_rn(ki.x, ki.x), __fmul_rn(ki.y, ki.y)), lc);
                const float2 t = prod(G, ki), gy = div_real(t, den), gm = make_float2(-gy.x, -gy.y);
                const float2 pm = prod(ki, Y);
                const float2 q = div_real(make_float2(__fadd_rn(Y.x, -pm.x), __fadd_rn(Y.y, -pm.y)), den);
                const float v = prod(make_float2(-t.x, -t.y), conj2(div_real(q, den))).x;
                s[i] = add2(add2(G, gy), prod(gm, conj2(ki)));
                const float2 direct = prod(G, conj2(q)), prediction = prod(gm, conj2(Y));
                const float2 power = make_float2(__fmul_rn(v, __fmul_rn(2.f, ki.x)), __fmul_rn(v, __fmul_rn(2.f, ki.y)));
                float2 term = add2(add2(add2(conj2(direct), prediction), make_float2(0.f, power.y)), make_float2(power.x, 0.f));
                if (b) term = add2(gkp[i], term);
                gkp[i] = term;
                gl_sum = __fadd_rn(gl_sum, v);
            }
        }
        __syncthreads();
        static_fft<H, 0, 1, true, true, W, 1, W>(s, tw_c, tid);
        static_fft<W, 0, 1, true, false, H, W, 1>(s, tw_r, tid);
        // Each interior pixel collects its own padded copy plus any wrapped margin copies.
        for (int i = threadIdx.x; i < h0 * w0; i += kThreads) {
            const int r = i / w0, q = i % w0;
            float sum = 0.f;
            for (int dr = -1; dr <= 1; ++dr) {
                const int R = r + pad + dr * h0;
                if (R < 0 || R >= H) continue;
                for (int dq = -1; dq <= 1; ++dq) {
                    const int Q = q + pad + dq * w0;
                    if (Q >= 0 && Q < W) sum = __fadd_rn(sum, s[R * W + Q].x);
                }
            }
            gx[offset + i] = sum;
        }
        __syncthreads();
    }
    store_block_sum(gl_sum, partial, gl + c, b);
}

template <int H, int W>
void launch_backward(const at::Tensor& x, const at::Tensor& g, const at::Tensor& k, const at::Tensor& l, at::Tensor& gx,
                     at::Tensor& gk, at::Tensor& gl, int pad, const float2* tr, const float2* tc) {
    auto kernel = static_s1_backward<H, W>;
    const int bytes = H * W * sizeof(float2);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    auto scratch = at::empty_like(k);
    // One launch per batch index, in order: gk accumulates deterministically.
    for (int b = 0; b < x.size(0); ++b)
        kernel<<<x.size(1), kThreads, bytes, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), g.data_ptr<float>(), reinterpret_cast<const float2*>(k.data_ptr()),
            l.data_ptr<float>(), gx.data_ptr<float>(), reinterpret_cast<float2*>(gk.data_ptr()),
            gl.data_ptr<float>(), reinterpret_cast<float2*>(scratch.data_ptr()), b, x.size(1), x.size(2), x.size(3),
            pad, tr, tc);
}

Axis axis_for(int n, int device);

// ---- s2/s3: low-res x, high-res prior x0 and output ------------------------
// One high-res (S*H) x (S*W) complex plane lives in shared memory; the low-res
// spectrum of x (H x W) is parked in an L2-resident scratch. Alias groups are
// {(h + a*H, w + b*W)}; production's solve (scale2/scale3 kernels):
//   pm = mean(k*p), km = mean(|k|^2), d = km + l, q = (y - pm)/d,
//   out = p + conj(k)*q at every alias.
template <int H, int W>
__device__ __forceinline__ void fft2_plane(float2* s, const float2* tr, const float2* tc, int tid) {
    static_fft<W, 0, 1, false, false, H, W, 1>(s, tr, tid);
    static_fft<H, 0, 1, false, true, W, 1, W>(s, tc, tid);
}
template <int H, int W>
__device__ __forceinline__ void ifft2_plane(float2* s, const float2* tr, const float2* tc, int tid) {
    static_fft<H, 0, 1, true, true, W, 1, W>(s, tc, tid);
    static_fft<W, 0, 1, true, false, H, W, 1>(s, tr, tid);
}

struct Twiddles {
    const float2 *low_rows, *low_cols, *high_rows, *high_cols;
};

template <int H, int W, int S>
__global__ void __launch_bounds__(kThreads, 1) static_s_forward(
    const float* __restrict__ x, const float* __restrict__ x0, const float2* __restrict__ k,
    const float* __restrict__ l, float* __restrict__ out, float2* __restrict__ scratch, int C, Twiddles tw,
    float prediction_factor, float power_factor) {
    extern __shared__ float2 s[];
    constexpr int SH = S * H, SW = S * W, N = SH * SW, n = H * W;
    const int bc = blockIdx.x, c = bc % C, tid = threadIdx.x;
    const float* xp = x + size_t(bc) * n;
    for (int i = tid; i < n; i += kThreads) s[i] = make_float2(xp[i], 0.f);
    __syncthreads();
    fft2_plane<H, W>(s, tw.low_rows, tw.low_cols, tid);
    float2* yp = scratch + size_t(bc) * n;
    for (int i = tid; i < n; i += kThreads) yp[i] = s[i];
    __syncthreads();
    const float* x0p = x0 + size_t(bc) * N;
    for (int i = tid; i < N; i += kThreads) s[i] = make_float2(x0p[i], 0.f);
    __syncthreads();
    fft2_plane<SH, SW>(s, tw.high_rows, tw.high_cols, tid);
    const float2* kp = k + size_t(c) * N;
    const float lc = l[c];
    for (int i = tid; i < n; i += kThreads) {
        const int h = i / W, w = i % W;
        float2 pm = make_float2(0.f, 0.f);
        float pw = 0.f;
#pragma unroll
        for (int j = 0; j < S * S; ++j) {
            const int off = (h + (j / S) * H) * SW + w + (j % S) * W;
            const float2 kk = __ldg(kp + off);
            pm = add2(pm, prod(kk, s[off]));
            pw = __fadd_rn(pw, __fadd_rn(__fmul_rn(kk.x, kk.x), __fmul_rn(kk.y, kk.y)));
        }
        pm = make_float2(__fmul_rn(pm.x, prediction_factor), __fmul_rn(pm.y, prediction_factor));
        const float den = __fadd_rn(__fmul_rn(pw, power_factor), lc);
        const float2 y = yp[i];
        const float2 q = div_real(make_float2(__fadd_rn(y.x, -pm.x), __fadd_rn(y.y, -pm.y)), den);
#pragma unroll
        for (int j = 0; j < S * S; ++j) {
            const int off = (h + (j / S) * H) * SW + w + (j % S) * W;
            s[off] = add2(s[off], prod(conj2(__ldg(kp + off)), q));
        }
    }
    __syncthreads();
    ifft2_plane<SH, SW>(s, tw.high_rows, tw.high_cols, tid);
    const float scale = float(1.0 / double(N));
    float* op = out + size_t(bc) * N;
    for (int i = tid; i < N; i += kThreads) op[i] = s[i].x * scale;
}

// Backward, one block per channel, one launch per batch index (as for s1).
//   t = sum k*G, gy = t/d, gd = Re((-t)*conj(q/d)), gm = (-gy)*(1/S^2, 0)
//   grad_p = G + gm*conj(k);  grad_k += conj(G*conj(q)) + gm*conj(p) + 2*k*gd/S^2
template <int H, int W, int S>
__global__ void __launch_bounds__(kThreads, 1) static_s_backward(
    const float* __restrict__ x, const float* __restrict__ x0, const float* __restrict__ g,
    const float2* __restrict__ k, const float* __restrict__ l, float* __restrict__ gx, float* __restrict__ gx0,
    float2* __restrict__ gk, float* __restrict__ gl, float2* __restrict__ scratch_y, float2* __restrict__ scratch_p,
    int b, int C, Twiddles tw, float prediction_factor, float power_factor, float inverse_aliases) {
    extern __shared__ float2 s[];
    __shared__ float partial[kThreads / 32];
    constexpr int SH = S * H, SW = S * W, N = SH * SW, n = H * W;
    const int c = blockIdx.x, tid = threadIdx.x;
    const size_t lo = (size_t(b) * C + c) * n, hi = (size_t(b) * C + c) * N;
    float2* yp = scratch_y + size_t(c) * n;
    float2* pp = scratch_p + size_t(c) * N;
    for (int i = tid; i < n; i += kThreads) s[i] = make_float2(x[lo + i], 0.f);
    __syncthreads();
    fft2_plane<H, W>(s, tw.low_rows, tw.low_cols, tid);
    for (int i = tid; i < n; i += kThreads) yp[i] = s[i];
    __syncthreads();
    for (int i = tid; i < N; i += kThreads) s[i] = make_float2(x0[hi + i], 0.f);
    __syncthreads();
    fft2_plane<SH, SW>(s, tw.high_rows, tw.high_cols, tid);
    for (int i = tid; i < N; i += kThreads) pp[i] = s[i];
    __syncthreads();
    for (int i = tid; i < N; i += kThreads) s[i] = make_float2(g[hi + i], 0.f);
    __syncthreads();
    fft2_plane<SH, SW>(s, tw.high_rows, tw.high_cols, tid);
    const float2* kp = k + size_t(c) * N;
    float2* gkp = gk + size_t(c) * N;
    const float lc = l[c], scale = float(1.0 / double(N));
    float gl_sum = 0.f;
    for (int i = tid; i < n; i += kThreads) {
        const int h = i / W, w = i % W;
        float2 pm = make_float2(0.f, 0.f), t = make_float2(0.f, 0.f);
        float pw = 0.f;
#pragma unroll
        for (int j = 0; j < S * S; ++j) {
            const int off = (h + (j / S) * H) * SW + w + (j % S) * W;
            const float2 kk = __ldg(kp + off), z = s[off], G = make_float2(z.x * scale, z.y * scale);
            pm = add2(pm, prod(kk, pp[off]));
            pw = __fadd_rn(pw, __fadd_rn(__fmul_rn(kk.x, kk.x), __fmul_rn(kk.y, kk.y)));
            t = add2(t, prod(G, kk));
        }
        pm = make_float2(__fmul_rn(pm.x, prediction_factor), __fmul_rn(pm.y, prediction_factor));
        const float den = __fadd_rn(__fmul_rn(pw, power_factor), lc);
        const float2 y = yp[i];
        const float2 q = div_real(make_float2(__fadd_rn(y.x, -pm.x), __fadd_rn(y.y, -pm.y)), den);
        const float2 gy = div_real(t, den);
        const float v = prod(make_float2(-t.x, -t.y), conj2(div_real(q, den))).x;
        const float2 gm = prod(make_float2(-gy.x, -gy.y), make_float2(inverse_aliases, 0.f));
        const float power_scale = __fmul_rn(v, inverse_aliases);
#pragma unroll
        for (int j = 0; j < S * S; ++j) {
            const int off = (h + (j / S) * H) * SW + w + (j % S) * W;
            const float2 kk = __ldg(kp + off), z = s[off], G = make_float2(z.x * scale, z.y * scale);
            s[off] = add2(G, prod(gm, conj2(kk)));
            const float2 direct = prod(G, conj2(q)), prediction = prod(gm, conj2(pp[off]));
            const float2 power = make_float2(__fmul_rn(power_scale, __fmul_rn(2.f, kk.x)),
                                             __fmul_rn(power_scale, __fmul_rn(2.f, kk.y)));
            float2 term = add2(add2(add2(conj2(direct), prediction), make_float2(0.f, power.y)), make_float2(power.x, 0.f));
            if (b) term = add2(gkp[off], term);
            gkp[off] = term;
        }
        yp[i] = gy;
        gl_sum = __fadd_rn(gl_sum, v);
    }
    __syncthreads();
    ifft2_plane<SH, SW>(s, tw.high_rows, tw.high_cols, tid);
    for (int i = tid; i < N; i += kThreads) gx0[hi + i] = s[i].x;
    __syncthreads();
    for (int i = tid; i < n; i += kThreads) s[i] = yp[i];
    __syncthreads();
    ifft2_plane<H, W>(s, tw.low_rows, tw.low_cols, tid);
    for (int i = tid; i < n; i += kThreads) gx[lo + i] = s[i].x;
    store_block_sum(gl_sum, partial, gl + c, b);
}

// Opt-in dynamic shared memory per block on this device; static shared
// memory (the 64-byte reduction buffer) comes out of the same budget.
int smem_capacity(int device) {
    int value = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess);
    return value;
}
constexpr int kStaticSmem = kThreads / 32 * sizeof(float);

Twiddles twiddles(int h, int w, int sh, int sw, int device) {
    return {axis_for(w, device).twiddle, axis_for(h, device).twiddle, axis_for(sw, device).twiddle,
            axis_for(sh, device).twiddle};
}

template <int H, int W, int S>
void scaled_forward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& k, const at::Tensor& l, at::Tensor& out) {
    constexpr int N = S * S * H * W;
    auto kernel = static_s_forward<H, W, S>;
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, N * 8) == cudaSuccess);
    auto scratch = at::empty({x.size(0), x.size(1), H, W}, k.options());
    const float factor = float(H * W) / float(N);
    kernel<<<x.size(0) * x.size(1), kThreads, N * 8, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), x0.data_ptr<float>(), reinterpret_cast<const float2*>(k.data_ptr()), l.data_ptr<float>(),
        out.data_ptr<float>(), reinterpret_cast<float2*>(scratch.data_ptr()), x.size(1),
        twiddles(H, W, S * H, S * W, x.get_device()), factor, factor);
}

template <int H, int W, int S>
void scaled_backward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& g, const at::Tensor& k,
                     const at::Tensor& l, at::Tensor& gx, at::Tensor& gx0, at::Tensor& gk, at::Tensor& gl) {
    constexpr int N = S * S * H * W;
    auto kernel = static_s_backward<H, W, S>;
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, N * 8) == cudaSuccess);
    auto scratch_y = at::empty({x.size(1), H, W}, k.options());
    auto scratch_p = at::empty({x.size(1), S * H, S * W}, k.options());
    const float factor = float(H * W) / float(N), inverse = 1.0f / float(S * S);
    const auto tw = twiddles(H, W, S * H, S * W, x.get_device());
    for (int b = 0; b < x.size(0); ++b)
        kernel<<<x.size(1), kThreads, N * 8, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(), x0.data_ptr<float>(), g.data_ptr<float>(), reinterpret_cast<const float2*>(k.data_ptr()),
            l.data_ptr<float>(), gx.data_ptr<float>(), gx0.data_ptr<float>(), reinterpret_cast<float2*>(gk.data_ptr()),
            gl.data_ptr<float>(), reinterpret_cast<float2*>(scratch_y.data_ptr()),
            reinterpret_cast<float2*>(scratch_p.data_ptr()), b, x.size(1), tw, factor, factor, inverse);
}

// Compiled shapes: s1 by padded plane side; s2/s3 by low-res side.
#define FLASH_S1_SIZES(X) X(100) X(96) X(64)
#define FLASH_S2_SIZES(X) X(64) X(48) X(32)
#define FLASH_S3_SIZES(X) X(48) X(32) X(24)

bool compiled(int side, int scale) {
#define FLASH_HAS(n) if (side == n) return true;
    if (scale == 1) { FLASH_S1_SIZES(FLASH_HAS) }
    if (scale == 2) { FLASH_S2_SIZES(FLASH_HAS) }
    if (scale == 3) { FLASH_S3_SIZES(FLASH_HAS) }
#undef FLASH_HAS
    return false;
}

struct Table {
    Axis axis;
    at::Tensor storage;
};
std::map<std::pair<int, int>, Table> tables;

Axis axis_for(int n, int device) {
    auto& t = tables[{n, device}];
    if (t.storage.defined()) return t.axis;
    Axis a{};
    a.n = n;
    int m = n;
    for (int r : {4, 2, 3, 5})
        while (m % r == 0 && !(r == 2 && m % 4 == 0)) {
            TORCH_CHECK(a.stages < kMaxStages, "too many stages");
            a.radix[a.stages++] = r;
            m /= r;
        }
    TORCH_CHECK(m == 1, "length ", n, " is not 2/3/5-smooth");
    auto host = at::empty({n, 2}, at::kFloat);
    for (int i = 0; i < n; ++i) {
        const double angle = -2.0 * 3.14159265358979323846264338327950288 * i / n;
        host[i][0] = float(std::cos(angle));
        host[i][1] = float(std::sin(angle));
    }
    t.storage = host.to(at::Device(at::kCUDA, device));
    a.twiddle = reinterpret_cast<const float2*>(t.storage.data_ptr<float>());
    t.axis = a;
    return a;
}

}  // namespace

// x: (B, C, h0, w0) float; k: (1, C, H, W) complex spectrum of the padded size; l: (C,) float.
// variant 0: generic runtime sizes; 1: compile-time sizes; 2: compile-time + batch pairs.
at::Tensor flash_s1_forward(const at::Tensor& x, const at::Tensor& k, const at::Tensor& l, int64_t pad, int64_t variant) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 4 && x.is_contiguous());
    TORCH_CHECK(k.scalar_type() == at::kComplexFloat && k.is_contiguous() && k.size(0) == 1 && k.size(1) == x.size(1));
    TORCH_CHECK(l.scalar_type() == at::kFloat && l.is_contiguous() && l.numel() == x.size(1));
    const int C = x.size(1), h0 = x.size(2), w0 = x.size(3), H = h0 + 2 * pad, W = w0 + 2 * pad;
    TORCH_CHECK(k.size(2) == H && k.size(3) == W && H * W <= kMaxElements);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    if (variant) {
        TORCH_CHECK(variant == 1 || x.size(0) % 2 == 0, "batch pairs need an even batch");
        const auto* tr = axis_for(W, x.get_device()).twiddle;
        const auto* tc = axis_for(H, x.get_device()).twiddle;
        if (variant == 1) dispatch_static<false>(x, k, l, out, pad, H, W, tr, tc);
        else dispatch_static<true>(x, k, l, out, pad, H, W, tr, tc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    const int bytes = H * W * sizeof(float2);
    TORCH_CHECK(cudaFuncSetAttribute(fused_s1_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    fused_s1_forward<<<x.size(0) * C, kThreads, bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), reinterpret_cast<const float2*>(k.data_ptr()), l.data_ptr<float>(), out.data_ptr<float>(), C,
        h0, w0, int(pad), axis_for(W, x.get_device()), axis_for(H, x.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Returns (grad_x, grad_k, grad_l) for the s1 shared-prior solve; x, g: (B, C, h0, w0).
std::vector<at::Tensor> flash_s1_backward(const at::Tensor& x, const at::Tensor& g, const at::Tensor& k,
                                          const at::Tensor& l, int64_t pad) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 4 && x.is_contiguous());
    TORCH_CHECK(g.sizes() == x.sizes() && g.scalar_type() == at::kFloat && g.is_contiguous());
    TORCH_CHECK(k.scalar_type() == at::kComplexFloat && k.is_contiguous() && k.size(0) == 1 && k.size(1) == x.size(1));
    TORCH_CHECK(l.scalar_type() == at::kFloat && l.is_contiguous() && l.numel() == x.size(1));
    const int h0 = x.size(2), w0 = x.size(3), H = h0 + 2 * pad, W = w0 + 2 * pad;
    TORCH_CHECK(k.size(2) == H && k.size(3) == W && pad <= h0 && pad <= w0);
    c10::cuda::CUDAGuard guard(x.device());
    auto gx = at::empty_like(x);
    auto gk = at::empty_like(k);
    auto gl = at::empty({x.size(1)}, l.options());
    const auto* tr = axis_for(W, x.get_device()).twiddle;
    const auto* tc = axis_for(H, x.get_device()).twiddle;
    if (H == 100 && W == 100) launch_backward<100, 100>(x, g, k, l, gx, gk, gl, pad, tr, tc);
    else if (H == 96 && W == 96) launch_backward<96, 96>(x, g, k, l, gx, gk, gl, pad, tr, tc);
    else if (H == 64 && W == 64) launch_backward<64, 64>(x, g, k, l, gx, gk, gl, pad, tr, tc);
    else TORCH_CHECK(false, "no backward instantiation for ", H, "x", W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, gk, gl};
}

// Capacity switch: true when this shape is compiled and its plane fits the
// device's opt-in shared memory per block. Callers fall back to production.
bool flash_supported(int64_t h0, int64_t w0, int64_t scale, int64_t pad, int64_t device) {
    if (h0 != w0 || scale < 1 || scale > 3 || (scale > 1 && pad)) return false;
    const int side = scale == 1 ? int(h0 + 2 * pad) : int(h0);
    if (!compiled(side, int(scale))) return false;
    const int64_t plane = int64_t(scale * side) * (scale * side) * 8;
    return plane + kStaticSmem <= smem_capacity(int(device));
}

int64_t flash_smem_capacity(int64_t device) { return smem_capacity(int(device)); }

// x: (B, C, H, W); x0: (B, C, S*H, S*W); k: (1, C, S*H, S*W); l: (C,).
at::Tensor flash_scaled_forward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& k, const at::Tensor& l,
                                int64_t scale) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.dim() == 4 && x.is_contiguous());
    TORCH_CHECK(x0.scalar_type() == at::kFloat && x0.is_contiguous() && x0.size(0) == x.size(0) &&
                x0.size(1) == x.size(1) && x0.size(2) == scale * x.size(2) && x0.size(3) == scale * x.size(3));
    TORCH_CHECK(k.scalar_type() == at::kComplexFloat && k.is_contiguous() && k.size(0) == 1 && k.size(1) == x.size(1) &&
                k.size(2) == x0.size(2) && k.size(3) == x0.size(3));
    TORCH_CHECK(l.scalar_type() == at::kFloat && l.is_contiguous() && l.numel() == x.size(1));
    TORCH_CHECK(flash_supported(x.size(2), x.size(3), scale, 0, x.get_device()), "shape not supported on this device");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x0);
    const int side = x.size(2);
#define FLASH_FWD2(n) if (scale == 2 && side == n) scaled_forward<n, n, 2>(x, x0, k, l, out);
#define FLASH_FWD3(n) if (scale == 3 && side == n) scaled_forward<n, n, 3>(x, x0, k, l, out);
    FLASH_S2_SIZES(FLASH_FWD2)
    FLASH_S3_SIZES(FLASH_FWD3)
#undef FLASH_FWD2
#undef FLASH_FWD3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<at::Tensor> flash_scaled_backward(const at::Tensor& x, const at::Tensor& x0, const at::Tensor& g,
                                              const at::Tensor& k, const at::Tensor& l, int64_t scale) {
    TORCH_CHECK(g.sizes() == x0.sizes() && g.scalar_type() == at::kFloat && g.is_contiguous());
    TORCH_CHECK(flash_supported(x.size(2), x.size(3), scale, 0, x.get_device()), "shape not supported on this device");
    c10::cuda::CUDAGuard guard(x.device());
    auto gx = at::empty_like(x);
    auto gx0 = at::empty_like(x0);
    auto gk = at::empty_like(k);
    auto gl = at::empty({x.size(1)}, l.options());
    const int side = x.size(2);
#define FLASH_BWD2(n) if (scale == 2 && side == n) scaled_backward<n, n, 2>(x, x0, g, k, l, gx, gx0, gk, gl);
#define FLASH_BWD3(n) if (scale == 3 && side == n) scaled_backward<n, n, 3>(x, x0, g, k, l, gx, gx0, gk, gl);
    FLASH_S2_SIZES(FLASH_BWD2)
    FLASH_S3_SIZES(FLASH_BWD3)
#undef FLASH_BWD2
#undef FLASH_BWD3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gx, gx0, gk, gl};
}
