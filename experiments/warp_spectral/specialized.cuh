#pragma once

#include <cuda/barrier>
#include <cuda_runtime.h>

// Experimental s=2, complex64, real-FFT interior solver. Include this AFTER
// converse2d_kernels.cu so its conjugate/squared_norm/read_frequency helpers
// are available. Launch exactly 64 threads: one synchronous-load warp and one
// compute/store warp. This is deliberately not a cp.async or TMA experiment.
//
// Group g maps to bc = g / (H*interior), h = (g/interior)%H,
// w = 1 + g%interior, interior = (W-1)/2. Its four HR aliases cover four
// distinct stored HR bins; aliases outside the stored half are conjugated and
// mapped back to their physical bins. The caller handles LR boundary columns.
//
// Two slots hold 2 * 32 * (8 complex inputs + 1 complex fy + 1 float lambda)
// = 4864 bytes, plus four sizeof(cuda::barrier<thread_scope_block>) objects.
// Per valid group: 76 global input bytes, 32 global output bytes, and 152 shared
// bytes (76 written + 76 read), before cache reuse, barriers and profile stores.
// A full 32-group tile therefore moves 3456 logical global bytes and 4864 shared
// bytes. Arithmetic is the same four-alias solve as the fused baseline.
//
// One-sided producer/consumer protocol: all four barriers expect 64 arrivals.
// Consumers initially arrive at both ready barriers, making both slots free.
// Producer: ready[slot].arrive_and_wait(), load, filled[slot].arrive().
// Consumer: filled[slot].arrive_and_wait(), compute, ready[slot].arrive().
// Each arrival is by all 32 threads of that role, completing 64 arrivals when
// the other role arrives. Barrier phase completion supplies shared-memory
// visibility and prevents overwrite until the prior consumer has finished.
// Even lanes with g >= groups take every synchronization operation.
//
// PROFILE timestamps use [role=2, event=8, iteration=64] uint32 layout, with
// exactly lane 0 writing, and must be interpreted only in a ONE-CTA launch:
//   role 0 (loader):   0/1 empty wait enter/exit; 2/3 load work begin/end;
//                     4 filled arrivals completed (publication marker).
//   role 1 (consumer): 0/1 filled wait enter/exit; 2/3 compute/store begin/end;
//                     5 ready arrivals completed (slot-return marker).
//   Remaining events are unused; initialize the timestamp buffer to zero.
// Dependencies: loader event 4[i] -> consumer event 1[i], and consumer event
// 5[i] -> loader event 1[i+2]. Post-arrival stamps can follow a wait exit by a
// few cycles: the actual arrival instruction precedes the marker store.
// PROFILE=false eliminates timestamp code; normal launches use
// ceil(groups / (32*iterations)) CTAs. PROFILE=true uses one CTA and
// groups <= 32*iterations, iterations <= 64. iterations must always be positive.

namespace warp_spectral_detail {

template <bool PROFILE>
__device__ __forceinline__ void stamp(unsigned int* stamps, int role,
                                      int event, int iteration) {
    if constexpr (PROFILE) {
        if (stamps && blockIdx.x == 0 && (threadIdx.x & 31) == 0 && iteration < 64)
            stamps[(role * 8 + event) * 64 + iteration] =
                static_cast<unsigned int>(clock());
    }
}

__device__ __forceinline__ float2 pack(c10::complex<float> value) {
    return make_float2(value.real(), value.imag());
}

__device__ __forceinline__ c10::complex<float> unpack(float2 value) {
    return {value.x, value.y};
}

}  // namespace warp_spectral_detail

template <bool PROFILE>
__global__ void ws_interior(const c10::complex<float>* fy,
                            const c10::complex<float>* prior,
                            const c10::complex<float>* fb,
                            const float* lambda,
                            c10::complex<float>* out,
                            int groups, int C, int H, int W, int KB, int KC,
                            int iterations, unsigned int* stamps) {
    using z = c10::complex<float>;
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    using namespace warp_spectral_detail;

    __shared__ float2 tile_prior[2][4][32];
    __shared__ float2 tile_filter[2][4][32];
    __shared__ float2 tile_fy[2][32];
    __shared__ float tile_lambda[2][32];
    __shared__ barrier filled[2];
    __shared__ barrier ready[2];

    if (threadIdx.x < 2) {
        init(&filled[threadIdx.x], 64);
        init(&ready[threadIdx.x], 64);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int role = threadIdx.x >> 5;
    const int interior = (W - 1) / 2;
    const int high_h = 2 * H;
    const int high_w = 2 * W;
    constexpr unsigned int all_lanes = 0xffffffffu;

    if (role == 1) {
        (void)ready[0].arrive();
        (void)ready[1].arrive();
    }

    for (int it = 0; it < iterations; ++it) {
        const int slot = it & 1;
        const int group = (static_cast<int>(blockIdx.x) * iterations + it) * 32 + lane;
        const bool valid = group < groups;

        if (role == 0) {
            stamp<PROFILE>(stamps, 0, 0, it);
            ready[slot].arrive_and_wait();
            stamp<PROFILE>(stamps, 0, 1, it);
            stamp<PROFILE>(stamps, 0, 2, it);
            if (valid) {
                const int w = 1 + group % interior;
                const int h = (group / interior) % H;
                const int bc = group / (interior * H);
                const int c = bc % C;
                const int kc = (KB == 1 ? 0 : bc / C) * KC + (KC == 1 ? 0 : c);
                tile_fy[slot][lane] = pack(fy[(bc * H + h) * (W / 2 + 1) + w]);
                tile_lambda[slot][lane] = lambda[c];
                #pragma unroll
                for (int alias = 0; alias < 4; ++alias) {
                    const int hh = h + (alias / 2) * H;
                    const int ww = w + (alias % 2) * W;
                    tile_prior[slot][alias][lane] = pack(
                        read_frequency<float, true>(prior, bc, hh, ww, high_h, high_w));
                    tile_filter[slot][alias][lane] = pack(
                        read_frequency<float, true>(fb, kc, hh, ww, high_h, high_w));
                }
            }
            __syncwarp(all_lanes);
            stamp<PROFILE>(stamps, 0, 3, it);
            (void)filled[slot].arrive();
            __syncwarp(all_lanes);
            stamp<PROFILE>(stamps, 0, 4, it);
        } else {
            stamp<PROFILE>(stamps, 1, 0, it);
            filled[slot].arrive_and_wait();
            stamp<PROFILE>(stamps, 1, 1, it);
            stamp<PROFILE>(stamps, 1, 2, it);
            if (valid) {
                z values[4], filters[4];
                z sum(0.0f, 0.0f);
                float power_sum = 0.0f;
                #pragma unroll
                for (int alias = 0; alias < 4; ++alias) {
                    values[alias] = unpack(tile_prior[slot][alias][lane]);
                    filters[alias] = unpack(tile_filter[slot][alias][lane]);
                    sum += filters[alias] * values[alias];
                    power_sum += squared_norm<float>(filters[alias]);
                }
                const z q = (unpack(tile_fy[slot][lane]) - sum / 4.0f) /
                            (power_sum / 4.0f + tile_lambda[slot][lane]);
                const int w = 1 + group % interior;
                const int h = (group / interior) % H;
                const int bc = group / (interior * H);
                #pragma unroll
                for (int alias = 0; alias < 4; ++alias) {
                    int hh = h + (alias / 2) * H;
                    int ww = w + (alias % 2) * W;
                    z result = values[alias] + conjugate<float>(filters[alias]) * q;
                    if (ww > W) {
                        hh = (high_h - hh) % high_h;
                        ww = high_w - ww;
                        result = conjugate<float>(result);
                    }
                    out[(bc * high_h + hh) * (W + 1) + ww] = result;
                }
            }
            __syncwarp(all_lanes);
            stamp<PROFILE>(stamps, 1, 3, it);
            (void)ready[slot].arrive();
            __syncwarp(all_lanes);
            stamp<PROFILE>(stamps, 1, 5, it);
        }
    }
}
