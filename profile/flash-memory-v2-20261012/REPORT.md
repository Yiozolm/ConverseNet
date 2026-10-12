# Flash FFT s1 kernels on RTX 5060 Ti / sm_120: memory-movement revision

2026-10-12, follows `profile/flash-bottleneck-20261012` (HEAD `6dd4bbb`, partial compensation).
Same device, same workload (B4 C128 96x96 circular pad 2, 100x100 planes), same Nsight Compute
2026.3.1, uncontrolled clocks and caches as in the baseline. The GPU carried 10-16% utilization
from another process during every measurement here (`nvidia-smi` samples before/after each run);
the baseline's load is unknown, so DRAM byte counts are good to a few MB and timings are compared
as ratios.

## What the baseline said and what was changed

The baseline report found the two fused kernels latency-bound at 33% occupancy: 38-39% of stall
samples were long-scoreboard waits at the pad/embed loads and the k loads, DRAM at 27-36% of peak,
one 80 KB block of 16 warps per SM. Reading the baseline's per-line counters and the kernel source
gave four concrete memory causes:

1. **One global load in flight per warp.** The pad-load loops (`load x`, `load g`) issue a load,
   store it to shared memory and only then issue the next. With 16 warps per SM the whole SM had
   16 outstanding 128 B requests against ~600+ cycles of DRAM latency.
2. **k re-read from DRAM once per batch (backward).** Blocks ran batch-major (`blockIdx = b*C + c`);
   between two uses of a channel's 80 KB spectrum, 128 blocks moved ~270 KB each through L2 (x, g,
   k, parked Y, grad_x, grad_k term), 35 MB against a 32 MB L2. The backward read 84 MB from DRAM
   where its inputs sum to 48 MB.
3. **Twiddle gathers were the largest L1TEX stream.** `__ldg(tw + t*r)` in every butterfly:
   57% of all global sectors in the forward (3.57 M of 6.26 M), 5.5 sectors per request, L1 hits
   that still queue behind the global path and count as long-scoreboard waits.
4. **k (and parked Y) loads in the pointwise phase** were consumed by the next instruction; after
   a barrier all 16 warps issued them together and waited together.

`tools/v4_flash_fft/fused.cu` now:

- batches every thread's pad/embed loads into registers before the first shared store
  (`load_padded`, `load_embedded`; 10-20 loads in flight per thread);
- orders the grid channel-major (`blockIdx = c*B + b`) in both kernels so one channel's k is read
  from DRAM once and hit in L2 by the other batches; partial slots stay `(b, c)`, so the
  reduction order is unchanged;
- copies the two twiddle tables (1.6 KB) into shared memory per block (`stage_twiddles`,
  `twiddle_at<TwShared>`); s2/s3 and the half-spectrum kernels keep the global table;
- loads k and Y one step ahead in the pointwise loops; issues `prefetch.global.L2` for the g
  plane and k at block start (backward) and for k (forward);
- makes the block size a template parameter (512 or 1024 threads, `set_s1_threads(forward,
  backward)`); 1024 threads give 32 resident warps per SM at a 64-register budget.

Every value is still loaded, rounded and stored exactly as before: `compare_commit.py --commit
HEAD` reports 8/8 eligible cases bitwise identical (output and all gradients) at 512 threads, and
the forward at 1024 threads performs the same per-element arithmetic (see the validation section).

## NCU, one call each, B4 C128 100x100 (`analysis/table.md`, `analysis/metrics.json`)

| metric | fwd baseline | fwd 512 | fwd 1024 | bwd baseline | bwd 512 | bwd 1024 |
|---|---:|---:|---:|---:|---:|---:|
| kernel time (us) | 412.6 | 306.9 | **282.0** | 864.8 | **626.2** | 648.6 |
| DRAM read (MB) | 31.9 | 39.2 | 44.7 | 83.8 | **54.1** | 56.5 |
| DRAM write (MB) | 17.7 | 18.4 | 18.4 | 57.2 | 59.3 | 74.0 |
| L2 read hit % | 44.5 | 58.5 | 54.4 | 39.1 | **78.5** | 83.7 |
| global load sectors (k) | 6261 | **2184** | 2192 | 10329 | **4201** | 4210 |
| global load requests (k) | 1168 | 333 | 341 | 1903 | 647 | 655 |
| shared wavefronts (k) | 10260 | 12269 | 12564 | 15511 | 18757 | 19788 |
| shared ld bank conflicts (k) | 548 | 1268 | 1280 | 790 | 1866 | 1900 |
| instructions (M) | 55.6 | 55.6 | 59.1 | 147.5 | 144.3 | 149.5 |
| local (spill) load instructions | 0 | 0 | 0 | 0 | 0 | 359936 |
| registers / thread | 125 | 128 | 64 | 128 | 128 | 64 |
| warps active % | 33.2 | 33.5 | **66.1** | 33.4 | 33.1 | 66.7 |
| eligible warps / scheduler / cycle | 0.68 | 0.96 | 2.14 | 0.92 | 1.56 | 3.12 |
| issue active % | 35.0 | 48.0 | 56.6 | 44.8 | 60.0 | 61.9 |
| SM throughput % peak | 33.2 | 45.1 | 53.2 | 42.2 | 57.4 | 59.6 |
| DRAM % peak | 26.9 | 41.9 | 50.0 | 36.4 | 40.5 | 45.0 |
| stall: long scoreboard (cycles/inst) | 4.41 | **1.29** | 1.41 | 3.52 | **0.36** | 0.64 |
| stall: mio throttle | 1.18 | 1.55 | 2.91 | 0.53 | 0.65 | 1.79 |
| stall: wait | 1.61 | 1.56 | 1.72 | 1.13 | 1.02 | 1.47 |
| stall: barrier | 0.80 | 0.65 | 1.86 | 0.54 | 0.42 | 1.27 |
| stall: short scoreboard | 0.70 | 0.62 | 1.23 | 0.62 | 0.88 | 1.27 |
| warp latency per issued instruction | 11.4 | 8.3 | 14.1 | 8.9 | 6.7 | 12.9 |

Reading:

- **The memory diagnosis held.** The backward's DRAM reads fell by the predicted 30 MB (k now
  read once per channel), its L2 read hit rate doubled, and the long-scoreboard stall per
  instruction fell 10x (3.52 -> 0.36). Global load sectors fell 2.9x / 2.5x once the twiddles
  left the global path.
- **Forward 1024 threads is the right block size, backward 512.** The forward fits 64 registers
  without spilling and gains another 8% from 32 resident warps. The two-term backward at 64
  registers spills (360 k local loads; the 74 MB DRAM write is the spill traffic) and loses 4%
  against 512. The defaults are therefore forward 1024 / backward 512; the forward at 1024 still
  performs the same per-element arithmetic as at 512, only the backward's grad_l block reduction
  order would change with the block size, and it stays at 512.
- **What is left is the shared-memory FFT, not DRAM.** At 1024 threads the forward's stalls are
  mio throttle (2.9), barrier (1.9) and wait (1.7); long scoreboard is 1.4. The shared twiddle
  gathers raised shared load bank conflicts 2.3x (548 k -> 1268 k; each late Stockham stage reads
  32 distinct twiddles per warp with a stride that collides). The Stockham store pattern already
  had 2.5-way conflicts. These are the next targets; see the roadmap below.
- DRAM reads of the forward rose 32 -> 39/45 MB although its kernel reads only 29 MB of inputs.
  The forward's k was already L2-resident in batch-major order (its L2 footprint between reuses
  is 20 MB), so the channel-major order and the k prefetch bought it nothing; four adjacent blocks
  prefetching the same lines, or the other process's traffic, account for the extra. It is not on
  the critical path (DRAM 42-50% of peak) and was not ablated separately.

## Operator timing (`artifacts/v4_flash_fft/flash_rtx5060ti_007_memory*`)

`flash_study.py`, forward+VJP against production, paired CUDA events, 6 rounds x 20 iterations,
10-16% foreign GPU load; the `_006_partial` column is this morning's baseline under 14% load.

| case | production us | flash 512/512 | flash 1024/1024 | `_006` speedup | 512/512 | 1024/1024 |
|---|---:|---:|---:|---:|---:|---:|
| circular s1 B4 C64 96 pad 2 | 1475 / 1481 | 855 | 828 | 1.48x | 1.73x | 1.79x |
| circular s1 B4 C128 96 pad 2 | 4080 / 4109 | 1533 | 1499 | 2.18x | 2.66x | 2.74x |
| s1 B4 C128 100 | 3909 / 3914 | 1533 | 1464 | 2.06x | 2.55x | 2.67x |
| s1 B4 C64 96 | 1439 / 1450 | 874 | 860 | 1.48x | 1.65x | 1.69x |

Every output, grad_x, grad_weight and grad_bias passes its budget at a 0.99-1.00x rel-L2 ratio
against production and is bitwise repeatable, at both block sizes. The 1024/1024 column is a
little faster end to end than NCU's kernel times suggest because the 96x96 backward spills less
at 1024 than at 512 (112 B against 208 B, ptxas) while the 100x100 backward spills 56 B at 1024
and none at 512.

The mixed default (forward 1024, backward 512; `flash_rtx5060ti_007_memory`,
`ops_rtx5060ti_005_memory`), bitwise identical to HEAD on the 8 eligible cases:

| case | production us | flash us | speedup | `_006` / `_004` baseline |
|---|---:|---:|---:|---:|
| circular s1 B4 C64 96 pad 2 | 1337 | 720 | **1.86x** | 1.48x |
| circular s1 B4 C128 96 pad 2 | 3706 | 1297 | **2.86x** | 2.18x |
| s1 B4 C128 100 | 3884 | 1484 | 2.62x | 2.06x |
| s1 B4 C64 96 | 1430 | 875 | 1.63x | 1.48x |
| data term s1 C64 96, KB 4, k7 | 1469 | 1028 | 1.43x | 1.27x |
| replicate s1 92 pad 4, k5 | 1507 | 728 | 2.07x | 1.70x |
| reflect s1 96 pad 2 | 1733 | 835 | 2.08x | 1.73x |
| zeros s1 96 pad 2 | 1646 | 788 | 2.09x | 1.69x |
| ops circular C64 / C128 | 1477 / 4063 | 811 / 1474 | 1.82x / 2.76x | 1.46x / 2.19x |

Every output, grad_x, grad_weight and grad_bias passes its budget (worst rel-L2 ratio 1.00) and
repeats bitwise.

## Whole-model meaning

In the baseline's USRNet training step the two fused kernels were 16.95% of GPU kernel time
(forward 4.77%, backward 12.18%). With the kernel-level ratios measured here (forward 1.46x,
backward 1.38x) the Amdahl estimate for the step is about 1.05x; the pointwise convolutions
(50%) and LayerNorm (16%) remain the larger items for throughput. This revision is about the s1
kernels' own ceiling on sm_120, not about the step.

## Roadmap in FlashAttention terms

- **v1** (October 6-8): the SRAM-resident plane. One block per (b, c) plane, the whole complex
  spectrum in 80 KB of shared memory, recomputation instead of saved spectra.
- **v2** (this revision): memory movement and work partitioning. Batched loads, L2-resident k by
  grid order, twiddles in shared memory, software prefetch of k/Y, L2 prefetch of g, 32 warps per
  SM for the forward. Bitwise identical arithmetic; 1.34-1.46x forward, 1.38x backward on the
  kernels; 2.55-2.74x operator-level against production at C128.
- **v3** (next): the shared-memory FFT itself. The stalls are now mio throttle, barrier and wait.
  Candidates, each an independent ablation with the bitwise/budget gate: a bank-conflict-free
  twiddle layout (per-stage tables, or (r=1..R-1) packed so one LDS.128 serves a butterfly) and
  padded/swizzled Stockham stores; register-resident radix-25/radix-16 passes so each axis takes
  two shared round trips and two barriers instead of three; getting the backward's pointwise
  phase under 64 registers (two passes over the plane, or two-term partials in the free 19 KB of
  shared memory) so the backward can also run 32 warps per SM.
- **v4** (Blackwell-specific): thread-block clusters with distributed shared memory for the s2
  (128x128, 128 KB) and s3 (144x144, 162 KB) planes that do not fit one SM's 99 KB; TMA bulk
  copies with an mbarrier so the g plane streams in behind FFT(x) instead of being prefetched to
  L2; neither has a measured benefit yet on this workload.

## Files

- `harness/kernel_worker.py`, `harness/launch_ncu.py`, `analysis/command.json`: the NCU run (4
  kernels, 14 passes each, explicit metric list; `reports/memory_v2.ncu-repz`, `reports/ncu.log`).
- `analysis/analyze.py` -> `analysis/metrics.json`, `analysis/table.md`: the table above against
  the baseline's `metrics_all_static_s1_*.json`.
- `.build/logs/flash-memory-v2-build.log`: ptxas `-v` for every instantiation;
  `.build/logs/flash-memory-v2-compare-head*.log`: the bitwise comparisons against HEAD.
