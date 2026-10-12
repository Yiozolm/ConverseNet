# Flash FFT on RTX 5060 Ti / sm_120: 768-thread forward, persistent blocks, two-CTA clusters (v4)

2026-10-12, follows `profile/flash-fft-v3-final-20261012` (v3, commit `b0b4db9`). Same device,
Nsight Compute 2026.3.1, uncontrolled clocks and caches, 8-16% foreign GPU load (compare ratios).

## 1. Forward at 768 threads (`profile/flash-fft-v3d-20261012`)

The fused passes stage 50 registers of data and were compiled for 512 and 768 threads; v3 ran the
forward at 1024 threads with single stages (277 us) because the 512-thread fused forward tied it
(282 us). 768 threads give 24 warps per SM at an 80-register budget, and the fused forward needs
67 registers there, no spills.

| forward, B4 C128 100x100 | 1024 single stages (v3) | 768 fused |
|---|---:|---:|
| kernel time (us) | 277.1 | **253.5** |
| shared wavefronts (k) | 11988 | 9590 |
| instructions (M) | 59.0 | 53.3 |
| warps active % | 65.3 | 49.7 |
| stall mio throttle / barrier / wait | 3.15 / 1.67 / 1.73 | 1.23 / 1.81 / 1.68 |
| registers | 56 | 67 |

The backward at 768 fused was 585 us against 574 us at 512 and stays at 512. Bitwise identical
to `6dd4bbb` (compare_commit, 8/8 single-SM cases). New default: forward 768, backward 512.

## 2. Persistent blocks, reverted (`profile/flash-fft-v3c-20261012`)

One block per SM walking the planes (grid = 36) and prefetching the next plane's x, g and k to
L2 during the current plane. Bitwise identical, 2-5x slower:

| | fwd 1024 v3 | fwd 1024 persistent | bwd 512 v3 | bwd 512 persistent |
|---|---:|---:|---:|---:|
| kernel time (us) | 277.1 | 557.8 | 574.3 | 2733.4 |
| local (spill) load instructions | 0 | 2139136 | 0 | 2737152 |
| DRAM write (MB) | 15.1 | 103.4 | 55.9 | 415.6 |
| stall long scoreboard | 1.26 | 5.63 | 0.40 | 10.50 |

ptxas (`.build/logs/flash-fft-v3c-build.log`): 596-1520 B of spill stores per thread in every s1
instantiation once the kernel body became a plane loop. The loop-carried state (pointers, the
prefetch addresses, the per-plane scalars) landed in the FFT's register budget. Reverted; the
prefetch idea would need the loop state in shared memory or a warp-specialized producer.

## 3. Two-CTA clusters for the s2/s3 planes (`cluster_s_forward`, `cluster_s_backward`)

Design: see the README section. Correctness evidence (`cluster_check_rtx5060ti_001.json`):

- On the four s2/s3 planes that fit one SM (s2 48, s2 32, s3 32, s3 24), cluster mode 1 against
  mode 2: output, grad_x, grad_x0, grad_weight and grad_bias **bitwise identical** to the
  single-CTA kernels. Every FFT line, every alias group and even the two-stage grad_l reduction
  round the same way.
- s2 B4 C64 64 -> 128 (128 KB plane) and s3 B2 C32 48 -> 144 (162 KB), clusters against production
  and the FP64 reference: every output and gradient passes (rel-L2 ratios 0.69-0.99, max-abs up to
  1.38x on the s3 grad_weight), bitwise repeatable.

Timing (`flash_rtx5060ti_012_cluster`, forward+VJP against production): s2 64 -> 128 **1.83x**
(3318 / 1814 us; production before, the A100's single-CTA fused kernel reaches 1.05x there);
s3 48 -> 144 0.93x (678 / 732 us). The s3 case has 64 planes = 128 CTAs on 36 SMs and pays the
exchanges, the duplicated low-res transform and the rank-0-only low-res inverse against a 678 us
production time. NCU of the s2 cluster kernels: `analysis/table.md`.

| s2 cluster kernels, B4 C64 64 -> 128, one call | forward | backward |
|---|---:|---:|
| kernel time (us) | 497.6 | 934.7 |
| grid (CTAs) / dynamic smem | 512 / 64 KB | 512 / 64 KB |
| DRAM read / write (MB) | 44.4 / 24.3 | 70.9 / 62.4 |
| L2 read hit % | 76.4 | 74.8 |
| shared wavefronts (k) | 15322 | 23140 |
| instructions (M) | 33.6 | 90.8 |
| warps active %; eligible / cycle | 33.1; 0.38 | 33.5; 0.63 |
| stall long scoreboard / mio throttle / barrier | 7.46 / 4.66 / 1.92 | 4.91 / 2.69 / 1.18 |
| warp latency / instruction | 23.1 | 15.9 |
| registers | 128 | 128 |

The cluster kernels stand where the s1 kernels stood before v2: latency-bound at 16 warps per SM,
with the DSMEM and global loads (long scoreboard) and the single-stage 128-point FFTs (mio throttle)
as the stalls. The s1 steps apply: batched loads, per-stage shared twiddles, fused stages (the
128-point axis is (4,4) (4,2) at 512 threads), a 768-thread forward.

## 4. Final configuration

Forward 768 fused / backward 512 fused / cluster mode 0 (`flash_rtx5060ti_013_f768`,
`ops_rtx5060ti_007_f768`), bitwise identical to `6dd4bbb` on the 8 single-SM cases, every budget passed:

| case | production us | flash us | speedup | v3 (`_010`) | baseline (`_006` / `_004`) |
|---|---:|---:|---:|---:|---:|
| circular s1 B4 C64 96 pad 2 | 1299 | 693 | 1.87x | 1.99x | 1.48x |
| circular s1 B4 C128 96 pad 2 | 3776 | 1227 | **3.08x** | 3.04x | 2.18x |
| s1 B4 C128 100 | 3606 | 1248 | 2.89x | 2.90x | 2.06x |
| s1 B4 C64 96 | 1361 | 727 | 1.87x | 1.90x | 1.48x |
| data term s1 C64 96, KB 4, k7 | 1590 | 1000 | 1.59x | 1.54x | 1.27x |
| replicate 92 pad 4 / reflect / zeros | 1616 / 1590 / 1539 | 734 / 683 / 687 | 2.20x / 2.33x / 2.24x | 2.33 / 2.39 / 2.25x | 1.69-1.73x |
| ops circular C64 / C128 | 1376 / 3807 | 699 / 1254 | 1.97x / 3.04x | 1.99x / 3.00x | 1.46x / 2.19x |
| s2 B4 C64 64 -> 128 (cluster) | 3318 | 1814 | **1.83x** | production | production |
| s3 B2 C32 48 -> 144 (cluster) | 678 | 732 | 0.93x | production | production |

Kernel level, B4 C128 100x100, against this morning's baseline: forward 413 -> 254 us (1.63x),
backward 865 -> 574 us (1.51x).

## 5. The s1 steps on the cluster kernels (v4b)

Batched half-plane loads (`load_half_rows`), per-stage shared twiddles and fused stages on the
high-res passes (`cluster_fft2<..., Shared = true>`), channel-major planes, and the low-res
transform and inverse split between the ranks (`cluster_low_spectrum`; rank r ends with the low-res
columns of parity r, the pixels it solves, so the yp park and read-back stay within one CTA and the
backward's low-res inverse no longer runs on rank 0 alone). `cluster_check_rtx5060ti_002.json`:
bitwise identical to the single-CTA kernels on the four single-SM planes; the large planes pass
every budget with the first version's ratios.

`analysis/table_v4b.md`, s2 B4 C64 64 -> 128:

| metric | fwd v4 | fwd v4b | bwd v4 | bwd v4b |
|---|---:|---:|---:|---:|
| kernel time (us) | 497.6 | **399.9** | 934.7 | **739.8** |
| DRAM read (MB) | 44.4 | 30.5 | 70.9 | 49.2 |
| L2 read hit % | 76.4 | 88.5 | 74.8 | 82.2 |
| shared wavefronts (k) | 15322 | 13232 | 23140 | 20570 |
| instructions (M) | 33.6 | 27.9 | 90.8 | 82.4 |
| stall long scoreboard | 7.46 | 3.01 | 4.91 | 1.48 |
| stall mio throttle | 4.66 | 6.00 | 2.69 | 2.99 |
| stall barrier | 1.92 | 2.01 | 1.18 | 1.13 |
| warp latency / instruction | 23.1 | 22.1 | 15.9 | 13.8 |
| registers | 128 | 94 | 128 | 127 |

`analysis/table_v4b_s3.md`, s3 B2 C32 48 -> 144 (128 CTAs on 36 SMs): forward 147 us, backward
288 us; mio throttle 5.0 / 2.8, barrier 3.8 / 2.4, long scoreboard 1.7 / 1.1. Operator level
(`flash_rtx5060ti_014_cluster_v4b`): s2 64 -> 128 2.33x (3937 / 1689 us), s3 48 -> 144 1.07x
(774 / 727 us), where the two kernels are 435 us of the 727 and the per-call kernel preparation
the rest.

## Roadmap after v4

- Cluster kernels: a 768-thread forward (the fused passes need 80 registers), conflict-free
  column-layout stores, overlapping the DSMEM exchange with the last row stage.
- s1: the backward's two-term pointwise phase (half its instructions) and the second Stockham
  stage's store conflicts; a warp-specialized producer for the next plane's loads without the
  persistent loop's register cost.
- Whole model: the fused kernels were 17% of the USRNet step this morning and are now about 11%;
  pointwise wgrad (50%) and LayerNorm (16%) are the larger items.

## Files

- `profile/flash-fft-v3c-20261012`, `profile/flash-fft-v3d-20261012`: NCU of the persistent and the
  768-thread builds (`analysis/table.md`).
- this directory: NCU of the s2 cluster kernels (`reports/cluster_v4.ncu-repz`).
- `.build/logs/flash-v4-*-build.log`, `flash-fft-v3c-build.log`: ptxas `-v`;
  `.build/logs/flash-v4-*compare*.log`, `flash-v4-cluster-check.log`: the bitwise checks.
