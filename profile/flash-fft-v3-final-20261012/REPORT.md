# Flash FFT s1 kernels on RTX 5060 Ti / sm_120: the shared-memory FFT (v3)

2026-10-12, follows `profile/flash-memory-v2-20261012` (v2: memory movement, commit `31b3c88`).
Same device, workload (B4 C128 96x96 circular pad 2, 100x100 planes), Nsight Compute 2026.3.1,
uncontrolled clocks and caches, 10-16% foreign GPU load throughout (compare ratios).

After v2 the forward's stalls were mio throttle (2.9 cycles per issued instruction), barrier (1.9)
and wait (1.7): the Stockham FFT's shared-memory traffic and its two barriers per stage. Three
changes were tried in sequence, each gated bitwise against commit `6dd4bbb` with
`compare_commit.py` (8/8 eligible cases identical every time):

1. **Per-stage twiddle layout in shared memory** (`profile/flash-fft-v3-20261012`). v2 copied the
   plain table of n roots and indexed it `t r`; a warp's 32 consecutive butterflies then read 32
   words at stride `span r` and conflicted. The copy is now laid out `(r - 1) NS + (j % NS)` per
   stage (`fill_twiddles`, `staged_twiddles`), so consecutive butterflies read consecutive words
   or broadcast. Same values, bitwise identical. Also lowered register pressure (forward 1024:
   64 -> 56 registers; backward 1024 spill 56 -> 8 B).
2. **Fused two-stage register pass** (`static_stage2`, `profile/flash-fft-v3b-20261012`). Two
   consecutive Stockham stages (radix R1 at NS1, R2 at NS1 R1) run in one pass: the R2 stage-1
   butterflies feeding the same R1 stage-2 butterflies form a closed group of R1 R2 elements
   (needs NS1 | N / (R1 R2), true for every compiled size), one thread per group. Each axis makes
   two shared round trips and four barriers instead of three and six (100: (4,5) 5; 96: (4,4)
   (2,3); 64: (4,4) 4). Same twiddles and `dft` calls in the same order: bitwise identical. The
   pass stages up to 24 complex values per thread, so it is compiled for 512 and 768 threads; the
   1024-thread kernels keep single stages.
3. **Padded line stride** (W + 1) for the fused passes, tried and rejected: see the ablation.

## NCU, one call each, B4 C128 100x100

Forward (1024 threads single stages unless noted) and backward (512 threads):

| metric | fwd v2 1024 | fwd v3a 1024 | fwd v3b 512 fused | bwd v2 512 | bwd v3a 512 | bwd v3b 512 fused |
|---|---:|---:|---:|---:|---:|---:|
| kernel time (us) | 282.0 | 277.3 | 285.5 | 626.2 | 613.5 | **578.3** |
| shared wavefronts (k) | 12564 | 11988 | **9922** | 18757 | 17798 | **15189** |
| shared ld bank conflicts (k) | 1280 | - | 1041 | 1900 | - | 1528 |
| instructions (M) | 59.07 | 59.04 | 54.79 | 144.34 | 141.50 | 142.59 |
| warps active % | 66.1 | 63.6 | 32.5 | 33.1 | 33.4 | 36.7 |
| stall mio throttle | 2.91 | 3.15 | **0.91** | 0.65 | 0.70 | 0.59 |
| stall barrier | 1.86 | 1.67 | **0.44** | 0.42 | 0.46 | 0.36 |
| stall long scoreboard | 1.41 | 1.13 | 0.96 | 0.36 | 0.40 | 0.93 |
| stall wait | 1.72 | 1.73 | 1.65 | 1.02 | 1.01 | 1.01 |
| warp latency / instruction | 14.05 | 14.05 | 7.81 | 6.66 | 6.58 | 6.17 |
| registers / thread | 64 | 56 | 64 | 128 | 125 | 119 |

(v3b columns are the padded-stride build; the ablation below shows pad 0 does the same work with
fewer wavefronts.) Backward block sizes 512 / 768 / 1024 in v3a: 613.5 / 620.6 / 617.8 us, so
the backward stays at 512.

Reading:

- The twiddle layout alone was worth 2% on each kernel.
- The fused pass removed a third of the FFT's shared round trips and barriers: the backward's
  shared wavefronts fell 15% and its time 6%; its mio-throttle and barrier stalls are now below
  0.6 cycles per instruction and the kernel issues an instruction every 6.2 cycles per warp. The
  forward at 512 threads with fused stages (285 us, 16 warps per SM, warp latency 7.8) ties the
  single-stage forward at 1024 threads (277 us, 32 warps, warp latency 14.1): fewer warps that
  stall less against more warps that stall more. The defaults stay forward 1024 / backward 512.
- What limits the kernels now is the remaining `wait` (fixed-latency dependency) stall of about
  1.0-1.7 cycles per instruction, the second-stage Stockham store conflicts, and the instruction
  count itself: the backward's two-term pointwise phase is 142 M instructions for a 10000-element
  plane times 512 blocks.

### Padding ablation (`profile/flash-fft-v3b-20261012/analysis/pad_ablation.md`)

Fused kernels, 512 threads, line stride W + 1 against W:

| metric | fwd pad 1 | fwd pad 0 | bwd pad 1 | bwd pad 0 |
|---|---:|---:|---:|---:|
| time us | 285.5 | 282.0 | 578.3 | 988.2 (*) |
| shared wavefronts (k) | 9922 | 9512 | 15189 | 14563 |
| shared st bank conflicts (k) | 1195 | 723 | 1766 | 1053 |
| shared ld bank conflicts (k) | 1041 | 1113 | 1528 | 1630 |
| instructions (M) | 54.79 | 52.36 | 142.59 | 136.55 |
| warp latency / instruction | 7.81 | 8.04 | 6.17 | 6.25 |

The padding cost 4% more instructions (the `(i / W) * LS + i % W` indexing) and 65% more store
conflicts for 7% fewer load conflicts: the predicted row-group conflicts were already small. Pad 0
is the default (`kS1Pad`). (*) The 988 us backward reading has the same per-instruction stalls as
the padded run and fewer instructions; it is foreign-load interference during that replay. The
final profile below re-measures it.

## Final configuration (forward 1024 single stages, backward 512 fused, pad 0)

`analysis/table.md` (`reports/fft_v3_final_002.ncu-repz`; the first collection,
`reports/fft_v3_final.ncu-repz`, returned a 713 us forward with the same wavefronts, instructions
and per-instruction stalls as every other 1024-thread forward: foreign-load interference, kept):

| metric | fwd baseline | fwd v2 | fwd final | bwd baseline | bwd v2 | bwd final |
|---|---:|---:|---:|---:|---:|---:|
| kernel time (us) | 412.6 | 282.0 | **277.1** | 864.8 | 626.2 | **574.3** |
| DRAM read (MB) | 31.9 | 44.7 | 39.6 | 83.8 | 54.1 | 59.3 |
| shared wavefronts (k) | 10260 | 12564 | 11988 | 15511 | 18757 | 14569 |
| instructions (M) | 55.6 | 59.1 | 59.0 | 147.5 | 144.3 | 136.6 |
| stall mio throttle | 1.18 | 2.91 | 3.15 | 0.53 | 0.65 | 0.60 |
| stall barrier | 0.80 | 1.86 | 1.67 | 0.54 | 0.42 | 0.38 |
| stall wait | 1.61 | 1.72 | 1.73 | 1.13 | 1.02 | 0.96 |
| warp latency / instruction | 11.4 | 14.1 | 13.9 | 8.9 | 6.7 | 6.4 |

Against this morning's baseline the forward kernel is 1.49x and the backward 1.51x faster, with
bitwise identical results.

Operator forward+VJP against production (`artifacts/v4_flash_fft/flash_rtx5060ti_010_fft_v3`,
`ops_rtx5060ti_006_fft_v3`; every budget passes at 1.00x, bitwise repeatable):

| case | production us | flash us | speedup | v2 | baseline `_006` / `_004` |
|---|---:|---:|---:|---:|---:|
| circular s1 B4 C64 96 pad 2 | 1421 | 714 | **1.99x** | 1.86x | 1.48x |
| circular s1 B4 C128 96 pad 2 | 3962 | 1301 | **3.04x** | 2.86x | 2.18x |
| s1 B4 C128 100 | 3769 | 1298 | **2.90x** | 2.62x | 2.06x |
| s1 B4 C64 96 | 1398 | 738 | 1.90x | 1.63x | 1.48x |
| data term s1 C64 96, KB 4, k7 | 1573 | 1022 | 1.54x | 1.43x | 1.27x |
| replicate 92 pad 4 / reflect / zeros | 1632 / 1711 / 1594 | 700 / 717 / 708 | 2.33x / 2.39x / 2.25x | 2.07-2.09x | 1.69-1.73x |
| ops circular C64 / C128 | 1434 / 4036 | 720 / 1344 | 1.99x / 3.00x | 1.82x / 2.76x | 1.46x / 2.19x |

The fused forward at 512 threads (`flash_rtx5060ti_009_fused_f512`: 1.92x / 3.01x / 2.87x / 1.90x)
and the default (`_f1024`: 1.93x / 2.98x / 2.85x / 1.85x) tie end to end as well.

## Roadmap

- **v1** SRAM-resident plane. **v2** memory movement (`31b3c88`). **v3** (this): conflict-free
  twiddles, fused two-stage passes. Bitwise identical throughout.
- **v3, remaining**: the backward's pointwise phase is the largest single item (about half its
  instructions); its two-term arithmetic is required by the grad_k / grad_l budgets, but the
  grad_Y assembly and the G scaling could move into the fused inverse pass's load phase to save
  one plane pass; the second Stockham stage's store conflicts (2-2.5-way) want a swizzle that
  keeps the arithmetic unchanged.
- **v4** (Blackwell): clusters with distributed shared memory for the s2 128x128 and s3 144x144
  planes; TMA staging of g behind FFT(x).

## Files

- `profile/flash-fft-v3-20261012`: twiddle layout, backward 512 / 768 / 1024 (`analysis/table.md`).
- `profile/flash-fft-v3b-20261012`: fused stages at 512 and 1024, pad 1; `reports/fft_v3b_pad0`
  and `analysis/pad_ablation.md` for the ablation.
- this directory: the final configuration (`analysis/table.md`, `reports/fft_v3_final.ncu-repz`).
- `.build/logs/flash-fft-v3*-build.log`: ptxas `-v`; `.build/logs/flash-fft-v3*-compare.log`:
  the bitwise comparisons.
