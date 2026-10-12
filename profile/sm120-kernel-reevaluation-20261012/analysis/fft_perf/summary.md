# Fused SRAM training: NVIDIA GeForce RTX 5060 Ti (sm_120)

- torch 2.11.0+cu130 / CUDA 13.0; opt-in shared memory per block 101376 B; s1 block sizes (forward, backward) [768, 512]

| case | plane | path | production us | flash us | speedup | worst rel-L2 ratio | budgets | repeatable |
|---|---|---|---|---|---|---|---|---|
| circular_s1_b4_c64_96_pad2 | 78 KB | fused | 1243 | 686 | 1.81x | 1.00 | pass | True |
| circular_s1_b4_c128_96_pad2 | 78 KB | fused | 3425 | 1195 | 2.87x | 1.00 | pass | True |
| forward_s1_b4_c128_100 | 78 KB | fused | 3294 | 1186 | 2.78x | 1.00 | pass | True |
| forward_s1_b4_c64_96 | 72 KB | fused | 1235 | 673 | 1.84x | 1.00 | pass | True |
| forward_s2_b4_c64_64 | 128 KB | fused | 3135 | 1471 | 2.13x | 0.90 | pass | True |
| forward_s2_b4_c64_48 | 72 KB | fused | 1289 | 856 | 1.51x | 0.93 | pass | True |
| forward_s2_b4_c64_32 | 32 KB | fused | 575 | 485 | 1.19x | 1.46 | FAIL | True |
| forward_s3_b2_c32_48 | 162 KB | fused | 649 | 611 | 1.06x | 0.96 | pass | True |
| forward_s3_b2_c32_32 | 72 KB | fused | 603 | 478 | 1.26x | 0.77 | pass | True |
| forward_s3_b2_c32_24 | 40 KB | fused | 538 | 465 | 1.16x | 1.21 | pass | True |

Operator-level forward+VJP; not a whole-model or convergence result.
