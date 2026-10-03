# s1 forward plane kernel

On an A100, Nsight showed `scale1_forward` (B4/C128/100x100, shared prior) executing about
194 instructions per element. Only about 29 were FP32; most of the rest was int64 div/mod
index math. The A100 run also showed about 24 MB of repeated reads of the broadcast `k`
plane. The A100 harness is on branch `colab/kernel-bound`.

`scale1_forward_planes` uses one thread per `D` element `(kb, c, offset)` and
`blockIdx.y = kb*C + c`. Each thread loads `k`, `l` and the denominator once and writes
`D` once. It then loops over the batches that share that plane. Each output element runs
the same FP32 operations in the same order, so results must be byte-identical. This is not
a budgeted arithmetic change. The original kernel remains the fallback when `KB*C > 65535`.

- `capture.py` writes SHA-256 for every output and VJP across 814 cases: B1-5, C1/3/8,
  odd and multi-block planes, KB/KC broadcast, shared and independent priors, weak
  regularization, 1e18 scaling, and zero kernel bins. It also covers the production
  spectral shapes and the B1/B4 module path. Use `--compare` against a baseline run.
- `perf.py` runs one timing round for a checked build. Alternate baseline and candidate
  roots in fresh processes.

Recorded on the RTX 5060 Ti, MSVC 14.44, CUDA 13.0 (local `artifacts/v4_scale1_planes/`):

- Byte capture: all 814 cases identical to the `347b040` build. This was checked on both
  candidate binaries, `045eea64` and the timed `34d7e691`.
- Release suite: 203/203 passed before the new test file was added. With it, 205/205
  passed on the timed binary `34d7e691`.
- Paired timing, 6 alternating rounds: the forward kernel median went from 334 µs to
  257 µs (1.30x). The complete B4 forward+VJP call median is about 4.9 ms and its
  round-to-round noise (4.6-5.4 ms) exceeds the about 80 µs saved. This shows no
  measurable whole-call change. It is not evidence of a whole-model speedup.
