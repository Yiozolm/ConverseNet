# Release suite and new-kernel checks on other GPUs (Colab CLI)

Google's `google-colab-cli` (June 2026) provisions a Colab A100 and runs code from the terminal,
replacing the notebook round trips in `tools/v4_fft_callbacks_prod/` and `tools/v4_flash_fft/`.
It supports Linux and macOS only; here it is installed in WSL (`uv tool install --python 3.13
google-colab-cli`) and driven as `wsl -d Ubuntu-24.04 -- colab ...`. The first call prints an
OAuth URL whose code the user pastes once; the token is then cached. A100 time is billed from
Colab compute units (5.3 per hour in October 2026); a complete run below costs about 0.1 unit.

Scripts (both run inside the remote Jupyter kernel via `colab exec -f`, configured with `--env`):

- `run_release.py`: clone and pin `CONVERSE_REF`, `pip install ninja`, checked build, the complete
  unmodified release suite at the full numerical level, then optionally `CONVERSE_GATE=1` (the
  split-K pointwise weight-gradient gate on its 14 synthetic geometries; the 4 GB captured
  fixtures are not uploaded, so the production-module probe is recorded as skipped) and
  `CONVERSE_PERF=1` (paired pointwise and LayerNorm timings with the USRNet B4 step). Writes
  `summary.json`, `summary.md`, every log and the `artifacts/fp32_release` reports into one
  directory and a tarball for `colab download`.
- `run_command.py`: clone, pin, build (reusing a finished checkout at the same commit) and run one
  repository command given in `CONVERSE_CMD` (`{out}` expands to the output directory).
- `layernorm_seeds.py`: every geometry of `test_layernorm_training.test_geometries_alignment_tails_and_eps`
  plus the production shape over independent seeds; per vector the single-run budget failures
  next to the seeded verdict. Runs on any GPU.

```sh
git push origin claude/v4-dev                       # the VM clones from the fork
colab new -s v4a100 --gpu A100
colab exec -s v4a100 --timeout 7200 --env CONVERSE_REF=<sha> --env CONVERSE_GATE=1 \
    --env CONVERSE_PERF=1 -f tools/v4_release_a100/run_release.py
colab download -s v4a100 /content/results/<name>.tgz artifacts/v4_release_a100/
colab stop -s v4a100
```

The default `colab exec` timeout is 30 s; raise it. A lost websocket (`Connection was lost.`)
leaves the session listed but its kernel unreachable: stop it and create a new one.

Failures are recorded, never hidden. The A100 baseline for comparison is the 54 FP64-budget
failures of 2026-10-03 (`tools/v4_fft_callbacks_prod/README.md`), identical at codex/v4.0.0.

## Run 1: A100-SXM4-40GB at 5c3d0f7 (`v4_release-NVIDIA_A100-SXM4-40GB-20261012T094745`)

torch 2.11.0+cu130, CUDA 13.0, driver 580.82.07, Python 3.13; checked build 81 s, binary
`d351230dc6fe`. Production code identical to 5c4a689 (5c3d0f7 adds only this directory).

**Release suite: 228 run, 56 failures.** 54 are the pre-existing sm_80 set, name for name
(`release_failures.txt` against the 2026-10-03 log). The two new ones are both in
`test_layernorm_training.test_geometries_alignment_tails_and_eps`, the first run of the fused
LayerNorm training VJP on anything but the RTX 5060 Ti:

| Geometry | Vector | candidate / baseline rel-L2 vs FP64 | ratio | limit |
|---|---|---|---|---|
| (1, 64, 20, 24), offset 0, eps 1e-5 | dweight | 1.110e-7 / 8.81e-8 | 1.261 | 1.25 |
| (1, 64, 96, 96), offset 1, eps 1e-6 | dbias | 1.180e-7 / 8.87e-8 | 1.331 | 1.25 |

Both max-abs ratios were 1.00 and 0.99. Attribution and the resulting kernel and test change are
in `tools/v4_layernorm_training/README.md` and commit c6c556a.

**Pointwise weight-gradient gate** (`pointwise_gate.json`): 14 synthetic geometries, 28/28
cases pass; split-K rel-L2 at most 1.00x of deterministic cuDNN (most cases far below).
Production-module probe skipped (no captured fixtures on the VM).

**Paired timing** (`pointwise_perf.json`, `layernorm_perf.json`; medians, dedicated GPU):

| Pointwise weight gradient | cuDNN deterministic | split-K | call | kernel |
|---|---:|---:|---:|---:|
| B4 C64 -> 128, 96x96 | 965 µs | 179 µs | 5.40x | 6.24x |
| B4 C128 -> 64, 96x96 | 860 µs | 160 µs | 5.37x | 5.83x |
| B1 C64 -> 128, 96x96 | 249 µs | 93 µs | 2.67x | 2.76x |
| B1 C128 -> 64, 96x96 | 247 µs | 93 µs | 2.66x | 2.78x |
| USRNet B4 step, only pointwise switched | 222.3 ms | 153.3 ms | **1.451x** | |

| LayerNorm forward + VJP | ATen | fused | call | kernel |
|---|---:|---:|---:|---:|
| B4 C64 96x96 | 1204 µs | 531 µs | 2.27x | 4.91x |
| B2 C128 100x100 | 1204 µs | 530 µs | 2.27x | 3.38x |
| B1 C64 96x96 | 1104 µs | 431 µs | 2.56x | 3.45x |
| USRNet B4 step, only LayerNorm switched | 199.1 ms | 154.1 ms | **1.292x** | |

The A100 call floor is higher than the RTX 5060 Ti's (Colab's Python/launch overhead), so the
B1 layer VJP with the split-K kernel is slower per call (0.82x) while its kernel time is 1.9x
faster. Operator- and step-level timings; not a convergence result.

## LayerNorm channel vectors over seeds (`layernorm_seeds_*.json`)

Eight seeds per geometry, same seed base on both GPUs, committed kernel (5c4a689 arithmetic):

| GPU | single-run failures (dweight, dbias) over 9 geometries x 8 seeds | seeded verdicts | dx rel-L2 ratio, geomean range |
|---|---|---|---|
| RTX 5060 Ti (`_rtx5060ti_001`) | 8, 4 of 72 each | 27/27 pass | 1.08–1.16 |
| A100 (`layernorm_seeds-...-20261012T095823`) | 9, 2 of 72 each | 27/27 pass | 1.09–1.16 |

The misses fall on the same geometries on both GPUs ((1,64,96,96) off 1 eps 1e-6: 3/8 dweight on
both; (2,64,5,7): 2/8 dweight on both). The single-seed channel-vector gate is a noise draw
against the ATen baseline's own reduction order, as the test already stated for the production
shape; the committed test passed on the RTX 5060 Ti by its seed choice. The dx column showed a
systematic 1.15x excess on every geometry and both GPUs, which led to the kernel change.

## Run 2: A100-SXM4-40GB at c6c556a (`v4_release-NVIDIA_A100-SXM4-40GB-20261012T102039`)

Same VM image (torch 2.11.0+cu130, CUDA 13.0, driver 580.82.07), fresh clone; build 81 s, binary
`d1449f25125b`. c6c556a carries the compensated LayerNorm backward and the seeded geometry gate.

**Release suite: 228 run, 54 failures, exactly the 2026-10-03 sm_80 set** (`release_failures.txt`
against the baseline: 54 common, 0 only here, 0 only there). The two LayerNorm failures of run 1
are gone; no test changed state in either direction.

**Pointwise gate**: 28/28 synthetic cases pass again (production probe skipped as before).

**Paired timing** (same protocol as run 1; Colab host noise is visible in the ATen references):

| Pointwise weight gradient | cuDNN deterministic | split-K | call | kernel |
|---|---:|---:|---:|---:|
| B4 C64 -> 128, 96x96 | 878 µs | 161 µs | 5.47x | 5.83x |
| B4 C128 -> 64, 96x96 | 862 µs | 160 µs | 5.38x | 5.82x |
| B1 C64 -> 128 / C128 -> 64 | 248 / 247 µs | 93 / 93 µs | 2.68x / 2.67x | 2.76x / 2.78x |
| USRNet B4 step, only pointwise switched | 224.0 ms | 148.9 ms | **1.505x** | |

| LayerNorm forward + VJP (compensated backward) | ATen | fused | call | kernel |
|---|---:|---:|---:|---:|
| B4 C64 96x96 | 1133 µs | 429 µs | 2.64x | 2.87x |
| B2 C128 100x100 | 1117 µs | 495 µs | 2.25x | 1.81x |
| B1 C64 96x96 | 1138 µs | 434 µs | 2.62x | 2.34x |
| USRNet B4 step, only LayerNorm switched | 199.9 ms | 153.2 ms | **1.305x** | |

The fused LayerNorm call got faster between runs (531 -> 429 µs at B4) while its profiler kernel
ratio fell (4.91x -> 2.87x); the two runs were on different Colab hosts and the ATen kernel sums
moved as well, so the per-kernel ratios on Colab are indicative only. The step-level numbers,
which are what the branch is judged on, are 1.29x and 1.31x for LayerNorm and 1.45x and 1.51x
for the pointwise gradient across the two runs.

## State of the sm_80 release suite

On the A100 the FP32 release suite has had the same 54 FP64-budget failures since codex/v4.0.0
(all on planes below 16 pixels a side: the 5x7 training matrix, PSF preparation, s2/s3 fusion).
Nothing on claude/v4-dev added to or removed from that set once the LayerNorm gate was corrected.
They remain recorded as failures, not relabelled; a release note has to state them as a known
sm_80 limitation or the release has to fix them.
