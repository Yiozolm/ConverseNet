# v4 campaign source and decision archive

`index.json` is compact provenance, not a gate report. It retains original
source/report paths and SHA-256, numerical rejection counts, compiler flags,
paired model speedups and unchanged decisions. Raw reports, binary extensions,
tensor captures, SASS and Nsight files remain in the ignored local artifacts;
none is copied here. Existing source records and decision JSON are copied
byte-for-byte, including their original `committed: false` fields. Archiving
those files does not turn a rejected optimization into an accepted one.

The current decisions recorded here are:

| Candidate | Recorded decision |
| --- | --- |
| Restricted B4 128-to-64 pointwise weight gradient | Accepted in `09f0f12`; the broad proposal remains rejected |
| Full-spectrum power FMA | Rejected: 12/1760 FP64 matrix checks, 40 test subcases failed; no performance admission |
| Division-VJP reuse | Rejected: 26/1760 FP64 matrix checks, 81 test subcases failed; no performance admission |
| Nearest k2/s2 residual | Rejected: 25/2163 cases failed, execution contracts passed |
| Residual with output FMA | Rejected: 19/2163 cases failed, execution contracts passed |
| Compensated residual | Numerical gate passed; complete-model speedups 1.0178724/1.0172057 failed the 1.03 requirement |
| Compensated residual with fused lambda prototype | Prototype gate/performance passed; this was not production admission |
| Initial padded production integration | Numerical/model checks passed; complete-model speedups 1.0275163/1.0264887 failed 1.03 |
| Overflow guard plus pad/crop cancellation | Accepted: fresh production 3519 direct + 3519 module checks, 168 release tests; whole-model speedups 1.0896529/1.0382488 |
| Nearest spectral and exact-phase recheck | Rejected: 10/96 and 9/96 tensors failed, respectively |

`fftfree_accepted.json` retains the final source/build identity, original report
hashes, per-round timings and decisions. The measured deployment is pretrained
SRResNet, B1 RGB 24x28 input, x4 output, on RTX 5060 Ti. Both repeats pass the
unchanged 1.03 threshold and have positive paired improvement in 7/9 and 8/9
rounds. The two checkpoint layers improve 5.91x to 7.13x in warm complete-call
timing; that local result must not be substituted for the whole-model result.
Model max-absolute deviation is 9.536743e-7. The extreme probe passes all 288
finite-baseline comparisons; 144 old-baseline-nonfinite cases are explicitly
excluded, not counted as passes. SASS contains no FP64 arithmetic. Short
training regressions are not convergence evidence.

`pointwise_determinism.json` preserves a separate release-reference correction.
Eight identical nondeterministic native cuDNN calls gave eight FP32 results;
the GEMM candidate and FP64 oracle remained fixed. The unit test now matches
the existing deterministic admission protocol, with cuDNN explicitly enabled.
The initial failure and the intermediate incorrect test configuration remain
recorded; no production algorithm or error threshold changed.

Files preserved below `rejected/`:

- Power-FMA and division-VJP patches against `09f0f12`, original decisions and
  checked source manifests. The patches plus that base preserve their code
  without duplicating whole source trees or retaining binary extensions.
- The unique output-FMA kernel and its original decision. The other source
  files match the residual implementation retained in `tools/v4_fftfree/`.
- All nine changed-source files for the rejected padded production integration,
  plus its original decision and source manifest. Its large full report and
  binary remain at the original paths identified in the index.

Keep the small source groups `tools/v4_arithmetic/`, `tools/v4_fftfree/`,
`tools/v4_fftfree_compensated/`, `tools/v4_fftfree_compensated_fused_lambda/`
and the three standalone FFT-free experiment helpers with the accepted
production change. They total roughly 235 KB at the recorded inventory and
contain no binary or large report. The production helper imports the fused
performance helper and the frozen fused-lambda study. The study imports its
inference wrapper immediately; the research loader and CPU mirror are lazy
imports. Historical provenance checks nevertheless hash the complete research
source group, so removing apparently unused files can invalidate that chain.

`pinned_cc244e3/` contains the four exact files required by
`tools/nearest_phase_repair/gate.py`, extracted using read-only Git access and
verified against that helper's pinned SHA-256. Commit `cc244e3` was locally
readable but had no named ref containing it when inspected. These source
copies are therefore preserved independently of ref reachability. The old
nearest-phase helper still requests `git show cc244e3`; on a clean clone
without that object, it needs the historical object or a separately reviewed
loader that reads these hash-verified copies. This archive does not silently
change that helper or claim that the old command already consumes these files.

The measured baseline s1 profile used input `[4,128,100,100]`, shared prior,
and kernel `[1,128,3,3]`. DRAM throughput was **88.33652% / 395.358185 GB/s**,
while SM throughput was **31.234921%**. This supports a memory-bandwidth
constraint for that measured kernel/shape; it is not a claim for all calls.
The index links the raw `.ncu-rep`, CSV, launcher and SASS comparison by their
exact paths and SHA. Profiler duration is not a complete-call benchmark.
The NCU baseline and independently rebuilt baseline had matching build inputs
but different binary SHA; that distinction is retained.

## Clean-checkout reproduction

The current actual-production helper can run fresh admission without historical
prototype reports. Select an existing CUDA PyTorch environment and the correct
MSVC/CUDA toolchain, run a checked build, and use a new report filename:

```powershell
./tools/run.ps1 test/extension_loader.py
./tools/run.ps1 -m unittest discover -s test -p 'test_*.py' -v
./tools/run_affinity.ps1 -Mask 0xFFFFFF -MetadataPath artifacts/v4_campaign/fresh_001.affinity.json tools/v4_fftfree_production.py --stage all --output artifacts/v4_campaign/fresh_001.json
```

Use the GPU owner's verified affinity mask if different; the wrapper rejects
an unavailable mask. Keep numerical level at the full default and the optional
exact-bit diagnostic disabled. `--stage all` runs fresh actual-production
3519 direct-op plus 3519 module probes, model checks, and both complete-caller
cache scopes and the full model performance protocol. It does not synthesize
historical evidence and does not load a research binary. A successful old
prototype result cannot replace any of these fresh checks.

Historical `--prototype-gate` and `--prototype-perf` arguments remain optional
and must be supplied together. They require authentic full reports, not this
index or an abbreviated JSON with invented `passed` fields. If those original
files are unavailable, regenerate independent checked prototype gate/perf
reports with the committed research sources in an isolated `09f0f12` baseline
checkout; keep that checkout and its report-referenced helper paths available
through the comparison. Preserve new failures and use new filenames. The
standalone historical prototype performance command cannot simply use the
already-optimized production module as its old baseline and retain its old
performance interpretation.

`archive_cpu.py` is the one-shot CPU archiver that created this directory. It
imports no Torch and never compiles, launches CUDA, stages files or alters Git
refs. Existing outputs are refused. Its only Git operations are read-only
`rev-parse`/`show` for the four pinned sources. The root may append a new accepted
decision to the index after validation; the copied rejected decisions and raw
reports should remain unchanged.
