# C128->64 restricted follow-up

The original two-direction experiment failed 63 numerical checks in
`artifacts/v4_campaign/wgrad_gate_v1.json`. Its source files and report hash are
preserved in `artifacts/v4_campaign/wgrad_v1_sources/source_manifest.json`.
Neither that report nor its failed status has been changed.

This experiment makes one structural change to deployment scope: only
**C128->64** may use the original FP32 GEMM weight VJP. C64->128 always uses
native convolution. The numerical budget is unchanged. There are no module,
seed, tensor-value, or example-specific exceptions in the candidate guard.

Use the same launcher as the original experiment, with these Python arguments:

```text
tools/v4_experiments/wgrad_study_c128_to64.py gate --fixtures artifacts/v4_campaign/wgrad_capture_v1.pt --output artifacts/v4_campaign/wgrad_gate_c128_to64_v1.json
tools/v4_experiments/wgrad_study_c128_to64.py capture --seed 29 --manifest artifacts/v4_campaign/dataset_absolute.json --fixtures artifacts/v4_campaign/wgrad_capture_seed29.pt --output artifacts/v4_campaign/wgrad_capture_seed29.json
tools/v4_experiments/wgrad_study_c128_to64.py gate --fixtures artifacts/v4_campaign/wgrad_capture_seed29.pt --output artifacts/v4_campaign/wgrad_gate_c128_to64_seed29.json
tools/v4_experiments/wgrad_study_c128_to64.py perf --gate artifacts/v4_campaign/wgrad_gate_c128_to64_v1.json --output artifacts/v4_campaign/wgrad_perf_c128_to64_v1.json
tools/v4_experiments/wgrad_model_c128_to64.py --gate artifacts/v4_campaign/wgrad_gate_c128_to64_v1.json --perf artifacts/v4_campaign/wgrad_perf_c128_to64_v1.json --manifest artifacts/v4_campaign/dataset_absolute.json --output artifacts/v4_campaign/wgrad_model_c128_to64_v1.json
```

Choose the actual existing fixture filename and new output paths. Both capture
and gate still cover all 140 full-model invocations: 70 active C128->64 calls
and 70 required-native C64->128 calls. Masks, missing bias, shared leaves,
noncontiguous layouts, spatial tails, higher-order derivatives, nondefault
stream, and CUDA Graph replay probe the active C128->64 direction. Small sizes,
unsupported convolution options, and other channel shapes verify fallback.

The performance phase measures every one of the 70 scoped invocations. The
model phase patches exactly the 14 matching modules and requires 70 active
calls per forward. The original precision, complete-call performance, source
identity, and full-model regression requirements remain unchanged. Both source
files implementing the candidate (restricted dispatcher and original VJP
class) are included in every gate/performance identity.
# Verified production scope

The production module is `models/pointwise.py`: only contiguous CUDA FP32
B4/C128->64/96x96 training inputs with trainable weights select GEMM. B1 and
other shapes, inference, frozen weights, unsupported layouts and the pytorch
backend retain native convolution. Parameter/checkpoint keys are unchanged.

The independently measured production gate passed 157 cases, including 70
actual candidate calls and 70 native fallbacks. Three seeds with three B4
Adam steps plus B1 fallback produced 6,670 passing tensor comparisons. Both
fresh production B4 complete-Adam runs improved (1.0228x and 1.0239x); the
70 complete-layer prototype calls separately improved 1.2393x–1.3407x.
The checked release suite passed all 154 tests. These are short regression
and timing results, not convergence evidence.

`wgrad_production_study.py` directly invokes the installed module. Its control
switches only the new PointwiseConv2d modules to `backend='pytorch'`; all other
Converse operators remain on the same CUDA binary. Do not run the older
prototype model script against integrated models and call it an independent
production comparison. The prototype's B1 speedup is not a production claim.
See [machine-readable evidence](results_c128_to64.json) for source/report SHA,
the frozen baseline, the rejected broad candidate and exact measurement scope.
