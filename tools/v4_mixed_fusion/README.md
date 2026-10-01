# Low-precision activation conversion and padding fusion

This next experiment follows `v4_mixed_precision/results.json`. That study
admitted FP16/BF16 activation storage with FP32 weights, bias, solver and output,
but independent conversion allocations made the complete operator slower.
The public production FP32 contract and all previous failures remain unchanged.

## Candidate scope fixed before implementation

The first candidate may replace only `pad(x_low.float())` with one CUDA kernel
that reads the low precision activation and writes the required padded FP32
buffer. It must preserve every value, padding convention and output layout.
The existing checked `forward(padded, padded, weight, bias, 1, eps)` and crop
remain responsible for all FFT, complex arithmetic, denominator and output.

The selected module path initially targets contiguous CUDA s1 **FP16 circular
padding**, `eps >= 1e-5` inference with FP32 weight/bias. The primitive can test four padding
modes and both low dtypes, but BF16 and other module padding modes retain the
old path until separately measured. Other layouts and differentiable GradMode
calls retain the existing module plus differentiable upcast. Frozen GradMode
must remain enabled: disabling it would change the existing solver route.
No cached training spectrum, output quantization, AMP, TF32, fast math, hidden
FFT resizing, parameter mutation or FP64 device compute is permitted.

The repository's native Windows checked-build, paired-timing, NCU and SASS
workflow is used. The generic optimizer skill's bundled GPU runner is not a
native Windows backend. All GPU work runs serially in the root process.

## Predeclared experiment gates

1. Before generating candidates, validate the unchanged mixed module baseline
   against frozen quantized FP32 and independent FP64 references, record its
   checked build/toolchain, and profile conversion/padding and the complete call.
2. Candidate padding must match `F.pad(x_low.float(), ...)` bit for bit on
   finite values, including signed zeros/subnormals, with FP32 contiguous output.
   Restrict dispatch or fall back where the original output layout differs.
3. The complete candidate must match the actual unchanged mixed module exactly
   for the same quantized input, and pass the existing FP32 arithmetic budgets
   against quantized-input references. Preserve mode selection, shared prior,
   broadcast reductions, higher-order fallback, caller state and invalid-input
   behavior. Unsupported cases must not silently change the operator.
4. Explore a bounded set of launch configurations only after those baseline
   gates pass. Select on complete-call timings, including all remaining casts,
   padding, solver, crop and Python dispatch. Keep every candidate result.
5. For performance admission, require at least 1.03x median wall and CUDA speed
   ratios in each of two independent paired repeats, each with nine AB/BA rounds
   and at least seven positive wall pairs. Require lower peak allocation on the
   large target. Cold samples clear cache outside each timed complete call.
6. An operator-only win is insufficient for a model claim. Require exact
   pretrained output equality with the matching quantized model baseline and
   the same performance thresholds on at least one declared full model.
   Quantization/dequantization common to both models stays inside model timing.
   A third original-FP32 model control may be reported, but cannot substitute
   for the equal-semantics quantized control.
7. Collect independent NCU traffic evidence and inspect SASS for the promoted
   kernel. Profiler durations are not latency benchmarks. Run the release suite
   plus meaningful boundary/dispatch tests before committing an accepted path.

The pre-generation baseline passed all 16 shape/mode comparisons. Its main
B4/C128/96x96 circular call has six conversion/padding kernels before the
unchanged solver, making circular padding the first selected module scope.
`proposal.json` pins the baseline and NCU evidence. Other tested primitive
modes are not advertised as optimized model dispatch.

The first unrestricted-epsilon candidate gate completed 523 cases but failed
eight inherited weak-regularization FP32 budgets. All 384 complete-module
outputs, layouts and routes were identical to the unchanged baseline, and all
recorded values were finite. That full gate remains failed. The next experiment
restricts accelerated dispatch to `eps >= 1e-5`; weaker regularization uses the
original module. No budget or reference changes. `restricted_scope.json` pins
the failed evidence and the new scope, while exact old adapter/gate sources are
kept under `history/`.

The new gate retains each legacy-budget failure and the overall `passed=false`
when applicable. Its separate `active_domain_admitted` flag requires strict
numerical success for every active-domain case and exact, finite, state-preserving
fallback behavior. The performance/model studies accept only that explicitly
named scope; they do not reinterpret full-matrix failures as numerical passes.

Research reports use fresh paths; source, binary, input and checkpoint hashes
identify each run. Failed candidates and inconclusive measurements stay failed
or inconclusive. This experiment does not recalibrate low-precision output
budgets or claim whole-model mixed-precision training/convergence.

## Accepted experimental entry and reproduction

The selected 256-thread build passed the restricted numerical domain, all eight
declared complete-call shape/cache requirements and both tested full models.
See [RESULTS.md](RESULTS.md) and [results.json](results.json). This remains an
explicit research entry; the production operator/module still requires FP32.
The gain is over the matching unfused mixed boundary. Original-FP32 whole-model
speed superiority was not consistently established.

Configure the existing project PyTorch/MSVC environment as in `test/README.md`.
Use fresh output/build paths rather than replacing any captured run:

```powershell
$env:CONVERSE2D_SKIP_BUILD='1'
./tools/run.ps1 tools/v4_mixed_fusion/baseline.py --output artifacts/v4_campaign/fusion_baseline_NEW.json
$env:TORCH_CUDA_ARCH_LIST='12.0'
./tools/run.ps1 tools/v4_mixed_fusion/loader.py --build --load-production-first --block-threads 256 --artifacts .build/mixed_fusion/pad_b256_NEW
./tools/run.ps1 tools/v4_mixed_fusion/gate.py --baseline artifacts/v4_campaign/fusion_baseline_NEW.json --artifacts .build/mixed_fusion/pad_b256_NEW --block-threads 256 --output artifacts/v4_campaign/fusion_gate_NEW.json
./tools/run_affinity.ps1 -Mask 0xC03C03 -MetadataPath artifacts/v4_campaign/fusion_perf_NEW.affinity.json tools/v4_mixed_fusion/study.py --artifacts .build/mixed_fusion/pad_b256_NEW --block-threads 256 --gate artifacts/v4_campaign/fusion_gate_NEW.json --output artifacts/v4_campaign/fusion_perf_NEW.json
./tools/run_affinity.ps1 -Mask 0xC03C03 -MetadataPath artifacts/v4_campaign/fusion_models_NEW.affinity.json tools/v4_mixed_fusion/model_study.py --artifacts .build/mixed_fusion/pad_b256_NEW --operator-gate artifacts/v4_campaign/fusion_gate_NEW.json --block-threads 256 --output artifacts/v4_campaign/fusion_models_NEW.json
```

The full gate intentionally still exits nonzero when inherited weak-budget rows
fail. Before proceeding, inspect `complete`, `active_domain_admitted`, exact
`admission_scope`, the full retained rows and checked identities. Downstream
programs enforce this restricted admission independently. Never turn the whole
matrix's `passed` flag into true. Select a valid recorded affinity mask on a
different host. This checked research build targets the measured SM120 device;
other architectures need their own build and measurements.

For final traffic comparisons, use `profile_ncu.py --worker study.py` twice with
the same gate/artifact/case and `--profile baseline` / `--profile candidate`.
Do not compare a different worker's initialization protocol as the final pair.
`sass_check.py --artifacts ... --output NEW_DIR` inspects the exact binary.

After loading the production and research libraries with `loader.py`, direct
layer experiments use `adapter.mixed_module_forward(layer, x_fp16, extension)`.
For the tested model protocol:

```python
from tools.v4_mixed_fusion.model_study import forward_scope

with torch.no_grad(), forward_scope(model, 'fused', extension):
    output = model(*fp32_model_inputs)
```

This scope keeps model masters and surrounding computation FP32, quantizes only
Converse2D inputs inside the model call, and restores forward methods afterward.
USRNet DataNet stays FP32; only its 35 prior Converse2D calls are quantized.
Scope setup/state verification is outside the reported warm resident-model
latency. The direct layer helper is a forward function; ordinary module hooks
are preserved by the enclosing real model calls in the model scope.
