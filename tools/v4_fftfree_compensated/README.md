# Compensated FP32 nearest k2/s2 inference experiment

This is a new arithmetic candidate derived from the explicit nearest experiment
in `tools/v4_fftfree/`, originally restored from
`cc244e3e49dcd896962e14c98095243b41ee578a:research/fftfree_inference/`.
It is not byte-identical to the historical residual or output-FMA kernel.
`candidate_origin.json` records the source group copied before these changes;
the old candidates, builds and failures remain untouched. Its independent
PyBind module does not register or replace production operators. The checked
loader records source, frozen-source, compiler, flags, Torch and binary identity,
requires an explicit CUDA architecture, and uses
`--fmad=false --ftz=false --prec-div=true`. Only explicit RN FMA intrinsics fuse.

`NearestK2Inference(x, weight, bias, eps)` explicitly defines the prior as
nearest upsampling of `x`. It never accepts or guesses an arbitrary prior.
It supports no_grad, inference_mode and GradMode with frozen inputs. A
differentiable input in GradMode is rejected. Frozen GradMode calls execute
the inference-only CUDA entry under no_grad. FP16/BF16/FP64 and autocast are
rejected. Python prepares the same FP32 `sigmoid(bias-9)+eps` on every call
and passes a `(1,C,1,1)` regularizer. CUDA resolves input layouts and prepares
the four-term kernel power per thread; the old ATen energy temporary kernels
are absent. This is intentionally a separate numerical experiment.

The kernel uses only `float` and `float2` arithmetic:

1. Full Knuth TwoSum (without a magnitude-order precondition) and explicit
   FMA TwoProduct accumulate `N=x-sum(w*x)` and `D=sum(w*w)+lambda` into hi/lo.
2. `qhi=Nhi/Dhi`; one residual correction computes
   `qlo=(fma(-qhi,Dhi,Nhi)+Nlo-qhi*Dlo)/Dhi`, with explicit RN operations.
3. Each phase output compensates `x+w*qhi+w*qlo`, then rounds to FP32 once.

Two-component addition still rounds its low component; this is an approximate
extended-significand calculation, not an arbitrary-precision or correctly
rounded-solution guarantee. Overflow or residuals below the FP32 subnormal
range invalidate the error-free-transform premise. A single division correction
cannot remove the error in the original FP32 regularizer. These limitations
are not silently handled with a wider production dtype or a changed formula.
Recomputing power per pixel adds ALU work; any performance gain needs measurement.

The default numerical gate contains 3,459 cases: three seeds; LR 1x1, 5x1,
5x6 and 7x5; four batch/channel broadcasts; softmax, signed, zero, 1e-6 and
1e-3 kernels; contiguous, sliced and transposed storage; and all three
inference contexts. Three additional kinds cover signed kernels with input
width multipliers `logspace(1e-3,1e3)`, near-zero inputs scaled by `1e-8`, and
a unit kernel (one coefficient equal to 1, all others 0) for exact residual
cancellation with the nearest prior. All three use eps=1e-5. The original
2,163 input/specification/stride records remain unchanged; CPU self-check
verifies their combined SHA. Weak fixtures use eps=1e-8, bias=-40 and
ordinary-amplitude inputs. The gate also includes the exact historical signed/transpose failure
and two synthetic layer-sized pad/crop timing fixtures. The timing fixtures
are not captured model activations or dataset-quality samples.

Every case compares the candidate with the frozen **half-spectrum FP32**
baseline from `test/fp32_baseline.py`, using identical quantized FP32 input
values promoted to the independent full-spectrum Python FP64 oracle.
`test/numerical_policy.py` supplies normal/weak relative-L2 and max-absolute
budgets. Normal cases also require atol=rtol=1e-5 against the frozen baseline,
except dynamic-range cases, whose allclose result is diagnostic under the
existing release policy; both normal FP64 budgets remain mandatory for them.
Every case records independent full-spectrum FP64 denominator minimum, p01,
median and maximum through `numerical_policy.denominator_statistics`, including
weak-case minimum denominators and the actual padded geometry.
Padded and cropped outputs are checked independently; repeat determinism,
input preservation, dtype/GradMode/autocast rejection, side-stream capture
and updated-input graph replay remain mandatory. Byte identity to the old
ATen residual expression is diagnostic only.

The original failure and offline screen are preserved in
`historical_fixture.json`, including original input hashes/strides and source
hashes. CPU self-check reproduces those exact input bytes and parses all Python
sources without initializing CUDA. `cpu_compensated.py` mirrors RN scalar
operations with NumPy binary32 and Windows UCRT `fmaf`, including a fused
cancellation probe. It checks the three distinct fixtures behind the prior
output-FMA failures plus the three new coverage kinds. This helper is never
imported by the candidate or GPU gate.
CPU numerical outputs are diagnostic; CUDA arithmetic and GPU sigmoid require
the independent full GPU gate.

```powershell
# CPU-only source/fixture check; does not build or initialize CUDA.
H:/Python/ConverseNet/.venv/Scripts/python.exe -B tools/v4_fftfree_compensated/study.py --self-check

# Root GPU owner only, when this candidate's turn is reached:
$env:CONVERSE_PYTHON = 'H:\Python\ConverseNet\.venv\Scripts\python.exe'
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
./tools/run.ps1 tools/v4_fftfree_compensated/loader.py --build --verbose --artifacts artifacts/v4_fftfree_compensated/build_001
./tools/run_affinity.ps1 -Mask 0xFFFFFF -MetadataPath artifacts/v4_fftfree_compensated/gate_001.affinity.json tools/v4_fftfree_compensated/study.py --artifacts artifacts/v4_fftfree_compensated/build_001 --output artifacts/v4_fftfree_compensated/gate_001.json

# A new run repeats the complete gate, then may time. The current production
# extension must already have a matching checked build; it is never rebuilt here.
./tools/run_affinity.ps1 -Mask 0xFFFFFF -MetadataPath artifacts/v4_fftfree_compensated/perf_001.affinity.json tools/v4_fftfree_compensated/study.py --artifacts artifacts/v4_fftfree_compensated/build_001 --output artifacts/v4_fftfree_compensated/perf_001.json --timing
```

Use the GPU owner's established affinity mask if different. Builds require a
fresh directory and reports a fresh filename. Every numerical or contract
failure blocks timing and yields a nonzero exit; the report retains all failed
cases. No baseline tensor, error budget or old result is recalibrated.

Optional timing compares the independent PyBind module and one current
production TORCH_LIBRARY in the same process, rotating AB/BA for six rounds.
The production timing fixtures must first pass the same current gate. Each
measured call includes replicate pad2, relevant nearest-prior creation,
kernel preparation, regularizer, solve, allocation, crop and dispatch. The
production kernel cache is cleared outside every measured call, so its
preparation remains inside timing. The candidate has no cache. Wall time,
CUDA events and extra PyTorch allocated memory are reported independently.
This measures complete **uncached** callers, not warm deployment performance.

No model replacement, training VJP, cuFFTDx integration, dataset quality or
convergence claim is included. Successful numerical admission alone does not
establish a performance improvement or authorize production integration.
