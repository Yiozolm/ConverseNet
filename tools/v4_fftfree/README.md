# Explicit nearest k2/s2 FP32 inference experiment

This directory restores the independent CUDA residual kernel from
`cc244e3e49dcd896962e14c98095243b41ee578a:research/fftfree_inference/`.
The CUDA kernel is byte-identical to that historical source. Its independent
PyBind module does not register or replace production operators. The checked
loader records source, frozen-source, compiler, flags, Torch and binary identity,
requires an explicit CUDA architecture, and retains `--fmad=false`.

`NearestK2Inference(x, weight, bias, eps)` explicitly defines the prior as
nearest upsampling of `x`. It never accepts or guesses an arbitrary prior.
It supports no_grad, inference_mode and GradMode with frozen inputs. A
differentiable input in GradMode is rejected. Frozen GradMode calls execute
the unchanged inference-only CUDA entry under no_grad. FP16/BF16/FP64 and
autocast are rejected; kernel power and regularization are prepared on every
call, with the power reduction retaining the original input layout.

The default numerical gate contains 2,163 cases: three seeds; LR 1x1, 5x1,
5x6 and 7x5; four batch/channel broadcasts; softmax, signed, zero, 1e-6 and
1e-3 kernels; contiguous, sliced and transposed storage; and all three
inference contexts. Weak fixtures use eps=1e-8, bias=-40 and ordinary-amplitude
inputs. The gate also includes the exact historical signed/transpose failure
and two synthetic layer-sized pad/crop timing fixtures. The timing fixtures
are not captured model activations or dataset-quality samples.

Every case compares the candidate with the frozen **half-spectrum FP32**
baseline from `test/fp32_baseline.py`, using identical quantized FP32 input
values promoted to the independent full-spectrum Python FP64 oracle.
`test/numerical_policy.py` supplies normal/weak relative-L2 and max-absolute
budgets. Normal cases also require atol=rtol=1e-5 against the frozen baseline.
Padded and cropped outputs are checked independently; repeat determinism,
input preservation, dtype/GradMode/autocast rejection, side-stream capture
and updated-input graph replay remain mandatory. Byte identity to the old
ATen residual expression is diagnostic only.

The original failure and offline screen are preserved in
`historical_fixture.json`, including original input hashes/strides and source
hashes. CPU self-check reproduces those exact input bytes and parses all Python
sources without initializing CUDA. Its numerical output is diagnostic only;
CPU checks do not admit a CUDA implementation.

```powershell
# CPU-only source/fixture check; does not build or initialize CUDA.
H:/Python/ConverseNet/.venv/Scripts/python.exe -B tools/v4_fftfree/study.py --self-check

# Root GPU owner only, when this candidate's turn is reached:
$env:CONVERSE_PYTHON = 'H:\Python\ConverseNet\.venv\Scripts\python.exe'
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
./tools/run.ps1 tools/v4_fftfree/loader.py --build --verbose --artifacts artifacts/v4_fftfree/build_001
./tools/run_affinity.ps1 -Mask 0xFFFFFF -MetadataPath artifacts/v4_fftfree/gate_001.affinity.json tools/v4_fftfree/study.py --artifacts artifacts/v4_fftfree/build_001 --output artifacts/v4_fftfree/gate_001.json

# A new run repeats the complete gate, then may time. The current production
# extension must already have a matching checked build; it is never rebuilt here.
./tools/run_affinity.ps1 -Mask 0xFFFFFF -MetadataPath artifacts/v4_fftfree/perf_001.affinity.json tools/v4_fftfree/study.py --artifacts artifacts/v4_fftfree/build_001 --output artifacts/v4_fftfree/perf_001.json --timing
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
