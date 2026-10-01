# v4 isolated pointwise-weight-gradient experiment

`pointwise_wgrad.py` changes only the first-order FP32 weight VJP of eligible
1x1 Conv2d modules. Forward, the combined native input/bias VJP, and the entire
higher-order VJP use native ATen. Eligibility is restricted to CUDA FP32,
64->128 or 128->64 channels, stride/dilation one, no padding/groups, and at least
4096 batch-spatial points. Other FP32 cases retain native convolution; non-FP32
inputs are rejected. AMP and TF32 are forbidden. Import never patches a model.

Each phase requires a **new** output path and preserves failures. The checked
CUDA extension must already have been built with `test/extension_loader.py`.
Use the repository's configured PyTorch interpreter / `tools/run.ps1` launcher.
Examples below are Python arguments to that launcher:

```text
tools/v4_experiments/wgrad_study.py capture --manifest artifacts/v4_campaign/dataset_absolute.json --fixtures artifacts/v4_campaign/capture_v1.pt --output artifacts/v4_campaign/capture_v1.json
tools/v4_experiments/wgrad_study.py gate --fixtures artifacts/v4_campaign/capture_v1.pt --output artifacts/v4_campaign/gate_v1.json
tools/v4_experiments/wgrad_study.py perf --gate artifacts/v4_campaign/gate_v1.json --output artifacts/v4_campaign/perf_v1.json
tools/v4_experiments/wgrad_model.py --gate artifacts/v4_campaign/gate_v1.json --perf artifacts/v4_campaign/perf_v1.json --manifest artifacts/v4_campaign/dataset_absolute.json --output artifacts/v4_campaign/model_v1.json
```

Capture loads the complete pretrained USRNet, uses B4 LR32 / scale 3, and saves
all 140 invocations of its 28 prior pointwise modules. The manifest selects the
existing real-photograph protocol with its declared synthetic degradation. If
`--manifest` is omitted, the report explicitly labels the input and target as
fixed-seed synthetic data. The fixture includes source/checkpoint identities;
the full fixture is approximately 4 GB and should remain an artifact.

The operator gate compares the identical FP32-origin inputs through native
FP32, candidate FP32 and independent native FP64 convolutions. Every output and
requested gradient uses `test/numerical_policy.py`, including its documented
floors. Bitwise equality is recorded only as a diagnostic. Coverage includes
all captures; B1/B4 synthetic activations in both channel directions; gradient
masks; missing bias; shared leaves; noncontiguous layouts; spatial tails; small
and unsupported-convolution fallbacks; higher derivatives; a nondefault stream;
and CUDA Graph replay after changing inputs and upstream gradients. Every
ordinary case asserts its expected dispatch route.

Performance requires a complete source/fixture/build-matching gate. Complete
forward plus input/weight/bias VJPs include Python/autograd overhead and all
required layout copies. Each captured invocation receives two independent
repeats of five warmups and nine alternating AB/BA rounds of ten calls. Both
repeats need median wall speedup >=1.03, positive improvement in >=7/9 rounds,
and no CUDA event-time slowdown above 2%. All raw rounds are retained.

The separate model phase requires both gates. Its unchanged checked CUDA model
is the control, so the comparison isolates this one pointwise VJP change. Three
seeds and three B4 Adam steps compare output, loss, every gradient, every
parameter and optimizer state with atol=rtol=3e-5. It asserts 140 active calls
per forward (70 if a future separately admitted scope selects one direction).
This is regression coverage, **not convergence evidence**. Full-model B1/B4
timing runs twice, restores identical parameters and populated Adam state before
each route/round outside timing, uses identical data sequences, and has no
timing hooks. After each timed ten-step sequence an untimed finite-state check
covers loss, gradients, parameters and Adam state. B4 must improve on both
repeats; neither B1 nor B4 may regress by more than 2%.

These tools do not themselves admit a production change or create a commit.
The campaign also requires the repository's checked build and release suite.
Historical failed experiments retain their original labels.
