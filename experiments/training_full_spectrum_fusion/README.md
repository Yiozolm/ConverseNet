# Full-spectrum CUDA training: default backend and research reproduction

As of 2026-09-22, fused full-spectrum training is the default Converse2D backend
for CUDA FP32 v7 calls with GradMode enabled and at least one differentiable
input. Rebuild the ordinary extension and restart Python after updating; no
opt-in wrapper or separate experimental loader is needed for normal training.
See [the package README](../../Converse2D/README.md) for the build command.
`no_grad()` / `inference_mode()` v7 inference retains the half-spectrum path.

This directory retains the separately compiled complex64 core and `FusedRoute`
for research reproduction. The wrapper accepts a caller-supplied operator
namespace and isolated core; it does not replace the normal installation
workflow. No files from the HPC experiments or their datasets are required by
this wrapper. Existing half-spectrum records and failed numerical results remain
historical evidence and are not relabeled as results of the new default.

The CUDA sources live in
`Converse2D/torch_converse2d/training/full_spectrum/full_fusion.cpp` and
`full_fusion.cu`. FP32 kernel pad/roll/FFT, input/prior FFT, lambda
parameterization and the final inverse FFT remain differentiable PyTorch
operations. The core fuses pointwise forward/VJP work and repeated-q use while
retaining alias reduction order and each broadcast `sum_to` boundary. It does
not fuse the FFT itself.

Scale 1 uses the dedicated `full_spectrum/scale1.cuh` pointwise kernels: one
forward launch and one fused backward pointwise launch. When the kernel matches
all input batches and channels, that backward launch also completes the kernel
gradient; otherwise it is followed by the existing
broadcast reductions and kernel-gradient completion. This removes the temporary
prediction/power buffers in forward and the product/negation buffers in backward.
Shared inputs omit the unused independent gy buffer and share one input layout
materialization. The complex gradient scratch layout, separate broadcast reductions, shared-input
gradient order and higher-order fallback are retained. Scale 2 uses
`full_spectrum/scale2.cuh` when LR width is greater than one and the HR complex
tensor occupies at most `INT32_MAX` bytes. It combines the ordered four-alias
reductions with pointwise forward/VJP work. Broadcast reductions, complex
denominator-gradient layout, final kernel-gradient addition order and the
higher-order fallback remain unchanged. Width-one and larger-byte-offset inputs
retain the generic route; scales 3 and above also retain the generic route.
The standalone loader freezes and hashes both `scale1.cuh` and `scale2.cuh`
together with the core while compiling only the `.cpp`/`.cu` files. Earlier
standalone manifests that lack the new header fail their source-identity check;
retain those artifacts and rebuild in a fresh directory.

The production build embeds the C++ core through
`training/full_spectrum/production.cpp`, suppressing its standalone library and
Python-module registration. The `converse2d::full_training` implementation
namespace prevents collisions with the retained half-spectrum implementation.
The production bridge also preserves a noncontiguous prior spectrum's layout
before IFFT. Both normal training and this research wrapper use independent
per-call kernel FFT graphs; the historical training scope cache is not used by
the default full-spectrum route.

The explicit multiply/FMA/add boundaries follow the measured ATen complex
arithmetic. The `y is p` scale-one case retains the original shared-input
gradient order `(G + gy) + gprediction`. Higher derivatives rebuild the
differentiable ATen reference, including a shared-input reference path. These
choices preserve the tested operation order; they are not a guarantee of
bitwise equality on every Torch release, GPU, layout or compiler.

## Standalone research build and warm load

These commands are for the isolated research loader. They are unnecessary for
normal use of the rebuilt production extension.

Use the repository's Python environment with matching PyTorch/CUDA and a CUDA
toolkit. Linux also needs a supported C++ compiler; Windows needs MSVC C++ build
tools. `ninja` must be available to PyTorch's extension loader. Building is
explicit and requires an empty artifact directory. A failed build is retained;
use a new artifact directory for its retry.

From the repository root, on Windows:

```powershell
$env:CONVERSE_FULL_TRAIN_ARTIFACTS = 'H:/Python/ConverseNet/artifacts/full_fusion_run_001'
$env:TORCH_CUDA_ARCH_LIST = '12.0'  # Example for the measured RTX 5060 Ti; select your GPU.
$env:CONVERSE_MSVC_VERSION = '14.44'  # Optional; otherwise use the installed default toolset.
./experiments/training_full_spectrum_fusion/run.ps1 --build
./experiments/training_full_spectrum_fusion/run.ps1 --warm
```

Linux, or an already configured compiler shell:

```bash
export CONVERSE_FULL_TRAIN_ARTIFACTS="$PWD/artifacts/full_fusion_run_001"
export TORCH_CUDA_ARCH_LIST="12.0"  # Replace with the intended architecture.
python experiments/training_full_spectrum_fusion/loader.py --build
python experiments/training_full_spectrum_fusion/loader.py --warm
```

If `TORCH_CUDA_ARCH_LIST` is absent, the loader selects the visible devices'
compute capabilities. Use the same architecture setting when loading. The
default artifact directory is `ROOT/artifacts/training_full_spectrum_fusion`;
`CONVERSE_FULL_TRAIN_ARTIFACTS` overrides it. The loader resolves `ROOT` from its
own location, not the current working directory.

Build identity includes Torch/CUDA, architecture and generated architecture
flags, Python/platform/ABI, compiler flags, current core hashes, frozen source
hashes and the binary hash. Warm loading verifies them before registration.
It neither silently rebuilds a changed source tree nor overwrites measured
artifacts. A process can register only one version of `converse_full_training`;
use a new process for another binary. This source/binary audit is not a numerical
test or a performance result.

The default-integration namespace/registration changes modify core source bytes.
Warm loading or `--adopt` against an earlier manifest therefore fails its source
identity check. Retain the old build and measurements; build the current source
in a fresh artifact directory and a fresh Python process. Do not rewrite old
manifests to make them appear compatible with changed source.

To reuse a binary from the original research loader, explicitly use
`loader.py --adopt --artifacts /absolute/path/to/research/artifacts`, or call
`load_fusion(artifacts=path, adopt=True)`. This is read-only: it verifies the old
manifest's Torch/CUDA versions, binary and frozen-source hashes, requires those
source bytes to match the current core, and reads architecture flags from the
retained `build.ninja`. It neither rewrites the research manifest nor claims the
old manifest recorded the new loader's additional ABI/platform fields. Missing
build evidence or mismatching identities cause an error. Ordinary new builds
use the stricter schema described above.

## Explicit standalone research use

```python
from experiments.training_full_spectrum_fusion.loader import load_fusion
from experiments.training_full_spectrum_fusion.api import FusedRoute

# existing_ops is the verified Converse2D namespace selected for this experiment.
candidate = FusedRoute(existing_ops, load_fusion())
output = candidate.forward(x, prior, weight, bias, scale, eps=1e-5)
```

Only CUDA float32 calls with GradMode enabled and at least one differentiable
input use the isolated full-spectrum core. `no_grad`, `inference_mode`, inputs with
no requested gradients, CPU and other dtypes delegate to
`existing_ops.forward(..., 'v7')`. The fallback therefore retains the original
operator's supported behavior; calling it does not imply every fallback uses a
CUDA half-spectrum kernel. Routing is independent of `model.train()`/`eval()`.
The compatibility `variant` argument is accepted, while fallback is explicitly
v7. `candidate.counts` reports `fused_full` and `original_v7` calls. Other operator
methods are delegated to `existing_ops`.

The wrapper neither detaches differentiable spectra nor caches them across
optimizer updates. It preserves `prior is x` before FFT preparation. It does
not patch existing modules automatically, alter AMP/TF32 settings, or establish
CUDA Graph support.

## Validation scope

The standalone research results below predate default integration and retain
their original source/binary identity. They are not new measurements of the
integrated production bridge. Associated experiment reports must identify the
tested GPU, Torch build, shapes, broadcast modes, gradient needs
and model fixtures. Arbitrary complex spectral contracts, higher derivatives
and source identity are separate from the required spatial Python FP32
zero-margin noninferiority checks. Short model-step equality is separate from
long training convergence and real-data quality. Only measurements attached to
the same accepted source/binary identity should be quoted as performance results.

The final research run on the existing RTX 5060 Ti/Torch 2.11 CUDA 13 setup
reports 96/96 spatial output/gradient tensors byte-identical to Python FP32,
322 spectral/execution contract cases passing, and 4,002 full-model state
tensors byte-identical over B1/B4 three-step comparisons. These are the specific
reported fixtures, not a claim of universal equality or longer convergence.
The installed wrapper was also rebuilt from source and checked through both warm
and explicit adopt loading: each path reproduced all 96 tensors and its original
v7 fallback routing. Repeat validation on another environment before relying on it.

The final fused core additionally passed 48 zero/near-zero-kernel cases (228
spatial tensors byte-identical to Python). In four paired rounds on this machine,
full USRNet HR96/s3 Adam steps changed as follows; allocator peaks are allocated,
not total process memory:

| Batch | Full ATen ms/step | Fused ms/step | Paired speedup | Full ATen → fused GiB |
|---|---:|---:|---:|---:|
| 1 | 200.666 | 171.330 | 1.172× | 3.442 → 3.118 |
| 4 | 754.325 | 620.860 | 1.215× | 12.035 → 10.679 |

In that original study, the fused full-spectrum path was slower than the
half-spectrum path. The independent operator measurements improved 1.34–1.60× over Python
full-spectrum, while some isolated operator peaks increased; do not generalize
the model memory reduction to every operator. This is not a convergence claim.

Run the derivative/execution suite after a build, using a fresh output filename:

```powershell
python test/study_training_full_spectrum_fusion.py --name contracts_run_001.json
```

For explicit research-binary adoption, set `CONVERSE_FULL_TRAIN_ARTIFACTS` to
that research directory and add `--adopt`. The suite records finite-difference
tolerances separately from the zero-margin spatial precision admission gate.
The local source/binary evidence and complete measurement report are retained
in the sibling HPC workspace under `docs/conversenet_full_spectrum_cuda_fusion.md`.

Retain the old half-spectrum and failed numerical reports. Default integration
and the standalone wrapper do not by themselves establish general noninferiority,
other-platform correctness or long-term convergence. Further optimization must
preserve the Python FP32 accuracy baseline before speed gains are accepted.
