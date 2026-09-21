# Production source layout

The production implementation is organized by execution mode, then scale.
This is a structural refactor of `2dcdbfc` on `codex/training-operator-optimization`;
it does not enable the shared-s1, low-precision, or warp-specialization candidates.

```text
torch_converse2d/
  converse2d.cpp                 operator schemas and registration only
  operator.cpp                  validation, dtype, FFT orchestration, backend choice
  common/spectrum_ops.h          self-contained ATen spectrum helpers
  reference/                    differentiable ATen spectral fallback
  inference/
    cache.cpp                   eager and graph cache state, defined once
    inference_preparation.cpp   ordinary kernel spectrum and cached power
    inference_prepare.cu        fused PSF pad/roll
    inference_dispatch.cu       scale dispatch and workspace allocation
    inference_scale1.cu
    inference_scale2.cu
    inference_scale3.cu
    inference_generic.cu
    detail/                     inference templates and device helpers
  training/
    autograd.cpp                spectral validation, first/higher-order autograd
    training_preparation.cpp    differentiable FP64 kernel FFT and per-use cast
    spectrum_scope.cpp          opt-in per-forward differentiable reuse
    training_dispatch.cu        unchanged scale1 threshold and workspace policy
    training_scale1.cu          existing specialized forward and backward
    training_generic.cu         existing generic forward and backward
    detail/                     training-specific device helpers
```

Training scale 2 and 3 currently use `training_generic.cu`. Add separate files
when there is an independently validated implementation; do not duplicate the
generic kernels just to mirror the inference directory. The production scale1
threshold remains `s == 1 && H*W >= 65536`. Gradient mode and input gradient
requirements select training, inference, or the ATen fallback; `model.eval()`
does not determine that choice.

## Build inputs

`build_config.py` is the shared, explicit source list for `setup.py` and
`test/extension_loader.py`. It computes the quoted-include closure, including
nested `.h` and `.cuh` files. The JIT manifest covers that closure, build
configuration, flags, compiler executable identity, and relevant environment
settings. Both C++ and CUDA compilation receive the revision fingerprint.
Changing a dependency invalidates `CONVERSE2D_SKIP_BUILD=1`; an already loaded
process must restart after build inputs change.

Keep candidate sources out of the explicit production list. Do not replace it
with a recursive glob. CUDA launchers and their device kernels live in the same
translation unit; no relocatable device code or extra device-link step is needed.
Use unique source basenames for predictable Windows object filenames.

## Isolated experiments and historical evidence

The root `converse2d_kernels.cu` and `converse2d_training.cu` are include-only
compatibility amalgamations for experiments. They are **not** production build
inputs. An experiment must compile either an amalgamation or its constituent
units, never both. `converse2d_training.h` retains the older standalone spatial
adapter for research; production autograd is in `training/autograd.cpp`.

`build_config.legacy_sources()` exports self-contained old-name source texts
from the **current** modules. Current s1 ablations and copied checkout builds use
this export rather than copying an incomplete set of facade files. Its text
hash identifies an amalgamation, not the on-disk facade. New research reports
should also record `test.extension_loader.production_source_hashes()`.

The nearest and warp experimental loaders retain their isolated registration
and include facades, with full production dependency fingerprints. These remain
research backends, not default dispatch choices.

Historical Git/snapshot loaders and old report validators are deliberately not
rewritten. In particular, `probe_shared_filter_workspace.py` and
`experiments/kernel_precision/extension.py` contain exact monolithic-source
patches. They are historical research scripts, not supported current-layout
entry points; reproduce them with the original source identity recorded in their
reports. Their checked patch contexts must fail rather than silently apply a
different optimization. Porting those candidate algorithms is a separate change.

Do not regenerate historical hashes or rewrite frozen HPC experiment snapshots.
For new snapshots include nested `.cuh` files and `build_config.py` in addition to
the C++/CUDA sources. The existing HPC 20260920 snapshot remains immutable.

## Validation

Run `python test/test_build_layout.py` for explicit-source, transitive-header,
and legacy-export contracts. Existing CPU/CUDA correctness, training fusion,
scale3, scope, cache, CUDA Graph, and spectral-I/O tests remain the numerical
contracts. Run tests in fresh processes when switching CPU/CUDA builds.

The local refactor plan and before/after evidence are maintained in the sibling
HPC repository under `docs/conversenet_refactor_plan.md` and
`artifacts/conversenet_refactor_20260921/`. The original strict full-model FP64
audit retains its known failure status; structural equivalence does not turn
that historical diagnostic into a passing precision claim.
