# Production source layout

The production implementation is organized by execution mode, then spectral
representation or scale. The original structural refactor of `2dcdbfc` is on
`codex/training-operator-optimization`. As of 2026-09-22, CUDA FP32 v7 training
defaults to the fused full-spectrum implementation; v7 inference retains the
half-spectrum path. Shared-s1, low-precision, warp-specialization and isolated
half-spectrum s2/s3 training candidates remain outside default dispatch.

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
    full_spectrum/
      production.cpp            production bridge, FP32 FFT preparation and IFFT
      full_fusion.h             internal full-spectrum declarations
      full_fusion.cpp           fused solve autograd and higher-order reference
      full_fusion.cu            generic kernels and scale1/scale2 includes
      scale1.cuh                s1 pointwise forward and analytic VJP fusion
      scale2.cuh                s2 ordered four-alias forward and VJP fusion
    autograd.cpp                retained internal half-spectrum autograd
    training_preparation.cpp    historical FP64 half-spectrum preparation
    spectrum_scope.cpp          historical scope API; default full path leaves it empty
    training_dispatch.cu        retained internal half-spectrum scale dispatch
    training_scale1.cu          retained half-spectrum specialized kernels
    training_generic.cu         retained half-spectrum generic kernels
    detail/                     half-spectrum device helpers
```

The full-spectrum training core dispatches s1 to `scale1.cuh`, which is included
inside the CUDA implementation namespace and compiled in the same translation
unit. Its pointwise forward/backward fusion retains the existing broadcast
reductions and complex scratch layout. Scale 2 uses `scale2.cuh` when LR width
is greater than one and the HR complex tensor fits within `INT32_MAX` bytes.
That kernel reproduces the ordered four-alias reduction while retaining separate
ATen broadcast reductions, complex denominator-gradient layout and higher-order
recomputation. Width-one and larger-byte-offset cases use generic alias
reductions, as do scales 3 and above. Both headers are quoted includes in the
existing CUDA translation unit and are covered by the dependency closure.

The scale-one wrapper materializes identical observation/prior spectra once
per call and omits the unused `gy` allocation/store for shared inputs while
retaining its register calculation. For `KB==B && KC==C`, a compile-time
adjoint specialization writes the final kernel gradient directly; its sixth
return slot lets the autograd wrapper bypass direct/prediction scratch and
the separate final pointwise kernel. Other shapes keep their broadcast
reductions. Complex `gd` storage, real-view stride, regularizer reduction,
explicit rounding boundaries and higher-order recomputation stay intact.
The CUDA translation unit explicitly includes `<type_traits>` for the
specialization tags; no new build source or public schema is added.

The retained internal half-spectrum path uses
`training_generic.cu` for s2/s3 and the historical
`s == 1 && H*W >= 65536` specialization for large s1. That threshold does not
select the current public CUDA FP32 v7 training backend. Gradient mode and input
gradient requirements select training, inference, or the ATen fallback;
`model.eval()` does not determine that choice.

The production bridge preserves independent FP32 kernel FFT graphs per call.
The old scope APIs and `reuse_training_spectra` model option remain compatible,
but default full-spectrum calls do not populate them and report `[0,0]` for
hits/misses. Scope nesting and exception cleanup remain supported. Reusing a
full-spectrum preparation graph would change gradient accumulation order and
requires its own precision validation. Before IFFT, the bridge restores a
noncontiguous prior spectrum's layout so the fused result follows the Python
reference's FFT layout rather than silently selecting a different transform
order.

## Build inputs

`build_config.py` is the shared, explicit source list for `setup.py` and
`test/extension_loader.py`. It computes the quoted-include closure, including
nested `.h`, `.cuh`, and the embedded full-spectrum `.cpp` file. The JIT manifest covers that closure, build
configuration, flags, compiler executable identity, and relevant environment
settings. Both C++ and CUDA compilation receive the revision fingerprint.
Changing a dependency invalidates `CONVERSE2D_SKIP_BUILD=1`; an already loaded
process must restart after build inputs change.

Keep candidate sources out of the explicit production list. Do not replace it
with a recursive glob. CUDA launchers and their device kernels live in the same
translation unit; no relocatable device code or extra device-link step is needed.
Use unique source basenames for predictable Windows object filenames.

`training/full_spectrum/production.cpp` includes `full_fusion.cpp` under
`CONVERSE2D_WITH_CUDA` and defines `CONVERSE_FULL_SPECTRUM_EMBEDDED` around the
include. The embedded form suppresses the standalone research
`TORCH_LIBRARY`/`PYBIND11_MODULE` registration; compile the bridge, not a second
copy of `full_fusion.cpp`, in production. The CPU build therefore never compiles
the CUDA-only autograd body. The full-spectrum C++ and CUDA implementation lives
in `converse2d::full_training`, avoiding helper-name collisions with retained
half-spectrum code when legacy source texts are amalgamated.

## Isolated experiments and historical evidence

The root `converse2d_kernels.cu` and `converse2d_training.cu` are include-only
compatibility amalgamations for experiments. They are **not** production build
inputs. An experiment must compile either an amalgamation or its constituent
units, never both. `converse2d_training.h` retains the older standalone spatial
adapter for research. Default full-spectrum autograd is embedded through
`training/full_spectrum/production.cpp`; `training/autograd.cpp` still implements
the internal half-spectrum entry.

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
and legacy-export contracts. Validate public full-spectrum training against the
Python FP32 reference, including output/gradient accuracy, weak regularization,
layout, shared inputs, selective gradients and higher derivatives. Retained
internal half-spectrum tests keep their original contracts. Scope tests check
empty counters and scoped/unscoped equality for the default route. Existing
CPU, inference, cache, CUDA Graph and spectral-I/O contracts still apply. Run
tests in fresh processes when switching CPU/CUDA builds. Historical profiling
and ablation scripts that identify half-spectrum kernels must not attribute
their old measurements or selectors to the new default full-spectrum route.

The local refactor plan and before/after evidence are maintained in the sibling
HPC repository under `docs/conversenet_refactor_plan.md` and
`artifacts/conversenet_refactor_20260921/`. The original strict full-model FP64
audit retains its known failure status; structural equivalence does not turn
that historical diagnostic into a passing precision claim.
