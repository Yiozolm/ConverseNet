# Converse2D

A differentiable regularized circular-convolution solver using a stable residual
formula. One C++ implementation provides the operator and cache; CUDA sources
provide fused inference and FP32 training kernels.

## Build

Install PyTorch and Ninja, then build with a CUDA toolkit compatible with PyTorch:

```sh
python -m pip install ./Converse2D --no-build-isolation
```

On Windows, use a Visual Studio Developer shell and set `CUDA_HOME`/`CUDA_PATH`
to the intended toolkit. Set `CONVERSE2D_CPU_ONLY=1` to build without custom CUDA
kernels. That build can still use CUDA tensors through ATen when PyTorch supports
them, but bypasses GPU caching.

After updating from the half-spectrum training implementation, rebuild the
extension with the command above and restart any Python process that has loaded
the previous binary. The full-spectrum training route is enabled by default;
no experimental loader or opt-in flag is required. Checkout JIT users should
rebuild through `test/extension_loader.py` before enabling
`CONVERSE2D_SKIP_BUILD=1` again.

## Usage

```python
import torch
import torch_converse2d
from models.util_converse import Converse2D

layer = Converse2D(32, 32, 3, scale=2, backend="cuda", variant="v7").cuda().eval()
x = torch.randn(1, 32, 128, 128, device="cuda")
with torch.inference_mode():
    y = layer(x)
```

The direct operator accepts an independent prior `x0`, including at scale=1:

```python
y = torch.ops.converse2d.forward(x, x0, weight, bias, scale, eps)
```

| Input | Shape |
|---|---|
| `x` | `(B, C, H, W)` |
| `x0` | `(B, C, H*scale, W*scale)` |
| `weight` | `(1 or B, 1 or C, kh, kw)` |
| `bias` | `(1, C, 1, 1)` |

Inputs share device and dtype; the kernel must fit the output dimensions.
`scale` is a positive integer and `eps` is finite and positive. Float32/float64
retain their activation FFT and solve precision. Float16/bfloat16 compute in float32, retain their gradient
connections and return the input dtype, including for arbitrary FFT sizes.

## Backends

`backend="auto"` uses the extension on CUDA when available; `"cuda"` requires it;
`"pytorch"` selects the stable full-FFT Python reference.

| Variant | Inference | Training |
|---|---|---|
| `v7` (default) | Real FFT/half spectrum with fused CUDA correction | FP32 CUDA full-spectrum fused solve/VJP; ATen otherwise |
| `v2` | Full-FFT ATen reference | Differentiable ATen |
| `v3`–`v6` | Compatibility aliases for shared full-FFT fused code | Differentiable ATen |

The FP32 CUDA v7 training backend is selected when gradients are enabled and
any input needs a gradient; `.eval()` alone does not disable autograd. Its
complex64 full-spectrum solve uses an analytic first-order backward, with
differentiable ATen recomputation for higher derivatives. Kernel preparation
uses FP32 pad/roll followed by `fft2`; observation/prior `fft2` and the final
`ifft2(...).real` also retain their FP32 autograd paths. CUDA fusion preserves
the reference's alias reduction order and broadcast-reduction boundaries. FFT
layout is retained at the inverse-transform boundary for noncontiguous prior spectra.

Scale 1 uses dedicated full-spectrum pointwise fusion in `scale1.cuh`, covering
both shared and independent priors. It combines forward preparation/output and
the pointwise backward stages while retaining separate broadcast reductions,
their layouts and the higher-order fallback. Scale 2 uses `scale2.cuh` when LR
width is greater than one and the HR complex tensor occupies at most
`INT32_MAX` bytes. It combines each four-alias reduction with forward/backward
pointwise work while preserving the reduction order and the separate broadcast
reductions. Width-one and larger-byte-offset cases retain the generic path, as
do scales 3 and above. These dispatches do not use the historical half-spectrum
scale-one area threshold.

Shared scale-one inputs reuse one forward layout materialization and omit the
unused independent-observation gradient buffer. Their combined gradient keeps
its existing addition order. When `KB==B && KC==C`, the pointwise backward
also completes the kernel gradient, avoiding the direct/prediction scratch
buffers and a separate final kernel. Broadcast cases retain their reductions;
the complex denominator-gradient buffer and its real-view stride remain
unchanged for the regularizer reduction. These changes introduce no spectrum
cache or reduced-precision arithmetic. The implementation and validation
record are tracked in the sibling [HPC report](../../HPC/docs/conversenet_full_spectrum_leet.md).

Trainable spectra are rebuilt on every call so optimizer updates cannot reuse
stale or detached kernels. CPU, FP64 and low-precision training retain their
ATen paths. Inference fusion requires `no_grad()` or `inference_mode()` and v7
inference continues to use the half spectrum. All four tensor inputs support
first and second derivatives. Training and inference implement the same
mathematical operator, but their floating-point outputs need not be identical.

`ConverseUSRNet(..., reuse_training_spectra=True)` and the private training scope
APIs remain accepted for compatibility. **The default full-spectrum path does
not reuse spectra**, even with this option enabled: each call retains its own
FP32 preparation graph, and scope hit/miss counts are `[0,0]`. Sharing that graph
would change gradient accumulation order and needs separate precision
validation. Nested scopes and exceptions still clean up correctly. The previous
FP64 half-spectrum preparation, large-s1 specialization and reuse experiments
remain available as historical/internal code; their thresholds and speedups do
not describe current default training. Their original results, including open
precision gates, are retained in [refinement results](../docs/training_refinements.md).

Further training optimization must preserve the Python FP32 accuracy baseline;
enabling the default route does not establish long-term convergence or accuracy
on every GPU, Torch version and input shape.

USRNet's dynamic-kernel data module uses the same operator:

```python
from models.converse_usrnet import ConverseUSRNet

model = ConverseUSRNet(backend="cuda", variant="v7").cuda().eval()
# Existing state_dict parameter names are preserved.
image = torch.rand(1, 3, 32, 40, device="cuda")
k = torch.ones(1, 1, 7, 7, device="cuda") / 49
with torch.inference_mode():
    y = model(image, k, sf=2)
```

For strict complete-model comparisons, disable cuDNN/matmul TF32 in the caller;
the library leaves application-wide precision settings unchanged.

### Spectral preparation in v7 inference

CUDA v7 inference writes the zero-padded, centered PSF in one kernel. For
uncached dynamic kernels, the correction kernel also accumulates `|FB|²` while
reading FB; it does not build and reconstruct a separate full power spectrum.
Fixed kernels still cache their prepared spectra and denominators. Grad-enabled
CUDA FP32 v7 calls needing gradients use the full-spectrum training preparation
described above; CPU and other variants/dtypes retain their existing paths.

The IFFT normalization remains after the transform. Moving it before the IFFT
was rejected because it worsened near-underflow precision. No fast-math or
reduced-precision arithmetic is enabled by these changes. See
[precision and performance measurements](../docs/nsight/spectral_io_optimization.md).

## CUDA Graph inference

For repeated USRNet inference, use the optional bounded graph runner. Rebuild
the extension first: old binaries do not have the capture-aware spectrum cache.
Load the checkpoint and move the model to CUDA before constructing the runner.

```python
from models.cuda_graph import USRNetCUDAGraph

runner = USRNetCUDAGraph(model, max_graphs=1)
with torch.inference_mode():
    y = runner(image, k, scale=2)  # First call warms up and captures.
    y_next = runner(next_image, next_kernel, scale=2)
runner.clear()  # Wait for pending work and release graph-owned buffers.
```

Input values can change freely; each call returns its own output tensor. Shapes,
batch, kernel broadcast shape, dtype, device and scale select an LRU entry.
Parameter replacement, normal in-place updates, checkpoint loading, built-in
module configuration and TF32/backend changes invalidate existing graphs.
FP32/FP64 CUDA inference is supported with autocast disabled and all modules in
eval mode. Training still calls `model(...)` directly. Hooks and nested graph
capture are rejected. Keep model creation outside `inference_mode()` so its
parameters have mutation version counters.

Use `max_graphs` to limit the number of private graph pools; the limit counts
graphs, not bytes. Shape changes can incur a warmup/capture pause and additional
memory. Sequential calls from different CUDA streams are ordered by events;
callers must make their input data ready on the calling stream as usual. Do not
edit the model or run unrelated CUDA work concurrently with graph capture. After
raw `.data`/storage edits or custom Python behavior changes, explicitly clear
both the runner and the operator cache. These constraints follow
[PyTorch CUDA Graph semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).

## Cache

Inference spectra are cached by retained tensor identity, mutation version,
storage, output grid, scale, CUDA stream and inference-mode state. Regular
in-place updates and `load_state_dict` invalidate cached weights. Calls with
autograd enabled and inference tensors without version counters bypass caching.
The cache is bounded to 64 entries and 256 MiB of accounted tensors.

During CUDA Graph capture, spectrum lookup and insertion both bypass this global
cache. Direct captures compute spectra inside their private pool. The runner
uses a separate warmup cache and retains its fixed-weight spectra for the graph's
lifetime, avoiding their recomputation on each replay. Dynamic spectra are still
computed inside the graph. Neither path depends on eager-cache entries, so later
eviction cannot invalidate captured pointers. The extension exposes
`torch.ops.converse2d.supports_cuda_graphs()` for detecting graph cache support;
the runner also checks for its graph-owned cache operations in the loaded binary.

After raw storage writes or `.data` mutations that bypass PyTorch's version
counter, explicitly clear it:

```python
torch.ops.converse2d.clear_cache()
```

See [tests and profiling](../test/README.md) for verification commands.

## Source organization

See [REFACTOR.md](REFACTOR.md) for the training/inference and scale layout, explicit build inputs, and compatibility with isolated experiments.
