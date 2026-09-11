# Converse2D 1.0

[Release notes](https://github.com/Yiozolm/ConverseNet/releases/tag/v1.0.0)

A differentiable regularized circular-convolution solver using a stable residual
formula. One C++ implementation provides the operator and cache; one CUDA source
provides fused inference kernels.

## Build

Install PyTorch and Ninja, then build with a CUDA toolkit compatible with PyTorch:

```sh
python -m pip install ./Converse2D --no-build-isolation
```

On Windows, use a Visual Studio Developer shell and set `CUDA_HOME`/`CUDA_PATH`
to the intended toolkit. Set `CONVERSE2D_CPU_ONLY=1` to build without custom CUDA
kernels. That build can still use CUDA tensors through ATen when PyTorch supports
them, but bypasses GPU caching.

## Usage

```python
import torch
import torch_converse2d
from models.util_converse import Converse2D

layer = Converse2D(32, 32, 3, scale=2, backend="cuda").cuda().eval()
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
compute natively. Float16/bfloat16 compute in float32, retain their gradient
connections and return the input dtype, including for arbitrary FFT sizes.

## Backends

`backend="auto"` uses the extension on CUDA when available; `"cuda"` requires it;
`"pytorch"` selects the stable full-FFT Python reference.

The native extension has one solver: the stable residual simplification with
real FFT, fused CUDA correction, fused PSF preparation, dynamic power accumulation
and graph-safe caches. Autograd-enabled calls use the differentiable real-FFT
ATen implementation of the same formula. The Python full-FFT path is retained
as an independent reference and a fallback when the extension is unavailable.

Fusion requires `no_grad()` or `inference_mode()`; `.eval()` alone does not
disable autograd. Training supports first and second derivatives for all four
tensor inputs. Floating-point outputs need not match bit for bit across FFT paths.

USRNet's dynamic-kernel data module uses the same operator:

```python
from models.converse_usrnet import ConverseUSRNet

model = ConverseUSRNet(backend="cuda").cuda().eval()
# Existing state_dict parameter names are preserved.
image = torch.rand(1, 3, 32, 40, device="cuda")
k = torch.ones(1, 1, 7, 7, device="cuda") / 49
with torch.inference_mode():
    y = model(image, k, sf=2)
```

For strict complete-model comparisons, disable cuDNN/matmul TF32 in the caller;
the library leaves application-wide precision settings unchanged.

### Spectral preparation

CUDA inference writes the zero-padded, centered PSF in one kernel. For
uncached dynamic kernels, the correction kernel also accumulates `|FB|²` while
reading FB; it does not build and reconstruct a separate full power spectrum.
Fixed kernels still cache their prepared spectra and denominators. CPU and
autograd-enabled calls keep their differentiable ATen preparation paths.

The IFFT normalization remains after the transform. Moving it before the IFFT
was rejected because it worsened near-underflow precision. No fast-math or
reduced-precision arithmetic is enabled by these changes. Precision and performance
measurements are summarized in the GitHub release notes.

## CUDA Graph inference

For repeated USRNet inference, use the optional bounded graph runner. Before
enabling graphs, rebuild the extension if needed: old binaries do not have the
capture-aware spectrum cache. Load the checkpoint and move the model to CUDA
before the first enabled call.

CUDA Graph is optional: `model(...)` always uses ordinary execution. A shared
call site can switch modes through `enabled`:

```python
from models.cuda_graph import USRNetCUDAGraph

use_cuda_graph = False  # Application configuration; no capture when disabled.
run = USRNetCUDAGraph(model, enabled=use_cuda_graph, max_graphs=1)
with torch.inference_mode():
    y = run(image, k, scale=2)

run.enabled = True   # Capture on the next supported inference call.
run.enabled = False  # Wait for pending replays and release graph caches.
```

When disabled, the wrapper calls the model directly, preserving CPU execution,
autograd and autocast. The graph constraints below apply when enabled. Explicit
`USRNetCUDAGraph(model)` construction still defaults to enabled for compatibility.

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

## Migrating to 1.0

Remove the `variant` keyword from model constructors and the trailing version
string from direct operator calls. There is no version selector or legacy
full-spectrum CUDA branch. Rebuild the extension after updating the source;
previously built binaries still expose the old ABI. Existing checkpoint keys
are unchanged. Historical reports retain their original version labels, and
frozen comparison code is loaded only by tests into a separate namespace.

After raw storage writes or `.data` mutations that bypass PyTorch's version
counter, explicitly clear it:

```python
torch.ops.converse2d.clear_cache()
```

See [tests and profiling](../test/README.md) for verification commands.
