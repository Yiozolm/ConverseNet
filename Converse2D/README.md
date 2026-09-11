# Converse2D

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
compute natively. Float16/bfloat16 compute in float32, retain their gradient
connections and return the input dtype, including for arbitrary FFT sizes.

## Backends

`backend="auto"` uses the extension on CUDA when available; `"cuda"` requires it;
`"pytorch"` selects the stable full-FFT Python reference.

| Variant | Inference | Training |
|---|---|---|
| `v7` (default) | Real FFT with fused CUDA correction | Differentiable real-FFT ATen |
| `v2` | Full-FFT ATen reference | Differentiable ATen |
| `v3`–`v6` | Compatibility aliases for shared full-FFT fused code | Differentiable ATen |

Fusion requires `no_grad()` or `inference_mode()`; `.eval()` alone does not
disable autograd. Training supports first and second derivatives for all four
tensor inputs. Floating-point outputs need not match bit for bit across FFT paths.

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
