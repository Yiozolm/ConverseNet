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

## Cache

Inference spectra are cached by retained tensor identity, mutation version,
storage, output grid, scale, CUDA stream and inference-mode state. Regular
in-place updates and `load_state_dict` invalidate cached weights. Calls with
autograd enabled and inference tensors without version counters bypass caching.
The cache is bounded to 64 entries and 256 MiB of accounted tensors.

After raw storage writes or `.data` mutations that bypass PyTorch's version
counter, explicitly clear it:

```python
torch.ops.converse2d.clear_cache()
```

See [tests and profiling](../test/README.md) for verification commands.
