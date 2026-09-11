# Corrected Converse2D

The operator solves the same regularized circular-convolution problem using a
stable residual correction. It avoids subtracting two large spectra and dividing
their difference by a small regularizer. The input prior `x0` remains independent
of `x`, even when `scale=1`.

## Implementations

| Runtime variant | Inference | Training |
|---|---|---|
| `v2` | Full FFT, ATen reference implementation | Differentiable ATen |
| `v3`, `v4`, `v5`, `v6` | Shared full FFT implementation with fused CUDA correction | Differentiable ATen |
| `v7` (default) | Real FFT with correct Hermitian alias indexing and fused CUDA correction | Differentiable real-FFT ATen |

The old filenames are not restored. Historical v3–v6 labels are compatibility
aliases for one corrected implementation, rather than separate copies of the
same code. Residual correction eliminates zero insertion, so the old v4 upsampler
and v5 reshape-only FFT wrapper are unnecessary. v6's untracked postprocessing
has been removed from training. v7 performs division in the frequency domain;
both row and column indices are mirrored when recovering conjugate frequencies.

CUDA fusion is used under `torch.no_grad()` or `torch.inference_mode()`. Calling
`.eval()` alone does not disable autograd. All four tensor inputs support first
and second derivatives through the ATen path. CPU builds and CPU tensors use ATen.

Inputs must share device and dtype. Float32/float64 compute natively;
float16/bfloat16 compute in float32 and return the input dtype, including for
non-power-of-two FFT sizes. `eps` must be finite and positive.

Full FFT and real FFT need not be bitwise identical in float32. When comparing
complete models strictly, disable TF32 in the surrounding convolution layers:

```python
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
```

The library leaves these application-wide settings to the caller. On the supplied
SRResNet checkpoint, allowing TF32 amplified small operator rounding differences
to about 8e-4 in a smoke test; disabling it reduced the difference to about 1e-6.

## Installation

Install PyTorch and Ninja first. To build fused kernels, use a local CUDA toolkit
compatible with the installed PyTorch and a supported host compiler.

```sh
cd Converse2D
python -m pip install . --no-build-isolation
```

Set `CONVERSE2D_CPU_ONLY=1` to build without the custom CUDA kernels. The ATen
extension can still operate on CUDA tensors if the installed PyTorch supports it.
On Windows, use a Visual Studio Developer shell and set `CUDA_HOME`/`CUDA_PATH`
to the intended toolkit. This change was verified on RTX 5060 Ti with Python
3.12, PyTorch 2.11.0+cu130, CUDA Toolkit 13.0 and MSVC 14.44.

## Usage

```python
import torch
import torch_converse2d

# The existing six-argument call is preserved; v7 is the default.
with torch.inference_mode():
    out = torch.ops.converse2d.forward(x, x0, weight, bias, scale, eps)
    reference = torch.ops.converse2d.forward(x, x0, weight, bias, scale, eps, "v2")

from models.util_converse import Converse2D
layer = Converse2D(32, 32, 3, scale=2, backend="cuda", variant="v7").cuda()
```

## Cache behavior

Inference spectra and reduced denominators are cached with retained tensor
identity, storage pointer, mutation version, output grid, scale, CUDA stream and
inference-mode state. Regular in-place updates and `load_state_dict` invalidate
the cache. Autograd-enabled calls and inference tensors without version counters
bypass it. The cache has limits of 64 entries and 256 MiB of accounted tensors.

Avoid mutations through `.data`, raw pointers or external storage writes that
bypass PyTorch's version counter. After such a write, explicitly call:

```python
torch.ops.converse2d.clear_cache()
```

See [tests and profiling](../test/README.md) and the
[GPU validation report](../analysis/gpu_validation.md) for measured results.
