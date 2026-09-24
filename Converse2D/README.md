# Converse2D FP32

Production inputs, weights, regularization parameters and outputs are `torch.float32`.
FFT spectra are `complex64`. Other tensor dtypes and historical v2–v6 variants
are rejected; `variant="v7"` remains for checkpoint/configuration compatibility.

With gradients enabled and at least one differentiable input, CUDA uses the
full-spectrum fused first-order solver. Kernel pad/roll/FFT, input FFT,
regularization and output IFFT remain differentiable FP32 operations. Every
call prepares its own kernel spectrum. Higher derivatives use differentiable
ATen operations. CPU training uses the full-spectrum ATen fallback.

`torch.no_grad()`, `torch.inference_mode()`, or all-frozen inputs select the
half-spectrum inference path, including versioned fixed-kernel caches and
graph-owned caches. `model.eval()` alone does not disable gradients.

## Build

Install PyTorch with the appropriate CUDA build and a compatible CUDA toolkit
and host C++ compiler. From this directory:

```sh
python setup.py build_ext --inplace
python -m pip install . --no-build-isolation
```

For a CPU-only extension, set `CONVERSE2D_CPU_ONLY=1`. On Windows,
`tools/run.ps1` initializes MSVC and CUDA; set `CONVERSE_PYTHON` to select Python
and `CONVERSE_MSVC_VERSION` if a particular installed toolset is needed.
Set `TORCH_CUDA_ARCH_LIST` for the GPU(s) being deployed. Restart Python after
rebuilding; an already loaded extension cannot be replaced in place.

From the repository root, run the checked JIT build and tests:

```sh
python -m unittest discover -s test -p 'test_*.py' -v
```

Windows equivalent: `./tools/run.ps1 -m unittest discover -s test -p 'test_*.py' -v`.
Build artifacts and source/binary fingerprints stay under `.build/`.

## Use

```python
import torch
import torch_converse2d

x = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
kernel = torch.rand(1, 3, 3, 3, device="cuda", requires_grad=True)
bias = torch.zeros(1, 3, 1, 1, device="cuda", requires_grad=True)
y = torch.ops.converse2d.forward(x, x, kernel, bias, 1)  # full spectrum
y.square().mean().backward()
with torch.inference_mode():
    prediction = torch.ops.converse2d.forward(x, x, kernel, bias, 1)  # half spectrum
```

The original `models.converse_core.converse2d_reference` and explicit
`backend="pytorch"` remain full-spectrum references. FP64 is supported only by
that independent Python reference for numerical checking. Production modules
require FP32; AMP and training-spectrum reuse are outside this release.

See [release validation](../docs/fp32_release.md) for measured scope and the
pre-cleanup source snapshot. No new speed or long-run convergence claim is made.
