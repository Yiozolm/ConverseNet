# Converse2D FP32

Production inputs, weights, regularization parameters and outputs are `torch.float32`.
FFT spectra are `complex64`. Other tensor dtypes and historical v2–v6 variants
are rejected; `variant="v7"` remains for checkpoint/configuration compatibility.

With gradients enabled and at least one differentiable input, CUDA uses the
full-spectrum fused first-order solver. Kernel pad/roll/FFT, input FFT,
regularization and output IFFT remain differentiable FP32 operations. Every
call prepares its own kernel spectrum. Higher derivatives use differentiable
ATen operations. CPU training uses the full-spectrum ATen fallback.
CUDA training combines PSF padding/centering and its first-order adjoint into
direct indexed copies. Eligible s2 kernels without batch/channel broadcast
also form their kernel gradient in the fused VJP, preserving the existing
FP32 operation boundaries and broadcast fallback.

Some CUDA training FFTs do their adjacent copies inside the transform, through
cuFFT LTO callbacks compiled at run time with NVRTC:
- the input FFT reads real or circularly padded `x` as `(x, +0)`;
- the output IFFT applies ATen's `1/N` scaling at its store;
- the real/crop VJP reads the gradient straight into the padded spectrum.

Each callback plan is admitted only if a random probe reproduces the replaced
ATen expression bit for bit. Shapes whose callback plan uses a different FFT
algorithm keep ATen; on the RTX 5060 Ti this includes the unpadded 96x96 IFFT.
The kernel FFT and every VJP expression stay ATen. Planes with a side below 16,
plans cuFFT cannot create, and stream capture without a cached plan also use
ATen. Set `CONVERSE2D_FFT_CALLBACKS=0` to force ATen everywhere, or a comma list
of `real,circular,inverse,crop_embed` to enable only those sites.

For the public arbitrary-prior `forward`, `torch.no_grad()`,
`torch.inference_mode()`, or all-frozen inputs select the half-spectrum
inference path, including versioned fixed-kernel caches and
graph-owned caches. `model.eval()` alone does not disable gradients.
Frozen inputs under GradMode keep their existing ATen half-spectrum solve;
they can reuse fixed-kernel preparation without switching its rounding order.
Cache entries snapshot kernel sizes, strides and storage offset as well as its
identity/version. First-order backward skips unused gradient outputs and
reductions; higher-order derivatives keep the ATen fallback.

The Python `Converse2D` module establishes an exact nearest-upsampled prior.
For CUDA inference with kernel size 2 and scale 2, its private `_nearest_k2_s2`
entry instead uses a pixel-local FP32 solve. Two-component FP32 accumulation
and division correction meet the same frozen-FP32/independent-FP64 budgets;
device arithmetic never uses FP64. Regularization is computed in that kernel.
An exceptional power-of-two rescaling handles finite intermediates that would
otherwise overflow the compensated calculation. This is not a guarantee of
accurate results for every finite FP32 input.

Valid small padding is omitted because cropping complete 2x2 phase blocks
cancels it. This eligible module output is contiguous with storage offset zero,
instead of a view into the padded output. Invalid padding retains the original
errors; larger valid padding still uses pad/solve/crop. The fast path requires
normal-range FP32 `eps`; other positive finite values retain padded FFT
semantics. CPU, differentiable calls, arbitrary priors and `backend="pytorch"`
keep their existing routing, gradient order and higher derivatives.

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

## Optional CUDA Graph execution

`USRNetCUDAGraph(model, enabled=False)` calls the model directly on CPU or CUDA,
preserving autograd and the model's FP32 dtype contract. Set `runner.enabled = True`
to capture supported FP32 CUDA inference calls with the model in eval mode and
gradients disabled. Explicit construction defaults to enabled. Switching back to
`False` waits for pending replays and clears captured graphs.
