# Isolated nearest-prior training candidate

This is an unintegrated FP32 training experiment, supporting `s=1` and `s=3`.
Production sources and the earlier `test/probe_nearest_training.py` remain
unchanged. The earlier ATen prior experiment passed its spatial FP64 gates but
did not improve complete forward/backward latency; it is a control here, not
evidence that this CUDA candidate is correct or faster.

Run in the existing CUDA build environment:

```powershell
F:/anaconda3/envs/vllm/python.exe test/probe_nearest_fused_training.py --cpu-adjoint-check
./experiments/training_speed/run.ps1 test/probe_nearest_fused_training.py
```

The second command builds and runs CUDA, and must be scheduled by the owner of
the shared GPU. No GPU or compilation was run while preparing these sources.
The default output is `artifacts/training_research/nearest_fused_training.json`;
existing reports are not overwritten. The loader creates a content-addressed
copy under `.build/training_nearest/`, substitutes a unique operator namespace
and external symbols, and records original/generated source hashes and flags.
It moves only the exact known wrapper `CL` options into explicit C++ flags for
the duration of `cpp_extension.load`; real compiler/build errors propagate.

## API and graph ownership

```
_training_nearest_spectral(y, k, regularizer, phaseH, phaseW, H, W, s)
```

`y` is `[B,C,H,W//2+1]`; `k` is `[KB,KC,sH,sW//2+1]`, where each of `KB`
and `KC` is either one or the corresponding input dimension. The regularizer
is `[1,C,1,1]`. The phases are one-dimensional complex arrays of length `sH`
and `sW`, with `requires_grad=False`; trainable phases are explicitly rejected.
All inputs share a CUDA device. FP32 execution uses complex64 and float32;
the internal operator also accepts complex128/float64 for gradient checks.

FFT/IFFT, kernel preparation and lambda parameterization are outside the
custom Function and retain normal autograd. The Python adapter uses exactly
the frozen prototype's differentiable FP64 kernel preparation followed by a
contiguous complex64 cast. Phases are generated on every invocation using
FP64 trigonometry and cast to complex64; this work is included in timing.
There is no cross-call parameter, phase or spectrum cache.

The Function saves `y,k,lambda,phaseH,phaseW,q,d`, never an HR prior. Its
ordinary first-order backward uses CUDA. `create_graph=True` rebuilds the
ATen nearest half spectrum and the reference solve from saved differentiable
inputs. Thus higher derivatives deliberately materialize a prior and may use
the ATen gather/scatter path; first-order memory/determinism claims do not
apply to this fallback.

## Mapping and complex VJP

Let `N=sH`, `M=sW`, `A[h,w]=phaseH[h]*phaseW[w]`, and
`phaseN[k]=sum(a=0..s-1) exp(-2*pi*i*k*a/N)`. There is no extra `s*s`
normalization in the nearest-prior transform. For an HR **stored** frequency
`0<=w<=floor(M/2)`:

```
lh = h % H; lw = w % W
if lw <= W//2: P[h,w] = A[h,w] * y[lh,lw]
else:         P[h,w] = A[h,w] * conj(y[(-lh)%H,W-lw])
```

For an HR **missing** column, the implementation first maps
`(h,w) -> ((N-h)%N,M-w)`, evaluates this stored-frequency expression, and
conjugates the entire result. This order matches
`full_spectrum(materialized_half_prior)` for arbitrary complex boundary
values. Assuming that arbitrary `y` has real-FFT Hermitian boundary values
would silently change the operator checked by complex gradcheck.

The original spectral core supplies its direct observation VJP `r` and its
hypothetical prior VJP at each stored HR frequency:

```
gp[h,w] = g[h,w] - conj(k[h,w]) * alias_adjoint(r)[h,w]
```

One CUDA thread owns each stored LR frequency `(h,w)` and produces
`dy = r + N^H gp`. It loops over `a=0..s-1`, then `b=0..s-1`, evaluating
direct before mirrored contributions at each pair:

```
direct:   hh=h+a*H;       ww=w+b*W
          if ww<=M//2: sum += conj(A[hh,ww]) * gp[hh,ww]
mirrored: hh=(-h)%H+a*H;  ww=W-w+b*W
          if w>0 and 2*w!=W and ww<=M//2:
              sum += A[hh,ww] * conj(gp[hh,ww])
```

LR DC and even Nyquist columns have only the direct branch. The two sets
otherwise enumerate every HR stored frequency mapping to that LR element
exactly once. For `s=1`, phases are one and this reduces to `dy=r+gp`.
There are no floating atomics or scatter writes. `gp` is recomputed at the
point of consumption, so no HR `gp` buffer is allocated.

The kernel VJP retains the original expression and alias adjoint, replacing
only its prior read with `P(y)`. A thread owns one filter coefficient and
accumulates broadcast examples/channels in the original fixed `b` then `c`
order. The lambda gradient retains the original `gd.sum({0,2,3},true)`.

## Memory and performance hypothesis

Write `L=BC*H*(W//2+1)` and `R=BC*sH*(sW//2+1)`. The candidate retains
the HR output and its upstream gradient, the kernel spectrum and kernel
gradient, LR `q/r/d/gd/dy`, and small phase vectors. It removes the HR prior
and HR prior VJP, each `8R` bytes in complex64. Compared with the spatial
control it also removes explicit nearest interpolation, its HR activation
RFFT and that FFT's autograd backward. Compared with the ATen spectral
control it removes the HR gather/where/phase-multiply intermediates and their
scatter-based reverse path.

For `B32,C32,H64,W80,s3`, `R=23,789,568`: each complex64 HR tensor is
190,316,544 bytes (181.5 MiB). Avoiding prior/gp materialization saves their
individual 181.5 MiB storages and removes associated writes. Eliminating a
materialized `gp` write plus its subsequent read alone removes `16R` logical
bytes, about 380.6 MB. These are tensor/traffic accounting terms, **not** an
assertion that all allocations overlap or that actual peak falls by their
sum. `dy` is now separate from `r`, costing another `8L` bytes (20.5 MiB).

Prior reads are replaced with LR spectrum and phase reads, not free values.
The LR gather recomputes `gp`, performs integer modulo/division and reads
`g/k/r` again; separate `dk` may duplicate some work. Small batches may be
limited by phase construction and launches. Measured speed/traffic can
therefore disprove the hypothesis. No fast math, precision reduction or
change to the solve denominator was introduced.

## Fixed gates and reproducibility

The new runner imports the frozen prototype's twelve fixtures without
changing seeds, scales, weak regularization, eps, lambda or upstream sizes.
It compares production, ATen spectral and this candidate against an
independent FP64 **spatial nearest + full FFT** reference:

- Normal output: absolute/relative tolerance `3e-5/3e-5`.
- All `dx/dw/db`: `5e-5/5e-5`.
- Weak output: `1e-6/1e-5`; gradient budget unchanged.
- Additional complex128 spectral checks include arbitrary complex values,
  four broadcast combinations, singleton/odd/even shapes, conjugate strided
  input, every nonempty `requires_grad` combination and s1 control. Direct
  spectral reference comparisons use output `1e-9/1e-9`, VJP `1e-8/1e-8`.
- Complex128 gradcheck/gradgradcheck for s1 and odd/even s3 use `1e-5/1e-4`.

An untimed profiler must show one LR activation RFFT, all five new CUDA
kernels, and no interpolation/index-select/scatter operation. Three repeated
calls on the unchanged B32 fixture must return bitwise-identical output and
all three spatial VJPs; source hashes and actual global deterministic setting
are recorded separately from `cudnn.deterministic=True`. Global deterministic
algorithms are not enabled by this script. Any gate failure blocks timing.

Formal timing uses one fixture at a time, five warmups and four alternating
rounds of twenty complete spatial FWD+VJP calls. It includes phase generation,
FP64 kernel preparation and all FFTs; no profiler/loss/optimizer is active.
Report synchronized wall, CUDA events and PyTorch allocated/reserved peaks
separately. There is no whole-model or convergence speed claim.

CPU-only preparation check passed seven arbitrary-complex prior/adjoint
cases, including singleton, odd/even and s1: largest gather-adjoint error
`1.78e-15`, canonical full-prior read error zero. Python AST passed. These
checks do not establish compiled CUDA correctness or performance.

Source basis SHA256 at creation:

```
converse2d_training.cu  f94db691427535aa630833bec39247dccb5f7c4510d50467704703c8734119a1
converse2d_training.h   a541350bdd3e595a74e05440a42c893ce5e733ddc81c2b662019085f9af2e767
probe_nearest_training.py da803392a7c6084bf3558ff3f050a47a0e7e04eb63a19d1559f915eb28a15e4f
probe_training_s1_shapes.py ddd47e99af0ace42fcff98f3733f44db2a3d18cd3e35f70662537954259c5ac7
```
