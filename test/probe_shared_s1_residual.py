"""ATen-only SAME-prior s1 residual; no production or CUDA source changes.

    python test/probe_shared_s1_residual.py --cpu-proof
    ./experiments/training_speed/run.ps1 test/probe_shared_s1_residual.py --cuda-accuracy --output artifacts/native_deconv_target/shared_s1_residual_accuracy.json

For prior IS x and s=1, C=conj(K)*(1-K)/(|K|^2+lambda), output is
x + irfft2(C*rfft2(x)). The identity term avoids an activation FFT/IFFT
roundtrip. Kernel FFT is differentiable FP64 -> complex64 for FP32 input;
activation FFT, residual solve and IFFT remain FP32. FP64 CPU proofs retain
complex128. No trainable values are cached or detached, no lambda is changed.

GPU execution, when explicitly requested, ONLY evaluates accuracy on the
original fixed s1/weak/pretrained/native fixtures. Each tensor's max_abs and
relative L2 error against the identical FP64 reference must be <= the original
Python FP32 baseline's corresponding errors, with NO margin. FP64 is the
shared error oracle, not a new pointwise tolerance gate. There is no timing.

CPU-only manual VJP proof for residual spectrum R=C*Y:
t=sum_B(conj(Y)*G) for shared KB; gY=G*conj(C);
gK=conj(t)/d-2*K*(real(t)+real(conj(t)*C))/d;
g_lambda=-real(conj(t)*C)/d. Reduce broadcast C/batch/frequency dimensions.
The spatial identity derivative is supplied by the outer x addition.
The GPU candidate itself uses ONLY ordinary ATen autograd, not this proof VJP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def correction_spectrum(y,k,regularizer):
    denominator = k.real.square()+k.imag.square()+regularizer
    correction = k.conj()*(1-k)/denominator
    return correction*y


def residual_core(x,weight,bias,eps):
    import torch
    from diagnose_boundary_precision import kernel_fft64
    spectrum_dtype = torch.complex128 if x.dtype==torch.float64 else torch.complex64
    k = kernel_fft64(weight,*x.shape[-2:]).to(dtype=spectrum_dtype,memory_format=torch.contiguous_format)
    regularizer = torch.sigmoid(bias-9.)+eps
    y = torch.fft.rfft2(x)
    residual = torch.fft.irfft2(correction_spectrum(y,k,regularizer),s=x.shape[-2:])
    return x+residual


def spatial(x,prior,weight,bias,scale=1,eps=1e-5):
    import torch
    from models.converse_core import validate_inputs
    if scale!=1 or prior is not x:
        raise ValueError("Residual candidate requires scale=1 and prior IS x; independent prior is unsupported")
    validate_inputs(x,prior,weight,bias,scale,eps)
    if x.dtype not in (torch.float32,torch.float64) or torch.is_autocast_enabled(x.device.type):
        raise ValueError("Residual candidate accepts only FP32/FP64 without autocast")
    return residual_core(x,weight,bias,eps)


def method(mode="residual",padding=0,eps=1e-5):
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    if mode not in ("residual","reference"):
        raise ValueError(mode)
    def forward(x,weight,bias):
        value = F.pad(x,(padding,)*4,mode="circular") if padding else x
        output = (spatial(value,value,weight,bias,1,eps) if mode=="residual" else
                  converse2d_reference(value,value,weight,bias,1,eps))
        return output[...,padding:-padding,padding:-padding] if padding else output
    return forward


def cpu_proof():
    import torch
    from probe_spatial_nonoverlap import matrix_solve
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU proof must run before initializing CUDA")
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(67218)

    class FormulaVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx,y,k,regularizer):
            if y.device.type!="cpu":
                raise RuntimeError("The handwritten proof VJP is CPU-only")
            ctx.save_for_backward(y,k,regularizer)
            return correction_spectrum(y,k,regularizer)

        @staticmethod
        def backward(ctx,g):
            y,k,regularizer = ctx.saved_tensors
            d = k.real.square()+k.imag.square()+regularizer
            c = k.conj()*(1-k)/d
            t = (y.conj()*g).sum_to_size(c.shape)
            cross = (t.conj()*c).real
            gy = g*c.conj()
            gk = (t.conj()/d-2*k*(t.real+cross)/d).sum_to_size(k.shape)
            gl = (-cross/d).sum_to_size(regularizer.shape)
            return gy,gk,gl

    spectral = []
    for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
        y = torch.randn(2,3,2,3,generator=rng,dtype=torch.complex128).requires_grad_()
        k = torch.randn(kb,kc,2,3,generator=rng,dtype=torch.complex128).requires_grad_()
        l = (torch.rand(1,3,1,1,generator=rng,dtype=torch.float64)+.2).requires_grad_()
        g = torch.randn(y.shape,generator=rng,dtype=y.dtype)
        actual = FormulaVJP.apply(y,k,l)
        expected = correction_spectrum(y,k,l)
        a = torch.autograd.grad(actual,(y,k,l),g)
        e = torch.autograd.grad(expected,(y,k,l),g)
        for av,ev in zip(a,e):
            torch.testing.assert_close(av,ev,atol=1e-12,rtol=1e-12)
        torch.autograd.gradcheck(FormulaVJP.apply,(y,k,l),fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
        torch.autograd.gradgradcheck(FormulaVJP.apply,(y,k,l),fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
        spectral.append(dict(KB=kb,KC=kc,gradcheck=True,gradgradcheck=True,
                             max_gradient_difference=max((av-ev).abs().max().item() for av,ev in zip(a,e))))
    # Independent pixel-space matrix/linear solve, not another FFT formula.
    pixels = []
    configs = [("normal",kb,kc,1.,1e-3) for kb,kc in ((1,1),(1,2),(2,1),(2,2))]
    configs += [(f"weak_{amplitude:g}",1,1,amplitude,1e-8) for amplitude in (0.,1e-6,1e-3)]
    for name,kb,kc,amplitude,eps in configs:
        x = torch.randn(2,2,3,4,generator=rng,dtype=torch.float64)
        k = torch.rand(kb,kc,3,3,generator=rng,dtype=torch.float64)/9
        b = torch.randn(1,2,1,1,generator=rng,dtype=torch.float64)
        g = torch.randn(x.shape,generator=rng,dtype=torch.float64)/x.numel()**.5
        if name.startswith("weak"):
            x.mul_(1e-5); k.mul_(amplitude); b.fill_(-40.); g.mul_(1e-5)
        values = tuple(value.requires_grad_() for value in (x,k,b))
        refs = tuple(value.detach().clone().requires_grad_() for value in values)
        actual = spatial(x,x,k,b,1,eps)
        expected = matrix_solve(refs[0],refs[0],refs[1],refs[2],1,eps)
        a = torch.autograd.grad(actual,values,g)
        e = torch.autograd.grad(expected,refs,g)
        torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)
        for av,ev in zip(a,e):
            torch.testing.assert_close(av,ev,atol=1e-8,rtol=1e-8)
        pixels.append(dict(name=name,KB=kb,KC=kc,eps=eps,
            output_max_abs=(actual-expected).abs().max().item(),
            gradient_max_abs=[(av-ev).abs().max().item() for av,ev in zip(a,e)]))
    # K=0 is an exact FP32 identity, not just within a tolerance. The lambda
    # derivative is zero without subtracting two nearly equal transfer terms.
    x = torch.randn(2,3,5,7,generator=rng).requires_grad_()
    k = torch.zeros(1,1,3,3,requires_grad=True)
    b = torch.full((1,3,1,1),-40.,requires_grad=True)
    g = torch.randn(x.shape,generator=rng)
    output = spatial(x,x,k,b,1,1e-8)
    dx,dw,db = torch.autograd.grad(output,(x,k,b),g)
    if not torch.equal(output,x) or not torch.equal(dx,g) or torch.count_nonzero(db):
        raise AssertionError("K=0 identity/zero lambda-gradient invariant failed")
    if not all(bool(torch.isfinite(value).all()) for value in (output,dx,dw,db)):
        raise AssertionError("Nonfinite zero-kernel result")
    try:
        spatial(x,x.clone(),k,b,1,1e-8)
    except ValueError as error:
        if "prior IS x" not in str(error):
            raise
    else:
        raise AssertionError("Independent prior was not rejected")
    return dict(passed=True,device="CPU",seed=67218,spectral=spectral,pixel_matrix=pixels,
                zero_kernel_fp32_output_identity=True,zero_kernel_fp32_dx_identity=True,
                zero_kernel_bias_gradient_exact_zero=True,independent_prior_rejected=True,
                limitation="CPU mathematical proof only; GPU accuracy/noninferiority remains separate")


def cuda_accuracy():
    import torch
    from probe_training_s1_shapes import cases
    from diagnose_boundary_precision import fixture
    from probe_pointwise_training import capture,tensor_hash,clear_cuda
    from audit_shared_s1_accuracy import compare_accuracy
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for requested accuracy evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    frozen = json.loads((ROOT/"artifacts/native_deconv_target/source_before/manifest.json").read_text())
    if worker.file_hash(ROOT/"models/converse_core.py")!=frozen["models/converse_core.py"]:
        raise RuntimeError("Original Python reference changed")
    work = list(cases())
    tensors = fixture()
    old = json.loads((ROOT/"artifacts/native_deconv_target/boundary_probe.json").read_text())
    if worker.tensor_hash(dict(zip(("x","weight","bias","upstream"),tensors)))!=old["input_sha256"]:
        raise RuntimeError("Pretrained fixture changed")
    work.append(dict(name="pretrained_module_pad2_crop2",tensors=tensors,eps=1e-5,weak=False,padding=2))
    config = native.CASES["s1_module"]
    cpu = native.cpu_data(config,seed=17)
    old = next(row for row in json.loads((ROOT/"artifacts/native_deconv_target/native_baseline.json").read_text())["cases"]
               if row["name"]=="s1_module")
    if worker.tensor_hash(cpu)!=old["complete_fixture_sha256"]:
        raise RuntimeError("Native fixture changed")
    work.append(dict(name="native_s1_module_seed17",tensors=(cpu["x"],cpu["weight"],
        cpu["bias"].reshape(1,config["shape"][1],1,1),cpu["upstream"]),eps=config["eps"],weak=False,padding=2))
    rows = []
    for case in work:
        settings = dict(padding=case.get("padding",0),eps=case["eps"])
        oracle = capture(case["tensors"],torch.float64,method("reference",**settings))
        baseline = capture(case["tensors"],torch.float32,method("reference",**settings))
        actual = capture(case["tensors"],torch.float32,method("residual",**settings))
        if any(value.dtype!=torch.float32 for value in actual.values()):
            raise RuntimeError("Residual route changed output/gradient dtype")
        row = dict(name=case["name"],fixture_sha256=tensor_hash(case["tensors"]),eps=case["eps"],weak=case["weak"],
                   padding=settings["padding"],comparison=compare_accuracy(actual,baseline,oracle))
        rows.append(row)
        print(json.dumps(dict(case=case["name"],passed=row["comparison"]["passed"])),flush=True)
        clear_cuda()
    return dict(passed=all(row["comparison"]["passed"] for row in rows),cases=rows,timing=False,
        acceptance="For every output/dx/dw/db, both max_abs and relative_l2 to the same FP64 oracle <= Python FP32 error; no margin",
        environment=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),tf32=False,deterministic_algorithms=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cpu-proof",action="store_true")
    modes.add_argument("--cuda-accuracy",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite prior evidence")
    if args.cuda_accuracy and args.output is None:
        parser.error("CUDA accuracy requires a new --output path")
    if args.cuda_accuracy:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="1":
            parser.error("TF32 override conflicts with FP32 protocol")
    sources = ("test/probe_shared_s1_residual.py","test/diagnose_boundary_precision.py",
               "test/probe_spatial_nonoverlap.py","test/probe_training_s1_shapes.py",
               "test/probe_pointwise_training.py","test/audit_shared_s1_accuracy.py",
               "test/benchmark_native_deconv.py","models/converse_core.py")
    report = dict(status="running",scope=__doc__,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                  for name in sources},production_eligible=False,timing=False)
    try:
        result = cuda_accuracy() if args.cuda_accuracy else cpu_proof()
        report["cuda_accuracy" if args.cuda_accuracy else "cpu_proof"] = result
        report["status"] = "passed" if result["passed"] else "accuracy_noninferiority_failed"
        return 0 if result["passed"] else 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if args.output is not None:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    raise SystemExit(main())
