"""Internal contract of the isolated SAME-prior shared-s1 spectral operator.

    ./experiments/training_speed/run.ps1 test/check_shared_s1_cuda_contract.py

FP32 numerical checks use the differentiable ATen ORIGINAL same-prior s1
residual expression at the same complex64 inputs. Output atol/rtol=3e-5/3e-5,
all VJPs=5e-5/5e-5. Promoting those same inputs to FP64 supplies separate
diagnostics, not a release gate. This is not the spatial backend=pytorch
quality suite: that suite must include the original Python FP32 FFT pipeline.

Arbitrary complex128 gradcheck/gradgradcheck remain mathematical derivative
checks with nondet_tol=0. Every FP32 first-order mask explicitly uses
create_graph=False; an untimed profiler verifies the expected handwritten
CUDA kernels for all seven masks. Alias, seven high-order masks, nondefault
stream, saved-input version checks and three-call bitwise repeatability are
separate checks. No timing, changed production source or GPU fallback.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.training_shared_s1.loader import load

MASKS = [mask for mask in itertools.product((False,True),repeat=3) if any(mask)]


def reference(y,k,regularizer):
    # Preserve the original s1 same-prior residual algebra, independently of
    # the candidate's factored transfer and handwritten derivatives.
    return y+k.conj()*((y-k*y)/(k.real.square()+k.imag.square()+regularizer))


def cpu_fixture(kb,kc,dtype,shape=(2,3,2,3)):
    import torch
    rng = torch.Generator().manual_seed(67218)
    b,c,h,w = shape
    real = torch.float32 if dtype==torch.complex64 else torch.float64
    y = torch.randn(b,c,h,w,generator=rng,dtype=dtype)
    k = torch.randn(kb,kc,h,w,generator=rng,dtype=dtype)
    regularizer = torch.rand(1,c,1,1,generator=rng,dtype=real)+.2
    upstream = torch.randn(y.shape,generator=rng,dtype=dtype)
    return (y,k,regularizer),upstream


def inputs_on(values,device,mask=(True,True,True),double=False):
    import torch
    return tuple(value.to(device=device,dtype=(torch.complex128 if value.is_complex() else torch.float64)
                          if double else value.dtype).clone().requires_grad_(needed)
                 for value,needed in zip(values,mask))


def numerical(actual,expected,output=False):
    import torch
    a,e = actual.detach().cpu(),expected.detach().cpu()
    dtype = torch.complex128 if e.is_complex() else torch.float64
    a,e = a.to(dtype),e.to(dtype)
    atol,rtol = (3e-5,3e-5) if output else (5e-5,5e-5)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return dict(passed=False,finite=False,atol=atol,rtol=rtol)
    error,budget = (a-e).abs(),atol+rtol*e.abs()
    return dict(passed=bool((error<=budget).all()),finite=True,atol=atol,rtol=rtol,
                max_abs=error.max().item(),relative_l2=(error.norm()/e.norm().clamp_min(1e-30)).item(),
                failed_elements=int((error>budget).sum()),max_budget_ratio=(error/budget).max().item())


def gradients(function,values,upstream,mask,retain_graph=False):
    import torch
    output = function(*values)
    requested = tuple(value for value,needed in zip(values,mask) if needed)
    grads = torch.autograd.grad(output,requested,upstream,create_graph=False,retain_graph=retain_graph)
    if any(value.grad is not None for value in values):
        raise AssertionError("Unexpected leaf gradient accumulation")
    return (output,*grads)


def first_order(ops,kb,kc,mask):
    import torch
    original,upstream = cpu_fixture(kb,kc,torch.complex64)
    values = inputs_on(original,"cuda",mask)
    g = upstream.cuda()
    function = ops.shared_s1_transfer
    profile = kb==1 and kc==1
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            actual = gradients(function,values,g,mask)
            torch.cuda.synchronize()
        names = [event.name for event in profiler.events() if str(event.device_type).endswith("CUDA")]
        expected_hits = dict(shared_s1_prepare=1,shared_s1_apply=1,shared_s1_grad_y=int(mask[0]),
            shared_s1_transfer_vjp=int(mask[1] or mask[2]),shared_s1_reduce_channels=int(mask[1]),
            shared_s1_reduce_lambda=int(mask[2]))
        hits = {name:sum(name in kernel for kernel in names) for name in expected_hits}
        if hits!=expected_hits:
            raise AssertionError(f"First-order kernel dispatch mismatch: {hits} != {expected_hits}")
        dispatch = dict(passed=True,kernel_hits=hits,expected_kernel_hits=expected_hits)
        del profiler
    else:
        actual = gradients(function,values,g,mask)
        dispatch = dict(profiled=False,create_graph=False)
    expected = gradients(reference,inputs_on(original,"cuda",mask),g,mask)
    diagnostic = gradients(reference,inputs_on(original,"cpu",mask,double=True),upstream.to(torch.complex128),mask)
    labels = ["output",*[name for name,needed in zip(("dy","dk","dlambda"),mask) if needed]]
    checks = {name:numerical(a,e,output=name=="output") for name,a,e in zip(labels,actual,expected)}
    diagnostics = {name:numerical(a,e,output=name=="output") for name,a,e in zip(labels,actual,diagnostic)}
    if not all(value["passed"] for value in checks.values()):
        raise AssertionError(f"FP32 ATen reference mismatch: KB={kb},KC={kc},mask={mask}: {checks}")
    return dict(passed=True,KB=kb,KC=kc,mask=list(mask),create_graph=False,
                fp32_aten=checks,fp64_diagnostic_only=diagnostics,dispatch=dispatch)


def complex_gradcheck(ops,kb,kc):
    import torch
    original,_ = cpu_fixture(kb,kc,torch.complex128)
    values = inputs_on(original,"cuda")
    torch.autograd.gradcheck(ops.shared_s1_transfer,values,fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
    torch.autograd.gradgradcheck(ops.shared_s1_transfer,values,fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
    return dict(passed=True,KB=kb,KC=kc,gradcheck=True,gradgradcheck=True,nondet_tol=0.)


def higher_order_mask(ops,mask):
    import torch
    original,_ = cpu_fixture(1,1,torch.complex128)
    values = inputs_on(original,"cuda",mask)
    torch.autograd.gradgradcheck(ops.shared_s1_transfer,values,fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
    return dict(passed=True,mask=list(mask),gradgradcheck=True,nondet_tol=0.)


def alias_check(ops):
    import torch
    original,upstream = cpu_fixture(2,3,torch.complex128)
    y,_,regularizer = inputs_on(original,"cuda")
    source = y.detach().clone().requires_grad_()
    lam = regularizer.detach().clone().requires_grad_()
    actual = gradients(lambda z,l:ops.shared_s1_transfer(z,z,l),(y,regularizer),upstream.cuda(),(True,True))
    expected = gradients(lambda z,l:reference(z,z,l),(source,lam),upstream.cuda(),(True,True))
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e,atol=1e-10,rtol=1e-10)
    function = lambda z,l:ops.shared_s1_transfer(z,z,l)
    torch.autograd.gradcheck(function,(y,regularizer),fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
    torch.autograd.gradgradcheck(function,(y,regularizer),fast_mode=True,atol=1e-6,rtol=1e-5,nondet_tol=0.)
    return dict(passed=True,same_y_kernel=True,ordinary_vjp=True,gradcheck=True,gradgradcheck=True)


def stream_check(ops):
    import torch
    original,upstream = cpu_fixture(1,1,torch.complex64,shape=(2,3,5,7))
    expected = gradients(reference,inputs_on(original,"cuda"),upstream.cuda(),(True,True,True))
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        values = inputs_on(original,"cuda")
        actual = gradients(ops.shared_s1_transfer,values,upstream.cuda(),(True,True,True))
        actual = tuple(value.detach().clone() for value in actual)
    side.synchronize()
    checks = {name:numerical(a,e,output=name=="output") for name,a,e in
              zip(("output","dy","dk","dlambda"),actual,expected)}
    if not all(value["passed"] for value in checks.values()):
        raise AssertionError(f"Nondefault stream mismatch: {checks}")
    return dict(passed=True,stream_id=side.cuda_stream,fp32_aten=checks)


def version_check(ops,index):
    import torch
    original,upstream = cpu_fixture(1,1,torch.complex64)
    values = inputs_on(original,"cuda")
    output = ops.shared_s1_transfer(*values)
    before = values[index]._version
    with torch.no_grad():
        values[index].add_(.125)
    try:
        torch.autograd.grad(output,values,upstream.cuda(),create_graph=False)
    except RuntimeError as error:
        if "modified by an inplace operation" not in str(error) or "version" not in str(error):
            raise
        return dict(passed=True,input=("y","kernel","lambda")[index],version_before=before,
                    version_after=values[index]._version,expected_error=str(error))
    raise AssertionError("Backward accepted a changed saved input")


def repeatability(ops):
    import torch
    from probe_pointwise_training import tensor_hash
    original,upstream = cpu_fixture(1,1,torch.complex64,shape=(8,3,17,19))
    values,g = inputs_on(original,"cuda"),upstream.cuda()
    first,hashes,rows = None,None,[]
    for index in range(3):
        actual = gradients(ops.shared_s1_transfer,values,g,(True,True,True))
        saved = tuple(value.detach().cpu().clone() for value in actual)
        current = [tensor_hash((value,)) for value in saved]
        if first is None:
            first,hashes = saved,current
        rows.append(dict(repetition=index+1,tensors={name:dict(sha256=sha,
            bitwise_equal_to_first=sha==initial,finite=bool(torch.isfinite(value).all()),
            max_abs_to_first=(value-reference).abs().max().item())
            for name,value,reference,sha,initial in zip(("output","dy","dk","dlambda"),saved,first,current,hashes)}))
    if not all(value["bitwise_equal_to_first"] and value["finite"] for row in rows for value in row["tensors"].values()):
        raise AssertionError(f"Bitwise repeatability failed: {rows}")
    return dict(passed=True,records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/native_deconv_target/shared_s1_cuda_contract.json")
    parser.add_argument("--verbose-build",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    report = dict(status="initializing",scope=__doc__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/"experiments/training_shared_s1"/name).read_bytes()).hexdigest()
                       for name in ("bindings.cpp","kernels.cu","loader.py")},checks=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")

    def check(name,function,*params):
        row = dict(check=name,parameters=params)
        report["checks"].append(row)
        save()
        try:
            row.update(function(*params))
        except Exception as error:
            row.update(passed=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            save()
            raise
        save()
        print(json.dumps(row,allow_nan=False),flush=True)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU substitute")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        ops,report["build"] = load(verbose=args.verbose_build)
        report["environment"] = dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),tf32=False,cudnn_deterministic=True,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        report["status"] = "checking"
        for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
            check("complex128_derivative_math",lambda kb,kc:complex_gradcheck(ops,kb,kc),kb,kc)
            for mask in MASKS:
                check("fp32_handwritten_first_order",lambda kb,kc,mask:first_order(ops,kb,kc,mask),kb,kc,mask)
        for mask in MASKS:
            check("selective_high_order",lambda mask:higher_order_mask(ops,mask),mask)
        check("same_input_alias",lambda:alias_check(ops))
        check("nondefault_stream",lambda:stream_check(ops))
        for index in range(3):
            check("saved_version",lambda index:version_check(ops,index),index)
        check("bitwise_repeatability",lambda:repeatability(ops))
        report["status"] = "passed_internal_contract"
        save()
    except Exception as error:
        report["status"] = "failed_internal_contract"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        save()
        raise


if __name__=="__main__":
    main()
