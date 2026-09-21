"""CPU-first contract checks for the isolated automatic-VJP adapter.

    python test/check_compiled_autograd_contract.py
    python test/check_compiled_autograd_contract.py --cuda --output artifacts/native_deconv_target/compiled_autograd_contract.json

Default uses an eager CPU FP64 stand-in for compiled code: it verifies graph
ownership/routing, not Inductor support. --cuda additionally uses real Inductor
FP32, original nearest fixtures/budgets, an independent FP64 high-order check,
and a nondefault CUDA stream. Bare AOT double-backward support is explicitly
observed; the adapter must pass regardless. No GPU work runs without --cuda.
No changes to frozen probes, fast math, tolerance relaxation or hand VJP.
This experiment disables AOT donated buffers before any compile/first call
and keeps that setting throughout. Dedicated nodonation cache directories
and fresh callables isolate it from the earlier incompatible compilation.
"""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def derivatives(function,inputs,upstream,directions):
    import torch
    output = function(*inputs)
    first = torch.autograd.grad(output,inputs,upstream,create_graph=True)
    terms = [(value*direction).sum() for value,direction in zip(first,directions) if value.requires_grad]
    if terms:
        second = torch.autograd.grad(sum(terms),inputs,allow_unused=True)
        second = tuple(torch.zeros_like(value) if grad is None else grad for value,grad in zip(inputs,second))
    else:
        second = tuple(torch.zeros_like(value) for value in inputs)
    return output,first,second


def cpu_check():
    import torch
    import torch.nn.functional as F
    import probe_spatial_nonoverlap_fast as fast
    from models.converse_core import converse2d_reference
    from experiments.training_nonoverlap.compiled_autograd import wrap_compiled
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contract must run before initializing CUDA")
    rng = torch.Generator().manual_seed(39271)
    source = (torch.randn(2,2,2,3,generator=rng,dtype=torch.float64),
              torch.rand(1,2,3,3,generator=rng,dtype=torch.float64)/9,
              torch.randn(1,2,1,1,generator=rng,dtype=torch.float64))
    upstream = torch.randn(2,2,6,9,generator=rng,dtype=torch.float64)/108**.5
    eager = lambda x,k,b:fast.nearest_route(x,k,b,3,1e-3,"lrpred_tensor")
    route = wrap_compiled(eager,eager)
    rows = []
    for mask in itertools.product((False,True),repeat=3):
        if not any(mask):
            continue
        actual_inputs = tuple(value.clone().requires_grad_(needed) for value,needed in zip(source,mask))
        expected_inputs = tuple(value.clone().requires_grad_(needed) for value,needed in zip(source,mask))
        actual = route(*actual_inputs)
        expected = eager(*expected_inputs)
        a = torch.autograd.grad(actual,tuple(value for value,needed in zip(actual_inputs,mask) if needed),upstream)
        e = torch.autograd.grad(expected,tuple(value for value,needed in zip(expected_inputs,mask) if needed),upstream)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        for av,ev in zip(a,e):
            torch.testing.assert_close(av,ev,atol=1e-12,rtol=1e-12)
        rows.append(dict(mask=list(mask),max_abs=max((av-ev).abs().max().item() for av,ev in zip(a,e))))
    inputs = tuple(value.clone().requires_grad_() for value in source)
    reference_inputs = tuple(value.clone().requires_grad_() for value in source)
    directions = tuple(torch.randn(value.shape,generator=rng,dtype=value.dtype)/value.numel()**.5 for value in source)
    a = derivatives(route,inputs,upstream,directions)
    reference = lambda x,k,b:converse2d_reference(x,F.interpolate(x,scale_factor=3,mode="nearest"),k,b,3,1e-3)
    e = derivatives(reference,reference_inputs,upstream,directions)
    torch.testing.assert_close(a[0],e[0],atol=1e-9,rtol=1e-9)
    for actual,expected in zip((*a[1],*a[2]),(*e[1],*e[2])):
        torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-8)
    torch.autograd.gradcheck(route,inputs,fast_mode=True,atol=1e-5,rtol=1e-4)
    torch.autograd.gradgradcheck(route,inputs,fast_mode=True,atol=1e-5,rtol=1e-4)

    # x and kernel can legally be the SAME leaf at this small spatial shape.
    shared = (torch.randn(1,1,3,3,generator=rng,dtype=torch.float64)*.1).requires_grad_()
    bias = torch.zeros(1,1,1,1,dtype=torch.float64,requires_grad=True)
    shared_ref = shared.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    g = torch.randn(1,1,9,9,generator=rng,dtype=torch.float64)/9
    direction = (torch.randn(shared.shape,generator=rng,dtype=shared.dtype),torch.ones_like(bias))
    av = derivatives(lambda x,b:route(x,x,b),(shared,bias),g,direction)
    ev = derivatives(lambda x,b:eager(x,x,b),(shared_ref,bias_ref),g,direction)
    for actual,expected in zip((*av[1],*av[2]),(*ev[1],*ev[2])):
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)

    # The same outer graph must support repeated ordinary backward and then
    # a create_graph=True query without consuming its compiled inner graph.
    inputs = tuple(value.clone().requires_grad_() for value in source)
    output = route(*inputs)
    first = torch.autograd.grad(output,inputs,upstream,retain_graph=True)
    again = torch.autograd.grad(output,inputs,upstream,retain_graph=True)
    higher = torch.autograd.grad(output,inputs,upstream,create_graph=True)
    for one,two,three in zip(first,again,higher):
        torch.testing.assert_close(one,two,atol=0,rtol=0)
        torch.testing.assert_close(one,three,atol=1e-12,rtol=1e-12)
    with torch.no_grad():
        no_grad = route(*inputs)
    assert not no_grad.requires_grad
    assert not route(*(value.detach() for value in inputs)).requires_grad

    version_inputs = tuple(value.clone().requires_grad_() for value in source)
    version_output = route(*version_inputs)
    with torch.no_grad():
        version_inputs[1].add_(.001)
    try:
        torch.autograd.grad(version_output,version_inputs,upstream)
    except RuntimeError as error:
        if "modified by an inplace operation" not in str(error):
            raise
    else:
        raise AssertionError("Saved original input mutation was not rejected")

    # Weak references observe Python Tensor ownership; no spy owns an input
    # or output strongly. Dropping/releasing outer graphs must release inner
    # proxies/output while the factory itself remains alive.
    observed = []
    def tracked(*values):
        result = eager(*values)
        observed.extend(weakref.ref(value) for value in (*values,result))
        return result
    tracked_route = wrap_compiled(eager,tracked)
    release_inputs = tuple(value.clone().requires_grad_() for value in source)
    result = tracked_route(*release_inputs)
    torch.autograd.grad(result,release_inputs,upstream)
    gc.collect()
    if any(ref() is not None for ref in observed):
        raise AssertionError("Inner tensors retained after ordinary outer backward")
    observed.clear()
    result = tracked_route(*release_inputs)
    del result
    gc.collect()
    if any(ref() is not None for ref in observed):
        raise AssertionError("Unconsumed inner graph retained after dropping outer output")
    for _ in range(2):
        result = route(*release_inputs)
        expected = eager(*release_inputs)
        torch.testing.assert_close(result,expected,atol=0,rtol=0)
        torch.autograd.grad(result,release_inputs,upstream)
        with torch.no_grad():
            release_inputs[1].add_(.001)
    if any(value.grad is not None for value in release_inputs):
        raise AssertionError("Contract checks accumulated leaf gradients")
    return dict(passed=True,device="CPU",compiled_substitute="eager FP64",gradient_masks=rows,
        independent_fp64_higher_order=True,gradcheck=True,gradgradcheck=True,alias_first_and_second=True,
        retain_graph_repeated_backward=True,no_grad=True,saved_version_check=True,
        released_inner_tensors=True,discarded_forward_releases_inner=True,no_cross_call_parameter_cache=True)


def cuda_check():
    import torch
    import torch._dynamo
    import torch._functorch.config as functorch_config
    import torch.nn.functional as F
    from experiments.training_nonoverlap.compiled_autograd import wrap_compiled
    from probe_compiled_nonoverlap import make_compiled,eager_method
    from probe_nearest_training import make_cases
    from probe_pointwise_training import capture,clear_cuda,fixture,tensor_hash
    from probe_training_s1_shapes import metrics
    from models.converse_core import converse2d_reference
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no substitute for requested real-compiled check")
    if functorch_config.donated_buffer is not False:
        raise RuntimeError("Configure donated_buffer=False before creating compiled callables")
    # No donation=True in-process Dynamo specialization can be reused. Disk
    # AOT/Inductor/Triton caches are separately selected before torch import.
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    cases = [case for case in make_cases() if case["name"] in
             ("small_odd_s3","b1_c32_64x80_s3","weak_s3_amplitude_0.001")]
    rows = []
    for case in cases:
        compiled = make_compiled(case)
        eager = eager_method(case)
        route = wrap_compiled(eager,compiled)
        s,eps = case["scale"],case["eps"]
        reference = lambda x,k,b:converse2d_reference(x,F.interpolate(x,scale_factor=s,mode="nearest"),k,b,s,eps)
        reference_values = capture(case["tensors"],torch.float64,reference)
        actual = capture(case["tensors"],torch.float32,route)
        checks = {key:metrics(value,reference_values[key],output=key=="output",weak=case["weak"]) for key,value in actual.items()}
        if not all(value["passed"] for value in checks.values()):
            raise AssertionError(f"Original FP32 budget failed: {case['name']}: {checks}")
        row = dict(name=case["name"],tensors_sha256=tensor_hash(case["tensors"]),first_order=checks)
        if case["name"]=="small_odd_s3":
            inputs = tuple(value.cuda().requires_grad_() for value in case["tensors"][:3])
            upstream = case["tensors"][3].cuda()
            # Explicitly observe the bare AOT limitation; only the documented
            # double-backward rejection is expected, not unrelated failures.
            raw,raw_first = None,None
            try:
                raw = compiled(*inputs)
                raw_first = torch.autograd.grad(raw,inputs,upstream,create_graph=True)
                torch.autograd.grad(sum(value.square().sum() for value in raw_first),inputs)
                row["bare_compiled_double_backward"] = dict(supported=True)
            except RuntimeError as error:
                if "double backward" not in str(error).lower():
                    raise
                row["bare_compiled_double_backward"] = dict(supported=False,error=str(error))
            finally:
                del raw,raw_first
            rng = torch.Generator().manual_seed(29417)
            directions = tuple(torch.randn(value.shape,generator=rng)/value.numel()**.5 for value in inputs)
            a = derivatives(route,inputs,upstream,tuple(value.cuda() for value in directions))
            ref_inputs = tuple(value.detach().double().requires_grad_() for value in inputs)
            e = derivatives(reference,ref_inputs,upstream.double(),tuple(value.cuda().double() for value in directions))
            second_checks = [metrics(actual.detach().cpu(),expected.detach().cpu(),output=False,weak=False)
                             for actual,expected in zip(a[2],e[2])]
            if not all(value["passed"] for value in second_checks):
                raise AssertionError(f"Unchanged gradient budget failed for high-order VJP: {second_checks}")
            # Also compare high-order reconstruction to the independent eager
            # automatic path, separately from the full-FFT FP64 comparison.
            eager_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
            e32 = derivatives(eager,eager_inputs,upstream,tuple(value.cuda() for value in directions))
            for actual,expected in zip((*a[1],*a[2]),(*e32[1],*e32[2])):
                torch.testing.assert_close(actual,expected,atol=5e-5,rtol=5e-5)
            output = route(*inputs)
            one = torch.autograd.grad(output,inputs,upstream,retain_graph=True)
            two = torch.autograd.grad(output,inputs,upstream)
            for av,ev in zip(one,two):
                torch.testing.assert_close(av,ev,atol=0,rtol=0)
            row["high_order_vs_fp64"] = second_checks
            row["high_order_vs_eager"] = True
            row["retain_graph_repeat"] = True
            del inputs,upstream,a,e,e32,ref_inputs,eager_inputs,output,one,two
        # Create forward on a side stream, consume its VJP on the caller's
        # stream with explicit dependencies. The wrapper must preserve normal
        # autograd stream semantics, with no hidden global-stream state.
        inputs = tuple(value.cuda().requires_grad_() for value in case["tensors"][:3])
        upstream = case["tensors"][3].cuda()
        caller = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(caller)
        with torch.cuda.stream(side):
            output = route(*inputs)
        caller.wait_stream(side)
        gradients = torch.autograd.grad(output,inputs,upstream)
        torch.cuda.synchronize()
        observed = dict(output=output,dx=gradients[0],dw=gradients[1],db=gradients[2])
        stream_checks = {key:metrics(value.detach().cpu(),reference_values[key],output=key=="output",weak=case["weak"])
                         for key,value in observed.items()}
        if not all(value["passed"] for value in stream_checks.values()):
            raise AssertionError(f"Nondefault stream budget failed: {stream_checks}")
        row["nondefault_stream"] = stream_checks
        rows.append(row)
        del inputs,upstream,output,gradients,observed,compiled,eager,route
        clear_cuda()
    import triton
    return dict(passed=True,device=torch.cuda.get_device_name(),torch=str(torch.__version__),
                cuda=torch.version.cuda,triton=str(triton.__version__),cases=rows,
                functorch_donated_buffer=functorch_config.donated_buffer,
                limitation="Contract validation only; adapter overhead/memory/steady performance not measured")


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuda",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    if args.cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for key,name in (("TORCHINDUCTOR_CACHE_DIR","inductor_nonoverlap_nodonation"),
                         ("TRITON_CACHE_DIR","triton_nonoverlap_nodonation")):
            path = ROOT/".build"/name
            path.mkdir(parents=True,exist_ok=True)
            os.environ[key] = str(path)
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="1":
            parser.error("TF32 override conflicts with FP32 contract")
    import torch
    import torch._functorch.config as functorch_config
    functorch_config.donated_buffer = False
    torch.set_num_threads(1)
    report = dict(status="checking",runtime=dict(functorch_donated_buffer=functorch_config.donated_buffer,
        donation_policy="Disabled before creating any compiled callable and retained throughout this experiment",
        inductor_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),triton_cache_dir=os.environ.get("TRITON_CACHE_DIR")),
        sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ("test/check_compiled_autograd_contract.py","experiments/training_nonoverlap/compiled_autograd.py",
         "test/probe_compiled_nonoverlap.py","test/probe_spatial_nonoverlap_fast.py","models/converse_core.py")})
    try:
        report["cpu"] = cpu_check()
        if args.cuda:
            report["cuda"] = cuda_check()
        report["status"] = "passed"
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
    main()
