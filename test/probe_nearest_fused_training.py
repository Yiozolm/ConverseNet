"""Isolated nearest-prior CUDA training candidate; production is unchanged.

    python test/probe_nearest_fused_training.py --cpu-adjoint-check
    python test/probe_nearest_fused_training.py

Compare production spatial nearest, the frozen ATen spectral-prior prototype,
and fused nearest-prior CUDA. Reuse the EXACT twelve fixed input fixtures and
independent full-FFT FP64 spatial reference from probe_nearest_training.py.
Normal output atol=rtol=3e-5, gradients 5e-5/5e-5; weak output 1e-6/1e-5.
All numerical, arbitrary-complex spectral, higher-order and dispatch gates
must pass before timing. No lambda/tolerance changes or parameter caching.

FFT/IFFT, differentiable FP64 kernel preparation -> complex64, and lambda
parameterization remain autograd. Phase generation is FP64 -> complex64 on
EVERY invocation and is timed. This candidate supports scale 1 and 3 only.
First-order CUDA computes nearest spectra on read and gathers the prior VJP
in a fixed LR-frequency thread, without an HR prior/gp or atomic scatter.
Higher derivatives reconstruct the differentiable materialized ATen formula.

Warm5, alternating four rounds of twenty FWD+dx/dw/db calls, no optimizer,
one fixture at a time, profiler off. Timings include all preparation. Peak
allocated/reserved is PyTorch fixture memory, not total process/device memory.
No speed claim about whole networks or equivalence with ConvTranspose2d.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_nearest_training import make_cases, nearest_half_spectrum, phase, prepare_kernel, spatial
from probe_pointwise_training import capture, clear_cuda, fixture, tensor_hash, timed_fixture
from probe_training_s1_shapes import metrics
from experiments.training_nearest.loader import load_candidate


def half_to_full(half, width):
    import torch
    tail = torch.roll(half[...,1:(width+1)//2].flip((-2,-1)),1,-2)
    if half.is_complex():
        tail = tail.conj()
    return torch.cat((half,tail),-1)


def spectral_reference(y, kernel, regularizer, ph, pw, height, width, scale):
    # An ATen expression independent of the CUDA VJP; identical half-spectrum
    # convention for arbitrary complex y/k, including non-Hermitian boundaries.
    import torch
    hs, ws = height*scale, width*scale
    rows = torch.arange(hs,device=y.device).remainder(height)
    columns = torch.arange(ws//2+1,device=y.device).remainder(width)
    mirror = columns > width//2
    stored_rows = torch.where(mirror[None,:],(-rows[:,None]).remainder(height),rows[:,None])
    stored_columns = torch.where(mirror,width-columns,columns)
    indices = (stored_rows*(width//2+1)+stored_columns[None,:]).flatten()
    p = y.flatten(-2).index_select(-1,indices).reshape(*y.shape[:2],hs,ws//2+1)
    p = torch.where(mirror[None,None,None,:],p.conj(),p)*(ph[:,None]*pw[None,:ws//2+1])

    def mean_alias(value):
        return value.reshape(*value.shape[:2],scale,height,scale,width).mean((2,4))

    power = kernel.real.square()+kernel.imag.square()
    if scale == 1:
        q = (y-kernel*p)/(power+regularizer)
    else:
        prediction = mean_alias(half_to_full(kernel*p,ws))[...,:width//2+1]
        denominator = mean_alias(half_to_full(power,ws))[...,:width//2+1]+regularizer
        q = (y-prediction)/denominator
        q = half_to_full(q,width).repeat(1,1,scale,scale)[...,:ws//2+1]
    return p+kernel.conj()*q


def cpu_adjoint_check():
    """Arbitrary complex y: enumerate CUDA's exact N^H direct/mirror sets."""
    import torch
    rng = torch.Generator().manual_seed(73219)
    records = []
    for height,width,scale in ((1,1,3),(1,4,3),(4,1,3),(2,3,3),(3,4,3),(3,5,3),(3,4,1)):
        y = torch.randn(1,1,height,width//2+1,generator=rng,dtype=torch.complex128).requires_grad_()
        ph,pw = phase(height*scale,scale,y),phase(width*scale,scale,y)
        p = nearest_half_spectrum(y,height,width,scale)
        g = torch.randn(p.shape,generator=rng,dtype=p.dtype)
        expected, = torch.autograd.grad(p,(y,),g)
        actual = torch.zeros_like(y)
        # Matches nearest_adjoint_y's loop; r addition is a separate direct
        # dependence and original solver's gp is replaced with arbitrary g.
        for h,w in itertools.product(range(height),range(width//2+1)):
            value = torch.zeros((),dtype=y.dtype)
            for a,b in itertools.product(range(scale),repeat=2):
                hh,ww = h+a*height,w+b*width
                if ww <= width*scale//2:
                    value += (ph[hh]*pw[ww]).conj()*g[0,0,hh,ww]
                if w > 0 and 2*w != width:
                    hh,ww = (-h)%height+a*height,width-w+b*width
                    if ww <= width*scale//2:
                        value += ph[hh]*pw[ww]*g[0,0,hh,ww].conj()
            actual[0,0,h,w] = value
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        # Canonical HR read used in q must equal full_spectrum(materialized p)
        # even when y contains arbitrary complex DC/Nyquist row values.
        full = half_to_full(p,width*scale)
        largest = 0.
        for h,w in itertools.product(range(height*scale),range(width*scale)):
            hh,ww,conjugate = h,w,w>width*scale//2
            if conjugate:
                hh,ww = (-h)%(height*scale),width*scale-w
            lh,lw = hh%height,ww%width
            lr_mirror = lw>width//2
            if lr_mirror:
                lh,lw = (-lh)%height,width-lw
            z = y[0,0,lh,lw]
            if lr_mirror:
                z = z.conj()
            z = z*(ph[hh]*pw[ww])
            if conjugate:
                z = z.conj()
            largest = max(largest,(z-full[0,0,h,w]).abs().item())
        if largest > 1e-12:
            raise AssertionError(f"Canonical prior read mismatch: {largest}")
        records.append(dict(height=height,width=width,scale=scale,
                            prior_adjoint_max_abs=(actual-expected).abs().max().item(),
                            canonical_read_max_abs=largest))
    return dict(passed=True,device="CPU",cases=records,
                limitation="Python formula checks only, not compiled CUDA validation")


def fused_spatial(ops,x,kernel,bias,scale,eps):
    import torch
    height,width = x.shape[-2:]
    x = x.contiguous()
    regularizer = torch.sigmoid(bias.contiguous()-9.)+eps
    k = prepare_kernel(kernel,height*scale,width*scale)
    y = torch.fft.rfft2(x)
    ph,pw = phase(height*scale,scale,y),phase(width*scale,scale,y)
    out = ops._training_nearest_spectral(y,k,regularizer,ph,pw,height,width,scale)
    return torch.fft.irfft2(out,s=(height*scale,width*scale))


def methods(ops,case):
    s,eps = case["scale"],case["eps"]
    return {"production":lambda x,k,b:spatial(x,k,b,s,eps,"production"),
            "aten_spectral":lambda x,k,b:spatial(x,k,b,s,eps,"aten_spectral"),
            "fused_nearest":lambda x,k,b:fused_spatial(ops,x,k,b,s,eps)}


def validate_spatial(ops,case):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    s,eps = case["scale"],case["eps"]
    expected = capture(case["tensors"],torch.float64,lambda x,k,b:converse2d_reference(
        x,x if s==1 else F.interpolate(x,scale_factor=s,mode="nearest"),k,b,s,eps))
    result = {}
    for name,method in methods(ops,case).items():
        actual = capture(case["tensors"],torch.float32,method)
        if any(value.dtype != torch.float32 for value in actual.values()):
            raise RuntimeError(f"Unexpected output or gradient dtype in {name}")
        errors = {key:metrics(actual[key],value,output=key=="output",weak=case["weak"])
                  for key,value in expected.items()}
        result[name] = dict(passed=all(item["passed"] for item in errors.values()),tensors=errors)
    return result


def validate_spectral(ops):
    import torch
    rng = torch.Generator().manual_seed(18371)
    rows = []
    # All broadcast forms; singleton/even/odd widths; s1; arbitrary complex
    # values, with y a conjugate noncontiguous view in one case.
    configs = ((1,1,3,1,1),(1,4,3,1,3),(4,1,3,2,1),
               (2,3,3,2,3),(3,4,3,1,1),(3,5,3,1,3),(3,4,1,2,3))
    for index,(h,w,s,kb,kc) in enumerate(configs):
        y = torch.randn(2,3,h,w//2+1,generator=rng,dtype=torch.complex128).cuda().requires_grad_()
        if index == 4:
            y = y.transpose(0,1).contiguous().transpose(0,1).conj()
        k = torch.randn(kb,kc,h*s,w*s//2+1,generator=rng,dtype=torch.complex128).cuda().requires_grad_()
        l = (torch.rand(1,3,1,1,generator=rng,dtype=torch.float64)+.4).cuda().requires_grad_()
        ph,pw = phase(h*s,s,y),phase(w*s,s,y)
        args = (y,k,l)
        fun = lambda y,k,l:ops._training_nearest_spectral(y,k,l,ph,pw,h,w,s)
        expected = spectral_reference(*args,ph,pw,h,w,s)
        actual = fun(*args)
        upstream = torch.randn(expected.shape,generator=rng,dtype=torch.complex128).cuda()
        a_grads = torch.autograd.grad(actual,args,upstream)
        e_grads = torch.autograd.grad(expected,args,upstream)
        torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)
        for a,e in zip(a_grads,e_grads):
            torch.testing.assert_close(a,e,atol=1e-8,rtol=1e-8)
        row = dict(height=h,width=w,scale=s,kernel_batch=kb,kernel_channels=kc,
                   output_max_abs=(actual-expected).abs().max().item(),
                   gradient_max_abs=[(a-e).abs().max().item() for a,e in zip(a_grads,e_grads)])
        # Each nonempty gradient mask validates allocation/dispatch pruning.
        for mask in itertools.product((False,True),repeat=3):
            if not any(mask):
                continue
            values = tuple(t.detach().requires_grad_(flag) for t,flag in zip(args,mask))
            requested = tuple(t for t,flag in zip(values,mask) if flag)
            a = torch.autograd.grad(fun(*values),requested,upstream)
            e = torch.autograd.grad(spectral_reference(*values,ph,pw,h,w,s),requested,upstream)
            for av,ev in zip(a,e):
                torch.testing.assert_close(av,ev,atol=1e-8,rtol=1e-8)
        with torch.no_grad():
            torch.testing.assert_close(fun(*args),expected,atol=1e-9,rtol=1e-9)
        row["all_seven_grad_masks"] = True
        rows.append(row)
    # Small numerical Jacobians explicitly test the ordinary CUDA backward
    # and the differentiable ATen higher-order fallback, separately from FP32.
    higher = []
    for h,w,s in ((2,3,3),(3,4,3),(2,3,1)):
        y = torch.randn(1,2,h,w//2+1,generator=rng,dtype=torch.complex128).cuda().requires_grad_()
        k = torch.randn(1,1,h*s,w*s//2+1,generator=rng,dtype=torch.complex128).cuda().requires_grad_()
        l = torch.full((1,2,1,1),.7,device="cuda",dtype=torch.float64,requires_grad=True)
        ph,pw = phase(h*s,s,y),phase(w*s,s,y)
        fun = lambda y,k,l:ops._training_nearest_spectral(y,k,l,ph,pw,h,w,s)
        torch.autograd.gradcheck(fun,(y,k,l),fast_mode=True,atol=1e-5,rtol=1e-4)
        torch.autograd.gradgradcheck(fun,(y,k,l),fast_mode=True,atol=1e-5,rtol=1e-4)
        higher.append(dict(height=h,width=w,scale=s,gradcheck=True,gradgradcheck=True))
    try:
        ops._training_nearest_spectral(y,k,l,ph.detach().requires_grad_(),pw,h,w,s)
    except RuntimeError as error:
        if "phase must not require gradients" not in str(error):
            raise
    else:
        raise AssertionError("Trainable phase was not rejected")
    clear_cuda()
    return dict(passed=True,cases=rows,higher_order=higher,trainable_phase_rejected=True)


def repeatability(ops,case):
    import torch
    clear_cuda()
    inputs,run = fixture(case["tensors"],torch.float32,methods(ops,case)["fused_nearest"])
    first,hashes,records = None,None,[]
    for index in range(3):
        result = run()
        values = {name:t.detach().cpu().clone() for name,t in result.items()}
        del result
        current_hashes = {name:tensor_hash((value,)) for name,value in values.items()}
        if first is None:
            first,hashes = values,current_hashes
        records.append(dict(repetition=index+1,tensors={name:dict(
            sha256=current_hashes[name],bitwise_equal_to_first=current_hashes[name]==hashes[name],
            max_abs_to_first=(value.double()-first[name].double()).abs().max().item())
            for name,value in values.items()}))
    if any(value.grad is not None for value in inputs):
        raise RuntimeError("Leaf gradients accumulated")
    del run,inputs,values,first
    clear_cuda()
    return dict(records=records,all_bitwise_equal=all(item["bitwise_equal_to_first"]
                for row in records for item in row["tensors"].values()))


def dispatch_check(ops,case):
    import torch
    clear_cuda()
    inputs,run = fixture(case["tensors"],torch.float32,methods(ops,case)["fused_nearest"])
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    counts = {event.key:event.count for event in profiler.key_averages()}
    names = [event.name for event in profiler.events() if str(event.device_type).endswith("CUDA")]
    hits = {token:sum(token in name for name in names) for token in
            ("nearest_solve_alias","nearest_solve_output","nearest_adjoint_q","nearest_adjoint_y","nearest_adjoint_filter")}
    # This B32 case uses separable prep: its one rfft2 is the LR activation.
    passed = counts.get("aten::fft_rfft2",0)==1 and counts.get("aten::upsample_nearest2d",0)==0
    passed &= all(value==1 for value in hits.values())
    # Phase construction uses where, but no HR gather or scatter is expected.
    forbidden = {name:count for name,count in counts.items() if any(token in name for token in
                 ("index_select","index_add","scatter","upsample_nearest"))}
    passed &= not forbidden
    del profiler,run,inputs
    clear_cuda()
    return dict(passed=bool(passed),kernel_hits=hits,forbidden_operations=forbidden,
                rfft2=counts.get("aten::fft_rfft2",0),scope="Untimed first-order FWD+VJP")


def time_case(ops,case,args):
    choices = methods(ops,case)
    names = list(choices)
    rounds = []
    for index in range(args.rounds):
        rotated = names[index%len(names):]+names[:index%len(names)]
        order = rotated if index%2==0 else list(reversed(rotated))
        variants = {name:timed_fixture(case["tensors"],choices[name],args) for name in order}
        rounds.append(dict(round=index+1,order=order,variants=variants))
    medians = {name:{key:statistics.median(row["variants"][name][key] for row in rounds)
                     for key in ("wall_ms","cuda_event_ms","peak_allocated_bytes","peak_reserved_bytes")}
               for name in names}
    ratios = {name:{key:[row["variants"][name][key]/row["variants"]["fused_nearest"][key] for row in rounds]
                    for key in ("wall_ms","cuda_event_ms")} for name in names[:-1]}
    return dict(rounds=rounds,medians=medians,paired_baseline_over_candidate=ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-adjoint-check",action="store_true")
    parser.add_argument("--verbose-build",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/training_research/nearest_fused_training.json")
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--rounds",type=int,default=4)
    parser.add_argument("--iters",type=int,default=20)
    args = parser.parse_args()
    if min(args.warmup,args.rounds,args.iters)<1:
        parser.error("warmup, rounds and iters must be positive")
    if args.cpu_adjoint_check:
        print(json.dumps(cpu_adjoint_check(),indent=2))
        return
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="1":
        parser.error("TF32 override conflicts with FP32 protocol")
    import torch
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    load_extension()
    ops,build = load_candidate(verbose=args.verbose_build)
    report = dict(status="validating",scope=__doc__,build=build,source_sha256=current_manifest(),
        script_sha256={name:hashlib.sha256((ROOT/"test"/name).read_bytes()).hexdigest() for name in
            ("probe_nearest_fused_training.py","probe_nearest_training.py","probe_training_s1_shapes.py",
             "probe_pointwise_training.py","test_fp32_training.py")},
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                         tf32=False,cudnn_deterministic=True,
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        settings=dict(warmup=args.warmup,rounds=args.rounds,iters=args.iters),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")

    try:
        report["cpu_adjoint"] = cpu_adjoint_check()
        report["internal_spectral"] = validate_spectral(ops)
        save()
        cases = make_cases()
        for case in cases:
            x,k,b,_ = case["tensors"]
            row = dict(name=case["name"],shape=list(x.shape),kernel_shape=list(k.shape),bias_shape=list(b.shape),
                       scale=case["scale"],eps=case["eps"],weak=case["weak"],tensors_sha256=tensor_hash(case["tensors"]),
                       validation=validate_spatial(ops,case),timing=None)
            report["cases"].append(row)
            save()
            print(json.dumps(dict(case=case["name"],validation=row["validation"])),flush=True)
        if not all(check["passed"] for row in report["cases"] for check in row["validation"].values()):
            report["status"] = "numerical_gate_failed_no_timing"
            save()
            raise SystemExit(1)
        representative = next(case for case in cases if case["name"]=="b32_c32_64x80_s3")
        report["repeatability"] = repeatability(ops,representative)
        report["dispatch_check"] = dispatch_check(ops,representative)
        save()
        if not report["dispatch_check"]["passed"] or not report["repeatability"]["all_bitwise_equal"]:
            report["status"] = "dispatch_or_repeatability_failed_no_timing"
            save()
            raise SystemExit(1)
        report["status"] = "timing"
        for case,row in zip(cases,report["cases"]):
            if case["timed"]:
                row["timing"] = time_case(ops,case,args)
                save()
                print(json.dumps(dict(case=case["name"],medians=row["timing"]["medians"])),flush=True)
        report["status"] = "complete_isolated_nearest_candidate"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        save()
        raise
    print(f"Saved {args.output}",flush=True)


if __name__=="__main__":
    main()
