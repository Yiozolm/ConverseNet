"""Inductor FP32 non-overlap experiment; no custom spatial backward or fallback.

    python test/probe_compiled_nonoverlap.py --smoke
    python test/probe_compiled_nonoverlap.py

Smoke compiles the original B1/C32/LR64x80/s3 k3 fixture and executes complete
FWD+dx/dw/db, checks the independent FP64 budget and three-call repeatability.
Compilation/backend failures are errors; no eager fallback is substituted.

Default reuses the original ten kh/kw<=scale nearest fixtures, unmodified.
Compile only k3/s3. The k1 case receives explicit eager/production validation
only; k7/s3 and k3/s1 are excluded. Numerical gates remain output 3e-5/3e-5,
dx/dw/db 5e-5/5e-5, weak output 1e-6/1e-5. All gates precede formal timing.

torch.compile: backend=inductor, fullgraph=True, dynamic=False and explicit
triton.cudagraphs=False. Global deterministic algorithms and cuDNN deterministic
are enabled for ALL routes, TF32/AMP off, CUBLAS_WORKSPACE_CONFIG=:4096:8.
Per-case compile-wrapper and first complete FWD+VJP setup times are separate
from warm5, alternating round4/iters20 timing. Setup CUDA-event spans include
host compilation/submission gaps and are not pure kernel times. Cache dirs
are under repository .build; they are not deleted, and cache state is reported.
Warm allocation/reservation peaks are PyTorch fixture memory, not total device
memory. This tests an isolated operator, not complete model training quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import probe_spatial_nonoverlap_fast as fast
from probe_nearest_training import make_cases
from probe_pointwise_training import capture, clear_cuda, fixture, tensor_hash, timed_fixture
from probe_training_s1_shapes import metrics

OPTIONS = {"triton.cudagraphs":False}


def eager_method(case):
    scale,eps = case["scale"],case["eps"]
    return lambda x,k,b:fast.nearest_route(x,k,b,scale,eps,"lrpred_tensor")


def compiled_graphs():
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def make_compiled(case):
    """Return a lazy compiled callable(x,k,b); caller owns environment/setup.

    No first execution or eager fallback occurs here. This factory supports
    only the exact k3/s3 formula; AOTAutograd compiles its ATen backward on the
    first VJP, which must be completed before any warm performance timing.
    """
    import torch
    if case["scale"]!=3 or tuple(case["tensors"][1].shape[-2:])!=(3,3):
        raise ValueError("Compiled experiment supports k3/s3 only")
    return torch.compile(eager_method(case),backend="inductor",fullgraph=True,dynamic=False,options=OPTIONS)


def prepare_compiled(case):
    import torch
    clear_cuda()
    before = compiled_graphs()
    began = time.perf_counter()
    method = make_compiled(case)
    wrapper_ms = (time.perf_counter()-began)*1000
    inputs,run = fixture(case["tensors"],torch.float32,method)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    began = time.perf_counter()
    start.record()
    values = run()  # Force both forward and AOTAutograd backward compilation.
    end.record()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter()-began)*1000
    setup = dict(compile_wrapper_wall_ms=wrapper_ms,first_fwd_vjp_wall_ms=first_ms,
        first_fwd_vjp_cuda_event_span_ms=start.elapsed_time(end),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        unique_graphs_before=before,unique_graphs_after=compiled_graphs(),
        scope="First FWD+all VJPs includes compile/cache lookup and execution; not a warm latency")
    if any(value.grad is not None for value in inputs) or not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError("Compiled setup accumulated gradients or produced nonfinite values")
    del values,run,inputs
    clear_cuda()
    return method,setup


def methods(case,compiled,include_production=True):
    result = {}
    if include_production:
        s,eps = case["scale"],case["eps"]
        result["production_fft"] = lambda x,k,b:fast.nearest_route(x,k,b,s,eps,"production")
    result["eager_lrpred_tensor"] = eager_method(case)
    if compiled is not None:
        result["compiled_lrpred_tensor"] = compiled
    return result


def validate(case,choices):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    s,eps = case["scale"],case["eps"]
    reference = capture(case["tensors"],torch.float64,lambda x,k,b:converse2d_reference(
        x,x if s==1 else F.interpolate(x,scale_factor=s,mode="nearest"),k,b,s,eps))
    result = {}
    for name,method in choices.items():
        values = capture(case["tensors"],torch.float32,method)
        if any(value.dtype != torch.float32 for value in values.values()):
            raise RuntimeError(f"{name} changed FP32 output/gradient dtype")
        checks = {key:metrics(values[key],value,output=key=="output",weak=case["weak"])
                  for key,value in reference.items()}
        result[name] = dict(passed=all(value["passed"] for value in checks.values()),tensors=checks)
    return result


def repeatability(case,choices):
    import torch
    result = {}
    for name,method in choices.items():
        clear_cuda()
        inputs,run = fixture(case["tensors"],torch.float32,method)
        first,first_hashes,rows = None,None,[]
        for index in range(3):
            values = run()
            saved = {key:value.detach().cpu().clone() for key,value in values.items()}
            del values
            hashes = {key:tensor_hash((value,)) for key,value in saved.items()}
            if first is None:
                first,first_hashes = saved,hashes
            rows.append(dict(repetition=index+1,tensors={key:dict(sha256=hashes[key],
                bitwise_equal_to_first=hashes[key]==first_hashes[key],finite=bool(torch.isfinite(value).all()),
                max_abs_to_first=(value.double()-first[key].double()).abs().max().item()) for key,value in saved.items()}))
        if any(value.grad is not None for value in inputs):
            raise RuntimeError("Repeatability accumulated leaf gradients")
        result[name] = dict(passed=all(value["finite"] and value["bitwise_equal_to_first"]
            for row in rows for value in row["tensors"].values()),repetitions=rows)
        del inputs,run,first,saved
        clear_cuda()
    return result


def benchmark(case,choices,args):
    names = list(choices)
    rounds = []
    for index in range(args.rounds):
        rotated = names[index%len(names):]+names[:index%len(names)]
        order = rotated if index%2==0 else list(reversed(rotated))
        before = compiled_graphs()
        values = {name:timed_fixture(case["tensors"],choices[name],args) for name in order}
        after = compiled_graphs()
        if after != before:
            raise RuntimeError("Unexpected graph compilation during warm benchmark")
        rounds.append(dict(round=index+1,order=order,variants=values,unique_graphs_before=before,unique_graphs_after=after))
    medians = {name:{key:statistics.median(row["variants"][name][key] for row in rounds)
                     for key in ("wall_ms","cuda_event_ms","peak_allocated_bytes","peak_reserved_bytes")}
               for name in names}
    ratios = {name:{key:[row["variants"][name][key]/row["variants"]["compiled_lrpred_tensor"][key] for row in rounds]
                    for key in ("wall_ms","cuda_event_ms")} for name in names if name!="compiled_lrpred_tensor"}
    return dict(rounds=rounds,medians=medians,paired_baseline_over_compiled=ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--rounds",type=int,default=4)
    parser.add_argument("--iters",type=int,default=20)
    args = parser.parse_args()
    if min(args.warmup,args.rounds,args.iters)<1:
        parser.error("Require positive warmup/rounds/iters")
    args.output = args.output or ROOT/"artifacts/native_deconv_target"/("compiled_nonoverlap_smoke.json" if args.smoke else "compiled_nonoverlap.json")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="1":
        parser.error("TF32 override conflicts with the FP32 protocol")
    # Set before importing torch or creating any CUDA context in this process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cache_dirs = {"TORCHINDUCTOR_CACHE_DIR":ROOT/".build/inductor_nonoverlap",
                  "TRITON_CACHE_DIR":ROOT/".build/triton_nonoverlap"}
    cache_state = {}
    for key,path in cache_dirs.items():
        cache_state[key] = dict(path=str(path),existed=path.exists(),
                               had_entries=path.exists() and any(path.iterdir()))
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key] = str(path)
    import torch
    import torch._dynamo
    from fp32_training_baseline import current_manifest
    torch._dynamo.config.suppress_errors = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    report = dict(status="initializing",scope=__doc__,smoke=args.smoke,source_sha256=current_manifest(),
        script_sha256={name:hashlib.sha256((ROOT/"test"/name).read_bytes()).hexdigest() for name in
            ("probe_compiled_nonoverlap.py","probe_spatial_nonoverlap_fast.py","probe_spatial_nonoverlap.py",
             "probe_nearest_training.py","probe_pointwise_training.py","probe_training_s1_shapes.py","test_fp32_training.py")},
        reference_sha256=hashlib.sha256((ROOT/"models/converse_core.py").read_bytes()).hexdigest(),
        compile_settings=dict(backend="inductor",fullgraph=True,dynamic=False,options=OPTIONS,suppress_errors=False),
        cache_directories=cache_state,settings=dict(warmup=args.warmup,rounds=args.rounds,iters=args.iters),
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,tf32=False,amp=False,
                         deterministic_algorithms=True,cudnn_deterministic=True,cublas_workspace_config=":4096:8"),
        excluded_cases=[],cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")

    try:
        import triton
        report["environment"]["triton"] = str(triton.__version__)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU/eager fallback")
        report["environment"]["gpu"] = torch.cuda.get_device_name()
        if not args.smoke:
            from extension_loader import load_extension
            load_extension()
        cases = []
        for case in make_cases():
            if args.smoke and case["name"]!="b1_c32_64x80_s3":
                continue
            if all(size<=case["scale"] for size in case["tensors"][1].shape[-2:]):
                cases.append(case)
            else:
                report["excluded_cases"].append(dict(name=case["name"],reason="Kernel support exceeds scale; requires production"))
        if len(cases)!=(1 if args.smoke else 10):
            raise RuntimeError("Original eligible fixture count changed")
        compiled_methods = {}
        report["status"] = "compiling_and_validating"
        for case in cases:
            compiled_eligible = case["scale"]==3 and tuple(case["tensors"][1].shape[-2:])==(3,3)
            row = dict(name=case["name"],shape=list(case["tensors"][0].shape),kernel_shape=list(case["tensors"][1].shape),
                       scale=case["scale"],eps=case["eps"],weak=case["weak"],tensors_sha256=tensor_hash(case["tensors"]),
                       compiled_eligible=compiled_eligible,setup=None,validation=None,repeatability=None,timing=None)
            report["cases"].append(row)
            save()
            print(f"Preparing {case['name']} (compiled={compiled_eligible})",flush=True)
            compiled = None
            if compiled_eligible:
                compiled,row["setup"] = prepare_compiled(case)
                compiled_methods[case["name"]] = compiled
                save()
            else:
                row["coverage_note"] = "k1 eager/production validation only; no compiled or timing claim"
            choices = methods(case,compiled,include_production=not args.smoke)
            row["validation"] = validate(case,choices)
            row["repeatability"] = repeatability(case,choices)
            save()
            print(json.dumps(dict(case=case["name"],setup=row["setup"],validation=row["validation"],
                                  repeatability_passed={name:value["passed"] for name,value in row["repeatability"].items()})),flush=True)
        if not all(all(value["passed"] for value in row["validation"].values()) and
                   all(value["passed"] for value in row["repeatability"].values()) for row in report["cases"]):
            report["status"] = "numerical_or_repeatability_gate_failed_no_timing"
            save()
            return 1
        if args.smoke:
            report["status"] = "complete_compiled_smoke_no_warm_performance_claim"
            save()
            return 0
        # Any new Dynamo guard specialization during timing is an error.
        torch._dynamo.config.error_on_recompile = True
        report["status"] = "timing"
        for case,row in zip(cases,report["cases"]):
            if case["timed"] and row["compiled_eligible"]:
                row["timing"] = benchmark(case,methods(case,compiled_methods[case["name"]]),args)
                save()
                print(json.dumps(dict(case=case["name"],medians=row["timing"]["medians"])),flush=True)
        report["status"] = "complete_isolated_compiled_nonoverlap"
        save()
        return 0
    except Exception as error:
        report["status"] = "failed_no_fallback"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        save()
        raise


if __name__=="__main__":
    raise SystemExit(main())
