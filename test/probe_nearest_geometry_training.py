"""Fixture-local immutable geometry phases; no trainable parameter caching.

    python test/probe_nearest_geometry_training.py

Compare production, fused_dynamic_phase and fused_geometry_phase using the
unchanged twelve cases from probe_nearest_training.py. Kernel preparation,
lambda and FFT/IFFT run on EVERY call in all three routes. Only phaseH/W,
which depend on H/W/scale/device/complex dtype, survive between calls of one
geometry fixture. The fixture owns its phases; there is no global cache.

Predeclared FP64 gates: normal output 3e-5/3e-5, all dx/dw/db 5e-5/5e-5,
weak output 1e-6/1e-5. Phase values must be byte-identical to dynamic creation.
The existing internal complex/higher-order gates run unchanged. All numerical
and repeatability gates precede timing. No tolerance, seed, eps or phase
formula is changed. This does not claim full-network or optimizer speed.

Each measured geometry fixture reports phase cache-miss generation separately
(wall/event time and allocated/reserved memory, in an already initialized
process). Its warm FWD+all-VJP timing includes all dynamic preparation and
resident phase memory, but excludes that explicitly reported miss cost.
Warm5, alternating round4/iters20, one fixture at a time, no profiler or leaf
gradient accumulation. Process/library cold start and total device memory
are outside these measurements.
"""
import argparse
from contextlib import contextmanager
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
from probe_nearest_training import make_cases, phase, prepare_kernel, spatial
from probe_nearest_fused_training import fused_spatial, validate_spectral
from probe_pointwise_training import capture, clear_cuda, fixture, tensor_hash, timed_fixture
from probe_training_s1_shapes import metrics
from experiments.training_nearest.loader import load_candidate

ROUTES = ("production","fused_dynamic_phase","fused_geometry_phase")


class GeometryPhases:
    """One immutable geometry pair, privately owned by one CUDA fixture."""

    def __init__(self,case,measure_generation=False):
        import torch
        height,width = case["tensors"][0].shape[-2:]
        scale = case["scale"]
        device = torch.device("cuda",torch.cuda.current_device())
        self.key = (height,width,scale,device.index,torch.complex64)
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.hits = 0
        self.misses = 1
        self.cold = None
        if measure_generation:
            torch.cuda.synchronize()
            initial_allocated = torch.cuda.memory_allocated()
            initial_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            began = time.perf_counter()
            start.record()
        # The phase implementation only reads device and dtype from `like`.
        # Its uninitialized scalar value is never consumed or retained.
        with torch.no_grad():
            like = torch.empty((),device=device,dtype=torch.complex64)
            self._values = (phase(height*scale,scale,like),phase(width*scale,scale,like))
            del like
        if measure_generation:
            end.record()
            torch.cuda.synchronize()
            self.cold = dict(wall_ms=(time.perf_counter()-began)*1000,
                cuda_event_ms=start.elapsed_time(end),initial_allocated_bytes=initial_allocated,
                initial_reserved_bytes=initial_reserved,final_allocated_bytes=torch.cuda.memory_allocated(),
                final_reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                scope="One phase cache miss in an initialized process; phase creation only, no spatial operator")
        self._versions = tuple(value._version for value in self._values)
        if any(value.requires_grad or value.grad_fn is not None for value in self._values):
            raise RuntimeError("Geometry phases must have no autograd graph")

    def get(self,y,height,width,scale):
        if (height,width,scale,y.device.index,y.dtype) != self.key:
            raise RuntimeError("Fixture geometry/device/dtype mismatch")
        if tuple(value._version for value in self._values) != self._versions:
            raise RuntimeError("Immutable geometry phases were modified")
        self.hits += 1
        return self._values

    def metadata(self):
        import torch
        if torch.cuda.current_stream(self._values[0].device).cuda_stream != self.stream:
            raise RuntimeError("This experiment requires creation and use on the same CUDA stream")
        return dict(key=dict(height=self.key[0],width=self.key[1],scale=self.key[2],
                             device_index=self.key[3],spectrum_dtype=str(self.key[4])),
                    creation_stream=self.stream,cache_hits=self.hits,cache_misses=self.misses,
                    resident_tensor_bytes=sum(value.numel()*value.element_size() for value in self._values),
                    phase_sha256=[tensor_hash((value,)) for value in self._values],
                    requires_grad=[value.requires_grad for value in self._values],
                    cold_generation=self.cold)

    def close(self):
        self._values = ()


def geometry_spatial(ops,geometry,x,kernel,bias,scale,eps):
    import torch
    height,width = x.shape[-2:]
    x = x.contiguous()
    regularizer = torch.sigmoid(bias.contiguous()-9.)+eps
    k = prepare_kernel(kernel,height*scale,width*scale)
    y = torch.fft.rfft2(x)
    ph,pw = geometry.get(y,height,width,scale)
    out = ops._training_nearest_spectral(y,k,regularizer,ph,pw,height,width,scale)
    return torch.fft.irfft2(out,s=(height*scale,width*scale))


@contextmanager
def route(ops,case,name,measure_generation=False):
    clear_cuda()
    s,eps = case["scale"],case["eps"]
    cache = None
    if name == "production":
        method = lambda x,k,b:spatial(x,k,b,s,eps,"production")
    elif name == "fused_dynamic_phase":
        method = lambda x,k,b:fused_spatial(ops,x,k,b,s,eps)
    elif name == "fused_geometry_phase":
        cache = GeometryPhases(case,measure_generation)
        method = lambda x,k,b:geometry_spatial(ops,cache,x,k,b,s,eps)
    else:
        raise ValueError(name)
    try:
        yield method,cache
    finally:
        if cache is not None:
            cache.close()
        del method,cache
        clear_cuda()


def phase_check(case):
    import torch
    clear_cuda()
    cache = GeometryPhases(case)
    x = case["tensors"][0].cuda()
    y = torch.fft.rfft2(x)
    h,w = x.shape[-2:]
    s = case["scale"]
    actual = cache.get(y,h,w,s)
    dynamic = (phase(h*s,s,y),phase(w*s,s,y))
    a_hashes = [tensor_hash((value,)) for value in actual]
    d_hashes = [tensor_hash((value,)) for value in dynamic]
    passed = a_hashes == d_hashes
    result = dict(passed=passed,geometry_phase_sha256=a_hashes,dynamic_phase_sha256=d_hashes,
                  max_abs=[(a-d).abs().max().item() for a,d in zip(actual,dynamic)],cache=cache.metadata())
    cache.close()
    del cache,x,y,actual,dynamic
    clear_cuda()
    return result


def validate(ops,case):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    s,eps = case["scale"],case["eps"]
    expected = capture(case["tensors"],torch.float64,lambda x,k,b:converse2d_reference(
        x,x if s==1 else F.interpolate(x,scale_factor=s,mode="nearest"),k,b,s,eps))
    result = {}
    for name in ROUTES:
        with route(ops,case,name) as (method,cache):
            actual = capture(case["tensors"],torch.float32,method)
            if any(value.dtype != torch.float32 for value in actual.values()):
                raise RuntimeError(f"Unexpected output/gradient dtype in {name}")
            errors = {key:metrics(actual[key],value,output=key=="output",weak=case["weak"])
                      for key,value in expected.items()}
            result[name] = dict(passed=all(item["passed"] for item in errors.values()),tensors=errors,
                                geometry_cache=cache.metadata() if cache is not None else None)
    return result


def repeatability(ops,case):
    import torch
    result = {}
    for name in ROUTES:
        with route(ops,case,name) as (method,cache):
            inputs,run = fixture(case["tensors"],torch.float32,method)
            first,first_hashes,records = None,None,[]
            for index in range(3):
                values = run()
                saved = {key:value.detach().cpu().clone() for key,value in values.items()}
                del values
                hashes = {key:tensor_hash((value,)) for key,value in saved.items()}
                if first is None:
                    first,first_hashes = saved,hashes
                records.append(dict(repetition=index+1,tensors={key:dict(
                    sha256=hashes[key],bitwise_equal_to_first=hashes[key]==first_hashes[key],
                    finite=bool(torch.isfinite(value).all()),
                    max_abs_to_first=(value.double()-first[key].double()).abs().max().item())
                    for key,value in saved.items()}))
            if any(value.grad is not None for value in inputs):
                raise RuntimeError("Repeatability accumulated leaf gradients")
            result[name] = dict(records=records,all_bitwise_equal=all(item["bitwise_equal_to_first"] and item["finite"]
                for row in records for item in row["tensors"].values()),
                geometry_cache=cache.metadata() if cache is not None else None)
            del inputs,run,first,saved
    return result


def dispatch_check(ops,case):
    import torch
    result = {}
    for name in ROUTES[1:]:
        with route(ops,case,name) as (method,cache):
            inputs,run = fixture(case["tensors"],torch.float32,method)
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
            cosine,sine = counts.get("aten::cos",0),counts.get("aten::sin",0)
            expected_trig = 0 if name=="fused_geometry_phase" else 2
            passed = all(value==1 for value in hits.values()) and cosine==sine==expected_trig
            passed &= counts.get("aten::fft_rfft2",0)==1 and counts.get("aten::upsample_nearest2d",0)==0
            result[name] = dict(passed=bool(passed),kernel_hits=hits,phase_cosine_ops=cosine,
                phase_sine_ops=sine,expected_each_trig=expected_trig,activation_rfft2=counts.get("aten::fft_rfft2",0),
                geometry_cache=cache.metadata() if cache is not None else None)
            del profiler,run,inputs
    return result


def benchmark(ops,case,args):
    rounds = []
    for index in range(args.rounds):
        names = list(ROUTES)
        rotated = names[index%len(names):]+names[:index%len(names)]
        order = rotated if index%2==0 else list(reversed(rotated))
        values = {}
        for name in order:
            with route(ops,case,name,measure_generation=True) as (method,cache):
                values[name] = timed_fixture(case["tensors"],method,args)
                values[name]["geometry_cache"] = cache.metadata() if cache is not None else None
                if cache is not None and (cache.misses != 1 or cache.hits != args.warmup+args.iters+1):
                    raise RuntimeError("Unexpected cache lifetime/hit count")
        rounds.append(dict(round=index+1,order=order,variants=values))
    medians = {name:{key:statistics.median(row["variants"][name][key] for row in rounds)
                     for key in ("wall_ms","cuda_event_ms","peak_allocated_bytes","peak_reserved_bytes")}
               for name in ROUTES}
    cold = {key:statistics.median(row["variants"]["fused_geometry_phase"]["geometry_cache"]["cold_generation"][key]
                                 for row in rounds) for key in ("wall_ms","cuda_event_ms","peak_allocated_bytes","peak_reserved_bytes")}
    ratios = {name:{key:[row["variants"][name][key]/row["variants"]["fused_geometry_phase"][key] for row in rounds]
                    for key in ("wall_ms","cuda_event_ms")} for name in ROUTES[:-1]}
    return dict(rounds=rounds,warm_medians=medians,phase_cache_miss_medians=cold,
                paired_baseline_over_geometry=ratios,
                accounting="Warm times include all kernel preparation and FFT/VJP; only the separately reported immutable phase miss is excluded. Resident phase tensors are included in each geometry initial/peak allocation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/native_deconv_target/nearest_geometry_training.json")
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--rounds",type=int,default=4)
    parser.add_argument("--iters",type=int,default=20)
    args = parser.parse_args()
    if min(args.warmup,args.rounds,args.iters)<1:
        parser.error("warmup, rounds and iters must be positive")
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
    ops,build = load_candidate()
    report = dict(status="validating",scope=__doc__,build=build,source_sha256=current_manifest(),
        script_sha256={name:hashlib.sha256((ROOT/"test"/name).read_bytes()).hexdigest() for name in
            ("probe_nearest_geometry_training.py","probe_nearest_fused_training.py","probe_nearest_training.py",
             "probe_training_s1_shapes.py","probe_pointwise_training.py","test_fp32_training.py")},
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                         tf32=False,cudnn_deterministic=True,
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        settings=dict(warmup=args.warmup,rounds=args.rounds,iters=args.iters),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")

    try:
        report["internal_spectral"] = validate_spectral(ops)
        save()
        cases = make_cases()
        for case in cases:
            x,k,b,_ = case["tensors"]
            row = dict(name=case["name"],shape=list(x.shape),kernel_shape=list(k.shape),bias_shape=list(b.shape),
                       scale=case["scale"],eps=case["eps"],weak=case["weak"],tensors_sha256=tensor_hash(case["tensors"]),
                       phase_equivalence=phase_check(case),validation=validate(ops,case),timing=None)
            report["cases"].append(row)
            save()
            print(json.dumps(dict(case=case["name"],phase_passed=row["phase_equivalence"]["passed"],validation=row["validation"])),flush=True)
        if not all(row["phase_equivalence"]["passed"] and all(check["passed"] for check in row["validation"].values())
                   for row in report["cases"]):
            report["status"] = "phase_or_numerical_gate_failed_no_timing"
            save()
            raise SystemExit(1)
        representatives = [case for case in cases if case["name"] in ("b32_c32_64x80_s3","s1_control")]
        report["repeatability"] = {case["name"]:repeatability(ops,case) for case in representatives}
        report["dispatch_check"] = dispatch_check(ops,representatives[0])
        save()
        if not all(value["all_bitwise_equal"] for rows in report["repeatability"].values() for value in rows.values()) or \
           not all(value["passed"] for value in report["dispatch_check"].values()):
            report["status"] = "repeatability_or_dispatch_failed_no_timing"
            save()
            raise SystemExit(1)
        report["status"] = "timing"
        for case,row in zip(cases,report["cases"]):
            if case["timed"]:
                row["timing"] = benchmark(ops,case,args)
                save()
                print(json.dumps(dict(case=case["name"],warm_medians=row["timing"]["warm_medians"],
                                      phase_cache_miss_medians=row["timing"]["phase_cache_miss_medians"])),flush=True)
        report["status"] = "complete_isolated_geometry_phase_candidate"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        save()
        raise
    print(f"Saved {args.output}",flush=True)


if __name__=="__main__":
    main()
