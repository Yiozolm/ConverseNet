"""Alternating baseline/current FP16/BF16 inference and forward+backward timing."""
import argparse
import json
import statistics
import sys
import time

import torch
from extension_loader import ROOT, load_extension
from low_precision_baseline import BASELINE_REF, load_baseline

sys.path.insert(0,str(ROOT))
from models.converse_core import converse2d_reference_nearest


def sample(fn,iters):
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return {"gpu_ms":start.elapsed_time(end)/iters,"wall_ms":(time.perf_counter()-t)*1000/iters,
            "peak_extra_bytes":torch.cuda.max_memory_allocated()-initial}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--iters",type=int,default=20)
    parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--mode",choices=("all","inference","training"),default="all")
    parser.add_argument("--dtype",choices=("all","float16","bfloat16"),default="all")
    parser.add_argument("--case",choices=("all","small_s1","small_s2","small_s3","batch16","batch17"),default="all")
    parser.add_argument("--output",default="artifacts/low_precision_benchmark.json")
    args = parser.parse_args()
    if min(args.iters,args.rounds)<1:
        parser.error("iters and rounds must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    load_extension()
    baseline = load_baseline()
    torch.manual_seed(482)
    rows = []
    for training in (False,True):
        if args.mode != "all" and training != (args.mode == "training"):
            continue
        for dtype in (torch.float16,torch.bfloat16):
            if args.dtype != "all" and str(dtype) != f"torch.{args.dtype}":
                continue
            cases = {"small_s1":(1,64,80,1),"small_s2":(1,64,80,2),"small_s3":(4,32,40,3)}
            if not training:
                cases.update(batch16=(16,64,64,2),batch17=(17,64,64,2))
            for case,(batch,h,w,scale) in cases.items():
                if args.case != "all" and args.case != case:
                    continue
                # Same low-precision inputs and parameters isolate implementation
                # changes; AMP/master-weight integration is tested separately.
                x = torch.randn(batch,32,h,w,device="cuda",dtype=dtype,requires_grad=training)
                weight = (torch.randn(1,32,3,3,device="cuda").flatten(2).softmax(-1)
                          .reshape(1,32,3,3).to(dtype).requires_grad_(training))
                bias = torch.zeros(1,32,1,1,device="cuda",dtype=dtype,requires_grad=training)
                with torch.no_grad():
                    reference = converse2d_reference_nearest(x.double(),weight.double(),bias.double(),scale)
                with torch.enable_grad() if training else torch.no_grad():
                    forwards = {"baseline":lambda:baseline.forward_nearest(x,weight,bias,scale),
                                "current":lambda:torch.ops.converse2d.forward_nearest(x,weight,bias,scale)}
                    outputs = {name:fn().detach() for name,fn in forwards.items()}
                    error = {name:{"max_abs":(out.double()-reference).abs().max().item(),
                                   "relative_l2":((out.double()-reference).norm()/reference.norm()).item()}
                             for name,out in outputs.items()}
                    for out in outputs.values():
                        torch.testing.assert_close(out.double(),reference,atol=0.008 if dtype==torch.float16 else 0.08,rtol=0.008)
                    calls = {name:(lambda fn=fn:torch.autograd.grad(fn().float().square().mean(),(x,weight,bias)))
                             if training else fn for name,fn in forwards.items()}
                    for fn in calls.values():
                        for _ in range(5):
                            fn()
                    measurements = {name:[] for name in calls}
                    for i in range(args.rounds):
                        for name in (list(calls) if i%2==0 else list(reversed(calls))):
                            measurements[name].append(sample(calls[name],args.iters))
                    medians = {name:{key:statistics.median(v[key] for v in values)
                                     for key in values[0]} for name,values in measurements.items()}
                    row = dict(training=training,dtype=str(dtype),shape=list(x.shape),scale=scale,
                               error=error,bitwise_equal=torch.equal(outputs["baseline"],outputs["current"]),
                               measurements=measurements,medians=medians,
                               speedup=medians["baseline"]["gpu_ms"]/medians["current"]["gpu_ms"])
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k!="measurements"}),flush=True)
                baseline.clear_cache()
                torch.ops.converse2d.clear_cache()
    if not rows:
        parser.error("no cases match the selected mode/dtype/case")
    output = ROOT / args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,baseline_ref=BASELINE_REF,iters=args.iters,rounds=args.rounds,
        selection=dict(mode=args.mode,dtype=args.dtype,case=args.case),
        scope="same low-precision inputs; training is forward+all input gradients, no optimizer",results=rows),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
