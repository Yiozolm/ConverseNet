"""Compare spectral I/O changes with the frozen v7 on identical GPU inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch
from extension_loader import production_source_hashes, ROOT, load_extension
from spectral_baseline import BASELINE_REF, load_baseline
sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference


def measure_pair(functions, iters, rounds):
    samples = {name: [] for name in functions}
    for repeat in range(rounds):
        order = list(functions)
        if repeat % 2: order.reverse()
        for name in order:
            fn = functions[name]
            for _ in range(5): fn()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            t0 = time.perf_counter()
            a.record()
            for _ in range(iters): fn()
            b.record(); b.synchronize()
            samples[name].append({'gpu_ms':a.elapsed_time(b)/iters,
                                  'wall_ms':(time.perf_counter()-t0)*1000/iters,
                                  'peak_extra_bytes':torch.cuda.max_memory_allocated()-allocated})
    medians = {name:{key:statistics.median(r[key] for r in rows)
                     for key in ('gpu_ms','wall_ms','peak_extra_bytes')}
               for name,rows in samples.items()}
    return {'medians':medians,'rounds':samples,
            'speedup':medians['baseline']['wall_ms']/medians['optimized']['wall_ms']}


def invoke(namespace, fn):
    # Both models and the graph runner use torch.ops.converse2d dynamically.
    # Switch the whole namespace, including the graph-owned cache operations.
    with patch.object(torch.ops, 'converse2d', namespace):
        return fn()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--iters',type=int,default=20)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--operators-only',action='store_true')
    parser.add_argument('--profile',action='store_true',help='NVTX-marked dynamic s2 baseline/optimized capture')
    parser.add_argument('--output',default='artifacts/spectral_io_benchmark.json')
    args = parser.parse_args()
    if min(args.iters,args.rounds) < 1: parser.error('positive iters and rounds required')
    load_extension()
    baseline = load_baseline()
    optimized = torch.ops.converse2d
    torch.manual_seed(921)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.profile:
        x = torch.randn(1,64,128,128,device='cuda')
        prior = torch.nn.functional.interpolate(x,scale_factor=2)
        weight = torch.randn(1,64,7,7,device='cuda')/49
        bias = torch.zeros(1,64,1,1,device='cuda')
        with torch.inference_mode():
            def step(ns):
                return ns.forward(x,prior,weight.clone(),bias,2,1e-5,'v7')
            torch.testing.assert_close(step(optimized),step(baseline),atol=1e-4,rtol=5e-5)
            for _ in range(5): step(baseline); step(optimized)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for label,ns in (('baseline',baseline),('optimized',optimized)):
                with torch.cuda.nvtx.range(label):
                    for _ in range(args.iters): step(ns)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        return
    rows = []
    cases = ((1,64,128,128,1),(1,64,128,128,2),(1,32,128,128,3),
             (1,32,127,129,3),(2,64,32,40,2))
    for b,c,h,w,s in cases:
        for dynamic in (False,True):
            x = torch.randn(b,c,h,w,device='cuda')
            prior = x if s==1 else torch.nn.functional.interpolate(x,scale_factor=s)
            weight = torch.randn(b if dynamic else 1,c,7,7,device='cuda').flatten(2).softmax(-1).reshape(-1,c,7,7)
            bias = torch.zeros(1,c,1,1,device='cuda')
            with torch.inference_mode():
                reference = converse2d_reference(x.double(),prior.double(),weight.double(),bias.double(),s,1e-5)
                def call(ns):
                    k = weight.clone() if dynamic else weight
                    return ns.forward(x,prior,k,bias,s,1e-5,'v7')
                old, new = call(baseline), call(optimized)
                for out in (old,new):
                    torch.testing.assert_close(out.double(),reference,atol=1e-4,rtol=5e-5)
                error = {'baseline_vs_fp64':(old.double()-reference).abs().max().item(),
                         'optimized_vs_fp64':(new.double()-reference).abs().max().item(),
                         'optimized_vs_baseline':(new-old).abs().max().item()}
                row = {'kind':'operator','shape':[b,c,h,w],'scale':s,'dynamic':dynamic,
                       'errors':error,**measure_pair({'baseline':lambda:call(baseline),
                                                     'optimized':lambda:call(optimized)},args.iters,args.rounds)}
            rows.append(row)
            print(json.dumps(row),flush=True)
            baseline.clear_cache(); optimized.clear_cache()
    if not args.operators_only:
        from models.converse_usrnet import ConverseUSRNet
        from models.cuda_graph import USRNetCUDAGraph
        model = ConverseUSRNet(backend='cuda').cuda().eval()
        model.load_state_dict(torch.load(ROOT/'model_zoo/converse_usrnet.pth',map_location='cuda',weights_only=True))
        for h,w in ((32,40),(64,80)):
            x = torch.rand(1,3,h,w,device='cuda')
            k = torch.rand(1,1,7,7,device='cuda'); k /= k.sum()
            with torch.inference_mode():
                old = invoke(baseline,lambda:model(x,k,2))
                new = invoke(optimized,lambda:model(x,k,2))
                torch.testing.assert_close(new,old,atol=3e-5,rtol=3e-5)
                error = (new-old).abs().max().item()
                runners = [USRNetCUDAGraph(model),USRNetCUDAGraph(model)]
                for graph in (False,True):
                    functions = {'baseline':lambda:invoke(baseline,lambda:runners[0](x,k,2) if graph else model(x,k,2)),
                                 'optimized':lambda:invoke(optimized,lambda:runners[1](x,k,2) if graph else model(x,k,2))}
                    for name,reference in (('baseline',old),('optimized',new)):
                        torch.testing.assert_close(functions[name](),reference,atol=3e-5,rtol=3e-5)
                    row = {'kind':'usrnet_graph' if graph else 'usrnet_eager','shape':list(x.shape),
                           'scale':2,'max_abs_vs_baseline':error,
                           **measure_pair(functions,args.iters,args.rounds)}
                    rows.append(row); print(json.dumps(row),flush=True)
                for runner in runners: runner.clear()
            baseline.clear_cache(); optimized.clear_cache()
    sources = ['Converse2D/torch_converse2d/converse2d.cpp','Converse2D/torch_converse2d/converse2d_kernels.cu']
    result = {'production_source_sha256':production_source_hashes(), 'baseline_ref':BASELINE_REF,'gpu':torch.cuda.get_device_name(),
              'torch':torch.__version__,'dtype':'float32','tf32':False,
              'iters':args.iters,'round_count':args.rounds,
              'method':'Same-process alternating frozen baseline/optimized, warmed caches; no profiler.',
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'results':rows}
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
