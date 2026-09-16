"""Production nearest entry versus explicit interpolation + independent-prior API."""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F
from extension_loader import ROOT, load_extension


def measure(fn, iterations):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)/iterations


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iters',type=int,default=50)
    parser.add_argument('--output',default='artifacts/nearest_fused/benchmark.json')
    args=parser.parse_args()
    if args.iters<1:parser.error('--iters must be positive')
    load_extension()
    torch.manual_seed(932)
    rows=[]
    cases=[(1,64,256,256,1,False),(1,32,32,40,2,False),
           (1,64,128,128,2,False),(1,32,128,128,3,False),
           (1,32,127,129,3,False),(1,32,64,64,4,False),
           (8,32,128,128,2,False),(1,64,128,128,2,True)]
    with torch.no_grad():
        for B,C,H,W,s,dynamic in cases:
            torch.ops.converse2d.clear_cache()
            x=torch.randn(B,C,H,W,device='cuda')
            weight=torch.randn(1,C,3,3,device='cuda').flatten(2).softmax(-1).reshape(1,C,3,3)
            bias=torch.zeros(1,C,1,1,device='cuda')
            if dynamic:
                with torch.inference_mode():weight=weight.clone()
            def baseline():
                prior=x if s==1 else F.interpolate(x,scale_factor=s,mode='nearest')
                return torch.ops.converse2d.forward(x,prior,weight,bias,s,1e-5)
            def fused():return torch.ops.converse2d.forward_nearest(x,weight,bias,s,1e-5)
            fns={'baseline':baseline,'fused':fused}
            for _ in range(10):baseline();fused()
            old,new=baseline(),fused()
            torch.testing.assert_close(new,old,atol=1e-4,rtol=5e-5)
            row={'shape':[B,C,H,W],'scale':s,'dynamic_uncached':dynamic,
                 'max_abs_diff':(new-old).abs().max().item(),'rounds':{},'peak_extra_bytes':{}}
            del old,new
            for name,fn in fns.items():
                torch.cuda.synchronize()
                used=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                fn();torch.cuda.synchronize()
                row['peak_extra_bytes'][name]=torch.cuda.max_memory_allocated()-used
            for rep in range(9):
                for name in (list(fns) if rep%2==0 else list(fns)[::-1]):
                    row['rounds'].setdefault(name,[]).append(measure(fns[name],args.iters))
            row['median_ms']={k:statistics.median(v) for k,v in row['rounds'].items()}
            row['latency_reduction_pct']=100*(1-row['median_ms']['fused']/row['median_ms']['baseline'])
            rows.append(row)
            print(json.dumps(row),flush=True)
    output=ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                                 'cuda':torch.version.cuda,'iterations':args.iters,'results':rows},indent=2))


if __name__=='__main__':main()
