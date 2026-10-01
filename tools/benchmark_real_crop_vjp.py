"""Interleaved operator-only real/crop VJP timing; no model/optimizer training."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import experiment_nan_fill as common

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.root=args.root.resolve(); args.output=args.output.resolve()
if args.output.exists(): parser.error('Use a new output path')
args.output.parent.mkdir(parents=True,exist_ok=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CONVERSE2D_SKIP_BUILD']='1'
common.affinity(0xC03C03)
sys.path[:0]=[str(args.root/'test'),str(args.root)]
import torch
import extension_loader
common.configure(torch,False); extension_loader.load_extension()
torch.manual_seed(831)
result={'manifest':json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),'cases':[]}
for batch in (1,4):
    z=torch.randn(batch,128,100,100,device='cuda',dtype=torch.complex64,requires_grad=True)
    outputs={'reference':z.real[...,2:-2,2:-2], 'candidate':torch.ops.converse2d._training_real_crop(z,2)}
    for layout in ('contiguous','transpose'):
        g=torch.randn(batch,128,96,96,device='cuda')
        if layout=='transpose': g=g.transpose(-1,-2).contiguous().transpose(-1,-2)
        row={'batch':batch,'layout':layout,'grad_stride':list(g.stride()),'rounds':[]}
        control=None
        for name,out in outputs.items():
            value=torch.autograd.grad(out,z,g,retain_graph=True)[0]
            record=common.tensor_record(value)
            if control is None: control=record
            else: assert record==control
            for _ in range(10): torch.autograd.grad(out,z,g,retain_graph=True)
        for index in range(6):
            pair={}
            for name in (('reference','candidate') if index%2==0 else ('candidate','reference')):
                out=outputs[name]
                torch.cuda.synchronize(); start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                began=time.perf_counter(); start.record()
                for _ in range(30): torch.autograd.grad(out,z,g,retain_graph=True)
                end.record(); end.synchronize()
                pair[name]={'wall_ms':(time.perf_counter()-began)*1000/30,'cuda_ms':start.elapsed_time(end)/30}
            row['rounds'].append(pair)
        row['summary']={name:statistics.median(p[name]['cuda_ms'] for p in row['rounds']) for name in outputs}
        row['summary']['paired_speedup']=statistics.median(p['reference']['cuda_ms']/p['candidate']['cuda_ms'] for p in row['rounds'])
        result['cases'].append(row)
        print(json.dumps({'batch':batch,'layout':layout,**row['summary']}),flush=True)
args.output.write_text(json.dumps(result,indent=2))
