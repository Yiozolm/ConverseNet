"""Direct original-binary frozen-model diagnostic, no hooks and no optimizer updates."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import experiment_nan_fill as common

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--arm',choices=('native','fused'),required=True)
args=parser.parse_args()
args.root,args.data_root,args.output=(p.resolve() for p in (args.root,args.data_root,args.output))
if args.output.exists(): parser.error('Use a new output directory')
args.output.mkdir(parents=True)
(args.output/'harness.py').write_bytes(Path(__file__).read_bytes())
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; os.environ['CONVERSE2D_SKIP_BUILD']='1'
common.affinity(0xC03C03)
sys.path[:0]=[str(args.root/'test'),str(args.root),str(args.root/'tools/roadmap_quality')]
import torch
import extension_loader
common.configure(torch,False); extension_loader.load_extension()
from models.converse_usrnet import ConverseUSRNet
from models.util_converse import Converse2D
import usrnet_training_data as data
data.ROOT=args.data_root
# Reproduce the original CPU-only profiler preflight, then release it completely.
probe=Converse2D(128,128,3,padding=2,backend='cuda').cuda().train()
probe_x=torch.zeros(4,128,96,96,device='cuda',requires_grad=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
    probe_out=probe(probe_x)
    node=probe_out.grad_fn.name()
    probe_grads=torch.autograd.grad(probe_out,(probe_x,probe.weight,probe.bias),torch.ones_like(probe_out))
assert ('RealCropBackward' in node)==(args.arm=='fused'),node
del probe,probe_x,probe_out,probe_grads,trace
gc.collect(); torch.cuda.empty_cache()
torch.manual_seed(17)
model=ConverseUSRNet(backend='cuda').cuda().train()
checkpoint=args.data_root/'artifacts/training_real_crop/v2/candidate_epoch/final.pth'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
model.load_state_dict(state['model'],strict=True)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,foreach=False,fused=False)
optimizer.load_state_dict(state['optimizer'])
initial={name:common.tensor_record(v) for name,v in model.state_dict().items()}
opt_initial={str(i)+'/'+k:common.tensor_record(v) for i,s in optimizer.state_dict()['state'].items() for k,v in s.items()}
protocol=data.DatasetProtocol(args.data_root/'artifacts/dataset_training/split_900_100.json',patch_size=96,scale=3,seed=17,noise_std=.01)
batch=tuple(t.cuda() for t in protocol.train_batch(12,4))
def run():
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    began=time.perf_counter(); start.record()
    out=model(batch[0],batch[1],3)
    loss=torch.nn.functional.mse_loss(out,batch[2])*1.
    loss.backward(); end.record(); end.synchronize()
    row={'fb_wall_ms':(time.perf_counter()-began)*1000,'fb_cuda_ms':start.elapsed_time(end),'loss':loss.item()}
    return row,out.detach()
for _ in range(3): run()
rows=[]
for i in range(20):
    row,out=run(); rows.append(row)
snapshot={'output':common.tensor_record(out),'gradients':{n:{'record':common.tensor_record(p.grad),'stride':list(p.grad.stride()),'offset':p.grad.storage_offset()} for n,p in model.named_parameters()}}
assert initial=={name:common.tensor_record(v) for name,v in model.state_dict().items()}
assert opt_initial=={str(i)+'/'+k:common.tensor_record(v) for i,s in optimizer.state_dict()['state'].items() for k,v in s.items()}
result={'status':'passed','arm':args.arm,'root':str(args.root),'harness_sha256':common.sha(__file__),'checked_manifest':json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),'checkpoint_sha256':common.sha(checkpoint),'rows':rows,'snapshot':snapshot,'mean_fb_ms':statistics.mean(r['fb_wall_ms'] for r in rows),'median_fb_ms':statistics.median(r['fb_wall_ms'] for r in rows),'parameters_unchanged':True,'optimizer_unchanged':True,'scope':'Original binary, no hooks, no NVML, old CPU-profiler preflight reproduced. Fixed trained checkpoint, fixed real data batch index12, 3 warmup and20 diagnostic FB calls; zero parameter/Adam updates.'}
common.write(args.output/'result.json',result)
print(json.dumps({k:result[k] for k in ('arm','mean_fb_ms','median_fb_ms')}),flush=True)
