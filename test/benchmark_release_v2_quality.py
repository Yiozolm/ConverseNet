"""Precision, weak-regularization and paired real-image quality for release notes.

The protocol is written before model execution. This is a short fine-tuning
check, separate from speed measurement and from full-convergence claims.
"""
import argparse
import copy
import importlib.util
import json
import statistics
from pathlib import Path
import sys

import numpy as np
import torch
import benchmark_release_v2 as bench


def helper(name, data_root):
    raw = bench.git_source('1b579ea', f'test/{name}.py')
    path = bench.OUT/f'{name}_frozen.py'
    path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location('_quality_'+name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = data_root
    return mod


def aggregate(rows):
    return {space:{metric:statistics.mean(row[space][metric] for row in rows)
                   for metric in ('psnr','ssim')} for space in ('rgb','y')}


def evaluate(model, batches, quality):
    model.eval()
    rows=[]
    with torch.inference_mode():
        for ids,x,k,target in batches:
            output=model(x.cuda(),k.cuda(),3)
            assert torch.isfinite(output).all()
            for i in range(len(ids)):
                hr=target[i].mul(255).round().byte().permute(1,2,0).numpy()
                row=dict(image=ids[i])
                for space in ('rgb','y'):
                    q=quality(output[i:i+1].cpu(),hr,space,3)
                    row[space]=dict(psnr=q['psnr_db'],ssim=q['ssim'])
                rows.append(row)
    return dict(mean=aggregate(rows),per_image=rows)


def stress(backends):
    from models.converse_core import converse2d_reference
    rows=[]
    for s in (1,2,3):
        for weak in (False,True):
            torch.manual_seed(2850+s)
            original=backends['original_python'].util_converse.Converse2D(3,3,7,scale=s,padding=0,eps=1e-8 if weak else 1e-5).cuda()
            if weak:
                with torch.no_grad():
                    original.weight.mul_(1e-6)
                    original.bias.fill_(-40)
            ms={'original_python':original}
            for name in ('v1','v2'):
                ms[name]=backends[name].util_converse.Converse2D(3,3,7,scale=s,padding=0,eps=original.eps,backend='cuda').cuda()
                ms[name].load_state_dict(original.state_dict())
            raw=torch.randn(2,3,7,9)*(1e-5 if weak else 1)
            upstream=torch.randn(2,3,7*s,9*s)/(2*3*7*9*s*s)**.5
            def ref(dtype):
                x=raw.cuda().to(dtype).requires_grad_()
                w=original.weight.detach().to(dtype).requires_grad_()
                b=original.bias.detach().to(dtype).requires_grad_()
                prior=x if s==1 else torch.nn.functional.interpolate(x,scale_factor=s,mode='nearest')
                output=converse2d_reference(x,prior,w,b,s,original.eps)
                gs=torch.autograd.grad(output,(x,w,b),upstream.cuda().to(dtype))
                return dict(zip(('output','dx','weight','bias'),[v.detach().cpu() for v in (output,*gs)]))
            reference=ref(torch.float64)
            results={}
            for name,m in ms.items():
                x=raw.cuda().requires_grad_()
                values=bench.snapshots(lambda:m(x),{'dx':x},m.named_parameters(),upstream.cuda())
                results[name]={key:bench.errors(v,reference[key]) for key,v in values.items()}
            results['stable_python_fp32']={key:bench.errors(v,reference[key]) for key,v in ref(torch.float32).items()}
            rows.append(dict(scale=s,weak=weak,eps=original.eps,errors=results))
    bench.clear()
    return rows


def full_model_precision(backends, checkpoint, batch):
    x,k,_=batch
    torch.manual_seed(30101)
    up=torch.randn(1,3,48,48,device='cuda')/(3*48*48)**.5
    ref=backends['v1'].converse_usrnet.ConverseUSRNet(backend='pytorch').cuda().double()
    ref.load_state_dict(checkpoint,strict=True)
    xr,kr=x.cuda().double().requires_grad_(),k.cuda().double().requires_grad_()
    reference=bench.snapshots(lambda:ref(xr,kr,3),{'dx':xr,'dk':kr},ref.named_parameters(),up.double())
    del ref,xr,kr
    bench.clear()
    result={}
    for name in (*bench.LABELS,'stable_python_fp32'):
        if name=='stable_python_fp32':
            m=backends['v1'].converse_usrnet.ConverseUSRNet(backend='pytorch').cuda()
            m.load_state_dict(checkpoint,strict=True)
        else:
            m=bench.model_factory(backends[name],name,checkpoint)
        xt,kt=x.cuda().requires_grad_(),k.cuda().requires_grad_()
        actual=bench.snapshots(lambda:m(xt,kt,3),{'dx':xt,'dk':kt},m.named_parameters(),up)
        result[name]={key:bench.errors(v,reference[key]) for key,v in actual.items()}
        del m,xt,kt,actual
        bench.clear()
        print('Full-model precision: '+name,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20)
    args=p.parse_args()
    torch.set_num_threads(24)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    backends=bench.load_backends()
    data=helper('usrnet_training_data',args.data_root.resolve())
    metrics=helper('evaluate_usrnet_quality',args.data_root.resolve())
    protocol=dict(seeds=[17,29,43],steps=args.steps,patch_size=48,scale=3,batch=1,
                  optimizer='Adam lr=1e-5, foreach=False, fused=False',loss='unclipped RGB MSE',
                  validation='all 100 fixed held-out center crops, RGB/Y uint8 PSNR/SSIM, crop border 3',
                  acceptance=dict(max_psnr_drop_db=.05,max_ssim_drop=.001,
                                  comparisons=['original_python','v1','stable_python_fp32']),
                  data_root=str(args.data_root.resolve()),manifest=str(args.manifest.resolve()),
                  manifest_sha256=bench.sha(args.manifest.read_bytes()),
                  script_sha256=bench.sha(Path(__file__).read_bytes()),
                  shared_benchmark_script_sha256=bench.sha(Path(bench.__file__).read_bytes()),
                  note='Short paired fine-tuning, not long-run convergence or time-to-quality.')
    bench.write_json(bench.OUT/'quality_protocol.json',protocol)
    base=data.DatasetProtocol(args.manifest,patch_size=48,scale=3,seed=17)
    val=list(base.validation_batches(1))
    checkpoint=torch.load(bench.ROOT/'model_zoo/converse_usrnet.pth',map_location='cpu',weights_only=True)
    report=dict(protocol=protocol,dataset=base.metadata,stress=stress(backends),runs=[],
                current_source_hashes=bench.production_source_hashes())
    report['full_model_precision']=full_model_precision(backends,checkpoint,base.train_batch(0,1))
    bench.write_json(bench.OUT/'quality.json',report)
    for seed in protocol['seeds']:
        ds=copy.copy(base);ds.seed=seed;ds._permutations={}
        batches=[ds.train_batch(step,1) for step in range(args.steps)]
        batch_hashes=[bench.sha(b''.join(t.numpy().tobytes() for t in batch)) for batch in batches]
        names=[*bench.LABELS,'stable_python_fp32']
        offset=protocol['seeds'].index(seed)
        names=names[offset:]+names[:offset]
        for name in names:
            torch.manual_seed(seed)
            if name=='stable_python_fp32':
                m=backends['v1'].converse_usrnet.ConverseUSRNet(backend='pytorch').cuda()
                m.load_state_dict(checkpoint,strict=True)
            else:m=bench.model_factory(backends[name],name,checkpoint)
            optimizer=torch.optim.Adam(m.parameters(),lr=1e-5,foreach=False,fused=False)
            before=evaluate(m,val,metrics.quality)
            m.train();losses=[]
            for batch in batches:
                x,k,hr=(t.cuda() for t in batch)
                optimizer.zero_grad(set_to_none=True)
                output=m(x,k,3);loss=(output-hr).square().mean()
                loss.backward()
                assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters())
                assert torch.stack([torch.isfinite(p.grad).all() for p in m.parameters()]).all()
                optimizer.step()
                assert torch.stack([torch.isfinite(p).all() for p in m.parameters()]).all()
                losses.append(loss.item())
            after=evaluate(m,val,metrics.quality)
            report['runs'].append(dict(seed=seed,backend=name,batch_hashes=batch_hashes,
                                       before=before,after=after,losses=losses,finite=True))
            bench.write_json(bench.OUT/'quality.json',report)
            print(f'Quality seed={seed} {name}: '+str(after['mean']),flush=True)
            del m,optimizer,x,k,hr,output,loss
            bench.clear()
    gates=[]
    for seed in protocol['seeds']:
        runs={r['backend']:r for r in report['runs'] if r['seed']==seed}
        for control in protocol['acceptance']['comparisons']:
            assert runs['v2']['batch_hashes']==runs[control]['batch_hashes']
            for space in ('rgb','y'):
                a,b=runs['v2']['after']['mean'][space],runs[control]['after']['mean'][space]
                gates.append(dict(seed=seed,control=control,space=space,
                                  delta_psnr_db=a['psnr']-b['psnr'],delta_ssim=a['ssim']-b['ssim'],
                                  passed=a['psnr']>=b['psnr']-.05 and a['ssim']>=b['ssim']-.001))
    report['quality_gates']=gates
    bench.write_json(bench.OUT/'quality.json',report)
    print('Quality gates: '+str(sum(r['passed'] for r in gates))+'/'+str(len(gates)),flush=True)


if __name__=='__main__':main()
