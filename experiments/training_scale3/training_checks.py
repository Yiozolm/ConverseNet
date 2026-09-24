from pathlib import Path
import sys,json,itertools,time,statistics,gc
from unittest.mock import patch
from locations import HERE,ROOT
sys.path[:0]=[str(ROOT/'test'),str(ROOT)]
import torch
from loader import load_all
from study import inputs,CASES,timed
from test_training_fusion import spectral_reference
ops=load_all(warm=True)
torch.manual_seed(220923)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def same(a,b):
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        return False
    return torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),b.detach().contiguous().reshape(-1).view(torch.uint8))
def boundaries():
    rows=[]
    for dtype in (torch.complex64,torch.complex128):
        real=torch.float32 if dtype==torch.complex64 else torch.float64
        for H,W in ((1,1),(1,2),(2,1),(3,2),(3,4),(4,6),(5,8),(5,7),(7,10),(2,3)):
            for KB,KC in ((1,1),(1,3),(2,1),(2,3)):
                data=(torch.randn(2,3,H,W//2+1,device='cuda',dtype=dtype).requires_grad_(),
                      torch.randn(2,3,3*H,3*W//2+1,device='cuda',dtype=dtype).requires_grad_(),
                      torch.randn(KB,KC,3*H,3*W//2+1,device='cuda',dtype=dtype).requires_grad_(),
                      torch.rand(1,3,1,1,device='cuda',dtype=real).add_(.2).requires_grad_())
                reference=spectral_reference(*data,H,W,3)
                old=ops['before']._training_spectral(*data,H,W,3)
                new=ops['fused']._training_spectral(*data,H,W,3)
                up=torch.randn_like(new)
                rg=torch.autograd.grad(reference,data,up)
                og=torch.autograd.grad(old,data,up)
                ng=torch.autograd.grad(new,data,up)
                tol=3e-5 if dtype==torch.complex64 else 1e-11
                for a,b in zip((new,*ng),(reference,*rg)):torch.testing.assert_close(a,b,atol=tol,rtol=tol)
                equality=[same(a,b) for a,b in zip((new,*ng),(old,*og))]
                if not all(equality):
                    diagnostic=dict(dtype=str(dtype),H=H,W=W,KB=KB,KC=KC,bitwise=equality,
                        max_abs=[(a-b).abs().max().item() for a,b in zip((new,*ng),(old,*og))])
                    (HERE/'boundary_failure.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
                    raise AssertionError(diagnostic)
                rows.append(dict(dtype=str(dtype),H=H,W=W,KB=KB,KC=KC,bitwise=equality))
    (HERE/'boundaries.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('BOUNDARIES',len(rows),'cases',sum(len(r['bitwise']) for r in rows),'tensors',flush=True)

def make_step(case,mode):
    cpu,up=inputs(case)
    x,p=(t.cuda().requires_grad_() for t in cpu[:2])
    k,b=(torch.nn.Parameter(t.cuda()) for t in cpu[2:])
    target=torch.zeros_like(p)
    opt=torch.optim.Adam((k,b),lr=1e-5,foreach=False,fused=False)
    def reset():
        with torch.no_grad():
            k.copy_(cpu[2]);b.copy_(cpu[3]);opt.zero_grad(set_to_none=True)
            x.grad=p.grad=None
            for param in (k,b):
                for value in opt.state.get(param,{}).values():
                    if isinstance(value,torch.Tensor):value.zero_()
    def step():
        opt.zero_grad(set_to_none=True);x.grad=p.grad=None
        y=ops[mode].forward(x,p,k,b,3,1e-5)
        loss=(y-target).square().mean();loss.backward();opt.step()
        return y,loss
    def snapshot():
        y,loss=step()
        values=[y,loss,x.grad,p.grad,k,b,k.grad,b.grad]
        for param in (k,b):values.extend(v for v in opt.state[param].values() if isinstance(v,torch.Tensor))
        return [v.detach().cpu().clone() for v in values]
    return step,reset,snapshot

def steps():
    results=[]
    for case in (CASES[2],CASES[3],CASES[6],CASES[7]):
        states={};row=dict(case=case,rounds=[],equivalence=[])
        for mode in ('before','fused'):
            fn,reset,snapshot=make_step(case,mode)
            states[mode]=[snapshot() for _ in range(3)]
            del fn,reset,snapshot;gc.collect();torch.cuda.empty_cache()
        equality=[same(a,b) for left,right in zip(states['before'],states['fused']) for a,b in zip(left,right)]
        assert all(equality),case
        row['equivalence']=dict(tensors=len(equality),bitwise_equal=sum(equality))
        for r in range(4):
            samples={}
            for mode in (('before','fused') if r%2==0 else ('fused','before')):
                fn,reset,snapshot=make_step(case,mode)
                for _ in range(5):fn()
                reset();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                samples[mode]=timed(fn,20)
                samples[mode]['allocated']=torch.cuda.max_memory_allocated()
                del fn,reset,snapshot;gc.collect();torch.cuda.empty_cache()
            row['rounds'].append(samples)
        row['paired_speedup']=statistics.median(r['before']['gpu_ms']/r['fused']['gpu_ms'] for r in row['rounds'])
        row['median_gpu_ms']={mode:statistics.median(r[mode]['gpu_ms'] for r in row['rounds']) for mode in ('before','fused')}
        results.append(row);print('STEP',case[0],row['paired_speedup'],row['median_gpu_ms'],flush=True)
        (HERE/'steps.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

def model():
    from models.converse_usrnet import ConverseUSRNet
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.manual_seed(221024)
    x=torch.rand(1,3,32,32,device='cuda');k=torch.rand(1,1,7,7,device='cuda');k=k/k.sum()
    target=torch.rand(1,3,96,96,device='cuda')
    checkpoint=torch.load(ROOT/'model_zoo/converse_usrnet.pth',map_location='cpu',weights_only=True)
    states={}
    for mode in ('before','fused'):
        with patch.object(torch.ops,'converse2d',ops[mode]):
            net=ConverseUSRNet(backend='cuda').cuda().train();net.load_state_dict(checkpoint)
            opt=torch.optim.Adam(net.parameters(),lr=1e-5,foreach=False,fused=False)
            saved={}
            for i in range(3):
                opt.zero_grad(set_to_none=True);y=net(x,k,3);loss=(y-target).square().mean();loss.backward();opt.step()
                saved[f'{i}/y']=y.detach().cpu().clone()
                for name,p in net.named_parameters():
                    saved[f'{i}/p/{name}']=p.detach().cpu().clone()
                    saved[f'{i}/g/{name}']=p.grad.detach().cpu().clone()
                    for key,v in opt.state[p].items():
                        if isinstance(v,torch.Tensor):saved[f'{i}/adam/{name}/{key}']=v.detach().cpu().clone()
            states[mode]=saved
            del y,loss,net,opt;gc.collect();torch.cuda.empty_cache()
    assert states['before'].keys()==states['fused'].keys()
    equal={name:same(value,states['fused'][name]) for name,value in states['before'].items()}
    assert all(equal.values())
    (HERE/'model.json').write_text(json.dumps(dict(shape='B1 HR96 s3 full pretrained USRNet, 3 Adam steps, fixed synthetic inputs',tensors=len(equal),bitwise_equal=sum(equal.values())),indent=2),encoding='utf-8')
    print('MODEL',len(equal),'bitwise tensors',flush=True)

if __name__=='__main__':boundaries();steps();model()
