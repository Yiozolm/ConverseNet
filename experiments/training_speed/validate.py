"""Independent FP64 spatial and complex-autograd training correctness checks."""
import argparse
import itertools
import json
import math
import sys
import time
import traceback

import torch
from torch.nn import functional as F
from extension import ROOT, load

SHAPES = [(1,1,1),(1,5,1),(4,1,1),(3,4,1),(3,5,1),(4,6,1),
          (1,5,2),(4,1,3),(5,7,2),(6,8,3),(3,5,4),(3,4,5)]
BROADCASTS = [(1,1),(1,3),(2,1),(2,3)]


def average(a,s):
    if s==1: return a
    h,w=a.shape[-2:]
    return a.reshape(*a.shape[:-2],s,h//s,s,w//s).mean((-4,-2))


def full(a,w):
    tail=a[...,1:(w+1)//2].flip((-2,-1)).roll(1,-2)
    return torch.cat((a,tail.conj() if a.is_complex() else tail),-1)


def spatial_ref(x,p,k,b,s,nearest):
    if nearest: p=x if s==1 else F.interpolate(x,scale_factor=s,mode='nearest')
    h,w=x.shape[-2:];kh,kw=k.shape[-2:]
    fk=torch.fft.fft2(F.pad(k,(0,w*s-kw,0,h*s-kh)).roll((-(kh//2),-(kw//2)),(-2,-1)))
    fy=torch.fft.fft2(x)
    fp=fy if p is x else torch.fft.fft2(p)
    lam=torch.sigmoid(b-9)+1e-3
    q=(fy-average(fk*fp,s))/(average(fk.real.square()+fk.imag.square(),s)+lam)
    if s>1: q=q.repeat(1,1,s,s)
    return torch.fft.ifft2(fp+fk.conj()*q).real


def spectral_ref(y,p,k,lam,h,w,s):
    power=k.real.square()+k.imag.square()
    pred=k*p
    if s>1:
        power=average(full(power,w*s),s)[...,:w//2+1]
        pred=average(full(pred,w*s),s)[...,:w//2+1]
    q=(y-pred)/(power+lam)
    if s>1: q=full(q,w).repeat(1,1,s,s)[...,:w*s//2+1]
    return p+k.conj()*q


def spatial_data(h,w,s,kb,kc):
    x=torch.randn(2,3,w,h,device='cuda').transpose(-1,-2)
    p=torch.randn(2,3,w*s,h*s,device='cuda').transpose(-1,-2)
    kh,kw=min(3,h*s),min(3,w*s)
    k=torch.rand(kb,kc,kw,kh,device='cuda').transpose(-1,-2)/(kh*kw)
    b=torch.randn(1,3,1,2,device='cuda')[...,:1]
    return tuple(t.requires_grad_() for t in (x,p,k,b))


def spectral_data(h,w,s,kb=1,kc=3,dtype=torch.complex128,strided=False):
    real=torch.float64 if dtype==torch.complex128 else torch.float32
    values=(torch.randn(2,3,h,w//2+1,device='cuda',dtype=dtype),
            torch.randn(2,3,h*s,w*s//2+1,device='cuda',dtype=dtype),
            torch.randn(kb,kc,h*s,w*s//2+1,device='cuda',dtype=dtype),
            torch.rand(1,3,1,1,device='cuda',dtype=real)+.2)
    if strided:
        values=tuple(torch.stack((t,t),-1)[...,0].conj() if t.is_complex()
                     else torch.stack((t,t),-1)[...,0] for t in values)
    return tuple(t.detach().requires_grad_() for t in values)


class Validation:
    def __init__(self,ops):
        self.ops,self.records=ops,[]
    def check(self,label,fn):
        self.current=dict(case=label,comparisons=[],status='running')
        start=time.perf_counter()
        try:
            fn()
            self.current['status']='passed'
        except Exception as exc:
            self.current.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            print('FAIL',label,exc,flush=True)
        self.current['elapsed_s']=time.perf_counter()-start
        self.records.append(self.current)
        if len(self.records)%25==0: print('Validated',len(self.records),flush=True)
    def compare(self,name,a,e,atol,rtol):
        dtype=torch.complex128 if e.is_complex() else torch.float64
        aa,ee=a.detach().to(dtype),e.detach().to(dtype)
        absolute=(aa-ee).abs().max().item()
        relative=((aa-ee).norm()/ee.norm().clamp_min(1e-30)).item()
        self.current['comparisons'].append(dict(tensor=name,atol=atol,rtol=rtol,
            max_abs=absolute if math.isfinite(absolute) else str(absolute),
            relative_l2=relative if math.isfinite(relative) else str(relative)))
        assert torch.isfinite(a).all() and torch.isfinite(e).all(),name
        torch.testing.assert_close(a.to(e.dtype),e,atol=atol,rtol=rtol)
    def spatial(self,values,s,mode,opt):
        x,p,k,b=values
        shared=mode=='same';nearest=mode=='nearest'
        if shared: p=x
        args=(x,k,b) if shared or nearest else (x,p,k,b)
        refs=tuple(t.detach().double().requires_grad_(t.requires_grad) for t in args)
        if shared or nearest:
            rx,rk,rb=refs;rp=rx if shared else None
        else: rx,rp,rk,rb=refs
        actual=self.ops.forward(x,p,k,b,s,1e-3,nearest,opt)
        expected=spatial_ref(rx,rp,rk,rb,s,nearest)
        self.compare('output',actual,expected,3e-5,3e-5)
        up=torch.randn_like(actual)/actual.numel()**.5
        inputs=[t for t in args if t.requires_grad]
        rin=[t for t in refs if t.requires_grad]
        ag=torch.autograd.grad(actual,inputs,up)
        eg=torch.autograd.grad(expected,rin,up.double())
        for i,(a,e) in enumerate(zip(ag,eg)): self.compare('gradient_'+str(i),a,e,5e-5,5e-5)
    def spectral(self,values,h,w,s,opt,higher=False):
        actual=self.ops.spectral(*values,h,w,s,opt)
        expected=spectral_ref(*values,h,w,s)
        double=values[0].dtype==torch.complex128
        self.compare('output',actual,expected,1e-12 if double else 1e-5,1e-12 if double else 1e-5)
        up=torch.randn_like(actual).conj()
        inputs=[t for t in values if t.requires_grad]
        eg=torch.autograd.grad(expected,inputs,up,retain_graph=higher)
        ag=torch.autograd.grad(actual,inputs,up,retain_graph=higher)
        for i,(a,e) in enumerate(zip(ag,eg)):
            self.compare('gradient_'+str(i),a,e,1e-11 if double else 2e-5,1e-11 if double else 2e-5)
        if higher:
            hg=torch.autograd.grad(actual,inputs,up,create_graph=True)
            for i,(a,e) in enumerate(zip(hg,eg)): self.compare('create_graph_'+str(i),a,e,1e-11,1e-11)
    def spatial_suite(self,opt):
        for h,w,s in SHAPES:
            for kb,kc in BROADCASTS:
                for mode in (('independent','same','nearest') if s==1 else ('independent','nearest')):
                    self.check(f'{opt}/spatial/{h},{w},{s}/{kb},{kc}/{mode}',
                        lambda:self.spatial(spatial_data(h,w,s,kb,kc),s,mode,opt))
        for s in (1,2):
            for mask in itertools.product((False,True),repeat=4):
                if not any(mask):continue
                values=tuple(t.detach().requires_grad_(need) for t,need in zip(spatial_data(3,4,s,1,3),mask))
                self.check(f'{opt}/spatial/selective/s{s}/{mask}',lambda:self.spatial(values,s,'independent',opt))
    def spectral_suite(self,opt):
        for h,w,s in SHAPES:
            for kb,kc in BROADCASTS:
                self.check(f'{opt}/arbitrary_complex/{h},{w},{s}/{kb},{kc}',
                    lambda:self.spectral(spectral_data(h,w,s,kb,kc),h,w,s,opt))
        for s in (1,2):
            for mask in itertools.product((False,True),repeat=4):
                if not any(mask):continue
                values=tuple(t.detach().requires_grad_(need) for t,need in zip(spectral_data(3,5,s,strided=True),mask))
                self.check(f'{opt}/complex/selective_strided_conj/s{s}/{mask}',
                    lambda:self.spectral(values,3,5,s,opt,higher=True))
        for h,w,s,kb,kc in [(1,1,1,1,1),(1,4,1,1,3),(4,1,1,2,1),(2,3,1,1,1),
                             (2,4,1,2,3),(2,4,2,1,3),(3,3,3,2,1)]:
            values=spectral_data(h,w,s,kb,kc)
            fn=lambda *v:self.ops.spectral(*v,h,w,s,opt)
            for name,checker in [('gradcheck',torch.autograd.gradcheck),('gradgradcheck',torch.autograd.gradgradcheck)]:
                self.check(f'{opt}/{name}/{h},{w},{s}/{kb},{kc}',
                    lambda:self.finite_difference(checker,fn,values))
        y,_,k,lam=spectral_data(2,3,1)
        fn=lambda y,k,lam:self.ops.spectral(y,y,k,lam,2,3,1,opt)
        for name,checker in [('gradcheck',torch.autograd.gradcheck),('gradgradcheck',torch.autograd.gradgradcheck)]:
            self.check(f'{opt}/{name}/shared',lambda:self.finite_difference(checker,fn,(y,k,lam)))
    @staticmethod
    def finite_difference(checker,fn,values):
        assert checker(fn,values,fast_mode=True,eps=1e-6,atol=1e-5,rtol=1e-3)
    def safety_suite(self,opt):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for s in (1,2,3):
                for mode in ('independent','nearest'):
                    values=spatial_data(7,8,s,1,3)
                    for update in range(3):
                        self.check(f'{opt}/side_stream/{mode}/s{s}/{update}',
                            lambda:self.spatial(values,s,mode,opt))
                        with torch.no_grad():
                            values[2].mul_(.98);values[3].add_(.02)
                for kb,kc in BROADCASTS:
                    self.check(f'{opt}/side_stream/complex64/s{s}/{kb},{kc}',
                        lambda:self.spectral(spectral_data(7,8,s,kb,kc,torch.complex64,True),7,8,s,opt))
        torch.cuda.current_stream().wait_stream(stream);stream.synchronize()
        for s in (1,2):
            for index in range(4):
                def mutation():
                    values=spectral_data(2,3,s)
                    out=self.ops.spectral(*values,2,3,s,opt)
                    with torch.no_grad():values[index].add_(.1)
                    try:out.real.sum().backward()
                    except RuntimeError as exc:
                        assert 'modified by an inplace' in str(exc)
                    else:raise AssertionError('saved tensor mutation accepted')
                self.check(f'{opt}/saved_version/s{s}/{index}',mutation)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',default='artifacts/training_speed/validation.json')
    parser.add_argument('--section',choices=('all','spatial','spectral','safety'),default='all')
    parser.add_argument('--variant',choices=('both','baseline','candidate'),default='both')
    args=parser.parse_args()
    ops,manifest=load()
    suite=Validation(ops);start=time.perf_counter()
    for opt in (False,True):
        name='candidate' if opt else 'baseline'
        if args.variant not in ('both',name):continue
        torch.manual_seed(9214)
        for section in ('spatial','spectral','safety'):
            if args.section in ('all',section):
                print('Validating',name,section,flush=True)
                getattr(suite,section+'_suite')(opt)
    failed=sum(r['status']=='failed' for r in suite.records)
    report=dict(status='failed' if failed else 'passed',seed=9214,torch=torch.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),manifest=manifest,
                summary=dict(total=len(suite.records),passed=len(suite.records)-failed,failed=failed),
                elapsed_s=time.perf_counter()-start,settings=vars(args),cases=suite.records)
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(report['summary'],flush=True)
    return int(failed>0)


if __name__=='__main__':sys.exit(main())
