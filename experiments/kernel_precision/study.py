"""Accuracy, cache lifecycle and amortized cost of high-precision kernel FFTs."""
import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from extension import ROOT, NAMES, load_all

sys.path[:0]=[str(ROOT),str(ROOT/'test')]
from accuracy_fp32_versions import metrics, arguments
from models.converse_core import converse2d_reference

OUT=ROOT/'artifacts/kernel_precision'
DATA=dict(accuracy=[],training=[],benchmarks=[],models=[],checks=[],errors=[])


def save():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(DATA,indent=2),encoding='utf-8')


def call(op,a,s,e):return op.forward(*a,s,e,'v7')


def make(case):
    a=arguments(case['shape'],case['scale'],case['seed'],case['kernel'],case['broadcast'],case['prior'],case['bias'])
    if case['kernel']=='gaussian':
        b,c=a[2].shape[:2]
        h,w=case['shape'][-2:]
        kh,kw=min(7,h*case['scale']),min(7,w*case['scale'])
        yy=torch.arange(kh,device='cuda')-kh//2
        xx=torch.arange(kw,device='cuda')-kw//2
        kernel=torch.exp(-(yy[:,None].square()+xx[None,:].square())/(2*1.2**2))
        a=(*a[:2],(kernel/kernel.sum())[None,None].repeat(b,c,1,1),a[3])
    return a


def accuracy(ops):
    old=json.loads((ROOT/'artifacts/accuracy_fp32_20260917/results.json').read_text(encoding='utf-8'))
    cases=[dict(r['case'],grid='previous') for r in old['rows'] if r['group']=='forward' and r['version']=='python_fp32']
    for seed,shape,s,kind,reg,prior in itertools.product((17,53,260917),
            ((1,4,5,7),(1,8,32,40),(1,8,127,129),(2,3,16,17)),(1,2,3),
            ('normalized','signed','box','gaussian'),((1e-3,0.),(1e-5,0.),(1e-7,-12.)),('nearest','independent')):
        cases.append(dict(seed=seed,shape=shape,scale=s,kernel=kind,eps=reg[0],bias=reg[1],prior=prior,
            broadcast=(shape[0]>1,kind!='box'),grid='crossed'))
    for index,case in enumerate(cases):
        a=make(case);s,e=case['scale'],case['eps']
        row=dict(case=case,variants={})
        with torch.no_grad():
            ref=converse2d_reference(*(t.double() for t in a),s,e)
            for name,op in ops.items():
                op.clear_cache();op.precision_reset()
                out=call(op,a,s,e)
                assert out.dtype==torch.float32
                row['variants'][name]=metrics(out,ref)
                if name=='adaptive':
                    row.update(risk=op.precision_last_risk(),selected_high=bool(op.precision_stats()[0]))
        DATA['accuracy'].append(row)
        if (index+1)%100==0:
            print('ACCURACY',index+1,'/',len(cases),flush=True);save()
    save()


def close(a,b,atol=5e-5,rtol=5e-5):torch.testing.assert_close(a.double(),b.double(),atol=atol,rtol=rtol)


def lifecycle(ops):
    case=dict(shape=(2,3,7,9),scale=2,seed=99,kernel='normalized',broadcast=(True,True),prior='independent',bias=0.)
    for name,op in ops.items():
        a=make(case);s=2;e=1e-3
        op.clear_cache();op.precision_reset()
        with torch.no_grad():
            first=call(op,a,s,e);second=call(op,a,s,e)
            assert torch.equal(first,second)
            assert op.precision_stats()[3]==1
            a[2].mul_(.8);a[2].add_(.01)
            changed=call(op,a,s,e)
            close(changed,converse2d_reference(*(t.double() for t in a),s,e))
            assert not torch.equal(changed,first)
            replacement=(*a[:2],a[2].clone()+.02,a[3])
            close(call(op,replacement,s,e),converse2d_reference(*(t.double() for t in replacement),s,e))
            a[2].set_(a[2].clone()*.9)
            close(call(op,a,s,e),converse2d_reference(*(t.double() for t in a),s,e))
            # Noncontiguous rectangular filters and explicit stream ownership.
            strided=tuple(t.transpose(-1,-2) for t in a)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):out=call(op,strided,s,e)
            torch.cuda.current_stream().wait_stream(stream)
            close(out,converse2d_reference(*(t.double() for t in strided),s,e))
            for ee in (1e-7,1e-3,1e-7):
                a[3].fill_(-12.)
                close(call(op,a,s,ee),converse2d_reference(*(t.double() for t in a),s,ee))
            # Cache must retain c64 spectra and f32 powers, not large c128 arrays.
            op.begin_graph_cache()
            call(op,a,s,e)
            owners=op.end_graph_cache()
            assert any(t.dtype==torch.complex64 for t in owners)
            assert all(t.dtype!=torch.complex128 and t.dtype!=torch.float64 for t in owners)
            del owners
            # Graph-owned warmup; replay remains valid after eager cache eviction.
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            op.begin_graph_cache()
            with torch.cuda.stream(stream):
                for _ in range(3):call(op,a,s,e)
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph,stream=stream):out=call(op,a,s,e)
            finally:owners=op.end_graph_cache()
            op.clear_cache();a[0].add_(.05)
            graph.replay();torch.cuda.synchronize()
            close(out,call(op,a,s,e))
            graph.reset();del graph,owners
            # Inference tensors have no version counter and must not cache.
            op.clear_cache();op.precision_reset()
            with torch.inference_mode():ai=tuple(t.clone() for t in a)
            call(op,ai,s,e);call(op,ai,s,e)
            assert op.precision_stats()[3]==0
        DATA['checks'].append(dict(variant=name,status='passed',checks=['repeat_hit','mutation','tensor_identity',
            'storage_replacement','strided_stream','epsilon_change','cache_dtype','graph_ownership_eviction','inference_tensor_bypass']))
    # Explicit policy invalidation: the same box kernel crosses the threshold.
    op=ops['adaptive'];op.clear_cache();op.precision_reset()
    a=arguments((1,2,16,16),1,77,'box',(False,True),'nearest',-12.)
    with torch.no_grad():
        call(op,a,1,1e-3);assert op.precision_stats()[:3]==[0,1,1]
        call(op,a,1,1e-7);assert op.precision_stats()[:3]==[1,1,2]
        call(op,a,1,1e-7);assert op.precision_stats()[3]==1
        a[3].fill_(2.)
        close(call(op,a,1,1e-7),converse2d_reference(*(t.double() for t in a),1,1e-7))
    DATA['checks'].append(dict(variant='adaptive',status='passed',checks=['epsilon_in_cache_key','bias_change_safe_lower_bound']))
    save();print('LIFECYCLE passed',flush=True)


def training(ops):
    for s,kind,eps in itertools.product((1,2,3),('normalized','signed','box'),(1e-3,1e-7)):
        a=arguments((2,3,5,7),s,912,kind,(True,False),'independent',-12. if eps<1e-5 else 0.)
        a=tuple(t.detach().requires_grad_() for t in a)
        r=tuple(t.detach().double().requires_grad_() for t in a)
        upstream=torch.randn_like(a[1])/math.sqrt(a[1].numel())
        ref=converse2d_reference(*r,s,eps)
        expected=torch.autograd.grad(ref,r,upstream.double())
        row=dict(scale=s,kernel=kind,eps=eps,variants={})
        for name,op in ops.items():
            out=call(op,a,s,eps)
            grads=torch.autograd.grad(out,a,upstream)
            row['variants'][name]=dict(forward=metrics(out,ref),gradients={k:metrics(g,rr) for k,g,rr in zip(('x','prior','weight','bias'),grads,expected)})
        DATA['training'].append(row)
    save();print('TRAINING completed',flush=True)


def timed(fn,n):
    torch.cuda.synchronize()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    t=time.perf_counter();begin.record()
    for _ in range(n):fn()
    end.record();end.synchronize()
    return dict(event_ms=begin.elapsed_time(end)/n,wall_ms=(time.perf_counter()-t)*1000/n)


def benchmark(ops,iters,rounds):
    configs=[((1,8,32,40),1,'signed'),((1,32,128,128),1,'box'),((1,64,128,128),2,'normalized'),
             ((1,8,127,129),1,'box'),((8,16,64,64),3,'signed'),((2,3,64,80),2,'gaussian')]
    for shape,s,kind in configs:
        case=dict(shape=shape,scale=s,seed=211,kernel=kind,broadcast=(False,True),prior='nearest',bias=0.)
        a=make(case);e=1e-5
        with torch.inference_mode():dynamic=tuple(t.clone() for t in a)
        for cache_mode in ('warm','cold','dynamic'):
            vals=dynamic if cache_mode=='dynamic' else a
            functions={}
            for name,op in ops.items():
                def fn(op=op):
                    if cache_mode=='cold':op.clear_cache()
                    return call(op,vals,s,e)
                functions[name]=fn
            with torch.no_grad():
                for op in ops.values():op.clear_cache()
                for fn in functions.values():
                    for _ in range(4):fn()
                samples={n:[] for n in NAMES}
                n=iters if cache_mode=='warm' else max(3,iters//3)
                for rr in range(rounds):
                    order=NAMES[rr%3:]+NAMES[:rr%3]
                    for name in order:samples[name].append(timed(functions[name],n))
                for name,op in ops.items():
                    op.clear_cache();torch.cuda.synchronize()
                    initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    temp=functions[name]();torch.cuda.synchronize()
                    peak=torch.cuda.max_memory_allocated()-initial
                    del temp
                    retained=torch.cuda.memory_allocated()-initial
                    # Logical cache payload is exact; allocator deltas can also
                    # contain FFT workspace and delayed stream reclamation.
                    op.begin_graph_cache()
                    call(op,a,s,e)
                    cache_tensors=op.end_graph_cache()
                    payload=sum(t.numel()*t.element_size() for t in cache_tensors)
                    spectrum_payload=sum(t.numel()*t.element_size() for t in cache_tensors if t.is_complex())
                    del cache_tensors
                    DATA['benchmarks'].append(dict(case=case,cache_mode=cache_mode,variant=name,
                        event_ms=statistics.median(v['event_ms'] for v in samples[name]),
                        wall_ms=statistics.median(v['wall_ms'] for v in samples[name]),samples=samples[name],
                        allocator_retained_delta_bytes=retained,spectrum_cache_payload_bytes=payload,
                        cached_complex_bytes=spectrum_payload,cold_peak_extra_bytes=peak,iterations=n))
            save();print('BENCH',shape,s,kind,cache_mode,flush=True)


def model_accuracy(ops):
    # Route only inside this test process; production wrappers are untouched.
    original=getattr(torch.ops.converse2d,'forward',None)
    torch.ops.converse2d.forward=ops['fp32'].forward
    from models.converse_dncnn import ConverseDnCNN
    from models.converse_srresnet import ConverseMSRResNet
    from models.converse_usrnet import ConverseUSRNet
    try:
        for name,factory,c in [('converse_dncnn',ConverseDnCNN,1),('converse_srresnet',ConverseMSRResNet,3),('converse_usrnet',ConverseUSRNet,3)]:
            torch.manual_seed(53)
            model=factory().cuda().eval()
            model.load_state_dict(torch.load(ROOT/'model_zoo'/f'{name}.pth',map_location='cuda',weights_only=True),strict=True)
            x=torch.rand(1,c,24,32,device='cuda');q=torch.arange(7,device='cuda')-3
            k=torch.exp(-(q[:,None].square()+q[None,:].square())/2.);k=(k/k.sum())[None,None]
            def run(dtype,backend):
                model.to(dtype)
                for layer in model.modules():
                    if hasattr(layer,'backend'):layer.backend=backend
                return model(x.to(dtype),k.to(dtype),2) if name.endswith('usrnet') else model(x.to(dtype))
            with torch.inference_mode():
                ref=run(torch.float64,'pytorch')
                for variant,op in ops.items():
                    op.clear_cache();op.precision_reset()
                    torch.ops.converse2d.forward=op.forward
                    out=run(torch.float32,'cuda')
                    DATA['models'].append(dict(model=name,variant=variant,**metrics(out,ref),stats=op.precision_stats()))
            print('MODEL',name,flush=True);save()
    finally:
        if original is None:delattr(torch.ops.converse2d,'forward')
        else:torch.ops.converse2d.forward=original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sections',nargs='+',default=['accuracy','lifecycle','training','models','benchmark'])
    parser.add_argument('--iters',type=int,default=30)
    parser.add_argument('--rounds',type=int,default=7)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    ops,hashes=load_all()
    DATA['manifest']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,
        time=time.strftime('%Y-%m-%d %H:%M:%S'),hashes=hashes,sections=args.sections,iters=args.iters,rounds=args.rounds,
        screening_threshold=1e4,screening_formula='max_filter ||kernel||_1^2 / (min_alias_mean_power + eps)',
        cache='same mutation/identity/stream/shape lifecycle; adaptive adds eps key',
        scope='Experimental private namespaces; FP32 input/output and solve, only kernel FFT optionally FP64')
    for section in args.sections:
        if section=='accuracy':accuracy(ops)
        elif section=='lifecycle':lifecycle(ops)
        elif section=='training':training(ops)
        elif section=='models':model_accuracy(ops)
        elif section=='benchmark':benchmark(ops,args.iters,args.rounds)
    save();print('DONE',OUT,flush=True)


if __name__=='__main__':main()
