"""Isolated shared-filter VJP materialization; no production file is changed.

Gate: scale>1, prior gradient required, kernel gradient required and broadcast
kernel. Existing adjoint_inputs writes each un-reduced gK contribution; a new
kernel sums them in the original b-then-c order, without atomics or fastmath.
This does NOT hit the current full-USRNet s3 DataNet (per-sample kernels), and
does not establish a full-network speedup. Exact formulas/reduction order do
not guarantee bitwise identity: materialization adds an FP32 rounding boundary.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'Converse2D/torch_converse2d'
SOURCES=('converse2d.cpp','converse2d_kernels.cu','converse2d_training.cu','converse2d_training.h')

REDUCER=r'''
// Workspace has the full prior shape; output retains the broadcast PSF shape.
template<class T>
__global__ void reduce_filter_workspace(const Z<T>* work,Z<T>* gk,
    I n,I B,I C,I plane,I KB,I KC) {
    const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const I kbc=i/plane,frequency=i%plane;
    const I first_b=KB==1?0:kbc/KC,last_b=KB==1?B:first_b+1;
    const I first_c=KC==1?0:kbc%KC,last_c=KC==1?C:first_c+1;
    Z<T> sum(0,0);
    for(I b=first_b;b<last_b;++b) for(I c=first_c;c<last_c;++c)
        sum+=work[(b*C+c)*plane+frequency];
    gk[i]=sum;
}

'''


def replace_once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Production source changed; review workspace patch context: '+old[:80])
    return text.replace(old,new,1)


def candidate_source(original):
    result=replace_once(original,'at::Tensor plain(const at::Tensor& t)',REDUCER+'at::Tensor plain(const at::Tensor& t)')
    result=replace_once(result,'const bool reduce_filter=need_k&&!no_broadcast;',
        'const bool reduce_filter=need_k&&!no_broadcast;\n    const bool materialize_filter=s>1&&need_p&&reduce_filter;')
    result=replace_once(result,'auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();',
        'auto gk=need_k?at::empty(k.sizes(),k.options()):at::Tensor();\n'
        '    auto work=materialize_filter?at::empty(p.sizes(),p.options()):at::Tensor();')
    marker='        } else {\n            adjoint_q<T>'
    if result.count(marker)!=1:raise RuntimeError('Generic backward dispatch context changed')
    prefix,suffix=result.split(marker,1)
    suffix=replace_once(suffix,'need_p?gp.data_ptr<Z<T>>():nullptr,need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr,',
        'need_p?gp.data_ptr<Z<T>>():nullptr,materialize_filter?work.data_ptr<Z<T>>():(need_k&&no_broadcast?gk.data_ptr<Z<T>>():nullptr),')
    suffix=replace_once(suffix,'            if(reduce_filter)\n                adjoint_filter<T>',
        '            if(materialize_filter)\n'
        '                reduce_filter_workspace<T><<<(k.numel()+255)/256,256,0,stream>>>(\n'
        '                    work.data_ptr<Z<T>>(),gk.data_ptr<Z<T>>(),k.numel(),p.size(0),p.size(1),\n'
        '                    p.size(2)*p.size(3),k.size(0),k.size(1));\n'
        '            else if(reduce_filter)\n                adjoint_filter<T>')
    return prefix+marker+suffix


def load_variants(verbose=False):
    import torch
    from torch.utils import cpp_extension
    originals={name:(SOURCE/name).read_text(encoding='utf-8') for name in SOURCES}
    candidate=candidate_source(originals['converse2d_training.cu'])
    source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in SOURCES}
    fingerprint=hashlib.sha256((json.dumps(source_hashes,sort_keys=True)+candidate).encode()).hexdigest()[:16]
    build=ROOT/'.build/shared_filter_workspace'/fingerprint
    if os.name=='nt':cpp_extension.SUBPROCESS_DECODE_ARGS=('utf-8','replace')
    loaded={};manifests={}
    for variant in ('current','candidate'):
        folder=build/variant;folder.mkdir(parents=True,exist_ok=True)
        namespace=f'ws_filter_{variant}_{fingerprint}_converse2d'
        prefix=namespace[:-2]
        files=[];generated={}
        for name,original in originals.items():
            content=candidate if variant=='candidate' and name=='converse2d_training.cu' else original
            content=content.replace('converse',prefix)
            path=folder/name.replace('converse',prefix)
            if not path.exists() or path.read_text(encoding='utf-8')!=content:path.write_text(content,encoding='utf-8')
            generated[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix!='.h':files.append(str(path))
        flags=['/O2','/std:c++17'] if os.name=='nt' else ['-O3','-std=c++17']
        flags+=['-DCONVERSE2D_WITH_CUDA=1','-DWORKSPACE_SOURCE_REV=0x'+fingerprint[:12]]
        wrapper_cl='/Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX'
        normalize_cl=os.name=='nt' and os.environ.get('CL')==wrapper_cl
        if normalize_cl:flags+=wrapper_cl.split()
        try:
            if normalize_cl:os.environ.pop('CL')
            cpp_extension.load(name=f'shared_filter_{variant}_{fingerprint}',sources=files,
                extra_include_paths=[str(folder)],extra_cflags=flags,extra_cuda_cflags=['-O3','-lineinfo'],
                with_cuda=True,is_python_module=False,build_directory=str(folder),verbose=verbose)
        finally:
            if normalize_cl:os.environ['CL']=wrapper_cl
        loaded[variant]=getattr(torch.ops,namespace)
        manifests[variant]=dict(namespace=namespace,generated_sources=generated,build_directory=str(folder),
                               cflags=flags,known_wrapper_cl_moved_to_cflags=normalize_cl,
                               nvcc_prepend_flags=os.environ.get('NVCC_PREPEND_FLAGS'))
    return loaded,dict(production_sources=source_hashes,fingerprint=fingerprint,variants=manifests,
                       cuda_flags=['-O3','-lineinfo'],production_unchanged=True)


def make_fixture(shape,scale,kb,kc,*,weak=None,needs=(True,True,True,True)):
    import torch
    b,c,h,w=shape
    x=torch.randn(shape)
    prior=torch.randn(b,c,h*scale,w*scale)
    weight=torch.rand(kb,kc,3,3)/9
    bias=torch.randn(1,c,1,1)
    eps=1e-3
    if weak is not None:
        x.mul_(1e-5);prior.mul_(1e-5);weight.mul_(weak);bias.fill_(-40.);eps=1e-8
    upstream=torch.randn_like(prior)*(1e-5 if weak is not None else prior.numel()**-.5)
    active=scale>1 and needs[1] and needs[2] and (kb!=b or kc!=c)
    return dict(values=(x,prior,weight,bias),upstream=upstream,shape=shape,scale=scale,eps=eps,
                kb=kb,kc=kc,weak=weak,needs=needs,
                expected_workspace_bytes=b*c*(h*scale)*(w*scale//2+1)*8 if active else 0)


def gpu_fixture(case):
    return tuple(t.cuda().requires_grad_(need) for t,need in zip(case['values'],case['needs'])),case['upstream'].cuda()


def capture(ops,case):
    import torch
    values,upstream=gpu_fixture(case)
    active=[t for t in values if t.requires_grad]
    output=ops.forward(*values,case['scale'],case['eps'],'v7')
    grads=torch.autograd.grad(output,active,upstream)
    names=[n for n,need in zip(('dx','dp','dw','db'),case['needs']) if need]
    snapshot={'output':output.detach().cpu(),**{n:t.detach().cpu() for n,t in zip(names,grads)}}
    del output,grads,active,values,upstream
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    return snapshot


def reference(case):
    import torch
    from models.converse_core import converse2d_reference
    values=case['values'];b=case['shape'][0]
    result={'output':torch.empty_like(values[1],dtype=torch.float64)}
    for name,value,need in zip(('dx','dp','dw','db'),values,case['needs']):
        if need:result[name]=torch.zeros_like(value,dtype=torch.float64)
    for index in range(b):
        sample=(values[0][index:index+1],values[1][index:index+1],
                values[2] if case['kb']==1 else values[2][index:index+1],values[3])
        args=tuple(t.cuda().double().requires_grad_(need) for t,need in zip(sample,case['needs']))
        active=[t for t in args if t.requires_grad]
        output=converse2d_reference(*args,case['scale'],case['eps'])
        grads=torch.autograd.grad(output,active,case['upstream'][index:index+1].cuda().double())
        result['output'][index:index+1].copy_(output.detach().cpu())
        names=[n for n,need in zip(('dx','dp','dw','db'),case['needs']) if need]
        for name,gradient in zip(names,grads):
            gradient=gradient.detach().cpu()
            if name in ('dx','dp') or (name=='dw' and case['kb']!=1):result[name][index:index+1].copy_(gradient)
            else:result[name].add_(gradient)
        del output,grads,args,active
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    return result


def compare(actual,expected,atol,rtol):
    import torch
    a,e=actual.double(),expected.double();error=(a-e).abs();budget=atol+rtol*e.abs()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    failures=int((error>budget).sum())
    return dict(passed=finite and failures==0,finite=finite,failed_elements=failures,
                max_abs=error.max().item(),relative_l2=(error.norm()/e.norm().clamp_min(1e-30)).item(),
                max_budget_ratio=(error/budget).max().item(),atol=atol,rtol=rtol)


def time_variant(ops,case,args,variant):
    import torch
    gc.collect();torch.cuda.empty_cache()
    values,upstream=gpu_fixture(case)
    active=[t for t in values if t.requires_grad]
    def run():
        output=ops.forward(*values,case['scale'],case['eps'],'v7')
        grads=torch.autograd.grad(output,active,upstream)
        # Never keep the previous output/VJP alive while evaluating the next
        # iteration. This matters for large HR gradients and peak memory.
        del output,grads
    for _ in range(args.warmup):run()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    initial=torch.cuda.memory_allocated()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    began=time.perf_counter();start.record()
    for _ in range(args.iters):run()
    end.record();end.synchronize()
    result=dict(wall_ms=(time.perf_counter()-began)*1000/args.iters,
                cuda_ms=start.elapsed_time(end)/args.iters,initial_allocated_bytes=initial,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                expected_extra_workspace_bytes=case['expected_workspace_bytes'] if variant=='candidate' else 0)
    del run,values,upstream,active
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--iters',type=int,default=20)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--verbose-build',action='store_true')
    args=parser.parse_args()
    if args.output.exists() or min(args.warmup,args.iters,args.rounds)<1:parser.error('Require a new output and positive counts')
    sys.path.insert(0,str(ROOT))
    import torch
    import train_usrnet_dataset as worker
    torch.manual_seed(9214)
    ops,manifest=load_variants(args.verbose_build)
    report=dict(status='validating',manifest=manifest,seed=9214,script_sha256=worker.file_hash(__file__),
        validation=[],timings=[],torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
        gate='s>1 && need_p && need_k && kernel broadcast',
        arithmetic='Unchanged per-frequency adjoint_inputs expression; deterministic original b-then-c serial reduction; no atomics/fastmath. Workspace materialization can alter rounding and is not predeclared bitwise equivalent.',
        scope='Public operator forward + requested VJP, not a whole training step; this gate does not hit current full-USRNet s3 dynamic kernels.',
        peak_method='One resident fixture per variant; outputs and VJPs destroyed each iteration; warmup precedes reset_peak_memory_stats; total allocator peaks, not process memory.')
    cases=[]
    for scale in (2,3,4):
        for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
            cases.append((f'normal/s{scale}/K{kb},{kc}',make_fixture((2,3,5,7),scale,kb,kc)))
    for amplitude in (0.,1e-6,1e-3):
        for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
            cases.append((f'weak/{amplitude}/K{kb},{kc}',make_fixture((2,3,3,4),3,kb,kc,weak=amplitude)))
    cases += [('unchanged_s1',make_fixture((2,3,5,7),1,1,3)),
              ('unchanged_weight_only',make_fixture((2,3,5,7),3,1,3,needs=(False,False,True,False)))]
    big=make_fixture((32,32,64,80),3,1,32)
    cases.append(('B32C32_64x80_s3',big))
    for label,case in cases:
        expected=reference(case)
        baseline=None;comparisons={}
        for name in ('current','candidate'):
            actual=capture(ops[name],case)
            if baseline is None:baseline=actual
            metrics={key:compare(value,expected[key],*( (1e-6,1e-5) if key=='output' and case['weak'] is not None
                        else (3e-5,3e-5) if key=='output' else (5e-5,5e-5))) for key,value in actual.items()}
            comparisons[name]=dict(passed=all(v['passed'] for v in metrics.values()),vs_fp64=metrics,
                bitwise_equal_to_current={key:torch.equal(value,baseline[key]) for key,value in actual.items()})
            del actual
        report['validation'].append(dict(case=label,shape=case['shape'],scale=case['scale'],
            kb=case['kb'],kc=case['kc'],needs=case['needs'],eps=case['eps'],
            expected_extra_workspace_bytes=case['expected_workspace_bytes'],comparisons=comparisons))
        del expected,baseline
        print('Validated',label,{name:row['passed'] for name,row in comparisons.items()},flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True);worker.write_json(args.output,report)
    if any(not v['passed'] for row in report['validation'] for v in row['comparisons'].values()):
        report['status']='numerical_gate_failed_no_timing';worker.write_json(args.output,report);raise SystemExit(1)
    if not args.validate_only:
        for label,case in [('B32C32_64x80_s3',big)]:
            rounds=[]
            for index in range(args.rounds):
                order=('current','candidate') if index%2==0 else ('candidate','current')
                rounds.append(dict(order=order,variants={name:time_variant(ops[name],case,args,name) for name in order}))
            ratios=[r['variants']['current']['wall_ms']/r['variants']['candidate']['wall_ms'] for r in rounds]
            report['timings'].append(dict(case=label,rounds=rounds,paired_ratios=ratios,paired_median=statistics.median(ratios)))
    report['status']='validation_passed' if args.validate_only else 'complete'
    worker.write_json(args.output,report)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
