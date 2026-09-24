"""Reproducible original-Python / v1.0.0 / FP32-release comparisons.

Frozen sources are loaded from Git. Only import/registration namespaces change.
No implementation, threshold, precision policy or original CPU allocation changes.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch
from torch.utils.cpp_extension import load
from extension_loader import ROOT, load_extension, production_source_hashes, build_config

sys.path.insert(0, str(ROOT))
ORIGINAL = '368aa39cebaa075b04926264c7494d969e8ee835'
V1 = 'e795a38'
LABELS = ('original_python', 'v1', 'v2')
OUT = ROOT/'artifacts/release_v2'
IDENTITIES = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_source(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ROOT)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def frozen_modules(ref, label, names):
    package = types.ModuleType(label)
    package.__path__ = []
    sys.modules[label] = package
    identity = {}
    for name in names:
        path = f'models/{name}.py'
        raw = git_source(ref, path)
        source = raw.decode('utf-8')
        source = source.replace('from models.', f'from {label}.').replace('from models import ', f'from {label} import ')
        if label == '_release1':
            source = source.replace('torch.ops.converse2d', 'torch.ops.converse2d_release1')
        module = types.ModuleType(f'{label}.{name}')
        module.__file__ = f'{ref}:{path}'
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        identity[path] = dict(original_sha256=sha(raw), derived_sha256=sha(source.encode()))
    IDENTITIES[label] = identity
    return package


def load_backends():
    load_extension()
    build = OUT/'build_v1'
    build.mkdir(parents=True, exist_ok=True)
    original_hashes, derived_hashes, sources = {}, {}, []
    for name in ('converse2d.cpp', 'converse2d_kernels.cu'):
        raw = git_source(V1, f'Converse2D/torch_converse2d/{name}')
        source = raw.decode('utf-8').replace('TORCH_LIBRARY(converse2d,', 'TORCH_LIBRARY(converse2d_release1,').replace('TORCH_LIBRARY_IMPL(converse2d,', 'TORCH_LIBRARY_IMPL(converse2d_release1,')
        target = build/name
        encoded = source.encode()
        if not target.exists() or target.read_bytes() != encoded:
            target.write_bytes(encoded)
        original_hashes[name], derived_hashes[name] = sha(raw), sha(encoded)
        sources.append(str(target))
    cxx, cuda = build_config.compile_flags()
    module = load(name='converse2d_release1_frozen', sources=sources,
                  extra_cflags=cxx+['-DCONVERSE2D_WITH_CUDA=1'], extra_cuda_cflags=cuda,
                  build_directory=str(build), with_cuda=True, verbose=True)
    IDENTITIES['v1_build'] = dict(original=original_hashes, derived=derived_hashes,
                                 binary_sha256=sha(Path(module.__file__).read_bytes()),
                                 cxx_flags=cxx+['-DCONVERSE2D_WITH_CUDA=1'], cuda_flags=cuda)
    original = frozen_modules(ORIGINAL, '_original', ('util_converse', 'converse_usrnet'))
    v1 = frozen_modules(V1, '_release1', ('converse_core', 'util_converse', 'converse_usrnet', 'cuda_graph'))
    from models import util_converse, converse_usrnet, cuda_graph
    v2 = types.SimpleNamespace(util_converse=util_converse, converse_usrnet=converse_usrnet, cuda_graph=cuda_graph)
    return dict(zip(LABELS, (original, v1, v2)))


def clear():
    torch.ops.converse2d.clear_cache()
    torch.ops.converse2d_release1.clear_cache()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def measure(functions, iters, rounds, reset=None):
    samples = {name: [] for name in functions}
    for r in range(rounds):
        order = list(functions)
        order = order[r % len(order):]+order[:r % len(order)]
        for name in order:
            fn = functions[name]
            if reset:
                reset(name)
            for _ in range(5):
                fn()
            if reset:
                reset(name)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            t0 = time.perf_counter()
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            samples[name].append(dict(wall_ms=(time.perf_counter()-t0)*1000/iters,
                                      gpu_ms=start.elapsed_time(end)/iters,
                                      peak_extra_bytes=torch.cuda.max_memory_allocated()-base))
    medians = {name: {key: statistics.median(v[key] for v in values) for key in values[0]}
               for name, values in samples.items()}
    return dict(medians=medians, rounds=samples, iterations=iters,
                speedups_vs_original={name: medians['original_python']['wall_ms']/values['wall_ms'] for name, values in medians.items()},
                v2_vs_v1=medians['v1']['wall_ms']/medians['v2']['wall_ms'])


def errors(actual, reference):
    actual, reference = actual.double(), reference.double()
    delta = actual-reference
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm()/reference.norm().clamp_min(1e-300)).item(),
                finite=bool(torch.isfinite(actual).all()))


def snapshots(fn, inputs, parameters, upstream):
    output = fn()
    targets = dict(inputs, **dict(parameters))
    values = torch.autograd.grad(output, tuple(targets.values()), upstream)
    return {'output': output.detach().cpu(), **{name: v.detach().cpu() for name, v in zip(targets, values)}}


def metadata(args):
    manifest = json.loads((ROOT/'.build/cuda/source_manifest.json').read_text())
    return dict(original_ref=ORIGINAL, v1_ref=subprocess.check_output(['git', 'rev-parse', 'v1.0.0^{commit}'], cwd=ROOT, text=True).strip(),
                v2_ref=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                cpu_threads=torch.get_num_threads(), tf32=False, autocast=False,
                cudnn_deterministic=True, script_sha256=sha(Path(__file__).read_bytes()),
                source_hashes=production_source_hashes(), frozen_identities=IDENTITIES,
                current_binary_sha256=manifest['binary_sha256'], build_inputs=manifest['inputs'],
                checkpoint_sha256=sha((ROOT/'model_zoo/converse_usrnet.pth').read_bytes()),
                protocol=vars(args), results=[])


def operators(backends, args, report):
    # Match every fixed/dynamic operator shape in the 1.0.0 release note.
    cases = [('fixed', b, c, h, w, s) for b, c, h, w, s in (
        (1,64,128,128,1), (1,64,128,128,2), (1,32,128,128,3),
        (1,32,127,129,3), (2,64,32,40,2))]
    cases += [('dynamic', b, 64, h, w, s) for b,h,w,s in (
        (1,128,128,2), (1,128,128,3), (2,32,40,2))]
    cases += [('fixed', 4,32,64,64,s) for s in (1,2,3)]
    for index, (kind,b,c,h,w,s) in enumerate(cases):
        torch.manual_seed(930+index)
        modules = {}
        eps = 1e-5 if kind == 'fixed' else 1e-3
        for name, package in backends.items():
            kw = {} if name == 'original_python' else dict(backend='cuda')
            m = (package.util_converse.Converse2D(c,c,7,scale=s,padding=0,eps=eps,**kw) if kind == 'fixed'
                 else package.converse_usrnet.ConvReverseDataNet(eps=eps,**kw))
            modules[name] = m.cuda()
        state = copy.deepcopy(modules['original_python'].state_dict())
        for m in modules.values():
            m.load_state_dict(state, strict=True)
        g = torch.Generator().manual_seed(19000+index)
        raw = torch.randn(b,c,h,w,generator=g)
        rawk = torch.randn(b,c,7,7,generator=g).flatten(2).softmax(-1).reshape(b,c,7,7)
        x, k = raw.cuda(), rawk.cuda()
        inference = {name: (lambda m=m: m(x)) if kind == 'fixed' else (lambda m=m: m(x,k.clone(),s)) for name,m in modules.items()}
        with torch.inference_mode():
            timing = measure(inference,args.iters,args.rounds)
            inference_outputs = {name: fn().cpu() for name,fn in inference.items()}
        ref = copy.deepcopy(modules['v1']).double()
        ref.backend = 'pytorch'
        xr, kr = raw.cuda().double().requires_grad_(), rawk.cuda().double().requires_grad_()
        upstream = torch.randn(b,c,h*s,w*s,generator=g)/((b*c*h*w*s*s)**.5)
        ref_fn = (lambda: ref(xr)) if kind == 'fixed' else (lambda: ref(xr,kr.clone(),s))
        reference = snapshots(ref_fn, {'dx':xr, **({'dk':kr} if kind == 'dynamic' else {})}, ref.named_parameters(), upstream.cuda().double())
        old64 = copy.deepcopy(modules['original_python']).double()
        original64 = snapshots((lambda:old64(xr)) if kind == 'fixed' else (lambda:old64(xr,kr.clone(),s)),
                               {'dx':xr, **({'dk':kr} if kind == 'dynamic' else {})}, old64.named_parameters(),upstream.cuda().double())
        math_equivalence = {key: errors(original64[key], value) for key,value in reference.items()}
        del ref, old64, xr, kr, ref_fn
        numerical = {}
        for name,m in modules.items():
            xt, kt = raw.cuda().requires_grad_(), rawk.cuda().requires_grad_()
            fn = (lambda:m(xt)) if kind == 'fixed' else (lambda:m(xt,kt.clone(),s))
            captured = snapshots(fn, {'dx':xt, **({'dk':kt} if kind == 'dynamic' else {})},m.named_parameters(),upstream.cuda())
            numerical[name] = {key:errors(value,reference[key]) for key,value in captured.items()}
        # Stable FP32 reference is an accuracy audit only, not the speed baseline.
        stable32 = copy.deepcopy(modules['v1'])
        stable32.backend = 'pytorch'
        xt, kt = raw.cuda().requires_grad_(), rawk.cuda().requires_grad_()
        stable = snapshots((lambda:stable32(xt)) if kind == 'fixed' else (lambda:stable32(xt,kt.clone(),s)),
                           {'dx':xt, **({'dk':kt} if kind == 'dynamic' else {})},stable32.named_parameters(),upstream.cuda())
        numerical['stable_python_fp32'] = {key:errors(v,reference[key]) for key,v in stable.items()}
        del stable32, stable, xt, kt, captured, fn
        training = {}
        for name,m in modules.items():
            xt, kt = raw.cuda().requires_grad_(), rawk.cuda().requires_grad_()
            targets = (xt, *m.parameters()) if kind == 'fixed' else (xt,kt,*m.parameters())
            up = upstream.cuda()
            def step(m=m,xt=xt,kt=kt,targets=targets,up=up):
                output = m(xt) if kind == 'fixed' else m(xt,kt.clone(),s)
                return torch.autograd.grad(output,targets,up)
            training[name] = step
        training_time = measure(training,args.train_iters,args.rounds)
        row = dict(kind=kind,shape=[b,c,h,w],scale=s,kernel=7,eps=eps,
                   input_sha256=sha(raw.numpy().tobytes()), weight_sha256=sha((state['weight'].cpu() if kind=='fixed' else rawk).numpy().tobytes()),
                   inference=timing, training_vjp=training_time,
                   inference_errors={name:errors(v,reference['output']) for name,v in inference_outputs.items()},
                   training_errors=numerical, original_fp64_vs_stable=math_equivalence)
        report['results'].append(row)
        write_json(OUT/'operators.json',report)
        print(f'{kind} {b}x{c}x{h}x{w}/s{s}: infer '+str({k:round(v['wall_ms'],4) for k,v in timing['medians'].items()})+' train '+str({k:round(v['wall_ms'],4) for k,v in training_time['medians'].items()}),flush=True)
        del modules,inference,training,reference,original64,x,k,state,step,xt,kt,targets,up
        clear()


def model_factory(package, name, checkpoint):
    model = package.converse_usrnet.ConverseUSRNet(**({} if name=='original_python' else dict(backend='cuda'))).cuda()
    model.load_state_dict(checkpoint,strict=True)
    return model


def gaussian():
    a = torch.arange(7,dtype=torch.float32)-3
    k = torch.exp(-(a[:,None].square()+a[None,:].square())/(2*1.2**2))
    return (k/k.sum())[None,None].cuda()


def models(backends,args,report):
    checkpoint = torch.load(ROOT/'model_zoo/converse_usrnet.pth',map_location='cpu',weights_only=True)
    for index,(h,w) in enumerate(((32,40),(64,80))):
        g=torch.Generator().manual_seed(21000+index)
        x=torch.rand(1,3,h,w,generator=g).cuda(); k=gaussian()
        ms={name:model_factory(p,name,checkpoint).eval() for name,p in backends.items()}
        ref=backends['v1'].converse_usrnet.ConverseUSRNet(backend='pytorch').cuda().double().eval()
        ref.load_state_dict(checkpoint,strict=True)
        runners={name:backends[name].cuda_graph.USRNetCUDAGraph(ms[name]) for name in ('v1','v2')}
        with torch.inference_mode():
            expected=ref(x.double(),k.double(),2).cpu()
            outputs={name:m(x,k,2).cpu() for name,m in ms.items()}
            for name,r in runners.items():
                outputs[name+'_graph']=r(x,k,2).cpu()
            del ref
            functions={name:lambda m=m:m(x,k,2) for name,m in ms.items()}
            functions.update({name+'_graph':lambda r=r:r(x,k,2) for name,r in runners.items()})
            timing=measure(functions,args.model_iters,args.model_rounds)
        row=dict(kind='usrnet_inference',shape=[1,3,h,w],scale=2, timing=timing,
                 errors_vs_fp64={name:errors(out,expected) for name,out in outputs.items()})
        report['results'].append(row);write_json(OUT/'models.json',report)
        print('USRNet inference '+str([h,w])+' '+str({k:round(v['wall_ms'],4) for k,v in timing['medians'].items()}),flush=True)
        for r in runners.values():r.clear()
        del ms,runners,functions,outputs,expected,x,k
        clear()
    for index,(h,w,s) in enumerate(((16,20,2),(32,40,2),(16,16,3))):
        g=torch.Generator().manual_seed(23000+index)
        x=torch.rand(1,3,h,w,generator=g).cuda();k=gaussian()
        target=torch.rand(1,3,h*s,w*s,generator=g).cuda()
        ms={name:model_factory(p,name,checkpoint).train() for name,p in backends.items()}
        optim={name:torch.optim.Adam(m.parameters(),lr=1e-5,foreach=False,fused=False) for name,m in ms.items()}
        functions={}
        for name,m in ms.items():
            o=optim[name]
            def step(m=m,o=o):
                o.zero_grad(set_to_none=True)
                output=m(x,k,s)
                loss=(output-target).square().mean()
                loss.backward()
                o.step()
                return loss
            functions[name]=step
        def reset(name):
            ms[name].load_state_dict(checkpoint,strict=True)
            optim[name].zero_grad(set_to_none=True)
            for state in optim[name].state.values():
                for value in state.values():
                    if isinstance(value,torch.Tensor):value.zero_()
        timing=measure(functions,args.model_train_iters,args.model_rounds,reset=reset)
        row=dict(kind='usrnet_adam_step',shape=[1,3,h,w],scale=s,timing=timing,
                 scope='zero_grad + full forward + MSE + backward for all parameters + Adam; data resident on GPU; no validation/I/O')
        report['results'].append(row);write_json(OUT/'models.json',report)
        print('USRNet training '+str([h,w,s])+' '+str({k:round(v['wall_ms'],4) for k,v in timing['medians'].items()}),flush=True)
        del ms,optim,functions,step,reset,x,k,target,m,o
        clear()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('build','operators','models'),required=True)
    p.add_argument('--iters',type=int,default=50)
    p.add_argument('--train-iters',type=int,default=20)
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--model-iters',type=int,default=10)
    p.add_argument('--model-train-iters',type=int,default=5)
    p.add_argument('--model-rounds',type=int,default=5)
    args=p.parse_args()
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY'):
        raise RuntimeError('Remove global backend/CPU-only overrides for this benchmark')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(24)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(930)
    backends=load_backends()
    report=metadata(args)
    if args.phase=='operators':operators(backends,args,report)
    elif args.phase=='models':models(backends,args,report)
    else:write_json(OUT/'build.json',report)


if __name__=='__main__':main()
