"""Paired FP32 accuracy audit. FP64 is used only as an oracle.

Run with the CUDA/MSVC environment in experiments/warp_spectral/run.ps1.
Builds are isolated from .build/cuda and never use its stale binary.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[1]
import sys as _layout_sys
_layout_sys.path.insert(0, str(ROOT / "test"))
from extension_loader import legacy_source_texts, production_source_hashes

sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference

OUT = ROOT / 'artifacts/accuracy_fp32_20260917'
BUILD = ROOT / '.build/accuracy_fp32_20260917'
ROWS, ERRORS, MANIFEST = [], [], {}


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(dict(manifest=MANIFEST, rows=ROWS, errors=ERRORS), indent=2), encoding='utf-8')


def apply_saved_patch(sources, patch):
    """Apply exact textual hunks, independent of checkout location/CRLF."""
    result = dict(sources)
    lines = patch.splitlines(keepends=True)
    name = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('+++ b/'):
            name = line.strip().split('/')[-1]
        if line.startswith('@@'):
            old, new = [], []
            i += 1
            while i < len(lines) and not lines[i].startswith(('@@','--- a/')):
                line = lines[i]
                if line.startswith((' ', '-')): old.append(line[1:])
                if line.startswith((' ', '+')): new.append(line[1:])
                i += 1
            before, after = ''.join(old), ''.join(new)
            assert result[name].count(before) == 1, (name, before)
            result[name] = result[name].replace(before, after, 1)
            continue
        i += 1
    return result


def build_source(label, sources, namespace=None):
    folder = BUILD / label
    folder.mkdir(parents=True, exist_ok=True)
    MANIFEST[label] = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in sources.items()}
    paths = []
    for name, content in sources.items():
        if namespace:
            content = content.replace('converse', namespace + '_converse')
            name = name.replace('converse', namespace + '_converse')
        dest = folder / name
        if not dest.exists() or dest.read_text(encoding='utf-8') != content:
            dest.write_text(content, encoding='utf-8')
        if dest.suffix in ('.cpp', '.cu'):
            paths.append(str(dest))
    print('BUILD', label, flush=True)
    cpp_extension.load(name='accuracy_' + label, sources=paths,
        extra_cflags=['/O2', '/std:c++17', '-DCONVERSE2D_WITH_CUDA=1'],
        extra_cuda_cflags=['-O3', '-lineinfo'], extra_ldflags=['cufft.lib'],
        with_cuda=True, is_python_module=False, build_directory=str(folder), verbose=False)
    return getattr(torch.ops, namespace + '_converse2d' if namespace else 'converse2d')


def load_all():
    cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
    source = ROOT / 'Converse2D/torch_converse2d'
    current = build_source('checkout', legacy_source_texts())
    versions = {'python_fp32': lambda a, s, e: converse2d_reference(*a, s, e)}
    versions.update({f'checkout_{v}': lambda a, s, e, v=v: current.forward(*a, s, e, v) for v in [f'v{i}' for i in range(2, 8)]})
    # Rebuild the immutable pre-FP32-training snapshot, then apply its saved patch
    # in an isolated directory to recover the actual post-optimization source.
    frozen = ROOT / 'artifacts/fp32_training/baseline'
    baseline = {p.name: p.read_text(encoding='utf-8') for p in frozen.iterdir() if p.suffix in ('.cpp', '.cu', '.h')}
    for item in json.loads((frozen.parent / 'baseline_sha256.json').read_text(encoding='utf-8-sig')):
        name = Path(item['Path']).name
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == item['Hash'].lower(), name
    before = build_source('training_before', baseline, 'audit_before')
    patch = ROOT / 'artifacts/fp32_training/changes.patch'
    after_src = apply_saved_patch(baseline, patch.read_text(encoding='utf-8'))
    MANIFEST['training_reconstruction'] = dict(method='Verified frozen raw hashes + exact-context saved patch; normalized line endings',
        patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest())
    after = build_source('training_after', after_src, 'audit_after')
    versions['snapshot_before_training'] = lambda a, s, e: before.forward(*a, s, e)
    versions['snapshot_after_training'] = lambda a, s, e: after.forward(*a, s, e)
    nearest = {}
    frozen_ops = []
    pre_io = {n: subprocess.check_output(['git','show',f'19c1bfc:Converse2D/torch_converse2d/{n}'],cwd=ROOT).decode('utf-8') for n in ('converse2d.cpp','converse2d_kernels.cu')}
    old = build_source('pre_spectral_io',pre_io,'audit_pre_io')
    frozen_ops.append(old)
    versions['pre_spectral_io_v7'] = lambda a,s,e,op=old:op.forward(*a,s,e,'v7')
    for label,directory in (('pre_batch_fft','batch_fft'),('nearest_before_c2r','b_direct_integration')):
        src = {name:(ROOT/'artifacts'/directory/f'before.{suffix}').read_text(encoding='utf-8') for name,suffix in (('converse2d.cpp','cpp'),('converse2d_kernels.cu','cu'))}
        op = build_source(label,src,'audit_'+label)
        frozen_ops.append(op)
        versions[label] = lambda a,s,e,op=op:op.forward(*a,s,e)
        if hasattr(op,'forward_nearest'):
            nearest[label+'_nearest'] = lambda a,s,e,op=op:op.forward_nearest(a[0],a[2],a[3],s,e)
    for name in ('nearest_cpp', 'nearest_spectral'):
        module = imported(f'experiments/{name}/study.py', 'audit_' + name)
        print('BUILD', name, flush=True)
        module.load(verbose=False)
        op = module.OP
        nearest[name] = lambda a, s, e, op=op: op.forward_nearest(a[0], a[2], a[3], s, e)
        MANIFEST[name] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'experiments' / name).iterdir() if p.suffix in ('.cpp', '.cu')}
    nearest['snapshot_nearest_before'] = lambda a, s, e: before.forward_nearest(a[0], a[2], a[3], s, e)
    nearest['snapshot_nearest_after'] = lambda a, s, e: after.forward_nearest(a[0], a[2], a[3], s, e)
    return versions, nearest, (current, before, after, *frozen_ops)


def metrics(actual, reference):
    a, r = actual.detach().double(), reference.detach().double()
    d = a - r
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    if not finite:
        return dict(finite=False, max_abs=None, relative_l2=None)
    return dict(finite=True, max_abs=d.abs().max().item(), mean_abs=d.abs().mean().item(),
        signed_mean=d.mean().item(), rmse=d.square().mean().sqrt().item(),
        relative_l2=(d.norm() / r.norm().clamp_min(1e-300)).item(), reference_max=r.abs().max().item(),
        violations_3e5=int((d.abs() > (3e-5 + 3e-5 * r.abs())).sum().item()), numel=r.numel())


def record(group, version, case, output, ref, baseline=None):
    row = dict(group=group, version=version, case=case, **metrics(output, ref))
    if baseline is not None:
        row['vs_python_fp32'] = metrics(output, baseline)
    ROWS.append(row)


def arguments(shape, scale, seed, kind, broadcast, prior, bias_value):
    torch.manual_seed(seed)
    b, c, h, w = shape
    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    p = F.interpolate(x, scale_factor=scale, mode='nearest') if prior == 'nearest' else torch.randn(b, c, h*scale, w*scale, device='cuda')
    kh, kw = min(3, h*scale), min(3, w*scale)
    kb, kc = (b if broadcast[0] else 1), (c if broadcast[1] else 1)
    k = torch.randn(kb, kc, kh, kw, device='cuda')
    if kind == 'normalized':
        k = k.flatten(2).softmax(-1).reshape(kb, kc, kh, kw)
    elif kind == 'box':
        k.fill_(1/(kh*kw))
    else:
        k /= kh*kw
    bias = torch.full((1, c, 1, 1), bias_value, device='cuda', dtype=torch.float32)
    return x, p, k, bias


def operators(versions, nearest, ops):
    shapes = [(1,2,1,1), (2,3,1,7), (2,3,7,1), (2,3,5,7), (2,3,8,10),
              (1,8,31,33), (1,16,64,80), (1,32,128,128), (1,8,127,129), (8,16,64,64)]
    index = 0
    for seed in (17, 53, 9214):
        for i, shape in enumerate(shapes):
            for s in (1,2,3,4):
                for prior in ('independent', 'nearest'):
                    # Cross all shapes/scales/seeds with a rotating stress design.
                    kind = ('normalized', 'signed', 'box')[(i+s+seed) % 3]
                    eps, bias = ((1e-3, 0.), (1e-5, 0.), (1e-7, -12.))[(i+seed+s) % 3]
                    bc = ((False,True),(True,False),(True,True),(False,False))[i % 4]
                    case = dict(seed=seed, shape=shape, scale=s, prior=prior, kernel=kind, eps=eps, bias=bias, broadcast=bc)
                    a = arguments(shape,s,seed,kind,bc,prior,bias)
                    with torch.no_grad():
                        ref = converse2d_reference(*(t.double() for t in a),s,eps)
                        py = versions['python_fp32'](a,s,eps)
                        candidates = {**versions, **(nearest if prior == 'nearest' else {})}
                        for name, fn in candidates.items():
                            try:
                                out = fn(a,s,eps)
                                record('forward',name,case,out,ref,py)
                            except Exception as exc:
                                ERRORS.append(dict(group='forward',version=name,case=case,error=str(exc)))
                    # Same vector-Jacobian product for every variant and FP64.
                    if i in (0,3,4,5,6) and s in (1,2,3):
                        ag = tuple(t.detach().requires_grad_() for t in a)
                        ar = tuple(t.detach().double().requires_grad_() for t in a)
                        torch.manual_seed(seed + 10000)
                        upstream = torch.randn_like(a[1]) / math.sqrt(a[1].numel())
                        er = converse2d_reference(*ar,s,eps)
                        grads = torch.autograd.grad(er, ar, upstream.double())
                        for name, fn in versions.items():
                            try:
                                out = fn(ag,s,eps)
                                actual = torch.autograd.grad(out,ag,upstream)
                                record('training_forward',name,case,out,er)
                                for key,g,r in zip(('x','prior','weight','bias'),actual,grads):
                                    record('gradient_'+key,name,case,g,r)
                            except Exception as exc:
                                ERRORS.append(dict(group='gradient',version=name,case=case,error=str(exc)))
                    for op in ops:
                        op.clear_cache()
                    index += 1
                    if index % 20 == 0:
                        print('OPERATORS',index,'/ 240', 'rows',len(ROWS),'errors',len(ERRORS),flush=True)
                        save()
    save()


def warp():
    module = imported('experiments/warp_spectral/study.py','audit_warp')
    module.load()
    MANIFEST['warp_sources'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/warp_spectral').iterdir() if p.suffix in ('.cpp','.cu')}
    for seed in (17,53,9214):
        torch.manual_seed(seed)
        for shape in ((1,1,1,1),(2,3,7,8),(1,4,31,33),(1,32,128,128),(1,8,127,129)):
            for lam in (1e-7,1e-5,1e-3):
                real,args = module.inputs(*shape,lam=lam)
                ref_spectrum = module.reference(real)
                ref = torch.fft.irfft2(ref_spectrum,s=(2*shape[-2],2*shape[-1]))
                for mode,name in enumerate(module.NAMES):
                    out = torch.ops.warp_spectral.run(*args,mode)
                    record('warp_image',name,dict(seed=seed,shape=shape,lam=lam),torch.fft.irfft2(out,s=ref.shape[-2:]),ref)
    save()
    print('WARP completed',flush=True)


def models(versions, nearest):
    from models.converse_dncnn import ConverseDnCNN
    from models.converse_srresnet import ConverseMSRResNet
    from models.converse_usrnet import ConverseUSRNet
    original = torch.ops.converse2d.forward
    candidates = {**versions, **nearest}
    factories = [('converse_dncnn',ConverseDnCNN,1),('converse_srresnet',ConverseMSRResNet,3),('converse_usrnet',ConverseUSRNet,3)]
    for name,factory,c in factories:
        checkpoint = ROOT/'model_zoo'/f'{name}.pth'
        MANIFEST[name+'_checkpoint'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = factory().cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
        for seed in (17,53):
            for h,w in ((12,16),(24,32)):
                torch.manual_seed(seed)
                x=torch.rand(1,c,h,w,device='cuda')
                coords=torch.arange(7,device='cuda')-3
                k=torch.exp(-(coords[:,None].square()+coords[None,:].square())/2.)
                k=(k/k.sum())[None,None]
                case=dict(model=name,seed=seed,shape=list(x.shape),scale=2 if name.endswith('usrnet') else None)
                def run(dtype,backend):
                    model.to(dtype)
                    for layer in model.modules():
                        if hasattr(layer,'backend'): layer.backend=backend
                    return model(x.to(dtype),k.to(dtype),2) if name.endswith('usrnet') else model(x.to(dtype))
                with torch.inference_mode():
                    ref=run(torch.float64,'pytorch').clone()
                    py=run(torch.float32,'pytorch').clone()
                    for version,fn in candidates.items():
                        try:
                            torch.ops.converse2d.forward = lambda x,p,k,b,s,e=1e-5,v='v7',fn=fn: fn((x,p,k,b),s,e)
                            # checkout closures use original dispatcher, avoiding recursion.
                            out=run(torch.float32,'cuda')
                            record('model',version,case,out,ref,py)
                        except Exception as exc:
                            ERRORS.append(dict(group='model',version=version,case=case,error=str(exc)))
                        finally:
                            torch.ops.converse2d.forward=original
                save()
                print('MODEL',name,seed,h,w,'errors',len(ERRORS),flush=True)
        del model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sections',nargs='+',default=['operators','warp','models'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    MANIFEST.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        tested_dtype='float32',oracle_dtype='float64',time=time.strftime('%Y-%m-%d %H:%M:%S'),
        scope='Numerical agreement; no dataset PSNR or convergence claim',sections=args.sections)
    versions,nearest,ops=load_all()
    # Capture dispatcher handles before temporary model routing.
    forward=ops[0].forward
    for i in range(2,8):
        v=f'v{i}'
        versions['checkout_'+v]=lambda a,s,e,v=v:forward(*a,s,e,v)
    save()
    for section in args.sections:
        if section=='operators': operators(versions,nearest,ops)
        elif section=='warp': warp()
        elif section=='models': models(versions,nearest)
    print('DONE',len(ROWS),'rows;',len(ERRORS),'errors;',OUT,flush=True)


if __name__=='__main__':
    main()
