"""User-requested original Python FP32 numerical baseline, new evidence only.

The immutable models.converse_core.converse2d_reference full-FFT implementation
is executed in FP32, as in the earlier backend=pytorch training comparison.
Existing normal/weak/HVP tolerances stay unchanged. Prior FP64 reports remain
diagnostics, never overwritten or relabelled. This is an operator gate, not
complete-model quality, training speed or convergence evidence.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def kernel_fft32(weight,height,width):
    import torch
    import torch.nn.functional as F
    kh,kw=weight.shape[-2:]
    psf=F.pad(weight,(0,width-kw,0,height-kh))
    return torch.fft.rfft2(psf.roll((-(kh//2),-(kw//2)),(-2,-1)))


def method(name,ops,eps,padding):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    from diagnose_boundary_precision import kernel_fft64
    def forward(x,k,b):
        x=F.pad(x,(padding,)*4,mode='circular') if padding else x
        if name=='python_fp32':
            output=converse2d_reference(x,x,k,b,1,eps)
        elif name=='production':
            output=torch.ops.converse2d.forward(x,x,k,b,1,eps,'v7')
        else:
            fk=(kernel_fft32(k,*x.shape[-2:]) if name.endswith('prep32') else kernel_fft64(k,*x.shape[-2:]))
            fk=fk.cfloat().contiguous()
            y=torch.fft.rfft2(x)
            lam=torch.sigmoid(b-9.)+eps
            if name.startswith('cuda_'):
                spectrum=ops.shared_s1_transfer(y,fk,lam)
            else:
                h=(fk.conj()+lam)/(fk.real.square()+fk.imag.square()+lam)
                spectrum=h*y
            output=torch.fft.irfft2(spectrum,s=x.shape[-2:])
        return output[...,padding:-padding,padding:-padding] if padding else output
    return forward


def capture(case,route,needs,higher):
    import torch
    inputs=tuple(v.detach().cuda().requires_grad_(need) for v,need in zip(case['tensors'][:3],needs))
    output=route(*inputs)
    values=dict(output=output.detach().cpu())
    requested=[v for v in inputs if v.requires_grad]
    labels=[name for name,need in zip(('dx','dw','db'),needs) if need]
    if requested:
        grads=torch.autograd.grad(output,requested,case['tensors'][3].cuda(),create_graph=higher)
        values.update({name:v.detach().cpu() for name,v in zip(labels,grads)})
        if higher:
            rng=torch.Generator().manual_seed(31917)
            directions=[torch.randn(g.shape,generator=rng).cuda() for g in grads]
            dot=sum((g*d).sum() for g,d in zip(grads,directions))
            second=torch.autograd.grad(dot,requested,allow_unused=True) if dot.requires_grad else [None]*len(requested)
            values.update({'hvp_'+name:None if g is None else g.detach().cpu() for name,g in zip(labels,second)})
    if any(v.grad is not None for v in inputs):raise RuntimeError('Leaf gradient accumulated')
    if any(v is not None and v.dtype!=torch.float32 for v in values.values()):raise RuntimeError('FP32 contract broken')
    return values


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--small-contract',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite old evidence')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from extension_loader import load_extension
    from experiments.training_shared_s1.loader import load
    from probe_training_s1_shapes import cases
    from diagnose_boundary_precision import fixture
    from validate_shared_s1_transfer import compare,cpu_cases
    from train_usrnet_dataset import tensor_hash,file_hash
    from probe_pointwise_training import tensor_hash as tuple_hash,clear_cuda
    os.environ['CONVERSE2D_SKIP_BUILD']='1';load_extension()
    ops,build=load()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    frozen=json.loads((ROOT/'artifacts/native_deconv_target/source_before/manifest.json').read_text())
    source='models/converse_core.py'
    if file_hash(ROOT/source)!=frozen[source]:raise RuntimeError('Original Python reference source changed')
    names=('production','aten_prep64','aten_prep32','cuda_prep64','cuda_prep32')
    report=dict(status='running',scope=__doc__,oracle=dict(backend='pytorch',dtype='float32',source=source,
        sha256=frozen[source],formulation='original full-FFT residual solve'),build=build,
        source_sha256={n:file_hash(ROOT/n) for n in ('test/validate_shared_s1_python_fp32.py',
            'test/validate_shared_s1_transfer.py','test/diagnose_boundary_precision.py',
            'test/probe_training_s1_shapes.py',source)},cases=[],production_eligible=False,
        environment=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
            tf32=False,deterministic_algorithms=True),candidate_passed={n:True for n in names})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        work=list(cases())
        tensors=fixture()
        old=json.loads((ROOT/'artifacts/native_deconv_target/boundary_probe.json').read_text())
        if tensor_hash(dict(zip(('x','weight','bias','upstream'),tensors)))!=old['input_sha256']:
            raise RuntimeError('Fixed pretrained fixture changed')
        work.append(dict(name='pretrained_module_pad2_crop2',tensors=tensors,eps=1e-5,weak=False,padding=2))
        if a.small_contract:work.extend(cpu_cases())
        for case in work:
            selective=case.get('selective',False)
            masks=list(itertools.product((False,True),repeat=3)) if selective else [(True,True,True)]
            row=dict(name=case['name'],fixture_sha256=tuple_hash(case['tensors']),eps=case['eps'],
                weak=case['weak'],padding=case.get('padding',0),checks=[])
            for needs in masks:
                for higher in ((False,True) if selective else (False,)):
                    expected=capture(case,method('python_fp32',ops,case['eps'],case.get('padding',0)),needs,higher)
                    checked={}
                    for name in names:
                        actual=capture(case,method(name,ops,case['eps'],case.get('padding',0)),needs,higher)
                        checked[name]=compare(actual,expected,case['weak'])
                        report['candidate_passed'][name] &= checked[name]['passed']
                    row['checks'].append(dict(needs=list(needs),higher=higher,candidates=checked))
            report['cases'].append(row);save();clear_cuda()
            print(case['name'],json.dumps({n:all(c['candidates'][n]['passed'] for c in row['checks']) for n in names}),flush=True)
        report['status']='passed' if all(report['candidate_passed'].values()) else 'numerical_gate_failed'
        save()
        return 0 if report['status']=='passed' else 1
    except Exception as e:
        report['status']='failed';report['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc());save();raise


if __name__=='__main__':raise SystemExit(main())
