"""Isolated compiled real arithmetic for exact same-input s1 transfer.

FFT, FP64 kernel preparation and their adjoints remain ATen. Compiled code
uses real components and automatically generated VJPs; high-order derivatives
use the established automatic eager reconstruction adapter. Original failed
pretrained fixture/FP64 budgets remain fixed. Only passing routes are timed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def spectral_method(high_precision):
    import torch
    def spectral(y,k,bias):
        kr,ki=k.real,k.imag
        lam=torch.sigmoid((bias.double() if high_precision else bias)-9.)+1e-5
        d=kr.square()+ki.square()+lam
        hr=((kr+lam)/d).float()
        hi=(-ki/d).float()
        yr,yi=y.real,y.imag
        return torch.complex(yr*hr-yi*hi,yr*hi+yi*hr)
    return spectral


def build_method(high_precision,compiled,wrapped):
    import torch
    import torch.nn.functional as F
    from diagnose_boundary_precision import kernel_fft64
    from experiments.training_nonoverlap.compiled_autograd import wrap_compiled
    spectral=spectral_method(high_precision)
    if compiled:
        optimized=torch.compile(spectral,backend='inductor',fullgraph=True,dynamic=False,
                                options={'triton.cudagraphs':False})
        spectral=wrap_compiled(spectral,optimized) if wrapped else optimized
    def forward(x,k,bias):
        x=F.pad(x,(2,2,2,2),mode='circular')
        fk=kernel_fft64(k,*x.shape[-2:])
        if not high_precision:fk=fk.cfloat().contiguous()
        y=torch.fft.rfft2(x.double() if high_precision else x).cfloat()
        result=torch.fft.irfft2(spectral(y,fk,bias),s=x.shape[-2:])
        return result[...,2:-2,2:-2]
    return forward


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iters',type=int,default=30)
    p.add_argument('--rounds',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():p.error('Use a new report path')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for key,folder in [('TORCHINDUCTOR_CACHE_DIR','inductor_s1_transfer'),('TRITON_CACHE_DIR','triton_s1_transfer')]:
        path=ROOT/'.build'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    import torch
    import torch._functorch.config as fc
    fc.donated_buffer=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch._dynamo.config.suppress_errors=False
    import train_usrnet_dataset as worker
    from diagnose_boundary_precision import fixture,comparison
    from probe_shared_s1_transfer import method
    from probe_pointwise_training import capture,timed_fixture
    tensors=fixture()
    fixture_hash=worker.tensor_hash(dict(zip(('x','weight','bias','upstream'),tensors)))
    original=json.loads((ROOT/'artifacts/native_deconv_target/boundary_probe.json').read_text())
    if fixture_hash!=original['input_sha256']:raise RuntimeError('Fixed fixture changed')
    sources=worker.source_hashes()
    for name in ['test/probe_compiled_s1_transfer.py','test/probe_shared_s1_transfer.py',
                 'test/diagnose_boundary_precision.py','experiments/training_nonoverlap/compiled_autograd.py']:
        sources[name]=worker.file_hash(ROOT/name)
    report=dict(status='running',source_sha256=sources,fixture_sha256=fixture_hash,scope=__doc__,
        settings=worker.json_safe(vars(a)),environment=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
            tf32=False,deterministic_algorithms=True,cudagraphs=False,donated_buffer=False),
        setup={},validation={},timing=[],production_eligible=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        expected=capture(tensors,torch.float64,method('reference'))
        passing={}
        for high in (False,True):
            for compiled,wrapped in ((False,False),(True,True)):
                name=('high64' if high else 'fp32')+('_compiled_wrapped' if compiled else '_eager_real')
                route=build_method(high,compiled,wrapped)
                start=time.perf_counter()
                actual=capture(tensors,torch.float32,route)
                report['setup'][name]=dict(first_capture_wall_s=time.perf_counter()-start,
                    scope='Includes compilation/cache lookup, CPU transfer and capture cleanup; not pure compile time')
                checks={key:comparison(actual[key],ref,*((3e-5,3e-5) if key=='output' else (5e-5,5e-5))) for key,ref in expected.items()}
                ok=all(v['passed'] for v in checks.values())
                report['validation'][name]=dict(passed=ok,tensors=checks)
                if ok:passing[name]=route
                print(name,json.dumps(dict(passed=ok,dw=checks['dw'])),flush=True)
                worker.write_json(a.output,report)
        names=list(passing)
        with torch._dynamo.config.patch(error_on_recompile=True):
            from torch._dynamo.utils import counters
            for i in range(a.rounds):
                before=int(counters['stats']['unique_graphs'])
                order=names[i%len(names):]+names[:i%len(names)] if names else []
                results={n:timed_fixture(tensors,passing[n],a) for n in order}
                after=int(counters['stats']['unique_graphs'])
                if before!=after:raise RuntimeError('Timing recompiled')
                report['timing'].append(dict(order=order,results=results,unique_graphs_before=before,unique_graphs_after=after))
                worker.write_json(a.output,report)
        report['status']='complete_fixed_fixture_screen'
    except Exception as e:
        report['status']='failed';report['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        raise
    finally:worker.write_json(a.output,report)


if __name__=='__main__':main()
