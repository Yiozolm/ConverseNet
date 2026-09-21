"""FP64 gate for isolated analytic shared-input s1 CUDA spectral transfer.

Uses original s1 normal/weak fixtures, unchanged pretrained failure fixture,
and all prior independent small broadcast/mask/HVP cases moved to CUDA.
Every candidate call is counted; no production mutation or timing. Numerical
failures are retained, with nonzero exit. No tolerance changes or CUDA fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verbose-build',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite old evidence')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from experiments.training_shared_s1.loader import load
    from diagnose_boundary_precision import kernel_fft64,fixture
    from probe_training_s1_shapes import cases
    import probe_shared_s1_transfer as base
    import validate_shared_s1_transfer as validate
    from train_usrnet_dataset import tensor_hash,file_hash
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    ops,metadata=load(verbose=a.verbose_build)
    original=base.method
    calls=0

    def factory(mode,padding=2,eps=1e-5):
        if mode=='reference':return original(mode,padding,eps)
        def forward(x,k,b):
            nonlocal calls
            value=torch.nn.functional.pad(x,(padding,)*4,mode='circular') if padding else x
            y=torch.fft.rfft2(value)
            fk=kernel_fft64(k,*value.shape[-2:]).cfloat().contiguous()
            lam=torch.sigmoid(b-9.)+eps
            result=ops.shared_s1_transfer(y,fk,lam)
            calls+=1
            output=torch.fft.irfft2(result,s=value.shape[-2:])
            return output[...,padding:-padding,padding:-padding] if padding else output
        return forward

    tensors=fixture()
    original_fixture=json.loads((ROOT/'artifacts/native_deconv_target/boundary_probe.json').read_text())
    if tensor_hash(dict(zip(('x','weight','bias','upstream'),tensors)))!=original_fixture['input_sha256']:
        raise RuntimeError('Fixed pretrained fixture changed')
    work=list(cases())
    work.append(dict(name='pretrained_module_pad2_crop2',tensors=tensors,eps=1e-5,weak=False,padding=2))
    work.extend(validate.cpu_cases())
    report=dict(status='running',scope=__doc__,build=metadata,
        source_sha256={name:file_hash(ROOT/name) for name in (
            'test/validate_shared_s1_cuda.py','test/validate_shared_s1_transfer.py',
            'test/probe_shared_s1_transfer.py','test/diagnose_boundary_precision.py')},
        cases=[],production_eligible=False,environment=dict(torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),deterministic_algorithms=True,tf32=False),calls=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        with patch.object(base,'method',factory),patch.object(validate,'MODES',('analytic_cuda',)):
            for case in work:
                before=calls
                row=validate.validate_case(case,'cuda')
                row['candidate_calls']=calls-before
                if row['candidate_calls']!=len(row['checks']):raise RuntimeError('Candidate route was skipped')
                report['cases'].append(row);report['calls']=calls
                print(json.dumps(dict(case=case['name'],passed=row['passed'])),flush=True)
                save()
        report['status']='passed' if all(r['passed'] for r in report['cases']) else 'numerical_gate_failed'
        save()
        return 0 if report['status']=='passed' else 1
    except Exception as e:
        report['status']='failed';report['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc());save();raise


if __name__=='__main__':raise SystemExit(main())
