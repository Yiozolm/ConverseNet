"""Accuracy non-inferiority to original Python FP32, not bitwise alignment.

Predeclared rule: for each output/dx/dw/db tensor, candidate maximum absolute
error AND relative L2 error to the identical FP64 reference must be no larger
than original Python FP32's corresponding error. No percentage slack, changed
lambda, rescaled fixture or pointwise FP64 tolerance is introduced. Individual
coordinate differences remain diagnostic. This strict per-tensor screen does
not replace the separate real-training quality gate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def accuracy(actual,reference):
    import torch
    delta=actual.double()-reference.double()
    norm=reference.double().norm().clamp_min(1e-30)
    return dict(max_abs=delta.abs().max().item(),relative_l2=(delta.norm()/norm).item(),
        finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()))


def compare_accuracy(actual,baseline,reference):
    result={}
    for key,ref in reference.items():
        a,b=accuracy(actual[key],ref),accuracy(baseline[key],ref)
        comparisons={m:a[m]<=b[m] for m in ('max_abs','relative_l2')}
        result[key]=dict(candidate=a,python_fp32=b,fp32_rounding_floor=accuracy(ref.float(),ref),
            metric_noninferior=comparisons,passed=a['finite'] and b['finite'] and all(comparisons.values()),
            ratio_to_python={m:(a[m]/b[m] if b[m]>0 else (1. if a[m]==0 else None)) for m in comparisons})
    return dict(passed=all(r['passed'] for r in result.values()),tensors=result)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('New report path required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from extension_loader import load_extension
    from experiments.training_shared_s1.loader import load
    from probe_training_s1_shapes import cases
    from diagnose_boundary_precision import fixture
    from validate_shared_s1_python_fp32 import method
    from probe_pointwise_training import capture,tensor_hash,clear_cuda
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    os.environ['CONVERSE2D_SKIP_BUILD']='1';load_extension()
    ops,build=load()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    frozen=json.loads((ROOT/'artifacts/native_deconv_target/source_before/manifest.json').read_text())
    if worker.file_hash(ROOT/'models/converse_core.py')!=frozen['models/converse_core.py']:
        raise RuntimeError('Original Python reference was modified')
    work=list(cases())
    tensors=fixture()
    original=json.loads((ROOT/'artifacts/native_deconv_target/boundary_probe.json').read_text())
    if worker.tensor_hash(dict(zip(('x','weight','bias','upstream'),tensors)))!=original['input_sha256']:
        raise RuntimeError('Pretrained fixture changed')
    work.append(dict(name='pretrained_module_pad2_crop2',tensors=tensors,eps=1e-5,weak=False,padding=2))
    config=native.CASES['s1_module'];cpu=native.cpu_data(config,seed=17)
    old=json.loads((ROOT/'artifacts/native_deconv_target/native_baseline.json').read_text())
    old=next(r for r in old['cases'] if r['name']=='s1_module')
    if worker.tensor_hash(cpu)!=old['complete_fixture_sha256']:raise RuntimeError('Native fixture changed')
    work.append(dict(name='native_s1_module_seed17',tensors=(cpu['x'],cpu['weight'],
        cpu['bias'].reshape(1,config['shape'][1],1,1),cpu['upstream']),eps=config['eps'],weak=False,padding=2))
    names=('production','aten_prep64','aten_prep32','cuda_prep64','cuda_prep32')
    report=dict(status='running',scope=__doc__,build=build,cases=[],
        source_sha256={n:worker.file_hash(ROOT/n) for n in ('test/audit_shared_s1_accuracy.py',
            'test/validate_shared_s1_python_fp32.py','models/converse_core.py',
            'test/probe_training_s1_shapes.py','test/diagnose_boundary_precision.py')},
        candidate_passed={n:True for n in names},production_eligible=False,timing=False,
        environment=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
            tf32=False,deterministic_algorithms=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for case in work:
            oracle=method('python_fp32',ops,case['eps'],case.get('padding',0))
            ref=capture(case['tensors'],torch.float64,oracle)
            baseline=capture(case['tensors'],torch.float32,oracle)
            row=dict(name=case['name'],fixture_sha256=tensor_hash(case['tensors']),eps=case['eps'],
                weak=case['weak'],padding=case.get('padding',0),candidates={})
            for name in names:
                actual=capture(case['tensors'],torch.float32,method(name,ops,case['eps'],case.get('padding',0)))
                row['candidates'][name]=compare_accuracy(actual,baseline,ref)
                report['candidate_passed'][name]&=row['candidates'][name]['passed']
            report['cases'].append(row);save();clear_cuda()
            print(case['name'],json.dumps({n:r['passed'] for n,r in row['candidates'].items()}),flush=True)
        report['status']='passed' if all(report['candidate_passed'].values()) else 'accuracy_noninferiority_failed'
        save()
        return 0 if report['status']=='passed' else 1
    except Exception as e:
        report['status']='failed';report['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc());save();raise


if __name__=='__main__':raise SystemExit(main())
