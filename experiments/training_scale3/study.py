from pathlib import Path
import sys,json,time,statistics,itertools,unittest,argparse
from unittest.mock import patch
from locations import HERE,ROOT
sys.path[:0]=[str(ROOT/'test'),str(ROOT)]
import torch
from loader import load_all
from models.converse_core import converse2d_reference
REPORT_FILE=HERE/'operators.json'

CASES=[('tiny',1,3,3,5,1,1,3),('b1_c32',1,32,32,40,1,32,3),
       ('b4_shared',4,32,64,80,1,32,3),('b4_dynamic',4,32,64,80,4,32,3),
       ('b8_shared',8,32,64,80,1,32,3),('large',1,32,128,128,1,32,3),
       ('data_b1',1,64,32,32,1,64,7),('data_b4',4,64,32,32,4,64,7)]
def inputs(case):
    _,B,C,H,W,KB,KC,K=case
    rng=torch.Generator().manual_seed(220922)
    x=torch.randn(B,C,H,W,generator=rng)*.1
    p=torch.randn(B,C,H*3,W*3,generator=rng)*.1
    k=torch.softmax(torch.randn(KB,KC,K*K,generator=rng),-1).reshape(KB,KC,K,K)
    b=torch.zeros(1,C,1,1)
    up=torch.randn(p.shape,generator=rng)/p.numel()**.5
    return (x,p,k,b),up
def capture(op,cpu,up,dtype):
    data=tuple(t.to('cuda',dtype).requires_grad_() for t in cpu)
    y=op(*data,3,1e-5)
    grads=torch.autograd.grad(y,data,up.to('cuda',dtype))
    return [v.detach().cpu().clone() for v in (y,*grads)]
def metric(a,b):
    d=a.double()-b.double()
    return dict(max_abs=d.abs().max().item(),relative_l2=(d.norm()/b.double().norm().clamp_min(1e-30)).item(),finite=bool(torch.isfinite(a).all()))
def timed(fn,iters):
    torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin=time.perf_counter();start.record()
    for _ in range(iters):fn()
    end.record();end.synchronize()
    return dict(gpu_ms=start.elapsed_time(end)/iters,wall_ms=(time.perf_counter()-begin)*1000/iters)
def run_operators(ops):
    report=dict(identity=json.loads((HERE/'builds.json').read_text()),scope='Isolated s3 forward and all four VJPs; FP32 independent prior; no optimizer',cases=[])
    for case in CASES:
        cpu,up=inputs(case)
        oracle=capture(converse2d_reference,cpu,up,torch.float64)
        py=capture(converse2d_reference,cpu,up,torch.float32)
        ref=capture(ops['before'].forward,cpu,up,torch.float32)
        pe=[metric(x,y) for x,y in zip(py,oracle)]
        row=dict(case=case,python_error=pe,validation={},rounds=[]);report['cases'].append(row)
        for name,op in ops.items():
            actual=capture(op.forward,cpu,up,torch.float32)
            errors=[metric(x,y) for x,y in zip(actual,oracle)]
            row['validation'][name]=dict(bitwise_before=[torch.equal(x.contiguous().reshape(-1).view(torch.uint8),y.contiguous().reshape(-1).view(torch.uint8)) for x,y in zip(actual,ref)],errors=errors,
                 python_noninferior=[all(a[m]<=b[m] for m in ('max_abs','relative_l2')) for a,b in zip(errors,pe)])
            assert all(e['finite'] for e in errors)
            for a,b in zip(actual,ref):torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)
        data=tuple(t.cuda().requires_grad_() for t in cpu);upgpu=up.cuda()
        def run(name):
            y=ops[name].forward(*data,3,1e-5)
            return torch.autograd.grad(y,data,upgpu)
        names=list(ops)
        for name in names:
            for _ in range(5):run(name)
        for i in range(6):
            values={}
            for name in names[i%len(names):]+names[:i%len(names)]:values[name]=timed(lambda:run(name),30)
            row['rounds'].append(values)
        row['median_gpu_ms']={name:statistics.median(r[name]['gpu_ms'] for r in row['rounds']) for name in names}
        row['paired_speedup']={name:statistics.median(r['before']['gpu_ms']/r[name]['gpu_ms'] for r in row['rounds']) for name in names}
        print(case[0],row['median_gpu_ms'],row['paired_speedup'],flush=True)
        del data,upgpu;torch.cuda.empty_cache()
        REPORT_FILE.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
def contracts(ops):
    import test_training_fusion as fusion
    import test_fp32_training as spatial
    import test_training_scale3_batch as large
    results={}
    for name in (n for n in ops if n!='before'):
        with patch.object(torch.ops,'converse2d',ops[name]),patch.object(fusion,'load_extension',lambda:None),patch.object(spatial,'load_extension',lambda:None),patch.object(large,'load_extension',lambda:None):
            suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(fusion),unittest.defaultTestLoader.loadTestsFromModule(spatial),unittest.defaultTestLoader.loadTestsFromModule(large)])
            result=unittest.TextTestRunner(verbosity=2).run(suite)
        results[name]=dict(tests=result.testsRun,failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors])
        (HERE/('contracts_'+name+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
        assert result.wasSuccessful()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['operators','fused','contracts','contracts_fused'],required=True);a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ops=load_all(warm=True)
    if a.phase=='fused':
        REPORT_FILE=HERE/'operators_fused.json'
        run_operators({n:ops[n] for n in ('before','int32','fused')})
    elif a.phase=='operators':run_operators({n:ops[n] for n in ('before','const64','int32')})
    elif a.phase=='contracts_fused':contracts({'fused':ops['fused']})
    else:contracts(ops)
