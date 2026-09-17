"""Eager-only FP32 training: isolated ATen, fused VJP, and new s1 CUDA kernels."""
import argparse
import collections
import copy
import datetime
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch.nn import functional as F

from extension import ROOT, HERE, load
from runtime import TrainingRunner
from workloads import backend, route, workload


def errors(a, b):
    a, b = a.detach().double(), b.detach().double()
    return dict(max_abs=(a-b).abs().max().item(),
                relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                finite=bool(torch.isfinite(a).all()))


def snapshot_compare(actual, expected, atol=3e-5, rtol=3e-4):
    result = {}
    for group in ('params', 'grads', 'momentum', 'buffers'):
        values = []
        assert actual[group].keys() == expected[group].keys()
        for name in actual[group]:
            a, b = actual[group][name], expected[group][name]
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), (group,name,'nonfinite')
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda msg: f'{group}/{name}: {msg}')
            values.append(errors(a,b))
        result[group] = dict(max_abs=max((v['max_abs'] for v in values), default=0),
                             max_relative_l2=max((v['relative_l2'] for v in values), default=0))
    torch.testing.assert_close(actual['loss'], expected['loss'], atol=atol, rtol=rtol)
    result['loss'] = errors(actual['loss'], expected['loss'])
    return result


def sample(fn, iters):
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return dict(event_ms=a.elapsed_time(b)/iters,wall_ms=(time.perf_counter()-t)*1000/iters,
                incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial)


def paired(calls, iters, rounds, reset=None, verify=None):
    rows = {name:[] for name in calls}
    names = list(calls)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:
            order = order[::-1]
        for name in order:
            if reset:
                reset(name)
            rows[name].append(sample(calls[name],iters))
            if verify:
                verify(name)
    median = {n:{k:statistics.median(v[k] for v in values) for k in values[0]}
              for n,values in rows.items()}
    return dict(rounds=rows,medians=median)


def profile(fn, iters=3):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    values = collections.defaultdict(lambda:dict(calls=0,us=0.))
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            values[e.name]['calls'] += 1/iters
            values[e.name]['us'] += e.time_range.elapsed_us()/iters
    return sorted([dict(name=n,**v) for n,v in values.items()], key=lambda row:-row['us'])


def operator_case(config,args):
    B,C,H,W,s = config
    x = torch.randn(B,C,H,W,device='cuda',requires_grad=True)
    w = torch.randn(1,C,3,3,device='cuda').flatten(2).softmax(-1).reshape(1,C,3,3).requires_grad_()
    b = torch.zeros(1,C,1,1,device='cuda',requires_grad=True)
    upstream = torch.randn(B,C,H*s,W*s,device='cuda')*.001
    inputs = (x,w,b)
    calls = {}
    for kind in ('checkout','fused','s1'):
        op = backend(kind)
        def fn(op=op):
            p = x if s==1 else F.interpolate(x,scale_factor=s,mode='nearest')
            y = op(x,p,w,b,s,1e-3)
            grads = torch.autograd.grad(y,inputs,upstream)
            return (y.detach(),)+grads
        calls[kind+'_eager'] = fn
        for _ in range(8): fn()
    result = paired(calls,args.iters,args.rounds)
    result.update(config=config,scope='operator forward + first-order x/weight/bias VJP; no optimizer or copies')
    result['profiles'] = {kind:profile(calls[kind+'_eager']) for kind in ('fused','s1')}
    print(json.dumps(dict(operator=config,medians=result['medians'])),flush=True)
    return result


def training_case(config,args):
    kind,shape,scale,accumulation = config
    model,batches,targets = workload(kind,shape,scale,accumulation)
    runners = {}
    for variant in ('checkout','fused','s1'):
        name = variant+'_eager'
        runners[name] = TrainingRunner(route(copy.deepcopy(model),variant),batches,targets)
    initial = {name:runner.snapshot() for name,runner in runners.items()}
    checks = []
    for step in range(4):
        changed = [tuple(t*(1+.02*step)+(.003*step if j==0 else 0) for j,t in enumerate(batch)) for batch in batches]
        changed_targets = [t*(1-.01*step)+.001*step for t in targets]
        snapshots = {}
        for name,runner in runners.items():
            runner.step(changed,changed_targets)
            snapshots[name] = runner.snapshot()
        comparisons = {}
        for variant in ('fused','s1'):
            comparisons[variant+'_vs_checkout'] = snapshot_compare(snapshots[variant+'_eager'],snapshots['checkout_eager'])
        comparisons['s1_vs_fused'] = snapshot_compare(snapshots['s1_eager'],snapshots['fused_eager'])
        checks.append(comparisons)
    activity = {}
    for name,runner in runners.items():
        final = runner.snapshot()
        changed_names = [key for key,t in final['params'].items() if not torch.equal(t,initial[name]['params'][key])]
        nonzero = [key for key,t in final['grads'].items() if torch.count_nonzero(t).item()>0]
        assert changed_names and nonzero,name
        solver = [key for key in final['params'] if key.endswith('weight') and final['params'][key].ndim==4 and final['params'][key].shape[-1]==3]
        assert any(key in changed_names for key in solver),(name,'solver parameters did not update')
        if kind=='usrnet':
            assert 'model.d.alpha' in nonzero and 'model.d.alpha' in changed_names,name
            assert any('.kernelnet.' in key for key in nonzero),name
            assert any('.kernelnet.' in key for key in changed_names),name
        activity[name] = dict(updated_parameters=changed_names,nonzero_gradients=nonzero)
    # Every timed path starts from the same initial state and consumes the same
    # device-resident inputs. Gradient clearing, loss/backward and SGD are timed.
    calls = {name:(lambda runner=runner:runner.step(batches,targets)) for name,runner in runners.items()}
    def verify(name):
        state = runners[name].snapshot()
        assert torch.isfinite(state['loss']).all(),name
        for group in ('params','grads','momentum'):
            assert all(torch.isfinite(t).all() for t in state[group].values()),(name,group)
    timing = paired(calls,args.iters,args.rounds,reset=lambda name:runners[name].reset(),verify=verify)
    row = dict(kind=kind,shape=shape,scale=scale,accumulation=accumulation,checks=checks,
               activity=activity,timing=timing,
               scope='Eager FP32 full step: zero_grad(set_to_none=True), MSE, accumulated backward, SGD(momentum=.9, lr=1e-4); device-resident data without extra copies',
               setup={name:{'warmup_ms':r.warmup_ms}
                      for name,r in runners.items()})
    print(json.dumps(dict(training=config,medians=timing['medians'])),flush=True)
    del runners,initial,model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--iters',type=int,default=20)
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--mode',choices=('all','operators','training'),default='all')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='artifacts/training_speed/eager_results.json')
    args = parser.parse_args()
    if min(args.iters,args.rounds)<1: parser.error('positive iterations/rounds required')
    _,manifest = load(include_checkout=True)
    torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out = ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                  manifest=manifest,settings=vars(args),operators=[],training=[])
    report['experiment_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.cpp','.cu','.ps1')}
    def save(): out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.mode in ('all','operators'):
        configs = [(1,32,64,80,1),(8,32,64,64,1),(1,32,256,256,1),(1,32,64,80,2)]
        for config in configs[:1] if args.quick else configs:
            report['operators'].append(operator_case(config,args));save()
    if args.mode in ('all','training'):
        configs = [('operator',(1,32,64,80),1,1),('operator',(4,32,64,64),1,1),
                   ('operator',(1,32,256,256),1,1),
                   ('operator',(1,32,64,80),2,1),('block',(1,16,32,40),1,1),
                   ('block',(1,16,32,40),1,2),('usrnet',(1,3,16,20),2,1)]
        for config in configs[:1] if args.quick else configs:
            report['training'].append(training_case(config,args));save()
    print('Saved',out,flush=True)


if __name__=='__main__': main()
