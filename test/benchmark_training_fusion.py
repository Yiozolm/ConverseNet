"""Paired training-fusion timings, gradient errors and short AMP training traces."""
import argparse
import contextlib
import copy
import json
import statistics
import sys
import time
from unittest import mock

import torch
from extension_loader import ROOT, load_extension
from training_baseline import BASELINE_REF, load_baseline

sys.path.insert(0,str(ROOT))
from models.converse_core import converse2d_reference, converse2d_reference_nearest


def sample(fn,iters):
    torch.cuda.synchronize()
    initial=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    t=time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return dict(gpu_ms=start.elapsed_time(end)/iters,wall_ms=(time.perf_counter()-t)*1000/iters,
                peak_extra_bytes=torch.cuda.max_memory_allocated()-initial)


def paired(calls,iters,rounds,contexts=None):
    contexts=contexts or {name:contextlib.nullcontext for name in calls}
    values={name:[] for name in calls}
    for name,fn in calls.items():
        with contexts[name]():
            for _ in range(5):
                fn()
    for i in range(rounds):
        for name in list(calls)[::1 if i%2==0 else -1]:
            with contexts[name]():
                values[name].append(sample(calls[name],iters))
    medians={name:{key:statistics.median(v[key] for v in rows) for key in rows[0]} for name,rows in values.items()}
    return dict(rounds=values,medians=medians,speedup=medians["baseline"]["gpu_ms"]/medians["current"]["gpu_ms"])


def error(a,b):
    delta=a.double()-b.double()
    return dict(max_abs=delta.abs().max().item(),relative_l2=(delta.norm()/b.double().norm().clamp_min(1e-30)).item(),
                finite=bool(torch.isfinite(a).all()),zero_fraction=(a==0).double().mean().item())


def operators(baseline,iters,rounds):
    rows=[]
    for dtype in (torch.float16,torch.bfloat16):
        for B,H,W,s,nearest,dynamic in ((1,64,80,1,True,False),(1,64,80,2,False,False),
                                       (4,32,40,3,True,True),(8,64,64,2,True,False)):
            x=torch.randn(B,32,H,W,device="cuda",dtype=dtype).requires_grad_()
            k=torch.randn(B if dynamic else 1,32,3,3,device="cuda").flatten(2).softmax(-1).reshape(-1,32,3,3).requires_grad_()
            b=torch.zeros(1,32,1,1,device="cuda",requires_grad=True)
            p=torch.randn(B,32,H*s,W*s,device="cuda",dtype=dtype).requires_grad_()
            data=(x,k,b) if nearest else (x,p,k,b)
            old=baseline.forward_nearest if nearest else baseline.forward
            new=torch.ops.converse2d.forward_nearest if nearest else torch.ops.converse2d.forward
            forwards={"baseline":lambda:old(*data,s,1e-3),"current":lambda:new(*data,s,1e-3)}
            outputs={name:fn() for name,fn in forwards.items()}
            upstream=torch.randn_like(outputs["current"])*0.001
            reference_data=tuple(t.detach().double().requires_grad_() for t in data)
            ref_fn=converse2d_reference_nearest if nearest else converse2d_reference
            reference=ref_fn(*reference_data,s,1e-3)
            ref_grads=torch.autograd.grad(reference,reference_data,upstream.double())
            gradients={name:torch.autograd.grad(out,data,upstream,retain_graph=True) for name,out in outputs.items()}
            accuracy={name:dict(output=error(outputs[name].detach(),reference.detach()),
                                 gradients=[error(a,e) for a,e in zip(gradients[name],ref_grads)]) for name in outputs}
            tol=0.008 if dtype==torch.float16 else 0.06
            for name,out in outputs.items():
                torch.testing.assert_close(out.double(),reference,atol=tol,rtol=tol)
                for a,e in zip(gradients[name],ref_grads):
                    torch.testing.assert_close(a.double(),e,atol=tol/10,rtol=tol)
            backward={name:(lambda out=out:torch.autograd.grad(out,data,upstream,retain_graph=True)) for name,out in outputs.items()}
            training={name:(lambda fn=fn:torch.autograd.grad(fn(),data,upstream)) for name,fn in forwards.items()}
            modes={"forward":forwards,"backward":backward,"forward_backward":training}
            row=dict(dtype=str(dtype),shape=list(x.shape),scale=s,nearest=nearest,dynamic_kernel=dynamic,accuracy=accuracy)
            for mode,calls in modes.items():
                row[mode]=paired(calls,iters,rounds)
            rows.append(row)
            print(json.dumps(dict(dtype=str(dtype),shape=list(x.shape),scale=s,nearest=nearest,
                results={mode:dict(speedup=row[mode]["speedup"],medians=row[mode]["medians"]) for mode in modes})),flush=True)
    return rows


def models(baseline,iters,rounds,steps):
    from models.converse_usrnet import ConverseUSRNet
    rows=[]
    contexts={"baseline":lambda:mock.patch("models.util_converse._converse2d_nearest",baseline.forward_nearest),
              "current":contextlib.nullcontext}
    for dtype in (torch.float16,torch.bfloat16):
        first=ConverseUSRNet(num_iterations=2,num_blocks=1,backend="cuda").cuda()
        nets={"baseline":first,"current":copy.deepcopy(first)}
        x=torch.rand(2,3,16,20,device="cuda")
        k=torch.rand(2,1,7,7,device="cuda")/49
        target=torch.rand(2,3,32,40,device="cuda")
        optimizers={name:torch.optim.SGD(net.parameters(),lr=1e-3) for name,net in nets.items()}
        scalers={name:torch.amp.GradScaler("cuda",init_scale=128,enabled=dtype==torch.float16) for name in nets}

        def step(name,record=False):
            net,optimizer,scaler=nets[name],optimizers[name],scalers[name]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=dtype):
                out=net(x,k,2)
                loss=(out.float()-target).square().mean()
            scaler.scale(loss).backward()
            if record:
                scaler.unscale_(optimizer)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            scaler.step(optimizer)
            scaler.update()
            return loss.detach()

        histories={name:[] for name in nets}
        initial=first.conv2.weight.detach().clone()
        for _ in range(steps):
            for name in nets:
                with contexts[name]():
                    histories[name].append(step(name,True).item())
        assert all(not torch.equal(initial,net.conv2.weight) for net in nets.values())
        torch.testing.assert_close(torch.tensor(histories["current"]),torch.tensor(histories["baseline"]),atol=0.002,rtol=0.02)
        parameter_error=max((a-b).abs().max().item() for a,b in zip(first.parameters(),nets["current"].parameters()))
        timings=paired({name:(lambda name=name:step(name)) for name in nets},iters,rounds,contexts)
        final_loss={}
        for name,net in nets.items():
            assert all(torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            with contexts[name](),torch.no_grad(),torch.autocast("cuda",dtype=dtype):
                value=(net(x,k,2).float()-target).square().mean()
                assert torch.isfinite(value)
                final_loss[name]=value.item()
        row=dict(dtype=str(dtype),model="ConverseUSRNet(iterations=2,blocks=1)",input_shape=list(x.shape),scale=2,
                 loss_trace=histories,trace_steps=steps,max_parameter_abs_difference=parameter_error,
                 full_step=timings,final_loss=final_loss,final_scaler_scale={n:s.get_scale() for n,s in scalers.items()})
        rows.append(row)
        print(json.dumps(dict(dtype=str(dtype),model=row["model"],loss_trace=histories,
            max_parameter_abs_difference=parameter_error,speedup=timings["speedup"],medians=timings["medians"])),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--iters",type=int,default=20)
    parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--trace-steps",type=int,default=10)
    parser.add_argument("--operators-only",action="store_true")
    parser.add_argument("--output",default="artifacts/training_fusion_benchmark.json")
    args=parser.parse_args()
    if min(args.iters,args.rounds,args.trace_steps)<1:
        parser.error("counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    load_extension()
    baseline=load_baseline()
    torch.manual_seed(395)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    report=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                baseline=BASELINE_REF,iters=args.iters,rounds=args.rounds,
                scope="FP32 master parameters; paired same-input operator timings; synthetic short-model training, not convergence")
    report["operators"]=operators(baseline,args.iters,args.rounds)
    report["models"]=[] if args.operators_only else models(baseline,args.iters,args.rounds,args.trace_steps)
    path=ROOT/args.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
