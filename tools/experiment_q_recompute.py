"""Isolated q-recomputation operator gate and one-epoch arm (no production switch)."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import experiment_nan_fill as common


def gate(torch, arm):
    cases = {}
    for batch, channels, size, kb, kc in ((1,128,100,1,128), (4,128,100,1,128), (4,64,96,4,64), (4,8,13,1,1)):
        for shared in (False, True):
            generator = torch.Generator().manual_seed(6317 + batch + channels)
            shape = (batch, channels, size, size)
            raw = [torch.randn(shape,generator=generator), torch.randn(shape,generator=generator),
                   torch.randn(kb,kc,3,3,generator=generator)/9, torch.randn(1,channels,1,1,generator=generator)]
            values = [t.cuda().requires_grad_() for t in raw]
            if shared:
                values[1] = values[0]
            upstream = torch.randn(shape,generator=generator).cuda()/((batch*channels*size*size)**.5)
            output = torch.ops.converse2d.forward(*values,1,1e-5)
            requested = [values[0],values[2],values[3]] if shared else values
            grads = torch.autograd.grad(output, requested, upstream)
            name = f"B{batch}C{channels}HW{size}KB{kb}KC{kc}_shared{shared}"
            cases[name] = {str(i):common.tensor_record(v) for i,v in enumerate((output,*grads))}
            print('gate',name,flush=True)
            del output, grads, values, requested, upstream
    torch.manual_seed(3187)
    values = [torch.randn(4,3,5,7,device='cuda',dtype=torch.complex64).requires_grad_(),
              torch.randn(4,3,5,7,device='cuda',dtype=torch.complex64).requires_grad_(),
              torch.randn(1,3,5,7,device='cuda',dtype=torch.complex64).requires_grad_(),
              torch.full((1,3,1,1),.1,device='cuda',requires_grad=True)]
    saved=[]
    def pack(t):
        saved.append(t)
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        output = torch.ops.converse2d._training_full_spectral(*values,1)
    pointers={v.data_ptr() for v in values}
    extras=[t for t in saved if t.is_complex() and t.data_ptr() not in pointers]
    extra_bytes=sum(t.numel()*t.element_size() for t in extras)
    expected = 0 if arm=='recompute' else 4*3*5*7*8
    if extra_bytes != expected:
        raise RuntimeError(f'Unexpected saved q bytes {extra_bytes}, expected {expected}')
    torch.autograd.grad(output.real.sum(),values)
    return dict(cases=cases, saved_extra_complex_bytes=extra_bytes, saved_tensor_count=len(saved))


def epoch(torch,args,result,write_progress):
    import usrnet_training_data as data
    import train_usrnet_dataset as worker
    from models.converse_usrnet import ConverseUSRNet
    data.ROOT=args.data_root
    protocol=data.DatasetProtocol(args.data_root/'artifacts/dataset_training/split_900_100.json',
                                  patch_size=96,scale=3,seed=17,noise_std=.01)
    batch_size=4
    assert len(protocol.train)==900
    steps=225
    recipe=argparse.Namespace(batch_size=4,microbatch_size=4,patch_size=96,scale=3,loss='mse')
    checkpoint=args.root/'model_zoo/converse_usrnet.pth'
    initial=torch.load(checkpoint,map_location='cpu',weights_only=True)
    torch.manual_seed(17)
    model=ConverseUSRNet(backend='cuda').cuda().train()
    model.load_state_dict(initial,strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,foreach=False,fused=False)
    for i in range(3):
        row=worker.train_step(model,optimizer,protocol.train_batch(i,batch_size),recipe)
        assert row['optimizer_applied']
    model.load_state_dict(initial,strict=True)
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    torch.manual_seed(17)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    result.update(dataset=protocol.metadata,checkpoint_sha256=common.sha(checkpoint),steps=rows)
    captured=[]
    hook=None
    data_seconds=0.
    began=time.perf_counter()
    try:
        for index in range(steps):
            data_began=time.perf_counter()
            batch=protocol.train_batch(index,batch_size)
            worker.check_cpu_batch(batch,recipe,batch_size)
            data_seconds+=time.perf_counter()-data_began
            inputs={name:common.tensor_record(t) for name,t in zip(('lr','kernel','hr'),batch)}
            if index==steps-1:
                hook=model.register_forward_hook(lambda _m,_i,out:captured.append(out.detach()))
            row=worker.train_step(model,optimizer,batch,recipe)
            row.update(step=index+1,inputs=inputs)
            rows.append(row)
            if not row['optimizer_applied']:
                raise RuntimeError(f'Nonfinite step {index+1}')
            if (index+1)%25==0:
                print(f"{args.arm} step={index+1}/225 loss={row['loss']:.9g} elapsed={time.perf_counter()-began:.2f}s",flush=True)
                write_progress()
        torch.cuda.synchronize()
        epoch_seconds=time.perf_counter()-began
    finally:
        if hook is not None: hook.remove()
    ids=[protocol.train[int(i)]['relative_path'] for i in protocol._permutations[0]]
    assert len(set(ids))==900
    result.update(epoch_wall_seconds=epoch_seconds,data_prepare_seconds=data_seconds,
                  training_wall_seconds=sum(r['training_step_wall_ms'] for r in rows)/1000,
                  training_step_mean_ms=statistics.mean(r['training_step_wall_ms'] for r in rows),
                  peak_memory=worker.cuda_peaks(),image_order=ids,
                  terminal=common.snapshot(model,optimizer,captured[0],rows[-1]))
    saved=args.output/'final.pth'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=1,optimizer_steps=225,
                    seed=17,arm=args.arm,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),saved)
    result['final_checkpoint_sha256']=common.sha(saved)
    print(json.dumps({key:result[key] for key in ('epoch_wall_seconds','training_wall_seconds','peak_memory')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('gate','epoch'),required=True)
    parser.add_argument('--arm',choices=('baseline','recompute'),required=True)
    args=parser.parse_args()
    args.root,args.data_root,args.output=(p.resolve() for p in (args.root,args.data_root,args.output))
    if args.output.exists(): parser.error('Choose a new output directory; preserve previous results')
    args.output.mkdir(parents=True)
    (args.output/'harness.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'experiment_nan_fill.py').write_bytes(Path(common.__file__).read_bytes())
    os.environ['CONVERSE2D_SKIP_BUILD']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    affinity=common.affinity(0xC03C03)
    sys.path[:0]=[str(args.root/'test'),str(args.root),str(args.root/'tools/roadmap_quality')]
    import torch
    import extension_loader
    common.configure(torch,False)
    extension_loader.load_extension()
    result=dict(status='running',arm=args.arm,mode=args.mode,root=str(args.root),harness_sha256=common.sha(__file__),
                helper_sha256=common.sha(common.__file__),
                checked_manifest=json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),
                commit=subprocess.check_output(['git','-C',str(args.root),'rev-parse','HEAD'],text=True).strip(),
                environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                  affinity=affinity,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                  deterministic_algorithms=True,fill_uninitialized_memory=False,tf32=False,amp=False,cudnn_benchmark=False,cudnn_deterministic=True),
                source_sha256={name:common.sha(args.root/name) for name in ('models/converse_usrnet.py','models/util_converse.py',
                    'models/converse_core.py','tools/roadmap_quality/train_usrnet_dataset.py','tools/roadmap_quality/usrnet_training_data.py')},
                scope='Gate has no model training. Epoch: B4, seed17, 900 unique images/225 Adam updates, warmup discarded. Loop includes data processing and diagnostic input hashes/progress; excludes startup, warmup, final hashes/checkpoint and evaluation. One epoch per arm; no convergence claim.')
    (args.output/'candidate.patch').write_bytes(subprocess.check_output(['git','-C',str(args.root),'diff','--binary']))
    # Preserve newly introduced headers as well as tracked production sources.
    for name in extension_loader.production_source_hashes():
        dest=args.output/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((args.root/name).read_bytes())
    def write_progress(): common.write(args.output/'result.json',result)
    write_progress()
    try:
        if args.mode=='gate': result.update(gate(torch,args.arm))
        else: epoch(torch,args,result,write_progress)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        write_progress()


if __name__=='__main__': main()
