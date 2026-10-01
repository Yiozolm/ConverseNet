"""Same-process real/crop backward ablation; fixed weights, no optimizer updates."""
import argparse
import ctypes
import gc
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import experiment_nan_fill as common


class Telemetry:
    def __init__(self,enabled=True):
        self.rows=[]; self.phase='setup'; self.stop=threading.Event(); self.error=None; self.enabled=enabled
        if not enabled: return
        try:
            self.lib=ctypes.WinDLL('nvml.dll')
            self.lib.nvmlInit_v2.restype=ctypes.c_int
            if self.lib.nvmlInit_v2()!=0: raise RuntimeError('NVML init failed')
            self.handle=ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)]
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))!=0: raise RuntimeError('NVML handle failed')
            for name in ('nvmlDeviceGetClockInfo','nvmlDeviceGetTemperature'):
                getattr(self.lib,name).argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)]
            for name in ('nvmlDeviceGetPowerUsage','nvmlDeviceGetPerformanceState'):
                getattr(self.lib,name).argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)]
        except Exception as error:
            self.error=repr(error)
    def uint(self,name,*values):
        out=ctypes.c_uint()
        status=getattr(self.lib,name)(self.handle,*values,ctypes.byref(out))
        return out.value if status==0 else None
    def run(self):
        if self.error or not self.enabled: return
        while not self.stop.is_set():
            self.rows.append(dict(time=time.perf_counter(),phase=self.phase,
                graphics_mhz=self.uint('nvmlDeviceGetClockInfo',0),sm_mhz=self.uint('nvmlDeviceGetClockInfo',1),
                memory_mhz=self.uint('nvmlDeviceGetClockInfo',2),temperature_c=self.uint('nvmlDeviceGetTemperature',0),
                power_mw=self.uint('nvmlDeviceGetPowerUsage'),pstate=self.uint('nvmlDeviceGetPerformanceState')))
            self.stop.wait(.2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iters',type=int,default=5)
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--hold-optimizer-state',action='store_true',help='Load Adam buffers for memory parity but never call step')
    parser.add_argument('--no-telemetry',action='store_true')
    parser.add_argument('--cpu-preflight',action='store_true',help='Reproduce the old CPU-profiler operator preflight before model setup')
    args=parser.parse_args()
    args.root,args.data_root,args.output=(p.resolve() for p in (args.root,args.data_root,args.output))
    if args.output.exists(): parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    (args.output/'harness.py').write_bytes(Path(__file__).read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; os.environ['CONVERSE2D_SKIP_BUILD']='1'
    affinity=common.affinity(0xC03C03)
    sys.path[:0]=[str(args.root/'test'),str(args.root),str(args.root/'tools/roadmap_quality')]
    import torch
    import extension_loader
    common.configure(torch,False); extension_loader.load_extension()
    from models.converse_usrnet import ConverseUSRNet
    from models.util_converse import Converse2D
    import usrnet_training_data as data
    data.ROOT=args.data_root
    if args.cpu_preflight:
        probe=Converse2D(128,128,3,padding=2,backend='cuda').cuda().train()
        probe_x=torch.zeros(4,128,96,96,device='cuda',requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as probe_trace:
            probe_out=probe(probe_x)
            probe_grads=torch.autograd.grad(probe_out,(probe_x,probe.weight,probe.bias),torch.ones_like(probe_out))
        del probe,probe_x,probe_out,probe_grads,probe_trace
        gc.collect(); torch.cuda.empty_cache()
    telemetry=Telemetry(enabled=not args.no_telemetry); sampler=threading.Thread(target=telemetry.run,daemon=True); sampler.start()
    result=dict(status='running',harness_sha256=common.sha(__file__),checked_manifest=json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),
                environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),affinity=affinity,
                    fill_uninitialized_memory=False,deterministic=True,tf32=False,amp=False,torch_threads=torch.get_num_threads()),
                scope='Same model and resident inputs, forward/backward plus finite/norm only. No optimizer updates; optional Adam buffers are held unchanged. Both arms build the same hook views; native arm returns the reconstructed native view of the identical IFFT base.',
                pairs=[])
    result['telemetry_enabled']=not args.no_telemetry
    result['cpu_profiler_preflight']=args.cpu_preflight
    def write(): common.write(args.output/'result.json',result)
    handles=[]
    try:
        torch.manual_seed(17)
        model=ConverseUSRNet(backend='cuda').cuda().train()
        checkpoint=args.data_root/'artifacts/training_real_crop/v2/candidate_epoch/final.pth'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True)
        optimizer=None
        optimizer_before=None
        if args.hold_optimizer_state:
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,foreach=False,fused=False)
            optimizer.load_state_dict(state['optimizer'])
            optimizer_before={str(i)+'/'+key:common.tensor_record(value) for i,entry in optimizer.state_dict()['state'].items() for key,value in entry.items()}
        result['hold_optimizer_state']=args.hold_optimizer_state
        initial={name:common.tensor_record(value) for name,value in model.state_dict().items()}
        result['checkpoint_sha256']=common.sha(checkpoint)
        protocol=data.DatasetProtocol(args.data_root/'artifacts/dataset_training/split_900_100.json',patch_size=96,scale=3,seed=17,noise_std=.01)
        cpu_batches={i:protocol.train_batch(i,4) for i in (0,12,224)}
        batches={i:tuple(t.cuda() for t in batch) for i,batch in cpu_batches.items()}
        result['inputs']={str(i):{name:common.tensor_record(t) for name,t in zip(('lr','kernel','hr'),batch)} for i,batch in cpu_batches.items()}
        control={'mode':'fused','validate':True,'calls':0}
        def hook(module,inputs,out):
            base=out._base
            node=out.grad_fn
            native=base.real[...,module.padding:-module.padding,module.padding:-module.padding]
            if control['validate']:
                assert node.name()=='RealCropBackward',node.name()
                assert base.ndim==4 and base.dtype==torch.complex64 and base.is_contiguous(),(base.shape,base.stride())
                base_node=base.grad_fn; edge=node.next_functions[0]
                assert edge[0] is base_node and edge[1]==base.output_nr,(edge,base_node,base.output_nr)
                assert native.shape==out.shape and native.stride()==out.stride() and native.storage_offset()==out.storage_offset()
                assert native.data_ptr()==out.data_ptr()
            control['calls']+=1
            return native if control['mode']=='native' else out
        for module in model.modules():
            if isinstance(module,Converse2D): handles.append(module.register_forward_hook(hook))
        result['hooked_modules']=len(handles)
        def run(mode,batch_index,timed=True):
            control['mode']=mode; control['calls']=0
            telemetry.phase=mode
            model.zero_grad(set_to_none=True)
            x,k,target=batches[batch_index]
            torch.cuda.synchronize(); start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            began=time.perf_counter(); cpu_began=time.process_time(); start.record()
            output=model(x,k,3)
            loss=torch.nn.functional.mse_loss(output,target)*1.0
            loss.backward()
            end.record(); end.synchronize()
            row=dict(mode=mode,batch_index=batch_index,start=began,end=time.perf_counter(),
                     fb_wall_ms=(time.perf_counter()-began)*1000,fb_cpu_ms=(time.process_time()-cpu_began)*1000,
                     fb_cuda_ms=start.elapsed_time(end),calls=control['calls'])
            assert row['calls']==35,row['calls']
            grads=[p.grad.detach() for p in model.parameters()]
            torch.cuda.synchronize(); start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            began=time.perf_counter(); start.record()
            finite=torch.stack([torch.isfinite(loss.detach()),*[torch.isfinite(g).all() for g in grads]]).all()
            norm=torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads]))
            values=torch.stack((loss.detach(),norm,(finite & torch.isfinite(norm)).to(loss.dtype))).cpu().tolist()
            end.record(); end.synchronize()
            row.update(finite_wall_ms=(time.perf_counter()-began)*1000,finite_cuda_ms=start.elapsed_time(end),loss=values[0],grad_norm=values[1],finite=bool(values[2]))
            assert row['finite']
            return row,output.detach()
        # Gate exact IFFT edge/layout and exact parameter gradients before timing.
        snapshots={}
        for mode in ('native','fused'):
            row,out=run(mode,12)
            snapshots[mode]=dict(output=common.tensor_record(out),loss=row['loss'],grad_norm=row['grad_norm'],
                grads={name:dict(record=common.tensor_record(p.grad),stride=list(p.grad.stride()),offset=p.grad.storage_offset(),contiguous=p.grad.is_contiguous()) for name,p in model.named_parameters()})
            del out
        assert snapshots['native']==snapshots['fused'],'Gradient bytes/layout differ'
        result['gate']=snapshots
        control['validate']=False
        for mode in ('native','fused'):
            for _ in range(3): run(mode,12)
        for i in range(args.rounds):
            pair=dict(round=i,batch_index=(0,12,224)[i%3],order=['native','fused'] if i%2==0 else ['fused','native'],arms={})
            for mode in pair['order']:
                rows=[]
                for _ in range(args.iters):
                    row,out=run(mode,pair['batch_index']); rows.append(row); del out
                pair['arms'][mode]=dict(rows=rows,fb_ms=statistics.mean(r['fb_wall_ms'] for r in rows),fb_cuda_ms=statistics.mean(r['fb_cuda_ms'] for r in rows),finite_ms=statistics.mean(r['finite_wall_ms'] for r in rows),cpu_ms=statistics.mean(r['fb_cpu_ms'] for r in rows))
                print(json.dumps(dict(round=i,mode=mode,batch=pair['batch_index'],**{k:v for k,v in pair['arms'][mode].items() if k!='rows'})),flush=True)
            assert [(r['loss'],r['grad_norm']) for r in pair['arms']['native']['rows']]==[(r['loss'],r['grad_norm']) for r in pair['arms']['fused']['rows']]
            pair['speedup']=pair['arms']['native']['fb_ms']/pair['arms']['fused']['fb_ms']
            result['pairs'].append(pair); write()
        if args.profile:
            result['profiles']={}
            for mode in ('native','fused'):
                telemetry.phase='profile_'+mode
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as trace:
                    with torch.profiler.record_function('real_crop_ablation/'+mode): row,out=run(mode,12)
                trace.export_chrome_trace(str(args.output/(mode+'.trace.json.gz')))
                result['profiles'][mode]={'counts':{e.key:e.count for e in trace.key_averages()},'row':row}
                del out,trace
        final={name:common.tensor_record(value) for name,value in model.state_dict().items()}
        assert final==initial,'Model parameters were modified'
        if optimizer is not None:
            optimizer_after={str(i)+'/'+key:common.tensor_record(value) for i,entry in optimizer.state_dict()['state'].items() for key,value in entry.items()}
            assert optimizer_after==optimizer_before,'Adam state was modified'
            result['optimizer_unchanged']=True
        result['parameters_unchanged']=True
        result['summary']=dict(paired_fb_speedup_median=statistics.median(p['speedup'] for p in result['pairs']),paired_fb_speedup_range=[min(p['speedup'] for p in result['pairs']),max(p['speedup'] for p in result['pairs'])])
        result['status']='passed'; print(json.dumps(result['summary']),flush=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error)); raise
    finally:
        for handle in handles: handle.remove()
        telemetry.stop.set(); sampler.join(timeout=3)
        result['telemetry_error']=telemetry.error
        common.write(args.output/'telemetry.json',telemetry.rows)
        write()


if __name__=='__main__': main()
