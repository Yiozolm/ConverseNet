"""One-epoch real/crop VJP comparison on top of q recomputation and pad fusion."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import experiment_nan_fill as common
from experiment_q_recompute import epoch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=('baseline','real_crop'),required=True)
    args=parser.parse_args()
    args.root,args.data_root,args.output=(p.resolve() for p in (args.root,args.data_root,args.output))
    if args.output.exists(): parser.error('Preserve evidence: output directory must be new')
    args.output.mkdir(parents=True)
    helpers=('experiment_training_real_crop.py','experiment_q_recompute.py','experiment_nan_fill.py')
    for name in helpers:
        (args.output/name).write_bytes((Path(__file__).parent/name).read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['CONVERSE2D_SKIP_BUILD']='1'
    affinity=common.affinity(0xC03C03)
    sys.path[:0]=[str(args.root/'test'),str(args.root),str(args.root/'tools/roadmap_quality')]
    import torch
    import extension_loader
    common.configure(torch,False)
    extension_loader.load_extension()
    from models.util_converse import Converse2D
    result=dict(status='running',arm=args.arm,root=str(args.root),
       commit=subprocess.check_output(['git','-C',str(args.root),'rev-parse','HEAD'],text=True).strip(),
       checked_manifest=json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),
       harness_sha256={name:common.sha(Path(__file__).parent/name) for name in helpers},
       environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
          affinity=affinity,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
          deterministic_algorithms=True,fill_uninitialized_memory=False,tf32=False,amp=False,
          cudnn_benchmark=False,cudnn_deterministic=True),
       source_sha256={name:common.sha(args.root/name) for name in ('models/converse_usrnet.py','models/util_converse.py',
          'models/converse_core.py','tools/roadmap_quality/train_usrnet_dataset.py','tools/roadmap_quality/usrnet_training_data.py')},
       scope='Both arms include q recomputation and circular-pad fusion. One B4/seed17 epoch: 900 unique images/225 Adam updates, warmup discarded, NaN fill off. Frozen epoch helper reused. Preflight is a single operator VJP with no optimizer and is outside timing. No convergence claim.')
    for name in extension_loader.production_source_hashes():
        dest=args.output/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((args.root/name).read_bytes())
    (args.output/'candidate.patch').write_bytes(subprocess.check_output(['git','-C',str(args.root),'diff','--binary']))
    def write_progress(): common.write(args.output/'result.json',result)
    write_progress()
    try:
        layer=Converse2D(128,128,3,padding=2,backend='cuda').cuda().train()
        x=torch.zeros(4,128,96,96,device='cuda',requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            out=layer(x)
            node_name=out.grad_fn.name()
            gradients=torch.autograd.grad(out,(x,layer.weight,layer.bias),torch.ones_like(out))
        events={event.key:event.count for event in trace.key_averages()}
        expected=args.arm=='real_crop'
        assert ('RealCropBackward' in node_name)==expected,(node_name,events)
        assert events.get('aten::fft_fft2')==2
        assert events.get('converse2d::_training_circular_s1')==1
        if expected:
            assert events.get('RealCropBackward')==1
            assert events.get('aten::slice_backward',0)==0
            assert events.get('aten::select_backward',0)==0
        result['preflight']=dict(events=events,output_node=node_name,output_stride=list(out.stride()),output_offset=out.storage_offset())
        del layer,x,out,gradients,trace
        gc.collect(); torch.cuda.empty_cache()
        epoch(torch,args,result,write_progress)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        write_progress()


if __name__=='__main__': main()
