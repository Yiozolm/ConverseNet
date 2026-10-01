"""One-epoch comparison for circular pad + complex64 preparation, after q recomputation."""
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
    parser.add_argument('--arm',choices=('baseline','circular_pad'),required=True)
    args=parser.parse_args()
    args.root,args.data_root,args.output=(p.resolve() for p in (args.root,args.data_root,args.output))
    if args.output.exists(): parser.error('Preserve previous evidence: use a new output directory')
    args.output.mkdir(parents=True)
    for name in ('experiment_training_pad.py','experiment_q_recompute.py','experiment_nan_fill.py'):
        (args.output/name).write_bytes((Path(__file__).parent/name).read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['CONVERSE2D_SKIP_BUILD']='1'
    affinity=common.affinity(0xC03C03)
    sys.path[:0]=[str(args.root/'test'),str(args.root),str(args.root/'tools/roadmap_quality')]
    import torch
    import extension_loader
    from models.util_converse import Converse2D
    common.configure(torch,False)
    extension_loader.load_extension()
    result=dict(status='running',arm=args.arm,root=str(args.root),
       commit=subprocess.check_output(['git','-C',str(args.root),'rev-parse','HEAD'],text=True).strip(),
       checked_manifest=json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),
       harness_sha256={name:common.sha(Path(__file__).parent/name) for name in
          ('experiment_training_pad.py','experiment_q_recompute.py','experiment_nan_fill.py')},
       environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
          affinity=affinity,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
          deterministic_algorithms=True,fill_uninitialized_memory=False,tf32=False,amp=False,
          cudnn_benchmark=False,cudnn_deterministic=True),
       source_sha256={name:common.sha(args.root/name) for name in ('models/converse_usrnet.py','models/util_converse.py',
          'models/converse_core.py','tools/roadmap_quality/train_usrnet_dataset.py','tools/roadmap_quality/usrnet_training_data.py')},
       scope='Both arms include s1 q recomputation. One B4/seed17 epoch: 900 unique images, 225 Adam updates. Exact frozen epoch helper reused. NaN fill off. Loop includes data processing/input hashes/progress, excludes setup/preflight/warmup/final hashes/checkpoint/evaluation. No convergence claim.')
    for name in extension_loader.production_source_hashes():
        dest=args.output/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((args.root/name).read_bytes())
    (args.output/'candidate.patch').write_bytes(subprocess.check_output(['git','-C',str(args.root),'diff','--binary']))
    def write_progress(): common.write(args.output/'result.json',result)
    write_progress()
    try:
        # One forward-only operator preflight, outside epoch and warmup timing.
        # Proves the actual model class selects the intended private entry.
        layer=Converse2D(128,128,3,padding=2,backend='cuda').cuda().train()
        x=torch.zeros(4,128,96,96,device='cuda',requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            out=layer(x)
        events={event.key:event.count for event in trace.key_averages()}
        fused=events.get('converse2d::_training_circular_s1',0)
        assert fused==(1 if args.arm=='circular_pad' else 0),events
        assert events.get('aten::fft_fft2')==2
        result['preflight']=dict(events=events,output_stride=list(out.stride()),output_offset=out.storage_offset())
        del layer,x,out,trace
        gc.collect(); torch.cuda.empty_cache()
        epoch(torch,args,result,write_progress)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        write_progress()


if __name__=='__main__': main()
