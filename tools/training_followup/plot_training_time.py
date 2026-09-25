"""Render recorded historical matched blocks; this plot makes no causal claim."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads(args.report.read_text(encoding='utf-8'))
    figure,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True)
    for axis,pair in zip(axes,report['pairs']):
        blocks=pair['blocks_250']
        x=[(r['first_step']+r['last_step'])/2 for r in blocks]
        for variant,color in (('before','#55657b'),('current','#c36928')):
            axis.plot(x,[r[variant+'_forward_backward_ms']['median'] for r in blocks],label=variant,color=color,lw=2,marker='o',ms=3)
        axis.set_title('Seed '+str(pair['seed']),loc='left',fontweight='bold')
        axis.set_xlabel('Optimizer update (250-step blocks)')
        axis.grid(alpha=.2)
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].set_ylabel('Median forward + loss + backward wall time (ms)')
    axes[0].legend(frameon=False)
    figure.suptitle('USRNet B4: time variation within matched historical trajectories',fontsize=14,fontweight='bold',y=.98)
    figure.text(.5,.035,'Identical data and numeric trajectories; sequential processes, not an interleaved benchmark.\nBlock medians do not identify clocks, temperature, contention, or a causal code regression.',ha='center',fontsize=9,color='#555555')
    figure.subplots_adjust(left=.075,right=.98,top=.84,bottom=.23,wspace=.12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output,dpi=160)
    manifest=dict(source=str(args.report),source_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  image_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
