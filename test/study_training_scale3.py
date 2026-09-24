"""Opt-in s3 study. Production dispatch remains unchanged."""
from pathlib import Path
import argparse,os,runpy,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--phase',choices=['build','compile','operators','fused','contracts','contracts_fused','training','profile'],required=True)
parser.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/training_scale3_20260922')
parser.add_argument('--route',choices=['before','fused'],default='fused')
parser.add_argument('--tool',choices=['nsys','ncu'],default='nsys')
args=parser.parse_args()
os.environ['CONVERSE_S3_ARTIFACTS']=str(args.artifacts.resolve())
artifact_dir=args.artifacts.resolve()
outputs={
    'operators':['operators.json'],
    'fused':['operators_fused.json'],
    'contracts_fused':['contracts_fused.json'],
    'training':['boundaries.json','steps.json','model.json'],
}
protected=[artifact_dir/name for name in outputs.get(args.phase,[])]
if args.phase in ('contracts','compile'):
    protected += list(artifact_dir.glob('contracts_*.json'))
if args.phase=='compile':
    protected += list(artifact_dir.glob('profile_*'))
    protected += [artifact_dir/name for name in ('operators.json','operators_fused.json','boundaries.json','steps.json','model.json')]
if any(path.exists() for path in protected):
    raise FileExistsError('Result files already exist; use a new --artifacts directory to preserve previous evidence.')
experiment=ROOT/'experiments/training_scale3'
sys.path[:0]=[str(experiment),str(ROOT/'test'),str(ROOT)]
if args.phase in ('build','compile'):
    if args.phase=='build':runpy.run_path(str(experiment/'generate.py'),run_name='__main__')
    from loader import load_all
    load_all()
elif args.phase=='training':
    runpy.run_path(str(experiment/'training_checks.py'),run_name='__main__')
elif args.phase=='profile':
    from profile_scale3 import launch
    launch(args.route,args.tool)
else:
    sys.argv=[str(experiment/'study.py'),'--phase',args.phase]
    runpy.run_path(sys.argv[0],run_name='__main__')
