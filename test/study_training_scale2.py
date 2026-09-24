"""Opt-in s2 study. Production dispatch remains unchanged."""
from pathlib import Path
import argparse,os,runpy,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--phase',choices=['build','compile','operators','fused','contracts','contracts_fused','training','profile'],required=True)
parser.add_argument('--artifacts',type=Path,default=ROOT/'artifacts/training_scale2_20260921')
parser.add_argument('--route',choices=['before','fused'],default='fused')
parser.add_argument('--tool',choices=['nsys','ncu'],default='nsys')
args=parser.parse_args()
os.environ['CONVERSE_S2_ARTIFACTS']=str(args.artifacts.resolve())
experiment=ROOT/'experiments/training_scale2'
sys.path[:0]=[str(experiment),str(ROOT/'test'),str(ROOT)]
if args.phase in ('build','compile'):
    if args.phase=='build':runpy.run_path(str(experiment/'generate.py'),run_name='__main__')
    from loader import load_all
    load_all()
elif args.phase=='training':
    runpy.run_path(str(experiment/'training_checks.py'),run_name='__main__')
elif args.phase=='profile':
    from profile_scale2 import launch
    launch(args.route,args.tool)
else:
    sys.argv=[str(experiment/'study.py'),'--phase',args.phase]
    runpy.run_path(sys.argv[0],run_name='__main__')
