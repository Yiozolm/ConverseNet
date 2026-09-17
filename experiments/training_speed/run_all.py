"""Run correctness, smoke, then three separate eager-only benchmark processes."""
import argparse
import pathlib
import subprocess
import sys

ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--iters',type=int,default=20)
    parser.add_argument('--rounds',type=int,default=6)
    args=parser.parse_args()
    out=ROOT/'artifacts/training_speed'
    out.mkdir(parents=True,exist_ok=True)
    jobs=[
        ('validation',[str(HERE/'validate.py')]),
        ('eager_smoke',[str(HERE/'study.py'),'--quick','--iters','3','--rounds','2',
                  '--output','artifacts/training_speed/eager_smoke.json']),
    ]
    for repeat in range(1,4):
        jobs.append((f'eager_benchmark_{repeat}',[str(HERE/'study.py'),'--iters',str(args.iters),
                    '--rounds',str(args.rounds),'--output',f'artifacts/training_speed/eager_results_{repeat}.json']))
    for name,command in jobs:
        path=out/(name+'.log')
        print('START',name,'log:',path,flush=True)
        with path.open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
        print('\n'.join(lines[-15:]),flush=True)
        print('END',name,'exit:',result.returncode,flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__=='__main__':sys.exit(main())
