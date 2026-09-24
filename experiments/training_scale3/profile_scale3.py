"""Profile a verified warm study binary; never mix capture with formal timing."""
import json,subprocess,sys
from pathlib import Path
from locations import HERE,ROOT

def worker(route,metadata):
    import torch
    from loader import load_all
    selected=load_all(warm=True)[route]
    torch.manual_seed(220922)
    torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(4,32,64,80,device='cuda',requires_grad=True)
    k=torch.softmax(torch.randn(1,32,9,device='cuda'),-1).reshape(1,32,3,3).requires_grad_()
    b=torch.zeros(1,32,1,1,device='cuda',requires_grad=True)
    up=torch.randn(4,32,192,240,device='cuda')*.001
    def step():
        with torch.cuda.nvtx.range('s3/'+route+'/forward'):
            prior=torch.nn.functional.interpolate(x,scale_factor=3,mode='nearest')
            y=selected.forward(x,prior,k,b,3,1e-5,'v7')
        with torch.cuda.nvtx.range('s3/'+route+'/backward'):
            return torch.autograd.grad(y,(x,k,b),up)
    for _ in range(5):step()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for _ in range(3):step()
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
    Path(metadata).write_text(json.dumps(dict(status='complete',route=route,
        scope='B4/C32/LR64x80/s3, nearest prior, forward+x/kernel/bias VJPs; diagnostics only',
        identity=json.loads((HERE/'builds.json').read_text())),indent=2),encoding='utf-8')

def launch(route,tool):
    from profile_nsight_training import locate
    out=HERE/f'profile_{tool}_{route}';out.mkdir(exist_ok=False)
    bootstrap=out/'worker.py';metadata=out/'metadata.json'
    bootstrap.write_text('import sys\n'+f'sys.path[:0] = {[str(Path(__file__).parent),str(ROOT/"test"),str(ROOT)]!r}\n'+
        'from profile_scale3 import worker\n'+f'worker({route!r},{str(metadata)!r})\n',encoding='utf-8')
    if tool=='nsys':
        cmd=[locate(tool),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=false',
             '--wait=primary','--show-output=true','--export=sqlite','-o',str(out/'capture')]
        report=out/'capture.nsys-rep'
    else:
        cmd=[locate(tool),'--target-processes','all','--profile-from-start','off',
             '--kernel-name','regex:.*(adjoint_filter|fused_backward_s3).*',
             '--launch-count','1' if route=='before' else '2',
             '--section','SpeedOfLight','--section','LaunchStats','--section','WarpStateStats',
             '-o',str(out/'capture')]
        report=out/'capture.ncu-rep'
    cmd += [sys.executable,'-u',str(bootstrap)]
    (out/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
    with (out/'capture.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode or not report.exists() or not metadata.exists():
        raise RuntimeError(f'{tool} capture incomplete; inspect {out}')
    print(report,flush=True)
