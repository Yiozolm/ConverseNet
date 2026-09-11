"""Dynamic-kernel DataNet benchmark: legacy, simplified and combined backends."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch
from extension_loader import ROOT, load_extension
from benchmark_corrected import measure

sys.path.insert(0,str(ROOT))


def load_class(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConvReverseDataNet


def main():
    load_extension()
    from models.converse_usrnet import ConvReverseDataNet
    folder=ROOT/".build"/"combined_baselines"
    folder.mkdir(parents=True,exist_ok=True)
    # Reconstruct the two math baselines from immutable commits. Both keep
    # their original Python tensor operations; only the import module differs.
    classes={}
    for label,ref in (("legacy_python","42f1d1c"),("simplified_python","dcc9896")):
        raw=subprocess.check_output(["git","show",f"{ref}:models/converse_usrnet.py"],cwd=ROOT)
        file=folder/f"{label}.py"
        file.write_bytes(raw)
        classes[label]=load_class(file,label)
    torch.manual_seed(721)
    rows=[]
    for training in (False,True):
        for b,h,w,s in ((1,64,80,1),(1,64,80,2),(1,128,128,3),(2,64,80,2)):
            x=torch.randn(b,64,h,w,device="cuda",requires_grad=training)
            k=(torch.randn(b,64,7,7,device="cuda")/49).requires_grad_(training)
            layers={label:cls().cuda() for label,cls in classes.items()}
            layers.update({v:ConvReverseDataNet(backend="cuda",variant=v).cuda() for v in ("v2","v7")})
            ref=ConvReverseDataNet(backend="pytorch").cuda().double()
            with torch.no_grad():
                expected=ref(x.double(),k.double(),s).float()
            with (torch.enable_grad() if training else torch.inference_mode()):
                for label,layer in layers.items():
                    # Model-produced kernels change identity each call. Avoid
                    # claiming hot-cache gains unavailable to dynamic kernels.
                    forward=lambda:layer(x,k.clone(),s)
                    out=forward().detach()
                    if label!="legacy_python":
                        torch.testing.assert_close(out,expected,atol=2e-4,rtol=5e-5)
                    fn=(lambda:torch.autograd.grad(forward().square().mean(),(x,k,layer.alpha))) if training else forward
                    row={"implementation":label,"B":b,"C":64,"H":h,"W":w,"scale":s,"training":training,
                         "max_abs_vs_float64":(out-expected).abs().max().item(),**measure(fn,5,10)}
                    rows.append(row)
                    print(json.dumps(row),flush=True)
            torch.ops.converse2d.clear_cache()
    result={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"kernel_size":7,
            "dynamic_kernel":True,"eps":1e-3,"results":rows}
    (ROOT/"analysis"/"combined_benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
