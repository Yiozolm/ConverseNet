"""Check pretrained USRNet outputs across the Python and CUDA backends."""
import json
import sys

import torch
from extension_loader import ROOT,load_extension
sys.path.insert(0,str(ROOT))


def main():
    if "--installed" in sys.argv:
        sys.path.insert(0,str(ROOT/"Converse2D"))
        import torch_converse2d
    else:
        load_extension()
    from models.converse_usrnet import ConverseUSRNet
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(53)
    model=ConverseUSRNet(backend="pytorch").cuda().eval()
    model.load_state_dict(torch.load(ROOT/"model_zoo/converse_usrnet.pth",map_location="cuda",weights_only=True),strict=True)
    rows=[]
    for dtype in (torch.float32,torch.float64):
        model.to(dtype)
        for b in (1,2):
            x=torch.rand(b,3,12,16,device="cuda",dtype=dtype)
            coords=torch.arange(7,device="cuda",dtype=dtype)-3
            kernels=[]
            for i in range(b):
                k=torch.exp(-(coords[:,None].square()+coords[None,:].square())/(2*(1.0+i*0.5)**2))
                kernels.append(k/k.sum())
            kernel=torch.stack(kernels)[:,None]
            for s in (1,2):
                for layer in model.modules():
                    if hasattr(layer,"backend"): layer.backend="pytorch"
                with torch.inference_mode(): reference=model(x,kernel,s)
                for layer in model.modules():
                    if hasattr(layer,"backend"): layer.backend="cuda"
                with torch.inference_mode(): actual=model(x,kernel,s)
                tol=3e-5 if dtype==torch.float32 else 1e-10
                torch.testing.assert_close(actual,reference,atol=tol,rtol=tol)
                assert torch.isfinite(actual).all()
                row={"dtype":str(dtype),"B":b,"scale":s,"max_abs":(actual-reference).abs().max().item()}
                rows.append(row)
                print(json.dumps(row),flush=True)
                torch.ops.converse2d.clear_cache()
    output=ROOT/"artifacts/usrnet_pretrained.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"tf32":False,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__": main()
