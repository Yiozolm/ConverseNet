"""End-to-end native-vs-FP32 FFT timings, including scaling/packing and errors."""
import argparse
import copy
import json
import sys

import torch
from extension_loader import ROOT, load_extension
from benchmark_training_fusion import paired

sys.path.insert(0,str(ROOT))
from models.native_fft import native_fft

CONTEXTS={"baseline":lambda:native_fft(False),"current":lambda:native_fft(True)}


def error(a,b):
    if b.is_complex(): a,b=a.cdouble(),b.cdouble()
    else: a,b=a.double(),b.double()
    return dict(relative_l2=((a-b).norm()/b.norm().clamp_min(1e-300)).item(),
                max_abs=(a-b).abs().max().item(),finite=bool(torch.isfinite(a).all()))


def primitives(iters,rounds):
    rows=[]
    for precision in (0,1):
        for B,H,W in ((1,64,64),(1,256,256),(8,64,64),(1,68,76)):
            x=torch.randn(B,32,H,W,device="cuda")
            f=torch.fft.rfft2(x)
            for inverse in (False,True):
                fn=(lambda:torch.ops.converse2d._native_irfft(f,H,W,precision)) if inverse else (
                    lambda:torch.ops.converse2d._native_rfft(x,precision))
                with native_fft(False): ref=fn()
                with native_fft(): actual=fn()
                accuracy=error(actual,ref)
                assert accuracy["finite"]
                torch.ops.converse2d.reset_native_fft_stats()
                timing=paired({"baseline":fn,"current":fn},iters,rounds,CONTEXTS)
                row=dict(precision="FP16" if precision==0 else "BF16",shape=list(x.shape),
                         direction="irfft" if inverse else "rfft",accuracy=accuracy,
                         stats=dict(torch.ops.converse2d.native_fft_stats()),timing=timing)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!="timing"}|dict(speedup=timing["speedup"],medians=timing["medians"])),flush=True)
    return rows


def operators(iters,rounds):
    rows=[]
    for dtype in (torch.float16,torch.bfloat16):
        for B,H,W,s in ((1,64,64,1),(1,64,64,2),(8,64,64,2),(1,68,76,2),(1,64,64,3)):
            x=(torch.randn(B,32,H,W,device="cuda")*0.1).to(dtype).requires_grad_()
            k=torch.randn(1,32,3,3,device="cuda").flatten(2).softmax(-1).reshape(1,32,3,3).requires_grad_()
            b=torch.zeros(1,32,1,1,device="cuda",requires_grad=True)
            fn=lambda:torch.ops.converse2d.forward_nearest(x,k,b,s,1e-3)
            with native_fft(False): ref=fn()
            with native_fft(): actual=fn()
            upstream=torch.randn_like(actual)*0.001
            rg=torch.autograd.grad(ref,(x,k,b),upstream)
            ag=torch.autograd.grad(actual,(x,k,b),upstream)
            accuracy=dict(output=error(actual.detach(),ref.detach()),gradients=[error(a,r) for a,r in zip(ag,rg)])
            assert accuracy["output"]["finite"] and all(g["finite"] for g in accuracy["gradients"])
            # Explicit experimental budgets, recorded rather than presented as
            # dataset-quality acceptance. Native FFT rounding is not bitwise.
            limit=0.01 if dtype==torch.float16 else 0.06
            accuracy["within_experimental_output_budget"]=accuracy["output"]["relative_l2"]<limit
            torch.ops.converse2d.reset_native_fft_stats()
            with torch.no_grad(): inference=paired({"baseline":fn,"current":fn},iters,rounds,CONTEXTS)
            inference_stats=dict(torch.ops.converse2d.native_fft_stats())
            torch.ops.converse2d.reset_native_fft_stats()
            train=lambda:torch.autograd.grad(fn(),(x,k,b),upstream)
            training=paired({"baseline":train,"current":train},iters,rounds,CONTEXTS)
            row=dict(dtype=str(dtype),shape=list(x.shape),scale=s,accuracy=accuracy,
                     inference=inference,training=training,inference_stats=inference_stats,
                     training_stats=dict(torch.ops.converse2d.native_fft_stats()))
            rows.append(row)
            print(json.dumps(dict(dtype=str(dtype),shape=list(x.shape),scale=s,accuracy=accuracy,
                inference_speedup=inference["speedup"],training_speedup=training["speedup"],
                inference_medians=inference["medians"],training_medians=training["medians"])),flush=True)
    return rows


def pretrained(iters,rounds):
    from models.converse_dncnn import ConverseDnCNN
    model=ConverseDnCNN().cuda().eval()
    model.load_state_dict(torch.load(ROOT/"model_zoo"/"converse_dncnn.pth",map_location="cuda",weights_only=True))
    x=torch.rand(1,1,12,12,device="cuda")
    rows=[]
    for dtype in (torch.float16,torch.bfloat16):
        with torch.inference_mode(),torch.autocast("cuda",dtype=dtype):
            with native_fft(False): ref=model(x)
            with native_fft(): actual=model(x)
            torch.ops.converse2d.reset_native_fft_stats()
            timing=paired({"baseline":lambda:model(x),"current":lambda:model(x)},iters,rounds,CONTEXTS)
            rows.append(dict(dtype=str(dtype),model="pretrained ConverseDnCNN",input_shape=list(x.shape),
                accuracy=error(actual,ref),timing=timing,stats=dict(torch.ops.converse2d.native_fft_stats())))
    return rows


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--iters",type=int,default=20)
    parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--output",default="artifacts/native_fft_benchmark.json")
    args=parser.parse_args()
    if min(args.iters,args.rounds)<1: parser.error("counts must be positive")
    if not torch.cuda.is_available(): parser.error("CUDA required")
    load_extension()
    torch.manual_seed(591)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    result=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                iters=args.iters,rounds=args.rounds,kernel_fft="FP32",scaling="per-transform power of two",
                scope="same extension; native policy vs FP32 FFT policy; all pack/unpack/scaling costs included")
    result["primitives"]=primitives(args.iters,args.rounds)
    result["operators"]=operators(args.iters,args.rounds)
    result["pretrained"]=pretrained(args.iters,args.rounds)
    path=ROOT/args.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__": main()
