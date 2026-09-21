"""Exact extraction parity for models.converse_training; no model integration.

    python test/test_converse_training_helpers.py
    ./experiments/training_speed/run.ps1 test/test_converse_training_helpers.py --cuda --output artifacts/native_deconv_target/training_helpers_cuda.json

CPU defaults to ATen spectral stand-ins and the 24 extraction checks. --cuda
independently runs the same helper checks plus all four solver broadcast forms
using the actual isolated SharedTransfer and production _training_spectral.
Ordinary VJPs explicitly use create_graph=False; fresh graphs check directional
HVPs through create_graph=True. Output, gradient dtype and every value must
match the frozen operation order exactly (atol=rtol=0).

This only checks extraction parity, not numerical superiority, performance,
training convergence or production dispatch. Source hashes are checked before
and after the run; the three reference scripts have immutable expected hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FROZEN = {
    "test/train_usrnet_mixed_shared_candidate.py":"45735b92b57b91cce9e39467e38cd2bf9c87498432bc49da03c6ccf6046101f9",
    "test/probe_nearest_training.py":"da803392a7c6084bf3558ff3f050a47a0e7e04eb63a19d1559f915eb28a15e4f",
    "test/probe_converse_boundaries.py":"02c8178fdd4149de2d7d25036e3a832243c11e1cf07acd81e1f4459512a4f1e6",
}


def source_hashes():
    paths = [ROOT/name for name in FROZEN]
    paths += [Path(__file__).resolve(),ROOT/"models/converse_training.py",ROOT/"test/extension_loader.py"]
    paths += [ROOT/"experiments/training_shared_s1"/name for name in ("bindings.cpp","kernels.cu","loader.py")]
    for suffix in ("*.cpp","*.cu","*.h"):
        paths += list((ROOT/"Converse2D/torch_converse2d").glob(suffix))
    return {path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def aten_spectral(y,prior,kernel,regularizer,height,width,scale):
    import torch
    def full(value,w):
        tail=value[...,1:(w+1)//2].flip((-2,-1)).roll(1,-2)
        return torch.cat((value,tail.conj() if value.is_complex() else tail),-1)
    def alias(value):
        return value.reshape(*value.shape[:2],scale,height,scale,width).mean((2,4))
    power=kernel.real.square()+kernel.imag.square()
    if scale==1:
        q=(y-kernel*prior)/(power+regularizer)
    else:
        q=(y-alias(full(kernel*prior,width*scale))[...,:width//2+1])/(alias(full(power,width*scale))[...,:width//2+1]+regularizer)
        q=full(q,width).repeat(1,1,scale,scale)[...,:width*scale//2+1]
    return prior+kernel.conj()*q


def run_checks(device,report,save,verbose_build=False):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from models import converse_training as helpers
    from probe_nearest_training import prepare_kernel as frozen_prepare
    torch.set_num_threads(1)
    torch.manual_seed(21874)
    rng=torch.Generator(device="cpu").manual_seed(21874)
    counts={"shared":0,"spectral":0,"same_s1_fft":0}
    if device=="cuda":
        if not torch.cuda.is_available() or os.environ.get("CONVERSE2D_CPU_ONLY")=="1":
            raise RuntimeError("Actual CUDA extensions are required; no CPU fallback")
        from extension_loader import load_extension
        from experiments.training_shared_s1.loader import load
        load_extension(verbose=verbose_build)
        isolated,report["shared_build"] = load(verbose=verbose_build)
        report["production_build"] = json.loads((ROOT/".build/cuda/source_manifest.json").read_text())
        raw_shared,raw_spectral=isolated.shared_s1_transfer,torch.ops.converse2d._training_spectral
        report["environment"]["gpu"]=torch.cuda.get_device_name()
    else:
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU extraction checks must not initialize CUDA")
        raw_shared=lambda y,k,l:((k.conj()+l)/(k.real.square()+k.imag.square()+l))*y
        raw_spectral=aten_spectral

    def shared(y,k,l):
        result=raw_shared(y,k,l)
        counts["shared"]+=1
        if device=="cuda" and "SharedTransfer" not in result.grad_fn.name():
            raise AssertionError("SharedTransfer CUDA autograd route was not used")
        return result

    def spectral(y,p,k,l,h,w,s):
        assert y.dtype==p.dtype==k.dtype==torch.complex64 and l.dtype==torch.float32
        assert k.is_contiguous()
        if s==1:
            assert p is y
            counts["same_s1_fft"]+=1
        result=raw_spectral(y,p,k,l,h,w,s)
        counts["spectral"]+=1
        if device=="cuda" and "SpectralSolve" not in result.grad_fn.name():
            raise AssertionError("Production SpectralSolve CUDA autograd route was not used")
        return result

    def random(shape,dtype=torch.float32):
        return torch.randn(shape,generator=rng,dtype=dtype).to(device)

    def same(actual,expected):
        assert (actual is None)==(expected is None)
        if actual is not None:
            torch.testing.assert_close(actual,expected,atol=0,rtol=0,check_dtype=True)

    def check(name,actual,reference,inputs):
        row=dict(name=name,passed=False,input_dtypes=[str(value.dtype) for value in inputs])
        report["checks"].append(row)
        save()
        versions=[value._version for value in inputs]
        a,e=actual(),reference()
        same(a,e)
        upstream=random(a.shape,a.dtype)
        # This comparison exercises ordinary handwritten spectral CUDA VJPs.
        ga=torch.autograd.grad(a,inputs,upstream,create_graph=False,allow_unused=True)
        ge=torch.autograd.grad(e,inputs,upstream,create_graph=False,allow_unused=True)
        for av,ev in zip(ga,ge):same(av,ev)
        row.update(output_dtype=str(a.dtype),vjp_dtypes=[None if g is None else str(g.dtype) for g in ga])
        # A nonlinear output objective gives a useful nonzero Hessian even
        # for linear FFT/padding extraction, while preserving matched order.
        def objective(value):
            return (value.real.square()+value.imag.square()).sum() if value.is_complex() else value.square().sum()
        ga=torch.autograd.grad(objective(actual()),inputs,create_graph=True,allow_unused=True)
        ge=torch.autograd.grad(objective(reference()),inputs,create_graph=True,allow_unused=True)
        for av,ev in zip(ga,ge):same(av,ev)
        directions=tuple(random(value.shape,value.dtype) for value in inputs)
        ta=[(grad*direction).sum() for grad,direction in zip(ga,directions) if grad is not None and grad.requires_grad]
        te=[(grad*direction).sum() for grad,direction in zip(ge,directions) if grad is not None and grad.requires_grad]
        assert bool(ta)==bool(te)
        ha=torch.autograd.grad(sum(ta),inputs,allow_unused=True) if ta else (None,)*len(inputs)
        he=torch.autograd.grad(sum(te),inputs,allow_unused=True) if te else (None,)*len(inputs)
        for av,ev in zip(ha,he):same(av,ev)
        assert versions==[value._version for value in inputs],"Helper mutated an input/parameter"
        assert all(value.grad is None for value in inputs),"Parity test accumulated leaf gradients"
        row.update(passed=True,output_vjp_hvp_exact=True,hvp_dtypes=[None if g is None else str(g.dtype) for g in ha])
        save()

    for name,shape,hw,dtype in (
        ("prepare_2d",(1,3,3,3),(7,9),torch.float32),
        ("prepare_separable_area",(1,2,3,3),(128,128),torch.float32),
        ("prepare_separable_filter_count",(1,256,3,3),(64,64),torch.float32),
        ("prepare_tall_kernel_fallback",(1,2,40,3),(128,128),torch.float32),
        ("prepare_mixed_double",(2,3,7,7),(9,11),torch.float64)):
        weight=random(shape,dtype).requires_grad_()
        check(name,lambda:helpers.prepare_kernel(weight,*hw),lambda:frozen_prepare(weight,*hw),(weight,))

    broadcasts=((1,3),) if device=="cpu" else ((1,1),(1,3),(2,1),(2,3))
    for kb,kc in broadcasts:
        # Transposition keeps the same non-square LR size while explicitly
        # testing the adapter's activation .contiguous() boundary.
        x=random((2,3,7,5)).transpose(-1,-2).requires_grad_()
        weight=random((kb,kc,3,3)).requires_grad_()
        bias=random((1,3,1,1)).requires_grad_()
        def frozen_shared():
            value=x.contiguous()
            y=torch.fft.rfft2(value)
            kernel=frozen_prepare(weight,*value.shape[-2:])
            regularizer=torch.sigmoid(bias.contiguous()-9.)+1e-3
            return torch.fft.irfft2(shared(y,kernel,regularizer),s=value.shape[-2:])
        check(f"shared_s1_KB{kb}_KC{kc}",lambda:helpers.shared_s1(x,weight,bias,1e-3,shared),
              frozen_shared,(x,weight,bias))

    broadcasts=((2,3),) if device=="cpu" else ((1,1),(1,3),(2,1),(2,3))
    for scale in (1,3):
        for kb,kc in broadcasts:
            x=random((2,3,7,5)).transpose(-1,-2).requires_grad_()
            kernel=random((kb,kc,3,3),torch.float64).transpose(-1,-2).requires_grad_()
            bias=random((1,3,1,1)).requires_grad_()
            def frozen_mixed():
                value=x.contiguous()
                height,width=value.shape[-2:]
                regularizer=torch.sigmoid(bias.contiguous()-9.)+1e-3
                prepared=frozen_prepare(kernel,height*scale,width*scale)
                y=torch.fft.rfft2(value)
                prior=y if scale==1 else torch.fft.rfft2(F.interpolate(value,scale_factor=scale,mode="nearest"))
                corrected=spectral(y,prior,prepared,regularizer,height,width,scale)
                return torch.fft.irfft2(corrected,s=(height*scale,width*scale))
            check(f"mixed_s{scale}_KB{kb}_KC{kc}",lambda:helpers.mixed_data(x,kernel,bias,scale,1e-3,spectral),
                  frozen_mixed,(x,kernel,bias))

    size=3 if device=="cpu" else 7
    for approximate in ("none","tanh"):
        net=SimpleNamespace(kernel_size=size,fc1=nn.Linear(size*size,64).to(device),
            fc2=nn.Linear(64,64,bias=False).to(device),fc3=nn.Linear(64,16*size*size).to(device),
            gelu=nn.GELU(approximate=approximate))
        value=random((2,1,size,size)).transpose(-1,-2).requires_grad_()
        parameters=tuple(p for layer in (net.fc1,net.fc2,net.fc3) for p in layer.parameters())
        def frozen_kernelnet():
            batch=value.shape[0]
            result=value.double().reshape(batch,-1)
            for index,layer in enumerate((net.fc1,net.fc2,net.fc3)):
                result=F.linear(result,layer.weight.double(),None if layer.bias is None else layer.bias.double())
                if index<2:result=F.gelu(result,approximate=net.gelu.approximate)
            return result.view(batch,16,net.kernel_size,net.kernel_size)
        check("kernelnet_"+approximate,lambda:helpers.kernelnet_fp64(value,net),frozen_kernelnet,(value,*parameters))
        assert all(p.dtype==torch.float32 for p in parameters)
    for use_bias in (False,True):
        layer=nn.Conv2d(16,64,1,bias=use_bias).to(device)
        value=random((2,16,size,size),torch.float64).requires_grad_()
        check("projection_bias_"+str(use_bias),lambda:helpers.projection_fp64(value,layer),
            lambda:F.conv2d(value,layer.weight.double(),None if layer.bias is None else layer.bias.double(),
                            layer.stride,layer.padding,layer.dilation,layer.groups),
            (value,*tuple(layer.parameters())))
        assert all(p.dtype==torch.float32 for p in layer.parameters())

    for layout in ("contiguous","transpose","channels_last","offset","stepped","overlap"):
        value=random((2,3,7,9),torch.float64)
        if layout=="transpose":value=value.transpose(-1,-2)
        elif layout=="channels_last":value=value.contiguous(memory_format=torch.channels_last)
        elif layout=="offset":value=random((2,3,9,11),torch.float64)[...,1:-1,1:-1]
        elif layout=="stepped":value=random((2,3,7,18),torch.float64)[...,::2]
        elif layout=="overlap":value=value[:1].expand(2,-1,-1,-1)
        value.requires_grad_()
        check("pad_"+layout,lambda:helpers.circular_pad(value,2).sin(),
              lambda:F.pad(value,(2,)*4,mode="circular").sin(),(value,))
        check("crop_"+layout,lambda:helpers.crop_view(value,2).sin(),lambda:value[...,2:-2,2:-2].sin(),(value,))
        assert helpers.circular_pad(value,0) is value and helpers.crop_view(value,0) is value
    report["zero_padding_identity"]=True
    rejected=[]
    for name,operation in (
        ("oversize_padding",lambda:helpers.circular_pad(torch.ones(1,1,1,1,device=device),2)),
        ("empty_crop",lambda:helpers.crop_view(torch.ones(1,1,4,4,device=device),2)),
        ("mixed_kernel_rounded",lambda:helpers.mixed_data(torch.ones(1,1,3,3,device=device),
            torch.ones(1,1,1,1,device=device),torch.zeros(1,1,1,1,device=device),1,1e-3,spectral)),
        ("wrong_shared_dtype",lambda:helpers.shared_s1(torch.ones(1,1,3,3,device=device,dtype=torch.float64),
            torch.ones(1,1,1,1,device=device),torch.zeros(1,1,1,1,device=device),1e-3,shared))):
        try:operation()
        except ValueError:rejected.append(name)
        else:raise AssertionError(f"Guard did not reject {name}")
    with torch.autocast(device,dtype=torch.bfloat16):
        try:helpers.prepare_kernel(torch.ones(1,1,1,1,device=device),3,3)
        except ValueError:rejected.append("autocast")
        else:raise AssertionError("Autocast was not rejected")
    report["guard_rejections"]=rejected
    report["spectral_call_counts"]=counts
    report["exact_check_count"]=len(report["checks"])
    if device=="cpu" and torch.cuda.is_initialized():
        raise AssertionError("CPU extraction checks initialized CUDA")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuda",action="store_true")
    parser.add_argument("--verbose-build",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    if args.cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="1":
            parser.error("TF32 override conflicts with this parity protocol")
    before=source_hashes()
    for name,expected in FROZEN.items():
        if before[name]!=expected:raise RuntimeError(f"Frozen reference changed: {name}")
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    report=dict(status="running",device="cuda" if args.cuda else "cpu",source_before=before,
        frozen_reference_sha256=FROZEN,checks=[],atol=0.,rtol=0.,
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,tf32=False,
                         deterministic_algorithms=True,cudnn_deterministic=True),
        scope="Extraction parity only; existing main-model calls remain unchanged")
    def save():
        if args.output is not None:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    try:
        run_checks(report["device"],report,save,args.verbose_build)
        after=source_hashes()
        report["source_after"]=after
        report["source_unchanged"]=after==before
        if after!=before:raise RuntimeError("Sources changed during helper parity checks")
        report["status"]="passed_extraction_parity"
    except Exception as error:
        report["status"]="failed_extraction_parity"
        report["source_after"]=source_hashes()
        report["source_unchanged"]=report["source_after"]==before
        report["error"]=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        save()
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
