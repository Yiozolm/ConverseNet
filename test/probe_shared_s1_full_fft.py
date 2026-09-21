"""Full-C2C shared-s1 precision experiment, with fixed fixtures and no timing.

    python test/probe_shared_s1_full_fft.py --cpu-proof
    ./experiments/training_speed/run.ps1 test/probe_shared_s1_full_fft.py --cuda-accuracy --output artifacts/native_deconv_target/shared_s1_full_fft_accuracy.json

All routes use differentiable full FP64 kernel FFT preparation and FP32
activation fft2 / final complex64 ifft2.real. The original four routes compute
lambda at input precision, then high-precision H/C calculations only promote
that value. The separately named fifth route promotes bias before sigmoid.

full_h32: cast K to c64, compute H32, multiply Y32.
full_h64_cast: compute H64 from K128/lambda32.double(), cast H to c64, multiply Y32.
full_h64_product: multiply H128 * Y32.cdouble(), cast product to c64 for IFFT.
full_c64_residual: compute C128=conj(K)*(1-K)/d, cast C to c64, multiply Y32,
                  then add spatial x to the real IFFT result.
full_h64_lambda64: explicitly compute sigmoid(bias.double()-9)+eps in FP64,
                  compute H128, cast H to c64, multiply Y32.

CPU mathematical proofs use FP64 inputs and preserve complex128 throughout.
GPU accuracy reuses the twelve original s1/weak/pretrained/native fixtures and
strict compare_accuracy: per tensor max_abs AND relative L2 to the same FP64
oracle must be <= original Python FP32's error, without margin or changed
eps/seed. This isolates full/half FFT rounding and shared spectral algebra;
it does not cherry-pick input values or establish a production dispatch.
Only s=1 with prior IS x is supported; no parameter cache or custom backward.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
VARIANTS = ("full_h32","full_h64_cast","full_h64_product","full_c64_residual","full_h64_lambda64")


def separable_kernel_fft64(weight,height,width):
    import torch
    import torch.nn.functional as F
    kh,kw = weight.shape[-2:]
    rows = F.pad(weight.double(),(0,width-kw)).roll(-(kw//2),-1)
    horizontal = torch.fft.fft(rows,dim=-1)  # Full C2C, never rfft here.
    columns = F.pad(horizontal,(0,0,0,height-kh)).roll(-(kh//2),-2)
    return torch.fft.fft(columns,dim=-2)


def kernel_fft64(weight,height,width):
    import torch
    import torch.nn.functional as F
    kh,kw = weight.shape[-2:]
    filters,area = weight.shape[0]*weight.shape[1],height*width
    if (area>=16384 or filters*area>=1048576) and kh<=height//4:
        return separable_kernel_fft64(weight,height,width)
    psf = F.pad(weight.double(),(0,width-kw,0,height-kh))
    return torch.fft.fft2(psf.roll((-(kh//2),-(kw//2)),(-2,-1)))


def full_core(x,weight,bias,eps,variant):
    import torch
    if variant not in VARIANTS:
        raise ValueError(variant)
    dtype = torch.complex128 if x.dtype==torch.float64 else torch.complex64
    k128 = kernel_fft64(weight,*x.shape[-2:])
    regularizer = torch.sigmoid((bias.double() if variant=="full_h64_lambda64" else bias)-9.)+eps
    y = torch.fft.fft2(x)
    if variant=="full_h32":
        k = k128.to(dtype=dtype,memory_format=torch.contiguous_format)
        h = (k.conj()+regularizer)/(k.real.square()+k.imag.square()+regularizer)
        spectrum = h*y
    else:
        lam64 = regularizer.double()
        d = k128.real.square()+k128.imag.square()+lam64
        if variant=="full_c64_residual":
            c = (k128.conj()*(1-k128)/d).to(dtype=dtype,memory_format=torch.contiguous_format)
            spectrum = c*y
        else:
            h = (k128.conj()+lam64)/d
            if variant=="full_h64_product":
                spectrum = (h*y.cdouble()).to(dtype=dtype,memory_format=torch.contiguous_format)
            else:
                spectrum = h.to(dtype=dtype,memory_format=torch.contiguous_format)*y
    output = torch.fft.ifft2(spectrum).real
    return x+output if variant=="full_c64_residual" else output


def spatial(x,prior,weight,bias,scale=1,eps=1e-5,variant="full_h32"):
    import torch
    from models.converse_core import validate_inputs
    if scale!=1 or prior is not x:
        raise ValueError("Full-FFT candidate requires scale=1 and prior IS x")
    validate_inputs(x,prior,weight,bias,scale,eps)
    if x.dtype not in (torch.float32,torch.float64) or torch.is_autocast_enabled(x.device.type):
        raise ValueError("Only FP32/FP64 without autocast is supported")
    return full_core(x,weight,bias,eps,variant)


def method(variant,padding=0,eps=1e-5):
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    if variant not in (*VARIANTS,"reference"):
        raise ValueError(variant)
    def forward(x,weight,bias):
        value = F.pad(x,(padding,)*4,mode="circular") if padding else x
        output = (converse2d_reference(value,value,weight,bias,1,eps) if variant=="reference" else
                  spatial(value,value,weight,bias,1,eps,variant))
        return output[...,padding:-padding,padding:-padding] if padding else output
    return forward


def cpu_proof():
    import torch
    import torch.nn.functional as F
    from probe_spatial_nonoverlap import matrix_solve
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU proof must run before CUDA initialization")
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(67218)
    preparation = []
    for h,w,kh,kw in ((5,7,3,3),(12,13,3,2),(128,128,3,3)):
        k = torch.randn(1,2,kh,kw,generator=rng,dtype=torch.float64).requires_grad_()
        direct = torch.fft.fft2(F.pad(k,(0,w-kw,0,h-kh)).roll((-(kh//2),-(kw//2)),(-2,-1)))
        separated = separable_kernel_fft64(k,h,w)
        selected = kernel_fft64(k,h,w)
        g = torch.randn(direct.shape,generator=rng,dtype=torch.complex128)
        expected_grad, = torch.autograd.grad(direct,k,g,retain_graph=True)
        grad, = torch.autograd.grad(separated,k,g)
        torch.testing.assert_close(selected,direct,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(separated,direct,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(grad,expected_grad,atol=1e-10,rtol=1e-10)
        preparation.append(dict(shape=[h,w],kernel=[kh,kw],
            forward_max_abs=(separated-direct).abs().max().item(),vjp_max_abs=(grad-expected_grad).abs().max().item()))
    configs = [("normal",kb,kc,1.,1e-3) for kb,kc in ((1,1),(1,2),(2,1),(2,2))]
    configs += [(f"weak_{amplitude:g}",1,1,amplitude,1e-8) for amplitude in (0.,1e-6,1e-3)]
    rows = []
    for name,kb,kc,amplitude,eps in configs:
        x = torch.randn(2,2,3,4,generator=rng,dtype=torch.float64)
        k = torch.rand(kb,kc,3,3,generator=rng,dtype=torch.float64)/9
        b = torch.randn(1,2,1,1,generator=rng,dtype=torch.float64)
        upstream = torch.randn(x.shape,generator=rng,dtype=torch.float64)/x.numel()**.5
        if name.startswith("weak"):
            x.mul_(1e-5);k.mul_(amplitude);b.fill_(-40.);upstream.mul_(1e-5)
        refs = tuple(value.clone().requires_grad_() for value in (x,k,b))
        reference = matrix_solve(refs[0],refs[0],refs[1],refs[2],1,eps)
        reference_gradients = torch.autograd.grad(reference,refs,upstream)
        for variant in VARIANTS:
            values = tuple(value.clone().requires_grad_() for value in (x,k,b))
            actual = spatial(values[0],values[0],values[1],values[2],1,eps,variant)
            gradients = torch.autograd.grad(actual,values,upstream)
            torch.testing.assert_close(actual,reference,atol=1e-9,rtol=1e-9)
            for av,ev in zip(gradients,reference_gradients):
                torch.testing.assert_close(av,ev,atol=1e-8,rtol=1e-8)
            rows.append(dict(case=name,variant=variant,KB=kb,KC=kc,
                output_max_abs=(actual-reference).abs().max().item(),
                gradient_max_abs=[(av-ev).abs().max().item() for av,ev in zip(gradients,reference_gradients)]))
    small = (torch.randn(1,1,3,4,generator=rng,dtype=torch.float64).requires_grad_(),
             (torch.rand(1,1,3,3,generator=rng,dtype=torch.float64)/9).requires_grad_(),
             torch.zeros(1,1,1,1,dtype=torch.float64,requires_grad=True))
    higher = []
    for variant in VARIANTS:
        function = lambda x,k,b:spatial(x,x,k,b,1,1e-3,variant)
        torch.autograd.gradcheck(function,small,fast_mode=True,atol=1e-5,rtol=1e-4,nondet_tol=0.)
        torch.autograd.gradgradcheck(function,small,fast_mode=True,atol=1e-5,rtol=1e-4,nondet_tol=0.)
        higher.append(dict(variant=variant,gradcheck=True,gradgradcheck=True))
    try:
        spatial(small[0],small[0].clone(),small[1],small[2])
    except ValueError as error:
        if "prior IS x" not in str(error):
            raise
    else:
        raise AssertionError("Independent prior accepted")
    return dict(passed=True,device="CPU",seed=67218,full_kernel_preparation=preparation,
        independent_pixel_matrix=rows,higher_order=higher,independent_prior_rejected=True,
        limitation="CPU FP64 proof of mathematical equivalence, not GPU FP32 noninferiority")


def cuda_accuracy(report,save):
    import torch
    from probe_training_s1_shapes import cases
    from diagnose_boundary_precision import fixture
    from probe_pointwise_training import capture,tensor_hash,clear_cuda
    from audit_shared_s1_accuracy import compare_accuracy
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    frozen = json.loads((ROOT/"artifacts/native_deconv_target/source_before/manifest.json").read_text())
    if worker.file_hash(ROOT/"models/converse_core.py")!=frozen["models/converse_core.py"]:
        raise RuntimeError("Original Python reference changed")
    work = list(cases())
    tensors = fixture()
    old = json.loads((ROOT/"artifacts/native_deconv_target/boundary_probe.json").read_text())
    if worker.tensor_hash(dict(zip(("x","weight","bias","upstream"),tensors)))!=old["input_sha256"]:
        raise RuntimeError("Pretrained fixture changed")
    work.append(dict(name="pretrained_module_pad2_crop2",tensors=tensors,eps=1e-5,weak=False,padding=2))
    config = native.CASES["s1_module"]
    cpu = native.cpu_data(config,seed=17)
    old = next(row for row in json.loads((ROOT/"artifacts/native_deconv_target/native_baseline.json").read_text())["cases"]
               if row["name"]=="s1_module")
    if worker.tensor_hash(cpu)!=old["complete_fixture_sha256"]:
        raise RuntimeError("Native fixture changed")
    work.append(dict(name="native_s1_module_seed17",tensors=(cpu["x"],cpu["weight"],
        cpu["bias"].reshape(1,config["shape"][1],1,1),cpu["upstream"]),eps=config["eps"],weak=False,padding=2))
    result = dict(cases=[],candidate_passed={variant:True for variant in VARIANTS},timing=False,
        lambda_policy=dict(original_four="Original FP32 sigmoid(bias-9)+eps, promoted only after parameterization for H64/C64",
                           full_h64_lambda64="Explicit FP64 sigmoid(bias.double()-9)+eps, with unchanged bias/eps inputs"),
        acceptance="Per tensor max_abs AND relative_l2 to same FP64 oracle <= original Python FP32 error; no margin",
        environment=dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),tf32=False,deterministic_algorithms=True))
    report["cuda_accuracy"] = result
    for case in work:
        settings = dict(padding=case.get("padding",0),eps=case["eps"])
        oracle = capture(case["tensors"],torch.float64,method("reference",**settings))
        baseline = capture(case["tensors"],torch.float32,method("reference",**settings))
        row = dict(name=case["name"],fixture_sha256=tensor_hash(case["tensors"]),eps=case["eps"],weak=case["weak"],
                   padding=settings["padding"],candidates={})
        result["cases"].append(row)
        for variant in VARIANTS:
            actual = capture(case["tensors"],torch.float32,method(variant,**settings))
            if any(value.dtype!=torch.float32 for value in actual.values()):
                raise RuntimeError("Candidate changed output/gradient dtype")
            row["candidates"][variant] = compare_accuracy(actual,baseline,oracle)
            result["candidate_passed"][variant] &= row["candidates"][variant]["passed"]
            save()
        print(case["name"],json.dumps({name:value["passed"] for name,value in row["candidates"].items()}),flush=True)
        clear_cuda()
    result["passed"] = all(result["candidate_passed"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cpu-proof",action="store_true")
    modes.add_argument("--cuda-accuracy",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite prior evidence")
    if args.cuda_accuracy and args.output is None:
        parser.error("CUDA accuracy requires a new --output path")
    if args.cuda_accuracy:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")=="1":
            parser.error("TF32 override conflicts with FP32 protocol")
    sources = ("test/probe_shared_s1_full_fft.py","test/probe_spatial_nonoverlap.py",
               "test/diagnose_boundary_precision.py","test/probe_training_s1_shapes.py",
               "test/probe_pointwise_training.py","test/audit_shared_s1_accuracy.py",
               "test/benchmark_native_deconv.py","models/converse_core.py")
    report = dict(status="running",scope=__doc__,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in sources},production_eligible=False,timing=False)
    def save():
        if args.output is not None:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    try:
        result = cuda_accuracy(report,save) if args.cuda_accuracy else cpu_proof()
        report["cuda_accuracy" if args.cuda_accuracy else "cpu_proof"] = result
        report["status"] = "passed" if result["passed"] else "accuracy_noninferiority_failed"
        return 0 if result["passed"] else 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        save()
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    raise SystemExit(main())
