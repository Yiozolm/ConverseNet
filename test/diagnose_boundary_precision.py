"""Fixed-fixture numerical decomposition of the boundary probe's inherited gap.

No production changes or timings. The independent oracle always remains the
same FP64 full-FFT residual solve. Native circular padding and two-slice crop
are used in every path. Mixed paths keep kernel preparation FP64 and expose
the effect of activation FFT and spectral VJP precision separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def fixture():
    import torch
    generator=torch.Generator().manual_seed(20260918)
    state=torch.load(ROOT/'model_zoo/converse_usrnet.pth',map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    x=torch.randn(4,128,96,96,generator=generator)
    upstream=torch.randn(x.shape,generator=generator)/x.numel()**.5
    return (x,state['p.m_body.0.conv1.3.weight'].clone(),
            state['p.m_body.0.conv1.3.bias'].clone(),upstream)


def kernel_fft64(weight,height,width):
    import torch
    from torch.nn import functional as F
    kh,kw=weight.shape[-2:]
    filters=weight.shape[0]*weight.shape[1]
    # Exactly the current differentiable preparation choice, including its
    # row crop before width-transform backward for these 100x100 spectra.
    if (height*width>=16384 or filters*height*width>=1048576) and kh<=height//4:
        rows=F.pad(weight.double(),(0,width-kw)).roll(-(kw//2),-1)
        horizontal=torch.fft.rfft(rows,dim=-1)
        columns=F.pad(horizontal,(0,0,0,height-kh)).roll(-(kh//2),-2)
        return torch.fft.fft(columns,dim=-2)
    psf=F.pad(weight.double(),(0,width-kw,0,height-kh))
    return torch.fft.rfft2(psf.roll((-(kh//2),-(kw//2)),(-2,-1)))


def capture(tensors,name,activation64=False,solve128=False):
    import torch
    from torch.nn import functional as F
    from models.converse_core import converse2d_reference
    dtype=torch.float64 if name=='reference_fp64' else torch.float32
    values=tuple(t.detach().to(device='cuda',dtype=dtype).requires_grad_() for t in tensors[:3])
    x,weight,bias=values
    upstream=tensors[3].to(device='cuda',dtype=dtype)
    padded=F.pad(x,(2,2,2,2),mode='circular')
    fk64=None
    if name=='current_cuda':
        output=torch.ops.converse2d.forward(padded,padded,weight,bias,1,1e-5,'v7')
    elif name in ('reference_fp64','python_fp32'):
        output=converse2d_reference(padded,padded,weight,bias,1,1e-5)
    else:
        height,width=padded.shape[-2:]
        fk64=kernel_fft64(weight,height,width)
        # Preserve the production complex64 boundary, even when the following
        # solve is lifted to complex128. This isolates solve arithmetic from
        # silently improving the kernel-spectrum values at the same time.
        k=fk64.to(dtype=torch.complex64,memory_format=torch.contiguous_format)
        y=torch.fft.rfft2(padded.double() if activation64 else padded).to(torch.complex64)
        p=y  # Shared identity is intentional; do not create a second FFT.
        lam=torch.sigmoid(bias-9.)+1e-5
        if solve128:
            y=y.cdouble()
            p=y
            k,lam=k.cdouble(),lam.double()
        power=k.real.square()+k.imag.square()
        q=(y-k*p)/(power+lam)
        spectrum=(p+k.conj()*q).cfloat()
        # Fixed FP32 output-IFFT boundary in every mixed path. The activation64
        # variant changes the shared input FFT and its autograd adjoint only.
        output=torch.fft.irfft2(spectrum,s=(height,width))
    output=output[...,2:-2,2:-2]
    requested=(*values,fk64) if fk64 is not None else values
    grads=torch.autograd.grad(output,requested,upstream)
    adjoint=None
    if fk64 is not None:
        # Replay ONLY the linear kernel-preparation adjoint with the same
        # incoming c128 VJP. No analytical/custom FFT backward is introduced.
        replay_weight=weight.detach().double().requires_grad_()
        replay_spectrum=kernel_fft64(replay_weight,*padded.shape[-2:])
        replay_grad=torch.autograd.grad(replay_spectrum,replay_weight,grads[3])[0]
        diff=grads[1].double()-replay_grad
        adjoint=dict(matches_replayed_fp64_adjoint_after_fp32_cast=torch.equal(grads[1],replay_grad.float()),
                     replay_cast_max_abs=diff.abs().max().item(),
                     replay_cast_relative_l2=(diff.norm()/replay_grad.norm().clamp_min(1e-30)).item(),
                     kernel_spectrum_vjp_dtype=str(grads[3].dtype),
                     statement='Exact replay equality checks the preparation adjoint for this incoming VJP; it does not imply the incoming VJP agrees with the independent full-FP64 reference.')
    result={key:value.detach().cpu() for key,value in
            zip(('output','dx','dw','db'),(output,*grads[:3]))}
    del values,grads,output,padded
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result,adjoint


def comparison(actual,expected,atol,rtol,max_failures=12):
    import torch
    a,e=actual.double(),expected.double()
    error=(a-e).abs();budget=atol+rtol*e.abs();ratios=error/budget
    finite=bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    failed=error>budget
    indices=failed.nonzero().tolist()
    indices.sort(key=lambda index:float(ratios[tuple(index)]),reverse=True)
    safe=lambda value:value if math.isfinite(value) else str(value)
    return dict(passed=finite and not indices,finite=finite,atol=atol,rtol=rtol,
                max_abs=safe(error.max().item()),relative_l2=safe((error.norm()/e.norm().clamp_min(1e-30)).item()),
                max_pointwise_budget_ratio=safe(ratios.max().item()),
                failed_elements=len(indices),failures=[dict(index=index,actual=a[tuple(index)].item(),
                    reference=e[tuple(index)].item(),absolute_error=error[tuple(index)].item(),
                    budget=budget[tuple(index)].item(),budget_ratio=ratios[tuple(index)].item())
                    for index in indices[:max_failures]])


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-python-fp32',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new diagnostic output; existing reports are immutable')
    import os
    import torch
    import train_usrnet_dataset as worker
    from extension_loader import load_extension
    os.environ['CONVERSE2D_SKIP_BUILD']='1'
    load_extension()  # Source/header/PyTorch/binary verification is mandatory.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    tensors=fixture()
    fixture_hash=worker.tensor_hash(dict(zip(('x','weight','bias','upstream'),tensors)))
    prior_path=ROOT/'artifacts/native_deconv_target/boundary_probe.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8-sig')) if prior_path.exists() else None
    if prior is not None and fixture_hash!=prior['input_sha256']:
        raise RuntimeError('The fixture differs from the original failed boundary probe')
    reference,_=capture(tensors,'reference_fp64')
    configurations=[('current_cuda',False,False),('aten_act_fp32_solve_c64',False,False),
                    ('aten_act_fp64_round_solve_c64',True,False),
                    ('aten_act_fp32_solve_c128',False,True),
                    ('aten_act_fp64_round_solve_c128',True,True)]
    if args.include_python_fp32:configurations.append(('python_fp32',False,False))
    report=dict(status='diagnostic_running',shape=[4,128,96,96],padding=2,scale=1,eps=1e-5,seed=20260918,
        input_sha256=fixture_hash,checkpoint_sha256=worker.file_hash(ROOT/'model_zoo/converse_usrnet.pth'),
        source_sha256=worker.source_hashes(),script_sha256=worker.file_hash(__file__),
        original_boundary_report=str(prior_path) if prior is not None else None,
        oracle='One fixed independent full-FFT FP64 converse2d_reference; FP32 fixture values are promoted without rerounding.',
        mixed_precision_boundaries=dict(kernel_fft='FP64 differentiable; rounded once to contiguous complex64 in ALL mixed candidates',
            activation_fft='Shared x FFT in FP32 or FP64, then always rounded to complex64 before solve',
            spectral_solve='complex64 or lifted complex128 on identical complex64 boundary values',
            regularizer='sigmoid(bias-9)+1e-5 formed in FP32, then lifted for c128 solve',
            output_ifft='Spectrum cast to complex64; FP32 irfft2 and native two-slice crop for ALL mixed candidates',
            caveat='Activation-FFT precision changes both that FFT and its automatic adjoint. Remaining output-IFFT, regularizer and spectrum casts remain FP32; this grid is not an all-FP64 replacement.'),
        interpretations=['Current vs matching ATen separates fused forward/VJP arithmetic from preparation.',
                         'Mixed FP64 input FFT isolates FFT preparation plus its automatic adjoint, preserving c64 solve boundary.',
                         'c128 solve isolates spectral arithmetic and local VJP on the same rounded input spectra.',
                         'Exact kernel-adjoint replay checks the final linear preparation/cast; upstream numerical error can still remain.'],
        cases={},no_timing=True,production_eligible=False)
    baseline=None
    for name,activation64,solve128 in configurations:
        actual,adjoint=capture(tensors,name,activation64,solve128)
        if baseline is None:baseline=actual
        errors={key:comparison(value,reference[key],*( (3e-5,3e-5) if key=='output' else (5e-5,5e-5)))
                for key,value in actual.items()}
        report['cases'][name]=dict(numerical_gate_passed=all(row['passed'] for row in errors.values()),
            against_fp64=errors,bitwise_equal_to_current={key:torch.equal(value,baseline[key]) for key,value in actual.items()},
            kernel_adjoint_replay=adjoint)
        print(name,json.dumps(errors['dw'],allow_nan=False),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        worker.write_json(args.output,report)
    report['status']='diagnostic_complete'
    worker.write_json(args.output,report)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
