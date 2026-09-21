"""Differentiable ATen nearest-spectrum training prototype; no production changes.

    python test/probe_nearest_training.py --cpu-prior-check
    python test/probe_nearest_training.py

Compare production spatial nearest+forward, an equivalent Python/ATen explicit
nearest control, and an ATen spectral-prior candidate using the SAME current
_training_spectral solver. Kernel preparation mirrors current differentiable
FP64 FFT -> contiguous complex64; lambda and all FFTs keep autograd. No new C++
entry, custom FFT adjoint, detached trainable tensor or cross-step cache is used.

P[k,l]=Y[k%H,l%W]*Phi_(sH)(k)*Phi_(sW)(l), with Phi_N(k)=sum_a exp(-2*pi*i*k*a/N).
There is no extra s^2 division. Reading the missing LR half spectrum reflects
BOTH frequency coordinates and conjugates. Phase generation uses FP64, exact
DC/alias-zero/Nyquist cases, then complex64 weights for FP32 execution.

This first prototype materializes the HR HALF spectrum through index_select,
where and multiplication. It removes spatial interpolation and its HR RFFT but
may replace their backward cost with gather/scatter reductions; speed and CUDA
determinism are unverified. Phase/index construction is included every call.
The old inference experiment supplies the formula, not training evidence.

All normal/weak output and x/kernel/bias VJP gates run BEFORE any timing. Normal
output atol=rtol=3e-5; gradients atol=rtol=5e-5. Weak output 1e-6/1e-5, same
gradient budgets. Reference is independent full-FFT FP64 with spatial nearest.
Profiler checks are separate from warm5, alternating round4/iters20 timing.
Timed scope is complete spatial operator FWD + all three VJPs, including kernel
preparation and phase construction, without loss/optimizer or model claims.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

from probe_pointwise_training import capture, clear_cuda, fixture, tensor_hash, timed_fixture
from probe_training_s1_shapes import metrics

ROOT = Path(__file__).resolve().parents[1]


def phase(length, scale, like):
    import torch
    index = torch.arange(length, device=like.device, dtype=torch.int64)
    reflected = index > length//2
    frequency = torch.where(reflected, length-index, index)
    angle = (-2*math.pi/length)*frequency.double()
    taps = torch.arange(scale, device=like.device, dtype=torch.float64)
    angles = angle[:, None]*taps[None, :]
    real, imag = angles.cos().sum(-1), angles.sin().sum(-1)
    imag = torch.where(reflected, -imag, imag)
    zero = (frequency != 0) & (frequency.remainder(length//scale) == 0)
    nyquist = frequency*2 == length
    real = torch.where(zero, torch.zeros_like(real), real)
    imag = torch.where(zero | nyquist | (frequency == 0), torch.zeros_like(imag), imag)
    real = torch.where(nyquist, torch.full_like(real, scale % 2), real)
    real = torch.where(frequency == 0, torch.full_like(real, scale), real)
    return torch.complex(real, imag).to(like.dtype)


def nearest_half_spectrum(y, height, width, scale):
    import torch
    if scale == 1:
        return y
    hs, ws = height*scale, width*scale
    rows = torch.arange(hs, device=y.device).remainder(height)
    columns = torch.arange(ws//2+1, device=y.device).remainder(width)
    mirror = columns > width//2
    stored_columns = torch.where(mirror, width-columns, columns)
    stored_rows = torch.where(mirror[None, :], (-rows[:, None]).remainder(height), rows[:, None])
    indices = (stored_rows*(width//2+1)+stored_columns[None, :]).reshape(-1)
    lifted = y.flatten(-2).index_select(-1, indices).reshape(*y.shape[:2], hs, ws//2+1)
    lifted = torch.where(mirror[None, None, None, :], lifted.conj(), lifted)
    weights = phase(hs, scale, y)[:, None]*phase(ws, scale, y)[None, :ws//2+1]
    return lifted*weights


def prepare_kernel(weight, height, width):
    import torch
    import torch.nn.functional as F
    kh, kw = weight.shape[-2:]
    filters, area = weight.shape[0]*weight.shape[1], height*width
    work = weight.double()
    if (area >= 16384 or filters*area >= 1048576) and kh <= height//4:
        rows = torch.roll(F.pad(work, (0, width-kw)), -(kw//2), -1)
        horizontal = torch.fft.rfft(rows, dim=-1)
        columns = torch.roll(F.pad(horizontal, (0, 0, 0, height-kh)), -(kh//2), -2)
        spectrum = torch.fft.fft(columns, dim=-2)
    else:
        psf = F.pad(work, (0, width-kw, 0, height-kh))
        spectrum = torch.fft.rfft2(torch.roll(psf, (-(kh//2), -(kw//2)), (-2, -1)))
    return spectrum.to(dtype=torch.complex64, memory_format=torch.contiguous_format)


def spatial(x, weight, bias, scale, eps, mode):
    import torch
    import torch.nn.functional as F
    height, width = x.shape[-2:]
    if mode == "production":
        prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
        return torch.ops.converse2d.forward(x, prior, weight, bias, scale, eps, "v7")
    x = x.contiguous()
    regularizer = torch.sigmoid(bias.contiguous()-9.)+eps
    kernel = prepare_kernel(weight, height*scale, width*scale)
    y = torch.fft.rfft2(x)
    if mode == "aten_spectral":
        prior = nearest_half_spectrum(y, height, width, scale)
    else:
        prior = y if scale == 1 else torch.fft.rfft2(F.interpolate(x, scale_factor=scale, mode="nearest"))
    corrected = torch.ops.converse2d._training_spectral(y, prior, kernel, regularizer, height, width, scale)
    return torch.fft.irfft2(corrected, s=(height*scale, width*scale))


def prior_cpu_check():
    import torch
    import torch.nn.functional as F
    rng = torch.Generator(device="cpu").manual_seed(9214)
    rows = []
    for height, width, scale in ((1,1,3), (1,4,3), (4,1,3), (3,5,3), (4,6,2), (5,4,4), (3,4,1)):
        x = torch.randn(1, 2, height, width, generator=rng, dtype=torch.float64).requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        actual = nearest_half_spectrum(torch.fft.rfft2(x), height, width, scale)
        expected = torch.fft.rfft2(F.interpolate(reference_x, scale_factor=scale, mode="nearest"))
        upstream = torch.randn(expected.shape, generator=rng, dtype=torch.complex128)
        dx, = torch.autograd.grad(actual, (x,), upstream)
        reference_dx, = torch.autograd.grad(expected, (reference_x,), upstream)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(dx, reference_dx, atol=1e-10, rtol=1e-10)
        rows.append(dict(height=height, width=width, scale=scale,
                         output_max_abs=(actual-expected).abs().max().item(),
                         gradient_max_abs=(dx-reference_dx).abs().max().item()))
    x = torch.randn(1, 1, 2, 3, generator=rng, dtype=torch.float64).requires_grad_()
    function = lambda value: nearest_half_spectrum(torch.fft.rfft2(value), 2, 3, 3)
    torch.autograd.gradcheck(function, (x,), fast_mode=True)
    torch.autograd.gradgradcheck(function, (x,), fast_mode=True)
    return dict(passed=True, device="CPU", cases=rows, gradcheck=True, gradgradcheck=True,
                limitation="Checks the differentiable prior formula only; not CUDA training or performance")


def make_cases():
    import torch
    rng = torch.Generator(device="cpu").manual_seed(9214)
    configs = [
        ("small_odd_s3",2,3,5,7,3,1,3,3,False),
        ("single_row_s3",2,3,1,5,3,1,1,1,False),
        ("shared_channels_s3",2,3,6,8,3,2,1,3,False),
        ("batched_full_s3",2,3,5,6,3,2,3,3,False),
        ("b1_c32_64x80_s3",1,32,64,80,3,1,32,3,True),
        ("b8_c32_64x80_s3",8,32,64,80,3,1,32,3,True),
        ("b32_c32_64x80_s3",32,32,64,80,3,1,32,3,True),
        ("datanet_b4_c64_32_s3",4,64,32,32,3,4,64,7,True),
        ("s1_control",1,32,64,80,1,1,32,3,True),
    ]
    cases = []
    for name,b,c,h,w,s,kb,kc,k,timed in configs:
        x = torch.randn(b,c,h,w,generator=rng)
        kernel = torch.rand(kb,kc,k,k,generator=rng)/(k*k)
        bias = torch.randn(1,c,1,1,generator=rng)
        upstream = torch.randn(b,c,h*s,w*s,generator=rng)/(b*c*h*w*s*s)**.5
        cases.append(dict(name=name, scale=s, eps=1e-3, weak=False, timed=timed,
                          tensors=(x,kernel,bias,upstream)))
    # Existing weak-regularizer parameters, now with the model's nearest prior.
    from test_fp32_training import FP32Training
    torch.manual_seed(9214)
    for amplitude in (0., 1e-6, 1e-3):
        x, independent_prior, kernel, bias = FP32Training.data(None,3,4,3,kb=1,kc=1)
        with torch.no_grad():
            x.mul_(1e-5)
            independent_prior.mul_(1e-5)
            kernel.mul_(amplitude)
            bias.fill_(-40.)
        upstream = torch.randn(2,3,9,12,device="cuda")*1e-5
        tensors = tuple(t.detach().cpu().clone() for t in (x,kernel,bias,upstream))
        cases.append(dict(name=f"weak_s3_amplitude_{amplitude:g}", scale=3, eps=1e-8,
                          weak=True, timed=False, tensors=tensors))
        del x, independent_prior, kernel, bias, upstream
    clear_cuda()
    return cases


def validate(case, modes):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    s, eps = case["scale"], case["eps"]
    reference = capture(case["tensors"], torch.float64,
        lambda x,k,b: converse2d_reference(x, x if s==1 else F.interpolate(x,scale_factor=s,mode="nearest"),k,b,s,eps))
    result = {}
    for mode in modes:
        actual = capture(case["tensors"], torch.float32,
                         lambda x,k,b: spatial(x,k,b,s,eps,mode))
        if any(value.dtype != torch.float32 for value in actual.values()):
            raise RuntimeError(f"{mode}: non-FP32 output/gradient")
        errors = {name: metrics(actual[name],reference[name],output=name=="output",weak=case["weak"])
                  for name in reference}
        result[mode] = dict(passed=all(row["passed"] for row in errors.values()), tensors=errors)
    return result


def operation_check(case, mode):
    import torch
    clear_cuda()
    inputs, run = fixture(case["tensors"], torch.float32,
                         lambda x,k,b: spatial(x,k,b,case["scale"],case["eps"],mode))
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    counts = {event.key:event.count for event in profiler.key_averages()}
    names = [event.name for event in profiler.events() if str(event.device_type).endswith("CUDA")]
    spectral_hits = {token:sum(token in name for name in names) for token in
                     ("solve_alias", "solve_output", "adjoint_q", "adjoint_inputs", "adjoint_filter")}
    interpolation = counts.get("aten::upsample_nearest2d", 0)
    # The representative case uses separable kernel preparation, so rfft2
    # counts here are activation transforms, not kernel FFT preparation.
    rfft2 = counts.get("aten::fft_rfft2", 0)
    expected = (0,1) if mode == "aten_spectral" else (1,2)
    passed = (interpolation,rfft2) == expected and all(value == 1 for value in spectral_hits.values())
    del profiler,run,inputs
    clear_cuda()
    return dict(passed=passed, expected_interpolation_rfft2=list(expected),
                interpolation=interpolation, rfft2=rfft2, spectral_kernel_hits=spectral_hits,
                selected_cpu_operations={name:count for name,count in counts.items() if any(
                    token in name for token in ("fft", "index_select", "index_add", "scatter", "upsample_nearest"))},
                scope="Untimed operation/dispatch evidence, separate from all timing")


def repeatability_check(case):
    import torch
    clear_cuda()
    inputs, run = fixture(case["tensors"], torch.float32,
                         lambda x,k,b: spatial(x,k,b,case["scale"],case["eps"],"aten_spectral"))
    first, first_hashes, records = None, None, []
    for index in range(3):
        values = run()
        saved = {name:value.detach().cpu().clone() for name,value in values.items()}
        del values
        hashes = {name:tensor_hash((value,)) for name,value in saved.items()}
        if first is None:
            first, first_hashes = saved, hashes
        records.append(dict(repetition=index+1, tensors={name:dict(
            sha256=hashes[name], bitwise_equal_to_first=hashes[name] == first_hashes[name],
            max_abs_to_first=(value.double()-first[name].double()).abs().max().item(),
            finite=bool(torch.isfinite(value).all())) for name,value in saved.items()}))
    if any(value.grad is not None for value in inputs):
        raise RuntimeError("Repeatability check accumulated leaf gradients")
    del run,inputs,saved,first
    clear_cuda()
    return dict(case=case["name"], repetitions=records,
                all_bitwise_equal=all(value["bitwise_equal_to_first"] for row in records for value in row["tensors"].values()),
                scope="Three untimed calls on the same unchanged CUDA fixture; observation, not a relaxed numerical gate",
                global_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic)


def time_case(case, modes, args):
    rounds = []
    for index in range(args.rounds):
        # Rotate and reverse so the candidate is not always measured last.
        rotated = modes[index % len(modes):]+modes[:index % len(modes)]
        order = rotated if index % 2 == 0 else list(reversed(rotated))
        values = {mode:timed_fixture(case["tensors"],
                    lambda x,k,b,mode=mode:spatial(x,k,b,case["scale"],case["eps"],mode),args) for mode in order}
        rounds.append(dict(round=index+1, order=order, variants=values))
    medians = {mode:{key:statistics.median(row["variants"][mode][key] for row in rounds)
                     for key in ("wall_ms","cuda_event_ms","peak_allocated_bytes","peak_reserved_bytes")}
               for mode in modes}
    ratios = {base:{key:[row["variants"][base][key]/row["variants"]["aten_spectral"][key] for row in rounds]
                    for key in ("wall_ms","cuda_event_ms")} for base in ("production","aten_explicit")}
    return dict(rounds=rounds, medians=medians, paired_baseline_over_candidate=ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-prior-check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/training_research/nearest_training.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if min(args.warmup,args.rounds,args.iters) < 1:
        parser.error("warmup, rounds and iters must be positive")
    if args.cpu_prior_check:
        print(json.dumps(prior_cpu_check(),indent=2))
        return
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    import torch
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    if not torch.cuda.is_available():
        parser.error("CUDA required for spatial training comparisons")
    sys.path.insert(0,str(ROOT))
    load_extension()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    modes = ["production","aten_explicit","aten_spectral"]
    cases = make_cases()
    report = dict(status="validating", scope=__doc__, cpu_prior_check=prior_cpu_check(),
                  source_sha256=current_manifest(),
                  script_sha256={name:hashlib.sha256((ROOT/"test"/name).read_bytes()).hexdigest() for name in
                                 ("probe_nearest_training.py","probe_training_s1_shapes.py","probe_pointwise_training.py","test_fp32_training.py")},
                  environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                                   tf32=False,cudnn_deterministic=True,
                                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                   determinism_note="Global deterministic algorithms are not enabled here (default false); cuDNN deterministic does not guarantee the ATen gather/scatter backward is deterministic"),
                  settings=dict(warmup=args.warmup,rounds=args.rounds,iters=args.iters),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")

    try:
        for case in cases:
            x,k,b,_ = case["tensors"]
            row = dict(name=case["name"],shape=list(x.shape),kernel_shape=list(k.shape),bias_shape=list(b.shape),
                       scale=case["scale"],eps=case["eps"],weak=case["weak"],tensors_sha256=tensor_hash(case["tensors"]),
                       validation=validate(case,modes),timing=None)
            report["cases"].append(row)
            save()
            print(json.dumps(dict(case=case["name"],validation=row["validation"])),flush=True)
        if not all(check["passed"] for row in report["cases"] for check in row["validation"].values()):
            report["status"] = "numerical_gate_failed_no_timing"
            save()
            raise SystemExit(1)
        representative = next(case for case in cases if case["name"] == "b32_c32_64x80_s3")
        report["repeatability"] = repeatability_check(representative)
        save()
        report["operation_check"] = {mode:operation_check(representative,mode) for mode in modes}
        save()
        if not all(value["passed"] for value in report["operation_check"].values()):
            report["status"] = "operation_check_failed_no_timing"
            save()
            raise SystemExit(1)
        report["status"] = "timing"
        for case,row in zip(cases,report["cases"]):
            if case["timed"]:
                row["timing"] = time_case(case,modes,args)
                save()
                print(json.dumps(dict(case=case["name"],medians=row["timing"]["medians"])),flush=True)
        report["status"] = "complete_local_training_prototype"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        save()
        raise
    print(f"Saved {args.output}",flush=True)


if __name__ == "__main__":
    main()
