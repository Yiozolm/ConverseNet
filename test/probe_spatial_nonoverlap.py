"""Exact spatial Converse prototype when both kernel dimensions are <= scale.

For periodic true convolution H and phase-zero decimation D, disjoint row
supports give A A^T = sum(k^2) I, A=DH. Thus the unchanged residual solution is
prior + A^T[(x-A prior)/(sum(k^2)+sigmoid(bias-9)+eps)]. This is not an
approximation, but FP32 output/gradient budgets still require validation.

Only ATen autograd is used. Kernel energy is accumulated in FP64 then cast to
the input dtype; batch/channel broadcasting stays differentiable. No cache,
detached trainable tensor or custom backward is introduced. k7/s3 DataNet and
k3/s1 prior layers are explicitly OUTSIDE this path and are rejected.

--cpu-proof runs only small independent dense-matrix/FFT proofs, including
arbitrary priors, odd/even/rectangular kernels and 1x1 spatial axes. It never
calls the original make_cases(), whose weak fixtures allocate CUDA tensors.
The CUDA entry uses that original make_cases() unchanged, keeps its eps/data,
filters only kh<=s and kw<=s, and gates ALL cases before any timing.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def expanded_filter(weight, batch, channels):
    kh, kw = weight.shape[-2:]
    return weight.expand(batch, channels, kh, kw).flip((-2, -1)).reshape(batch * channels, 1, kh, kw)


def blur_downsample(prior, weight, scale):
    import torch.nn.functional as F
    batch, channels, height, width = prior.shape
    kh, kw = weight.shape[-2:]
    padded = F.pad(prior, (kw - 1 - kw // 2, kw // 2, kh - 1 - kh // 2, kh // 2), mode="circular")
    grouped = padded.reshape(1, batch * channels, height + kh - 1, width + kw - 1)
    output = F.conv2d(grouped, expanded_filter(weight, batch, channels), stride=scale, groups=batch * channels)
    return output.reshape(batch, channels, height // scale, width // scale)


def transpose_correction(q, weight, scale):
    import torch
    import torch.nn.functional as F
    batch, channels, height, width = q.shape
    kh, kw = weight.shape[-2:]
    output = F.conv_transpose2d(q.reshape(1, batch * channels, height, width),
                                expanded_filter(weight, batch, channels), stride=scale, padding=0,
                                output_padding=(scale - kh, scale - kw), groups=batch * channels)
    output = output.reshape(batch, channels, height * scale, width * scale)
    return torch.roll(output, (kh // 2 - (kh - 1), kw // 2 - (kw - 1)), dims=(-2, -1))


def spatial_nonoverlap(x, prior, weight, bias, scale=1, eps=1e-5):
    """Independent-prior interface; invalid support is rejected, never approximated."""
    import torch
    from models.converse_core import validate_inputs
    validate_inputs(x, prior, weight, bias, scale, eps)
    kh, kw = weight.shape[-2:]
    if kh > scale or kw > scale:
        raise ValueError(f"Non-overlap requires kh,kw<=scale; got kernel={kh}x{kw}, scale={scale}; use production fallback")
    if x.dtype not in (torch.float32, torch.float64) or torch.is_autocast_enabled(x.device.type):
        raise ValueError("This isolated path supports FP32/FP64 without autocast only")
    residual = x - blur_downsample(prior, weight, scale)
    energy = weight.double().square().sum(dim=(-2, -1), keepdim=True).to(x.dtype)
    regularizer = torch.sigmoid(bias - 9.0) + eps
    q = residual / (energy + regularizer)
    return prior + transpose_correction(q, weight, scale)


def nearest_nonoverlap(x, weight, bias, scale, eps):
    import torch.nn.functional as F
    prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
    return spatial_nonoverlap(x, prior, weight, bias, scale, eps)


def explicit_matrix(weight, batch, channels, height, width, scale):
    """Differentiable A built from scalar circular indices, independently of conv/FFT."""
    import torch
    kh, kw = weight.shape[-2:]
    hs, ws = height * scale, width * scale
    expanded = weight.expand(batch, channels, kh, kw)
    matrix = torch.zeros(batch, channels, height * width, hs * ws, dtype=weight.dtype, device=weight.device)
    for ky in range(kh):
        for kx in range(kw):
            selector = torch.zeros(height * width, hs * ws, dtype=weight.dtype, device=weight.device)
            for row in range(height):
                for column in range(width):
                    iy, ix = (row * scale + kh // 2 - ky) % hs, (column * scale + kw // 2 - kx) % ws
                    selector[row * width + column, iy * ws + ix] = 1
            matrix = matrix + expanded[..., ky, kx, None, None] * selector
    return matrix


def matrix_solve(x, prior, weight, bias, scale, eps):
    import torch
    batch, channels, height, width = x.shape
    matrix = explicit_matrix(weight, batch, channels, height, width, scale)
    p = prior.flatten(-2).unsqueeze(-1)
    residual = x.flatten(-2).unsqueeze(-1) - matrix @ p
    gram = matrix @ matrix.transpose(-1, -2)
    eye = torch.eye(height * width, dtype=x.dtype, device=x.device)
    regularizer = torch.sigmoid(bias - 9.0) + eps
    q = torch.linalg.solve(gram + regularizer * eye, residual)
    return (p + matrix.transpose(-1, -2) @ q).squeeze(-1).reshape_as(prior)


def cpu_proof():
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    if torch.cuda.is_initialized():
        raise RuntimeError("Run the CPU proof before initializing CUDA")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = torch.Generator(device="cpu").manual_seed(20260918)
    # name, B,C,H,W,s,kh,kw,KB,KC
    configurations = [
        ("s1_1x1", 1, 2, 3, 4, 1, 1, 1, 1, 2),
        ("s3_1x1_single_pixel", 2, 3, 1, 1, 3, 1, 1, 1, 1),
        ("s3_odd_shared_batch", 2, 3, 2, 3, 3, 3, 3, 1, 3),
        ("s3_odd_single_row_shared_channel", 2, 3, 1, 4, 3, 3, 3, 2, 1),
        ("s2_even_single_column_batched", 2, 2, 3, 1, 2, 2, 2, 2, 2),
        ("s3_rect_2x3", 2, 3, 2, 3, 3, 2, 3, 1, 1),
        ("s3_rect_3x2", 1, 2, 3, 2, 3, 3, 2, 1, 2),
        ("s4_even_4x4", 1, 2, 2, 2, 4, 4, 4, 1, 1),
        ("s4_rect_1x4_single_row", 2, 1, 1, 3, 4, 1, 4, 2, 1),
    ]
    rows = []
    try:
        for index, config in enumerate(configurations + [(f"weak_cpu_{a:g}", 2, 3, 3, 4, 3, 3, 3, 1, 1)
                                                        for a in (0.0, 1e-6, 1e-3)]):
            name, batch, channels, height, width, scale, kh, kw, kb, kc = config
            x = torch.randn(batch, channels, height, width, generator=rng, dtype=torch.float64)
            prior = torch.randn(batch, channels, height * scale, width * scale, generator=rng, dtype=torch.float64)
            weight = torch.rand(kb, kc, kh, kw, generator=rng, dtype=torch.float64) / (kh * kw)
            bias = torch.randn(1, channels, 1, 1, generator=rng, dtype=torch.float64)
            upstream = torch.randn(prior.shape, generator=rng, dtype=torch.float64) / prior.numel() ** 0.5
            eps = 1e-3
            if index >= len(configurations):
                weight = weight * (0.0, 1e-6, 1e-3)[index - len(configurations)]
                x, prior, upstream = x * 1e-5, prior * 1e-5, upstream * 1e-5
                bias.fill_(-40.0)
                eps = 1e-8
            matrix = explicit_matrix(weight, batch, channels, height, width, scale)
            gram = matrix @ matrix.transpose(-1, -2)
            energy = weight.expand(batch, channels, kh, kw).square().sum((-2, -1))
            expected_gram = energy[..., None, None] * torch.eye(height * width, dtype=torch.float64)
            torch.testing.assert_close(gram, expected_gram, atol=1e-12, rtol=1e-12)
            ap = blur_downsample(prior, weight, scale)
            expected_ap = (matrix @ prior.flatten(-2).unsqueeze(-1)).squeeze(-1).reshape_as(x)
            q = torch.randn(x.shape, generator=rng, dtype=torch.float64)
            atq = transpose_correction(q, weight, scale)
            expected_atq = (matrix.transpose(-1, -2) @ q.flatten(-2).unsqueeze(-1)).squeeze(-1).reshape_as(prior)
            torch.testing.assert_close(ap, expected_ap, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(atq, expected_atq, atol=1e-12, rtol=1e-12)
            values = {}
            for method, function in (("spatial", spatial_nonoverlap), ("matrix", matrix_solve), ("fft", converse2d_reference)):
                inputs = tuple(value.clone().requires_grad_() for value in (x, prior, weight, bias))
                output = function(*inputs, scale, eps)
                values[method] = (output.detach(), *[value.detach() for value in torch.autograd.grad(output, inputs, upstream)])
            error = {}
            for reference in ("matrix", "fft"):
                errors = []
                for actual, expected in zip(values["spatial"], values[reference]):
                    torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)
                    errors.append((actual - expected).abs().max().item())
                error[reference] = dict(zip(("output", "dx", "dprior", "dw", "db"), errors))
            nearest_values = []
            for reference in (False, True):
                xx, ww, bb = [value.clone().requires_grad_() for value in (x, weight, bias)]
                pp = xx if scale == 1 else F.interpolate(xx, scale_factor=scale, mode="nearest")
                output = (converse2d_reference(xx, pp, ww, bb, scale, eps) if reference
                          else spatial_nonoverlap(xx, pp, ww, bb, scale, eps))
                nearest_values.append((output.detach(), *torch.autograd.grad(output, (xx, ww, bb), upstream)))
            for actual, expected in zip(*nearest_values):
                torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)
            rows.append(dict(name=name, config=config[1:], eps=eps, gram_max_abs=(gram - expected_gram).abs().max().item(),
                             forward_A_max_abs=(ap - expected_ap).abs().max().item(),
                             transpose_A_max_abs=(atq - expected_atq).abs().max().item(),
                             independent_prior_errors=error, nearest_and_shared_x_vjp_passed=True))
        tiny = (torch.randn(1, 2, 1, 2, generator=rng, dtype=torch.float64).requires_grad_(),
                torch.randn(1, 2, 2, 4, generator=rng, dtype=torch.float64).requires_grad_(),
                torch.rand(1, 1, 2, 1, generator=rng, dtype=torch.float64).requires_grad_(),
                torch.randn(1, 2, 1, 1, generator=rng, dtype=torch.float64).requires_grad_())
        function = lambda x, p, k, b: spatial_nonoverlap(x, p, k, b, 2, 1e-3)
        torch.autograd.gradcheck(function, tiny, fast_mode=True)
        torch.autograd.gradgradcheck(function, tiny, fast_mode=True)
        rejected = []
        for scale, kernel in ((3, 7), (1, 3)):
            x = torch.ones(1, 1, 4, 5, dtype=torch.float64)
            try:
                spatial_nonoverlap(x, F.interpolate(x, scale_factor=scale, mode="nearest"),
                                  torch.ones(1, 1, kernel, kernel, dtype=torch.float64),
                                  torch.zeros(1, 1, 1, 1, dtype=torch.float64), scale)
            except ValueError as error:
                rejected.append(dict(scale=scale, kernel=kernel, reason=str(error)))
            else:
                raise AssertionError("Overlapping support was not rejected")
        assert not torch.cuda.is_initialized()
        return dict(passed=True, device="CPU", dtype="float64", cases=rows, gradcheck=True, gradgradcheck=True,
                    rejected_overlap_cases=rejected, cuda_initialized=False,
                    proof="A uses explicit periodic indices j*s+floor(k/2)-tap. Nonoverlapping row supports make AAT spatial-energy times identity; independent dense solve and full FFT agree.",
                    scope="Small mathematical/autograd proof only; does not replace the unchanged original CUDA FP32 fixture gates")
    finally:
        torch.set_num_threads(previous_threads)


def validate_case(case):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    from probe_nearest_training import spatial
    from probe_pointwise_training import capture
    from probe_training_s1_shapes import metrics
    scale, eps = case["scale"], case["eps"]
    reference = capture(case["tensors"], torch.float64, lambda x, k, b: converse2d_reference(
        x, x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest"), k, b, scale, eps))
    result = {}
    for name, function in (("production", lambda x, k, b: spatial(x, k, b, scale, eps, "production")),
                           ("spatial_nonoverlap", lambda x, k, b: nearest_nonoverlap(x, k, b, scale, eps))):
        actual = capture(case["tensors"], torch.float32, function)
        if any(value.dtype != torch.float32 for value in actual.values()):
            raise RuntimeError("FP32 dtype contract violated")
        errors = {key: metrics(actual[key], value, output=key == "output", weak=case["weak"])
                  for key, value in reference.items()}
        result[name] = dict(passed=all(value["passed"] for value in errors.values()), tensors=errors)
    return result


def time_case(case, args):
    from probe_nearest_training import spatial
    from probe_pointwise_training import timed_fixture
    scale, eps = case["scale"], case["eps"]
    methods = dict(production=lambda x, k, b: spatial(x, k, b, scale, eps, "production"),
                   spatial_nonoverlap=lambda x, k, b: nearest_nonoverlap(x, k, b, scale, eps))
    rounds = []
    for index in range(args.rounds):
        order = ["production", "spatial_nonoverlap"] if index % 2 == 0 else ["spatial_nonoverlap", "production"]
        rounds.append(dict(round=index + 1, order=order,
                           variants={name: timed_fixture(case["tensors"], methods[name], args) for name in order}))
    return dict(rounds=rounds,
                medians={name: {key: statistics.median(row["variants"][name][key] for row in rounds)
                                for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
                         for name in methods},
                paired_production_over_spatial={key: [row["variants"]["production"][key] /
                                                       row["variants"]["spatial_nonoverlap"][key] for row in rounds]
                                                for key in ("wall_ms", "cuda_event_ms")})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cpu-proof", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/spatial_nonoverlap.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iters) < 1 or args.output.exists():
        parser.error("Require positive counts and a new output path")
    sys.path.insert(0, str(ROOT))
    if args.cpu_proof:
        report = cpu_proof()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]), cuda_initialized=False, output=str(args.output))))
        return 0
    # Prove the small CPU systems before any CUDA initialization.
    proof = cpu_proof()
    import torch
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    from probe_nearest_training import make_cases
    from probe_pointwise_training import tensor_hash
    if not torch.cuda.is_available() or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        parser.error("CUDA FP32 production build required")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with the FP32 protocol")
    load_extension()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    original = make_cases()  # Do not alter its RNG stream, eps, scale or upstream.
    cases, rejected = [], []
    for case in original:
        kh, kw = case["tensors"][1].shape[-2:]
        if kh <= case["scale"] and kw <= case["scale"]:
            cases.append(case)
        else:
            rejected.append(dict(name=case["name"], kernel=[kh, kw], scale=case["scale"], reason="Overlapping supports; not eligible, original production remains required"))
    report = dict(status="validating", scope=__doc__, cpu_proof=proof, source_sha256=current_manifest(),
                  script_sha256={name: hashlib.sha256((ROOT / "test" / name).read_bytes()).hexdigest() for name in
                                 ("probe_spatial_nonoverlap.py", "probe_nearest_training.py", "probe_pointwise_training.py", "probe_training_s1_shapes.py", "test_fp32_training.py")},
                  reference_sha256=hashlib.sha256((ROOT / "models/converse_core.py").read_bytes()).hexdigest(),
                  settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_deterministic=True),
                  excluded_cases=rejected, production_eligible=False, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        for case in cases:
            row = dict(name=case["name"], scale=case["scale"], eps=case["eps"], weak=case["weak"],
                       shape=list(case["tensors"][0].shape), kernel_shape=list(case["tensors"][1].shape),
                       tensors_sha256=tensor_hash(case["tensors"]), validation=validate_case(case), timing=None)
            report["cases"].append(row)
            save()
        if not all(value["passed"] for row in report["cases"] for value in row["validation"].values()):
            report["status"] = "original_precision_gate_failed_no_timing"
            save()
            return 1
        report["status"] = "timing"
        for case, row in zip(cases, report["cases"]):
            if case["timed"]:
                row["timing"] = time_case(case, args)
                save()
                print(json.dumps(dict(case=case["name"], medians=row["timing"]["medians"])), flush=True)
        report["status"] = "complete_isolated_nonoverlap_prototype"
        save()
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
