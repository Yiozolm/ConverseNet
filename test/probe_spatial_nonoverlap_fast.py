"""ATen-only optimizations of the exact non-overlap spatial formula.

Five routes: production FFT; unchanged old_folded; shared_batch using native B
and groups=C for KB=1; k3/s3 LR prediction plus transpose; and k3/s3 LR prediction
plus addcmul phase blocks/pixel_shuffle. KB=B still uses the old folded adjoint.
LR specializations fall back to the old correct spatial path for other legal
kernels. All spatial routes reject kh>s or kw>s; k7/s3 DataNet remains excluded.

No production/old-probe edits, custom backward, detached trainable values or
cross-call cache. The separately supplied CPU audit checks matrices/FFT and
higher derivatives; its artifact hash is recorded when available. CUDA validation
reuses original nearest make_cases, unchanged seeds/eps/upstream and budgets,
before any same-fixture FWD+dx/dw/db timing. No rate or quality claim is assumed.
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

import probe_spatial_nonoverlap as old

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("production", "old_folded", "shared_batch", "lrpred_transpose", "lrpred_tensor")


def validate_nearest(x, weight, bias, scale, eps):
    import torch
    if not isinstance(scale, int) or scale < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Require positive integer scale and finite positive eps")
    if x.ndim != 4 or any(size == 0 for size in x.shape):
        raise ValueError("Input must be nonempty BCHW")
    batch, channels = x.shape[:2]
    if weight.ndim != 4 or weight.shape[0] not in (1, batch) or weight.shape[1] not in (1, channels):
        raise ValueError("Invalid kernel batch/channel broadcasting")
    if any(size < 1 or size > scale for size in weight.shape[-2:]):
        raise ValueError("Non-overlap requires kh<=scale and kw<=scale; use production fallback")
    if bias.shape != (1, channels, 1, 1) or any(value.dtype != x.dtype or value.device != x.device for value in (weight, bias)):
        raise ValueError("Bias shape/device/dtype mismatch")
    if x.dtype not in (torch.float32, torch.float64) or torch.is_autocast_enabled(x.device.type):
        raise ValueError("Only FP32/FP64 without autocast is supported")


def shared_filter(weight, channels):
    kh, kw = weight.shape[-2:]
    return weight.expand(1, channels, kh, kw).flip((-2, -1)).reshape(channels, 1, kh, kw)


def shared_transpose(q, weight, scale):
    import torch
    import torch.nn.functional as F
    if weight.shape[0] != 1:
        return old.transpose_correction(q, weight, scale)
    kh, kw = weight.shape[-2:]
    correction = F.conv_transpose2d(q, shared_filter(weight, q.shape[1]), stride=scale,
                                     output_padding=(scale - kh, scale - kw), groups=q.shape[1])
    return torch.roll(correction, (kh // 2 - (kh - 1), kw // 2 - (kw - 1)), dims=(-2, -1))


def shared_spatial(x, prior, weight, bias, scale, eps):
    import torch
    import torch.nn.functional as F
    from models.converse_core import validate_inputs
    validate_nearest(x, weight, bias, scale, eps)
    validate_inputs(x, prior, weight, bias, scale, eps)
    if weight.shape[0] != 1:
        return old.spatial_nonoverlap(x, prior, weight, bias, scale, eps)
    kh, kw = weight.shape[-2:]
    padded = F.pad(prior, (kw - 1 - kw // 2, kw // 2, kh - 1 - kh // 2, kh // 2), mode="circular")
    prediction = F.conv2d(padded, shared_filter(weight, x.shape[1]), stride=scale, groups=x.shape[1])
    energy = weight.double().square().sum((-2, -1), keepdim=True).to(x.dtype)
    q = (x - prediction) / (energy + torch.sigmoid(bias - 9.0) + eps)
    del prediction, padded
    return prior + shared_transpose(q, weight, scale)


def lr_prediction_q(x, weight, bias, eps):
    """k3/s3: phase-zero prediction uses current/previous LR neighbours."""
    import torch
    kernel64 = weight.double()
    coeff00 = kernel64[..., :2, :2].sum((-2, -1), keepdim=True).to(x.dtype)
    coeff01 = kernel64[..., :2, 2:].sum((-2, -1), keepdim=True).to(x.dtype)
    coeff10 = kernel64[..., 2:, :2].sum((-2, -1), keepdim=True).to(x.dtype)
    coeff11 = kernel64[..., 2:, 2:].sum((-2, -1), keepdim=True).to(x.dtype)
    prediction = x * coeff00
    prediction = torch.addcmul(prediction, torch.roll(x, 1, -1), coeff01)
    prediction = torch.addcmul(prediction, torch.roll(x, 1, -2), coeff10)
    prediction = torch.addcmul(prediction, torch.roll(x, (1, 1), (-2, -1)), coeff11)
    energy = kernel64.square().sum((-2, -1), keepdim=True).to(x.dtype)
    return (x - prediction) / (energy + torch.sigmoid(bias - 9.0) + eps)


def phase_tensor_output(x, q, weight):
    """HR phases use kernel [1,0,2]; last phase receives NEXT LR q."""
    import torch
    import torch.nn.functional as F
    base = x[:, :, None, None]
    current = q[:, :, None, None]
    block00 = torch.addcmul(base, current, weight[..., :2, :2].flip((-2, -1))[..., None, None])
    next_w = torch.roll(q, -1, -1)[:, :, None, None]
    block01 = torch.addcmul(base, next_w, weight[..., :2, 2:].flip(-2)[..., None, None])
    top = torch.cat((block00, block01), dim=3)
    del block00, block01, next_w
    next_h = torch.roll(q, -1, -2)[:, :, None, None]
    block10 = torch.addcmul(base, next_h, weight[..., 2:, :2].flip(-1)[..., None, None])
    next_hw = torch.roll(q, (-1, -1), (-2, -1))[:, :, None, None]
    block11 = torch.addcmul(base, next_hw, weight[..., 2:, 2:][..., None, None])
    bottom = torch.cat((block10, block11), dim=3)
    del block10, block11, next_h, next_hw, current, base
    phases = torch.cat((top, bottom), dim=2)  # B,C,phase_row,phase_col,H,W
    del top, bottom
    batch, channels, height, width = x.shape
    result = F.pixel_shuffle(phases.reshape(batch, channels * 9, height, width), 3)
    del phases
    return result


def nearest_route(x, weight, bias, scale, eps, route):
    import torch.nn.functional as F
    if route == "production":
        from probe_nearest_training import spatial
        return spatial(x, weight, bias, scale, eps, "production")
    validate_nearest(x, weight, bias, scale, eps)
    if route == "old_folded":
        return old.nearest_nonoverlap(x, weight, bias, scale, eps)
    if route == "shared_batch":
        prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
        return shared_spatial(x, prior, weight, bias, scale, eps)
    if route not in ("lrpred_transpose", "lrpred_tensor"):
        raise ValueError(route)
    if scale != 3 or tuple(weight.shape[-2:]) != (3, 3):
        return old.nearest_nonoverlap(x, weight, bias, scale, eps)
    q = lr_prediction_q(x, weight, bias, eps)
    if route == "lrpred_tensor":
        return phase_tensor_output(x, q, weight)
    correction = shared_transpose(q, weight, 3)
    result = F.interpolate(x, scale_factor=3, mode="nearest") + correction
    del correction, q
    return result


def validate_case(case):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    from probe_pointwise_training import capture
    from probe_training_s1_shapes import metrics
    scale, eps = case["scale"], case["eps"]
    reference = capture(case["tensors"], torch.float64, lambda x, k, b: converse2d_reference(
        x, x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest"), k, b, scale, eps))
    result = {}
    for route in ROUTES:
        actual = capture(case["tensors"], torch.float32,
                         lambda x, k, b, route=route: nearest_route(x, k, b, scale, eps, route))
        if any(value.dtype != torch.float32 for value in actual.values()):
            raise RuntimeError("Candidate changed output/gradient dtype")
        checks = {key: metrics(actual[key], value, output=key == "output", weak=case["weak"])
                  for key, value in reference.items()}
        result[route] = dict(passed=all(value["passed"] for value in checks.values()), tensors=checks)
    return result


def time_case(case, args):
    from probe_pointwise_training import timed_fixture
    rounds = []
    for index in range(args.rounds):
        order = list(ROUTES[index % len(ROUTES):] + ROUTES[:index % len(ROUTES)])
        values = {route: timed_fixture(case["tensors"],
                   lambda x, k, b, route=route: nearest_route(x, k, b, case["scale"], case["eps"], route), args)
                  for route in order}
        rounds.append(dict(round=index + 1, order=order, variants=values))
    return dict(rounds=rounds,
                medians={route: {key: statistics.median(row["variants"][route][key] for row in rounds)
                                 for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
                         for route in ROUTES},
                paired_production_over_candidate={route: {key: [row["variants"]["production"][key] / row["variants"][route][key]
                                                                  for row in rounds]
                                                          for key in ("wall_ms", "cuda_event_ms")}
                                                     for route in ROUTES[1:]})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/spatial_nonoverlap_fast.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    sys.path.insert(0, str(ROOT))
    proof_path = ROOT / "artifacts/training_research/nearest_pixel_shuffle_audit.json"
    proof = dict(available=proof_path.is_file(), path=str(proof_path))
    if proof_path.is_file():
        proof["sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        proof["passed"] = json.loads(proof_path.read_text(encoding="utf-8"))["passed"]
        if not proof["passed"]:
            raise RuntimeError("Independent CPU audit did not pass")
    import torch
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    from probe_nearest_training import make_cases
    from probe_pointwise_training import tensor_hash
    if not torch.cuda.is_available() or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        parser.error("CUDA FP32 build required")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with protocol")
    load_extension()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cases, excluded = [], []
    for case in make_cases():
        if all(size <= case["scale"] for size in case["tensors"][1].shape[-2:]):
            cases.append(case)
        else:
            excluded.append(dict(name=case["name"], reason="kernel support exceeds scale; original production required"))
    report = dict(status="validating", scope=__doc__, cpu_proof=proof, source_sha256=current_manifest(),
                  script_sha256={name: hashlib.sha256((ROOT / "test" / name).read_bytes()).hexdigest() for name in
                                 ("probe_spatial_nonoverlap_fast.py", "probe_spatial_nonoverlap.py", "probe_nearest_training.py",
                                  "probe_pointwise_training.py", "probe_training_s1_shapes.py", "test_fp32_training.py")},
                  reference_sha256=hashlib.sha256((ROOT / "models/converse_core.py").read_bytes()).hexdigest(),
                  settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_deterministic=True),
                  production_eligible=False, excluded_cases=excluded, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        for case in cases:
            report["cases"].append(dict(name=case["name"], scale=case["scale"], eps=case["eps"], weak=case["weak"],
                                        shape=list(case["tensors"][0].shape), kernel_shape=list(case["tensors"][1].shape),
                                        tensors_sha256=tensor_hash(case["tensors"]), validation=validate_case(case), timing=None))
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
        report["status"] = "complete_isolated_spatial_implementations"
        save()
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
