"""Level1A/B quantized-reference matrix; no production changes or performance claims.

The 972 cases sample, rather than Cartesian-expand, geometry, regularization,
layout and gradient masks. Both dtypes and both weight-storage ablations are
crossed with all 243 base fixtures. Twelve rows are explicit range probes.
--self-check selects a small CPU subset and can never grant GPU admission.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "test")]
import torch
import fp32_baseline as frozen
import numerical_policy as fp32_policy
from models import converse_core
from tools.v4_mixed_precision import policy
from tools.v4_mixed_precision.adapter import mixed_converse2d

PLAN_SHA256 = "9fb6a7b79743a1f797f11df04cfcba85f56b37de9d2d2bff3109cf6cfcd57127"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DEPENDENCIES = ("tools/v4_mixed_precision/adapter.py", "tools/v4_mixed_precision/policy.py",
                "tools/v4_mixed_precision/gate.py", "test/fp32_baseline.py",
                "test/numerical_policy.py", "test/extension_loader.py",
                "models/converse_core.py", "Converse2D/build_config.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {name: sha(ROOT / name) for name in DEPENDENCIES}


def record(value):
    dense = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                finite=bool(torch.isfinite(value).all()),
                sha256=hashlib.sha256(dense.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def case_specs():
    shapes = ((8, 8), (17, 19), (31, 37), (7, 1))
    regs = ((1e-5, "normal"), (1e-5, "minus40"), (1e-8, "normal"), (1e-8, "minus40"))
    masks = ((True,)*4, (True, True, False, False), (False, False, True, True), (False, False, False, True))
    index = 0
    base = []
    for scale in (1, 2, 3, 4):
        for broadcast, (kb, kc) in enumerate(((1, 1), (1, 3), (2, 1), (2, 3))):
            for storage, layout in enumerate(("contiguous", "strided", "transpose")):
                for kind_index, kind in enumerate(("normal", "normalized", "1e-3", "1e-6", "zero")):
                    h, w = shapes[(scale + broadcast + 2*storage + kind_index) % 4]
                    eps, bias = regs[(index//4 + storage + kind_index) % 4]
                    training = (index + index//5) % 5 == 0
                    mode = "training" if training else ("no_grad", "inference_mode", "frozen")[(index//5 + storage) % 3]
                    shared = scale == 1 and (kind_index + storage) % 2 == 0
                    base.append(dict(id=f"base{index:03d}", scale=scale, height=h, width=w, kb=kb, kc=kc,
                                     layout=layout, kernel=kind, eps=eps, bias=bias, shared=shared,
                                     mode=mode, needs=masks[(index//5) % 4] if training else (False,)*4,
                                     regime="weak" if eps == 1e-8 and bias == "minus40" else "normal",
                                     range_probe=None))
                    index += 1
    for probe in ("input_fp16_range", "input_bf16_range", "output_fp16_range"):
        base.append(dict(id=probe, scale=1, height=3, width=5, kb=1, kc=3, layout="contiguous",
                         kernel="1e-6" if probe == "output_fp16_range" else "normalized",
                         eps=1e-8 if probe == "output_fp16_range" else 1e-5,
                         bias="minus40" if probe == "output_fp16_range" else "normal",
                         shared=True, mode="no_grad", needs=(False,)*4,
                         regime="weak" if probe == "output_fp16_range" else "normal", range_probe=probe))
    for spec in base:
        for dtype in DTYPES:
            for ablation in ("activation_only", "activation_and_weight"):
                yield dict(spec, dtype=dtype, ablation=ablation, name=f"{spec['id']}/{dtype}/{ablation}")


def fixture(spec):
    s, h, w = spec["scale"], spec["height"], spec["width"]
    generator = torch.Generator(device="cpu").manual_seed(76103 + 13*s + 7*h + w + 31*spec["kb"] + spec["kc"])
    kh, kw = min(3, h*s), min(3, w*s)
    x = torch.randn(2, 3, h, w, generator=generator)
    prior = x if spec["shared"] else torch.randn(2, 3, h*s, w*s, generator=generator)
    weight = torch.randn(spec["kb"], spec["kc"], kh, kw, generator=generator)
    kind = spec["kernel"]
    if kind == "normalized":
        weight = weight.flatten(-2).softmax(-1).reshape_as(weight)
    else:
        weight = weight / (kh*kw)**.5
        if kind != "normal":
            weight = weight * (0.0 if kind == "zero" else float(kind))
    bias = torch.randn(1, 3, 1, 1, generator=generator) * .2
    if spec["bias"] == "minus40":
        bias.fill_(-40)
    probe = spec["range_probe"]
    if probe == "input_fp16_range":
        x.fill_(70000.)
    elif probe == "input_bf16_range":
        x.fill_(3.4e38)
    elif probe == "output_fp16_range":
        x.fill_(10000.)
    upstream = torch.randn(prior.shape, generator=generator) / prior.numel()**.5
    return (x, prior, weight, bias), upstream


def layout(value, name):
    if name == "strided":
        return torch.stack((value, value), -1)[..., 0]
    if name == "transpose":
        return value.transpose(-1, -2).contiguous().transpose(-1, -2)
    return value.contiguous()


def prepare(raw, dtypes, spec, device):
    # Detach only fixture leaves; the actual adapter's casts remain differentiable.
    needs = list(spec["needs"])
    if spec["mode"] in ("no_grad", "inference_mode"):
        needs = [True]*4  # Explicitly verify disabled GradMode overrides trainable leaves.
    if spec["shared"]:
        needs[0] = needs[1] = needs[0] or needs[1]
    memo, values = {}, []
    for value, dtype, need in zip(raw, dtypes, needs):
        key = id(value)
        if key not in memo:
            memo[key] = layout(value.to(device=device, dtype=dtype), spec["layout"]).detach().requires_grad_(need)
        values.append(memo[key])
    return tuple(values)


def context(mode):
    return {"training": torch.enable_grad, "frozen": torch.enable_grad,
            "no_grad": torch.no_grad, "inference_mode": torch.inference_mode}[mode]()


def evaluate(function, raw, dtypes, spec, device, upstream):
    values = prepare(raw, dtypes, spec, device)
    with context(spec["mode"]):
        output = function(*values, spec["scale"], spec["eps"])
        result = {"output": output.detach()}
        if spec["mode"] == "training":
            labels, requested = [], []
            for index, (name, value) in enumerate(zip(("dx", "dprior", "dweight", "dbias"), values)):
                if spec["shared"] and index == 1:
                    continue
                if value.requires_grad:
                    labels.append("dx_shared" if spec["shared"] and index == 0 else name)
                    requested.append(value)
            gradients = torch.autograd.grad(output, requested, upstream.to(device=device, dtype=output.dtype))
            result.update({name: value.detach() for name, value in zip(labels, gradients)})
    return result


def boundary_reference(core, spec, *, output_dtype=torch.float32):
    low = DTYPES[spec["dtype"]]
    return {name: value.to(output_dtype if name == "output" else
                          (torch.float32 if name == "dbias" or (name == "dweight" and spec["ablation"] == "activation_only") else low))
            for name, value in core.items()}


def reference_diagnostics(raw, spec):
    """Actual FP32 full/half RQ expressions; candidate q is not exposed by its API."""
    with torch.no_grad():
        x, prior, weight, bias = (v.detach().float() for v in raw)
        h, w = x.shape[-2:]
        s, kh, kw = spec["scale"], weight.shape[-2], weight.shape[-1]
        psf = torch.nn.functional.pad(weight, (0, w*s-kw, 0, h*s-kh))
        psf = torch.roll(psf, (-(kh//2), -(kw//2)), (-2, -1))
        training = spec["mode"] == "training"
        transform = torch.fft.fft2 if training else torch.fft.rfft2
        kernel, fy = transform(psf), transform(x)
        fp = fy if spec["shared"] else transform(prior)
        power = kernel.real.square() + kernel.imag.square()
        prediction = kernel * fp
        def full(value, width):
            tail = value[..., 1:(width+1)//2].flip((-2, -1)).roll(1, -2)
            return torch.cat((value, tail.conj() if value.is_complex() else tail), -1)
        if s > 1:
            if not training:
                power, prediction = full(power, w*s), full(prediction, w*s)
            power, prediction = frozen.alias_mean(power, s), frozen.alias_mean(prediction, s)
            if not training:
                power, prediction = power[..., :w//2+1], prediction[..., :w//2+1]
        regularizer = torch.sigmoid(bias - 9.0) + spec["eps"]
        denominator = power + regularizer
        q = (fy - prediction) / denominator
        finite = bool(torch.isfinite(denominator).all())
        positive = finite and bool((denominator > 0).all())
        stats = [None]*4
        if finite:
            shape = (*x.shape[:-2], h, w if training else w//2+1)
            expanded = denominator.expand(shape).flatten()
            stats = torch.quantile(expanded, expanded.new_tensor([0., .01, .5, 1.])).tolist()
        q_finite = bool(torch.isfinite(q).all())
        return dict(denominator=dict(zip(("min", "p01", "median", "max"), stats)),
                    min_denominator=stats[0], p01_denominator=stats[1],
                    median_denominator=stats[2], max_denominator=stats[3],
                    denominator_finite=finite, denominator_positive=positive, q_finite=q_finite,
                    lambda_finite=bool(torch.isfinite(regularizer).all()),
                    passed=bool(positive and q_finite and torch.isfinite(regularizer).all()),
                    arithmetic_dtype="float32/complex64", spectrum="full" if training else "half",
                    scope="Quantized FP32 reference diagnostics; candidate internal q is not directly observable")


def run_case(spec, device):
    raw, upstream = fixture(spec)
    low = DTYPES[spec["dtype"]]
    storage_types = (low, low, low if spec["ablation"] == "activation_and_weight" else torch.float32, torch.float32)
    storage = prepare(raw, storage_types, dict(spec, needs=(False,)*4, mode="frozen"), device)
    quantized = tuple(v.detach().float() for v in storage)
    if spec["shared"]:
        quantized = (quantized[0], quantized[0], *quantized[2:])
    input_states = {name: policy.representation_status(before, after.cpu())
                    for name, before, after in zip(("x", "prior", "weight", "bias"), raw, storage)}
    result = dict(name=spec["name"], spec=spec, dtype=spec["dtype"], ablation=spec["ablation"], mode=spec["mode"],
                  inputs=[record(v) for v in storage], original_inputs=[record(v) for v in raw],
                  input_representation=input_states, passed=False, expected_range_probe=bool(spec["range_probe"]),
                  expected_range_rejection=bool(spec["range_probe"] and
                      (spec["dtype"] == "fp16" or spec["range_probe"] == "input_bf16_range")))
    if any(state != "finite" for state in input_states.values()):
        return dict(result, representation_status="input_representation_overflow", status="rejected_representation",
                    candidate_executed=False, rejection_as_expected=result["expected_range_rejection"])
    result["input_quantization_error"] = {name: policy.metrics(after, before.to(device))
        for name, before, after in zip(("x", "prior", "weight", "bias"), raw, quantized)}
    diagnostics = reference_diagnostics(quantized, spec)
    original32 = prepare(raw, (torch.float32,)*4, dict(spec, mode="frozen", needs=(False,)*4), device)
    result["diagnostics"] = dict(original_fp32=reference_diagnostics(original32, spec), quantized_fp32=diagnostics)
    if not diagnostics["passed"]:
        return dict(result, representation_status="finite_inputs", status="rejected_reference_nonfinite",
                    candidate_executed=False, rejection_as_expected=False)
    backend = "cuda" if device.startswith("cuda") else "pytorch"
    adapter_a = lambda *args: mixed_converse2d(*args, output_dtype=torch.float32, backend=backend)
    adapter_b = lambda *args: mixed_converse2d(*args, output_dtype=low, backend=backend)
    core = torch.ops.converse2d.forward if backend == "cuda" else converse_core.converse2d_fp32
    r64 = evaluate(frozen.converse2d_reference, raw, (torch.float64,)*4, spec, device, upstream)
    r32 = evaluate(frozen.converse2d_fp32, raw, (torch.float32,)*4, spec, device, upstream)
    rq = evaluate(frozen.converse2d_fp32, quantized, (torch.float32,)*4, spec, device, upstream)
    rq64 = evaluate(frozen.converse2d_reference, quantized, (torch.float64,)*4, spec, device, upstream)
    candidate_a = evaluate(adapter_a, storage, storage_types, spec, device, upstream)
    cq = evaluate(core, quantized, (torch.float32,)*4, spec, device, upstream) if spec["mode"] == "training" else candidate_a
    matched_a = boundary_reference(rq, spec)
    level1a = {name: policy.comparison(candidate_a[name], r32[name], matched_a[name], r64[name], regime=spec["regime"])
               for name in candidate_a}
    # This isolates core FP32 VJPs before the low-precision leaf-gradient cast.
    core_checks = {name: fp32_policy.comparison(cq[name], rq[name], rq64[name], regime=spec["regime"])
                   for name in cq}
    gradient_cast = {name: policy.metrics(matched_a[name], rq[name]) for name in rq if name != "output"}
    a_passed = all(v["passed"] for v in level1a.values()) and all(v["passed"] for v in core_checks.values())
    grad_overflow = [name for name in matched_a if name != "output"
                     and policy.representation_status(rq[name], matched_a[name]) == "representation_overflow"]
    candidate_b = evaluate(adapter_b, storage, storage_types, spec, device, upstream.to(low))
    if spec["mode"] == "training":
        # Identical physical upstream for all B references; its low-dtype
        # representation loss is explicit rather than charged to the kernel.
        up_b = upstream.to(low).float()
        rb64 = evaluate(frozen.converse2d_reference, raw, (torch.float64,)*4, spec, device, up_b)
        rb32 = evaluate(frozen.converse2d_fp32, raw, (torch.float32,)*4, spec, device, up_b)
        rqb = evaluate(frozen.converse2d_fp32, quantized, (torch.float32,)*4, spec, device, up_b)
    else:
        rb64, rb32, rqb = r64, r32, rq
    matched_b = boundary_reference(rqb, spec, output_dtype=low)
    level1b = {"output": policy.output_cast_comparison(candidate_b["output"], candidate_a["output"], rq["output"],
                                                       r32["output"], r64["output"], regime=spec["regime"], level1a_passed=a_passed)}
    for name in candidate_b:
        if name != "output":
            check = policy.comparison(candidate_b[name], rb32[name], matched_b[name], rb64[name], regime=spec["regime"])
            check["level1a_passed"] = a_passed
            check["passed"] &= a_passed
            level1b[name] = check
    b_overflow = [name for name in matched_b
                  if policy.representation_status(rqb[name], matched_b[name]) == "representation_overflow"]
    overflow = bool(grad_overflow or b_overflow)
    passed = bool(a_passed and all(v["passed"] for v in level1b.values()) and not overflow)
    result.update(candidate_executed=True, representation_status="boundary_representation_overflow" if overflow else "finite",
                  gradient_representation_overflow=grad_overflow, level1b_representation_overflow=b_overflow,
                  status="complete" if passed else ("rejected_representation" if overflow else "failed_numerical"),
                  passed=passed, level1a=dict(passed=bool(a_passed and not grad_overflow), tensors=level1a),
                  level1b=dict(passed=bool(a_passed and all(v["passed"] for v in level1b.values()) and not b_overflow), tensors=level1b),
                  fp32_core_vs_rq64=dict(passed=all(v["passed"] for v in core_checks.values()), tensors=core_checks),
                  gradient_boundary_cast_error=gradient_cast,
                  level1b_upstream_cast_error=policy.metrics(upstream.to(low), upstream),
                  output_records=dict(candidate_a=record(candidate_a["output"]), candidate_b=record(candidate_b["output"]),
                                      R64=record(r64["output"]), R32=record(r32["output"]), RQ=record(rq["output"]), RQ64=record(rq64["output"])),
                  rejection_as_expected=bool(result["expected_range_rejection"] and overflow), model_approval=False)
    return result


def load_checked():
    path = ROOT / ".build/cuda/source_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        arch = manifest["inputs"]["toolchain"]["environment"]["TORCH_CUDA_ARCH_LIST"]
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        else:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
        spec = importlib.util.spec_from_file_location("mixed_gate_checked_loader", ROOT / "test/extension_loader.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.load_extension()
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return manifest, loader


def summarize(rows):
    supported = [r for r in rows if not r["expected_range_probe"]]
    inference = [r for r in supported if r["mode"] != "training"]
    training = [r for r in supported if r["mode"] == "training"]
    groups = []
    for dtype in DTYPES:
        for ablation in ("activation_only", "activation_and_weight"):
            group = [r for r in supported if r["dtype"] == dtype and r["ablation"] == ablation]
            item = dict(dtype=dtype, ablation=ablation, cases=len(group))
            for mode, selection in (("inference", [r for r in group if r["mode"] != "training"]),
                                    ("training", [r for r in group if r["mode"] == "training"])):
                item[mode + "_cases"] = len(selection)
                for level in ("level1a", "level1b"):
                    item[mode + "_" + level + "_passed"] = bool(selection) and all(r.get(level, {}).get("passed", False) for r in selection)
            groups.append(item)
    probes = [r for r in rows if r["expected_range_probe"]]
    return dict(cases=len(rows), supported_cases=len(supported), supported_passed=bool(supported) and all(r["passed"] for r in supported),
                inference_passed=bool(inference) and all(r["passed"] for r in inference),
                training_passed=bool(training) and all(r["passed"] for r in training),
                failed_cases=[r["name"] for r in rows if not r["passed"]],
                expected_range_rejections=[r["name"] for r in rows if r.get("rejection_as_expected")],
                unexpected_supported_failures=[r["name"] for r in supported if not r["passed"]],
                by_dtype_ablation=groups,
                range_probe_classification_passed=bool(probes) and all(
                    r.get("rejection_as_expected", False) if r["expected_range_rejection"] else r["passed"] for r in probes),
                representation_failures=[r["name"] for r in rows if "overflow" in r.get("representation_status", "")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh unique output path; previous reports are retained")
    if args.self_check:
        args.device = "cpu"
    specs = list(case_specs())
    if len(specs) != 972 or len({v["name"] for v in specs}) != 972:
        raise AssertionError("Matrix is incomplete or has duplicate IDs")
    if args.self_check:
        specs = specs[:960:61] + specs[960:]
    torch.set_num_threads(1) if args.device == "cpu" else None
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    report = dict(kind="mixed_precision_level1_gate", status="running", complete=False, passed=False,
                  created_utc=datetime.now(timezone.utc).isoformat(), device=args.device,
                  gpu_admission=False, self_check=args.self_check, expected_cases=len(specs),
                  metadata=dict(source_sha256=source_identity(), policy=policy.POLICY,
                                frozen_fp32_sha256=fp32_policy.BASELINE_SHA256, plan_sha256=PLAN_SHA256,
                                torch=str(torch.__version__), cuda=torch.version.cuda,
                                tf32=False, amp=False), rows=[], model_approval=False,
                  coverage="240 sampled fixtures x 2 storage dtypes x 2 weight ablations, plus 12 explicit range probes. "
                           "All scale/broadcast/layout/kernel combinations occur; shape, eps/bias, mode and VJP mask are sampled. "
                           "Shapes include 8x8,17x19,31x37,7x1; s1 has shared/independent priors. Output cast is separate. "
                           "Expected range rejection is a successful classification, never numerical admission.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    try:
        if args.device == "cuda":
            manifest, loader = load_checked()
            report["metadata"]["checked_manifest"] = manifest
            report["metadata"]["production_sources"] = loader.production_source_hashes()
            report["metadata"]["gpu"] = torch.cuda.get_device_name()
        for index, spec in enumerate(specs):
            row = run_case(spec, args.device)
            report["rows"].append(row)
            if not row["passed"]:
                print("NOT_ADMITTED", row["name"], row["status"], flush=True)
            if (index+1) % 25 == 0:
                save()
                print(f"Checked {index+1}/{len(specs)}", flush=True)
        if source_identity() != report["metadata"]["source_sha256"]:
            raise RuntimeError("A measured source dependency changed during the gate")
        if args.device == "cuda":
            if loader.production_source_hashes() != report["metadata"]["production_sources"]:
                raise RuntimeError("Production source changed during gate")
            if sha(ROOT / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
                raise RuntimeError("Production binary changed during gate")
        report.update(status="complete", complete=True, summary=summarize(report["rows"]))
        report["passed"] = all(r["passed"] for r in report["rows"])
        report["gpu_admission"] = args.device == "cuda" and not args.self_check and report["passed"]
        report["cuda_initialized"] = torch.cuda.is_initialized()
    except Exception as error:
        report.update(status="error", complete=False, passed=False,
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        save()
        raise
    save()
    print(json.dumps(dict(status=report["status"], passed=report["passed"], summary=report["summary"],
                          output=str(args.output.resolve())), indent=2, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
