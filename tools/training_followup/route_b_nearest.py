"""Reproduce ONE frozen nearest_spectral failure; diagnostic only, no extension.

CPU: .venv/Scripts/python.exe tools/training_followup/route_b_nearest.py --output artifacts/training_followup/route_b_cpu.json --verify-legacy
GPU: same command with --device cuda and a fresh --output, only in an assigned GPU window.
Standalone: copy this script plus *_inputs.json; pass --inputs that.json. Only torch is required.
Exit 0 means the counterexample was reproduced, never that the candidate passed admission.
"""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time

sys.dont_write_bytecode = True
import torch


# The following five functions and nearest_spectral are exact AST copies of
# cc244e3 research/algorithms.py. --verify-legacy checks those ASTs before running.
def check(x, p, k, b, s, eps):
    if any(v.dtype != torch.float32 for v in (x, p, k, b)):
        raise ValueError('FP32 candidates only; FP64 belongs to independent references')
    if p.shape != (*x.shape[:-2], x.shape[-2]*s, x.shape[-1]*s):
        raise ValueError('Invalid prior dimensions')


def aliases(t, s):
    return t if s == 1 else t.reshape(*t.shape[:-2], s, t.shape[-2]//s, s, t.shape[-1]//s).mean((-4, -2))


def kernel_fft(k, shape, real=False):
    kh, kw = k.shape[-2:]
    p = torch.nn.functional.pad(k, (0, shape[1]-kw, 0, shape[0]-kh))
    p = torch.roll(p, (-(kh//2), -(kw//2)), (-2, -1))
    return torch.fft.rfft2(p) if real else torch.fft.fft2(p)


def spectral(x, p, k, b, s, eps, *, K=None, P=None, Y=None):
    check(x, p, k, b, s, eps)
    K = kernel_fft(k, p.shape[-2:]) if K is None else K
    Y = torch.fft.fft2(x) if Y is None else Y
    P = (Y if p is x else torch.fft.fft2(p)) if P is None else P
    power = K.real.square()+K.imag.square()
    lam = torch.sigmoid(b-9.0)+eps
    q = (Y-aliases(K*P, s))/(aliases(power, s)+lam)
    if s > 1:
        q = q.repeat(1, 1, s, s)
    return torch.fft.ifft2(P+K.conj()*q).real


_geometry = {}


def nearest_spectral(x, p, k, b, s, eps=1e-5):
    """Caller declares p=nearest(x); this is never inferred from shape alone."""
    check(x, p, k, b, s, eps)
    if s < 2:
        raise ValueError('Upsampling required')
    hs, ws = p.shape[-2:]
    key = (hs, ws, s, x.device)
    if key not in _geometry:
        with torch.no_grad():
            phases = []
            for n in (hs, ws):
                f = torch.arange(n, device=x.device, dtype=torch.float32)
                z = torch.zeros(n, device=x.device, dtype=torch.complex64)
                for a in range(s):
                    angle = f*(-2*math.pi*a/n)
                    z = z+torch.polar(torch.ones_like(angle), angle)
                phases.append(z)
            if len(_geometry) >= 16:
                _geometry.clear()
            _geometry[key] = phases
    row, col = _geometry[key]
    Y = torch.fft.fft2(x)
    P = Y.repeat(1, 1, s, s)*row[:, None]*col[None, :]
    return spectral(x, p, k, b, s, eps, P=P, Y=Y)


def alias_mean(a, scale):
    """Average corresponding entries in frequency blocks, not adjacent pixels."""
    if scale == 1:
        return a
    h, w = a.shape[-2:]
    return a.reshape(*a.shape[:-2], scale, h // scale, scale, w // scale).mean((-4, -2))


def validate_inputs(x, x0, weight, bias, scale, eps):
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.ndim != 4 or any(d == 0 for d in x.shape):
        raise ValueError("x must have nonempty shape (B,C,H,W)")
    b, c, h, w = x.shape
    if x0.shape != (b, c, h * scale, w * scale):
        raise ValueError("x0 must have shape (B,C,H*scale,W*scale)")
    if weight.ndim != 4 or weight.shape[0] not in (1, b) or weight.shape[1] not in (1, c):
        raise ValueError("weight must have shape (1|B,1|C,kh,kw)")
    if not (0 < weight.shape[2] <= h * scale and 0 < weight.shape[3] <= w * scale):
        raise ValueError("kernel must fit within the output spatial dimensions")
    if bias.shape != (1, c, 1, 1):
        raise ValueError("bias must have shape (1,C,1,1)")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("reference accepts FP32 or FP64 tensors")
    if any(t.device != x.device or t.dtype != x.dtype for t in (x0, weight, bias)):
        raise ValueError("all inputs must have the same device and dtype")


def converse2d_reference(x, x0, weight, bias, scale=1, eps=1e-5):
    """Stable solution with independent x0, including when scale == 1.

    FP64 is reserved for independent numerical validation.
    """
    validate_inputs(x, x0, weight, bias, scale, eps)
    output_dtype = x.dtype
    same_prior = x0 is x
    h, w = x.shape[-2:]
    kh, kw = weight.shape[-2:]
    psf = torch.nn.functional.pad(weight, (0, w * scale - kw, 0, h * scale - kh))
    fb = torch.fft.fft2(torch.roll(psf, (-(kh // 2), -(kw // 2)), (-2, -1)))
    power = fb.real.square() + fb.imag.square()
    fy = torch.fft.fft2(x)
    fx0 = fy if same_prior else torch.fft.fft2(x0)
    regularizer = torch.sigmoid(bias - 9.0) + eps
    correction = (fy - alias_mean(fb * fx0, scale)) / (alias_mean(power, scale) + regularizer)
    if scale != 1:
        correction = correction.repeat(1, 1, scale, scale)
    result = torch.fft.ifft2(fx0 + fb.conj() * correction).real
    return result.to(output_dtype)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw_bytes(value):
    return value.detach().resolve_conj().resolve_neg().cpu().contiguous().numpy().tobytes()


def snapshot(value, include_data=False):
    data = raw_bytes(value)
    result = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=sha(data), bytes=len(data))
    if include_data:
        result["base64_little_endian"] = base64.b64encode(data).decode("ascii")
    return result


def components(value):
    value = value.detach().cpu().contiguous()
    return torch.view_as_real(value) if value.is_complex() else value


def comparison(actual, control):
    a, b = components(actual).float(), components(control).float()
    au = a.view(torch.int32).to(torch.int64) & 0xffffffff
    bu = b.view(torch.int32).to(torch.int64) & 0xffffffff
    # Monotone representable-value distance, identifying -0 and +0. Near zero,
    # a large distance is not a large physical error; absolute error is adjacent.
    order = lambda u: torch.where((u & 0x80000000) != 0,
                                  0x80000000 - (u & 0x7fffffff), 0x80000000 + u)
    ulp = (order(au) - order(bu)).abs()
    delta = actual.detach().to(torch.complex128 if actual.is_complex() else torch.float64).cpu() - control.detach().cpu()
    return dict(byte_equal=raw_bytes(actual) == raw_bytes(control),
                different_components=int((a != b).sum()), max_abs=float(delta.abs().max()),
                component_ulp_max=int(ulp.max()), component_ulp_gt0=int((ulp > 0).sum()),
                component_ulp_gt1=int((ulp > 1).sum()), signed_zero_is_same_for_ulp=True)


def error(value, high):
    dtype = torch.complex128 if high.is_complex() else torch.float64
    v, h = value.detach().cpu().to(dtype), high.detach().cpu()
    diff = v - h
    denominator = torch.linalg.vector_norm(h).clamp_min(1e-300)
    return dict(max_abs=float(diff.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(diff) / denominator),
                l2=float(torch.linalg.vector_norm(diff)), finite=bool(torch.isfinite(v).all()))


def gate(actual, control, high):
    a, c = error(actual, high), error(control, high)
    per_metric = {name: a[name] <= c[name] for name in ("max_abs", "relative_l2")}
    return dict(candidate=a, python_fp32=c, per_metric_passed=per_metric,
                passed=a["finite"] and c["finite"] and all(per_metric.values()))


def phases_with_trace(shape, scale, device, dtype):
    phases, trace = [], {}
    for axis, n in zip(("row", "col"), shape):
        f = torch.arange(n, device=device, dtype=dtype)
        z = torch.zeros(n, device=device, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)
        for a in range(scale):
            angle = f * (-2 * math.pi * a / n)
            unit = torch.polar(torch.ones_like(angle), angle)
            z = z + unit
            trace[f"{axis}/angle{a}"] = angle
            trace[f"{axis}/polar{a}"] = unit
            trace[f"{axis}/sum{a}"] = z
        phases.append(z)
    return phases, trace


def stages(x, k, b, *, candidate=False):
    # Instrumentation only. Both outputs are checked against the independent,
    # uninstrumented functions above; no stage uses double for a FP32 candidate.
    s, eps = 2, 1e-5
    p = torch.nn.functional.interpolate(x, scale_factor=s, mode="nearest")
    t = {"prior": p}
    kh, kw = k.shape[-2:]
    psf = torch.nn.functional.pad(k, (0, p.shape[-1] - kw, 0, p.shape[-2] - kh))
    t["psf"] = torch.roll(psf, (-(kh // 2), -(kw // 2)), (-2, -1))
    t["K"] = torch.fft.fft2(t["psf"])
    t["power"] = t["K"].real.square() + t["K"].imag.square()
    t["Y"] = torch.fft.fft2(x)
    if candidate:
        (row, col), phase = phases_with_trace(p.shape[-2:], s, x.device, x.dtype)
        p_row = t["Y"].repeat(1, 1, s, s) * row[:, None]
        t["P"] = p_row * col[None, :]
    else:
        phase = {}
        t["P"] = torch.fft.fft2(p)
    t["lambda"] = torch.sigmoid(b - 9.0) + eps
    t["prediction_product"] = t["K"] * t["P"]
    t["alias_prediction"] = alias_mean(t["prediction_product"], s)
    t["alias_power"] = alias_mean(t["power"], s)
    t["numerator"] = t["Y"] - t["alias_prediction"]
    t["denominator"] = t["alias_power"] + t["lambda"]
    t["q"] = t["numerator"] / t["denominator"]
    t["q_tiled"] = t["q"].repeat(1, 1, s, s)
    t["correction"] = t["K"].conj() * t["q_tiled"]
    t["spectrum"] = t["P"] + t["correction"]
    t["complex_output"] = torch.fft.ifft2(t["spectrum"])
    t["output"] = t["complex_output"].real
    return t, phase


def build_inputs():
    generator = torch.Generator().manual_seed(41191)
    x = torch.randn(2, 3, 5, 7, generator=generator)
    # The original runner consumes this unused independent prior before k/b.
    torch.randn(2, 3, 10, 14, generator=generator)
    k = torch.rand(1, 1, 3, 3, generator=generator) / 9
    b = torch.randn(1, 3, 1, 1, generator=generator)
    impulse = torch.tensor([[[[1., 0.], [0., 0.]]]], dtype=torch.float32)
    identity = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    identity[..., 1, 1] = 1
    return {"historical_seed41191": (x, k, b),
            "minimal_identity_impulse": (impulse, identity, torch.zeros(1, 1, 1, 1))}


def encode_inputs(cases):
    return dict(format="route_b_nearest_f32_v1", candidate="nearest_spectral", scale=2, eps=1e-5,
                source_seed=41191,
                minimality="B=C=1; H=W=ceil(3/2)=2 is the minimum valid input for fixed s2/k3. "
                           "One nonzero x and one nonzero kernel coefficient; deterministic, no RNG. "
                           "This is not a claim of global minimality over other kernel sizes or formulas.",
                cases={name: {key: snapshot(value, True) for key, value in zip(("x", "weight", "bias"), values)}
                       for name, values in cases.items()})


def decode_inputs(packet):
    if packet["format"] != "route_b_nearest_f32_v1" or packet["scale"] != 2 or packet["eps"] != 1e-5:
        raise ValueError("Only the selected frozen s2 case is supported")
    result = {}
    for name, case in packet["cases"].items():
        values = []
        for key in ("x", "weight", "bias"):
            item = case[key]
            raw = base64.b64decode(item["base64_little_endian"], validate=True)
            if sha(raw) != item["sha256"] or item["dtype"] != "torch.float32":
                raise ValueError("Input hash or dtype mismatch")
            value = torch.tensor(struct.unpack("<" + "f" * (len(raw) // 4), raw), dtype=torch.float32).reshape(item["shape"])
            if raw_bytes(value) != raw:
                raise ValueError("Input byte reconstruction failed")
            values.append(value)
        result[name] = tuple(values)
    return result


def verify_legacy(root):
    algorithm = root / "research/algorithms.py"
    reference = root / "models/converse_core.py"
    current = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    ours = {n.name: ast.dump(n, include_attributes=False) for n in current.body if isinstance(n, ast.FunctionDef)}
    requested = {algorithm: ("check", "aliases", "kernel_fft", "spectral", "nearest_spectral"),
                 reference: ("alias_mean", "validate_inputs", "converse2d_reference")}
    result, modules = {}, {}
    for path, names in requested.items():
        data = path.read_bytes()
        tree = ast.parse(data.decode("utf-8"))
        old = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            if ours[name] != old[name]:
                raise RuntimeError(f"Frozen function AST mismatch: {name}")
        spec = importlib.util.spec_from_file_location("frozen_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path.stem] = module
        result[str(path)] = dict(sha256=sha(data), exact_function_asts=list(names))
    return result, modules


def analyze(name, values, device, legacy=None):
    x, k, b = [value.to(device) for value in values]
    control, _ = stages(x, k, b)
    actual, phase = stages(x, k, b, candidate=True)
    high, _ = stages(x.double(), k.double(), b.double())
    p = control["prior"]
    direct_a = nearest_spectral(x, p, k, b, 2, 1e-5)
    direct_c = converse2d_reference(x, p, k, b, 2, 1e-5)
    direct_h = converse2d_reference(x.double(), p.double(), k.double(), b.double(), 2, 1e-5)
    checks = dict(candidate_trace_exact=raw_bytes(actual["output"]) == raw_bytes(direct_a),
                  python_trace_exact=raw_bytes(control["output"]) == raw_bytes(direct_c),
                  independent_fp64_trace_exact=raw_bytes(high["output"]) == raw_bytes(direct_h))
    if legacy:
        checks["legacy_candidate_exact"] = raw_bytes(direct_a) == raw_bytes(legacy["algorithms"].nearest_spectral(x, p, k, b, 2, 1e-5))
        checks["legacy_python_exact"] = raw_bytes(direct_c) == raw_bytes(legacy["converse_core"].converse2d_reference(x, p, k, b, 2, 1e-5))
    if not all(checks.values()):
        raise RuntimeError(f"Instrumentation changed arithmetic: {name} {checks}")
    stage_rows = {}
    for label in control:
        stage_rows[label] = dict(candidate_vs_python=comparison(actual[label], control[label]),
                                 candidate_vs_fp64=error(actual[label], high[label]),
                                 python_vs_fp64=error(control[label], high[label]),
                                 candidate=snapshot(actual[label], name.startswith("minimal")),
                                 python_fp32=snapshot(control[label], name.startswith("minimal")))
    first = next((label for label, row in stage_rows.items() if not row["candidate_vs_python"]["byte_equal"]), None)
    _, high_phase = phases_with_trace(p.shape[-2:], 2, device, torch.float64)
    phase_rows = {label: dict(error_to_fp64=error(value, high_phase[label]),
                              versus_fp64_rounded_to_fp32=comparison(value, high_phase[label].to(value.dtype)),
                              values=snapshot(value, True)) for label, value in phase.items()}
    # Existing old solver's P argument provides a one-variable localization
    # intervention. It is diagnostic, not a new candidate or a proposed repair.
    control_p = spectral(x, p, k, b, 2, 1e-5, P=control["P"], Y=actual["Y"])
    injected_p = spectral(x, p, k, b, 2, 1e-5, P=actual["P"], Y=control["Y"])
    interventions = dict(replacing_only_P_with_original_HR_FFT_restores_control_bytes=raw_bytes(control_p) == raw_bytes(direct_c),
                         injecting_only_candidate_P_reproduces_candidate_bytes=raw_bytes(injected_p) == raw_bytes(direct_a))
    result = dict(shape=list(x.shape), output_shape=list(p.shape), scale=2, eps=1e-5,
                  output_gate=gate(direct_a, direct_c, direct_h), instrumentation=checks,
                  first_different_common_stage=first, stages=stage_rows,
                  phase_construction=phase_rows, single_variable_interventions=interventions)
    if name.startswith("minimal"):
        # Analytic DFT of [1,1,0,0], independent of FFT and trig libraries.
        exact_axis = torch.tensor([2, 1-1j, 0, 1+1j], dtype=torch.complex128, device=device)
        result["analytic_axis"] = dict(values=snapshot(exact_axis, True),
            candidate_row_error=error(phase["row/sum1"], exact_axis),
            candidate_col_error=error(phase["col/sum1"], exact_axis),
            row_components=components(phase["row/sum1"]).tolist(),
            col_components=components(phase["col/sum1"]).tolist())
        result["analytic_output_is_nearest_input"] = dict(
            python32_exact=raw_bytes(direct_c) == raw_bytes(p),
            fp64_exact=raw_bytes(direct_h) == raw_bytes(p.double()),
            candidate_exact=raw_bytes(direct_a) == raw_bytes(p),
            candidate_values=direct_a.detach().cpu().tolist(), reference_values=p.cpu().tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Replay exact bytes instead of regenerating seed input")
    parser.add_argument("--verify-legacy", action="store_true")
    parser.add_argument("--research-root", type=Path, default=Path(".build/roadmap-research"))
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("Explicit input packets require a little-endian host")
    inputs_path = args.output.with_name(args.output.stem + "_inputs.json")
    if args.output.exists() or inputs_path.exists():
        parser.error("Use fresh output names; old evidence must remain unchanged")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    identity, legacy = verify_legacy(args.research_root.resolve()) if args.verify_legacy else ({}, None)
    packet = json.loads(args.inputs.read_text(encoding="utf-8")) if args.inputs else encode_inputs(build_inputs())
    cases = decode_inputs(packet)
    if set(cases) != {"historical_seed41191", "minimal_identity_impulse"}:
        raise ValueError("This script diagnoses exactly the historical case and one minimal counterexample")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    report = dict(kind="one_route_b_counterexample", candidate="nearest_spectral", status="running",
                  scope="Forward-only diagnosis of one old candidate; no production change, repair, performance or VJP admission",
                  device=args.device, torch=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  script_sha256=sha(Path(__file__).read_bytes()), input_packet=str(inputs_path),
                  input_packet_sha256=sha(inputs_path.read_bytes()), old_source=identity,
                  minimality=packet["minimality"], cases={},
                  ulp_note="Componentwise ordered FP32 distance; signed zeros identified. Near-zero ULP distance must be read with absolute error.",
                  fp64_scope="Independent Python full-spectrum reference and diagnostic phase oracle only; the candidate remains FP32/complex64.")
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for name, values in cases.items():
                report["cases"][name] = analyze(name, values, args.device, legacy)
        report["frozen_sources_unchanged"] = all(sha(Path(path).read_bytes()) == item["sha256"] for path, item in identity.items())
        reproduced = all(not case["output_gate"]["passed"] for case in report["cases"].values())
        report.update(status="counterexample_reproduced" if reproduced else "counterexample_not_reproduced",
                      all_selected_output_gates_fail=reproduced, candidate_admitted=False)
    except Exception as exc:
        report.update(status="error", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": args.device,
                      "first_divergence": {name: c["first_different_common_stage"] for name, c in report["cases"].items()},
                      "output": str(args.output)}, indent=2))
    return 0 if report["all_selected_output_gates_fail"] and report["frozen_sources_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
