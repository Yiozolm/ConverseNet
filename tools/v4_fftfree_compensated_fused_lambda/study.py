"""Compensated FP32 nearest k2/s2 with fused lambda; full independent gate.

--self-check is CPU-only. GPU runs need a separately checked research build.
The numerical baseline is frozen half-spectrum FP32; FP64 is only an oracle.
No model is modified and no production route is registered by this experiment.
"""
import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import torch
from torch.nn import functional as F
from fp32_baseline import converse2d_fp32 as frozen_half
from numerical_policy import BUDGETS, BASELINE_SHA256, comparison, denominator_statistics
from models.converse_core import converse2d_reference as reference64
from inference import NearestK2Inference

SEEDS = (17, 29, 43)
MODES = ("no_grad", "inference_mode", "frozen_gradmode")
HISTORY_NAME = "synthetic/seed17/h7w5/kb1kc3/signed/transpose"
NEW_KINDS = ("dynamic", "near_zero_input", "cancellation")
ORIGINAL_INPUT_SHA256 = "2d191b5111fc4dfa52ff258190f5d2ad3aab423e91b6fd3e8765595e423f1df6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(value):
    dense = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "sha256": hashlib.sha256(dense.numpy().tobytes()).hexdigest()}


def layout(value, name):
    if name == "transpose":
        return value.transpose(-1, -2).contiguous().transpose(-1, -2)
    if name == "strided":
        return torch.stack((value, value), -1)[..., 0]
    return value.contiguous()


def case_specs():
    for seed in SEEDS:
        for h, w in ((1, 1), (5, 1), (5, 6), (7, 5)):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                for kind in ("softmax", "signed", "zero", "weak_1e-6", "weak_1e-3", *NEW_KINDS):
                    for storage in ("contiguous", "strided", "transpose"):
                        for mode in MODES:
                            yield dict(name=f"matrix/{seed}/{h}x{w}/kb{kb}kc{kc}/{kind}/{storage}/{mode}",
                                       seed=seed, shape=(2, 3, h, w), kb=kb, kc=kc,
                                       kind=kind, layout=storage, mode=mode, padding=0,
                                       eps=1e-8 if kind in ("zero", "weak_1e-6", "weak_1e-3") else 1e-5)
    # Exact seed arithmetic, value generation and strides from the old study.
    yield dict(name=HISTORY_NAME, seed=17, shape=(2, 3, 7, 5), kb=1, kc=3,
               kind="signed", layout="transpose", mode="inference_mode", padding=0,
               eps=1e-5, historical=True)
    # Explicit synthetic layer-sized callers, not captured activations or quality data.
    for h, w in ((24, 28), (48, 56)):
        yield dict(name=f"timing_fixture/B1C64/{h}x{w}/replicate_pad2", seed=17,
                   shape=(1, 64, h, w), kb=1, kc=64, kind="softmax",
                   layout="contiguous", mode="no_grad", padding=2, eps=1e-3,
                   timing_fixture=True)
    # Appended coverage only: the original 3459 cases retain their inputs/order.
    # Each kind uses the same fixed quantized weights across the bias/eps sweep.
    for kind in ("signed", "softmax"):
        for bias_value in (-40, -10, 0, 9, 20):
            for eps in (1e-5, 1e-8):
                for mode in MODES:
                    yield dict(name=f"bias_sweep/{kind}/bias{bias_value}/eps{eps}/{mode}",
                               seed=17, shape=(2, 3, 5, 6), kb=1, kc=3,
                               kind=kind, layout="contiguous", mode=mode, padding=0,
                               eps=eps, bias_value=bias_value, bias_sweep=True,
                               regime="weak" if bias_value <= -30 and eps == 1e-8 else "normal")


def materialize(spec, device):
    b, c, h, w = spec["shape"]
    gen = torch.Generator(device="cpu").manual_seed(spec["seed"] + 101*h + 17*w + spec["kb"] + spec["kc"])
    x = torch.randn(b, c, h, w, generator=gen)
    weight = torch.randn(spec["kb"], spec["kc"], 2, 2, generator=gen)
    bias = torch.randn(1, c, 1, 1, generator=gen) * .2
    kind = spec["kind"]
    if kind == "softmax":
        weight = weight.flatten(-2).softmax(-1).reshape_as(weight)
    elif kind in ("signed", "dynamic", "near_zero_input"):
        weight = weight * .25
        if kind == "dynamic":
            x = x * torch.logspace(-3, 3, w)
        elif kind == "near_zero_input":
            x = x * 1e-8
    elif kind == "cancellation":
        weight.zero_()
        weight[..., 1, 1] = 1
    else:
        weight = weight * (0.0 if kind == "zero" else float(kind.split("_")[1]))
        bias.fill_(-40)
        # Ordinary-amplitude inputs stress amplification under weak regularization.
    if spec.get("bias_sweep"):
        bias.fill_(float(spec["bias_value"]))
    return tuple(layout(value.to(device), spec["layout"]) for value in (x, weight, bias))


def mode_context(mode):
    return {"no_grad": torch.no_grad, "inference_mode": torch.inference_mode,
            "frozen_gradmode": torch.enable_grad}[mode]()


def regime_for(spec):
    # Existing fixtures keep their original classification. The new bias sweep
    # explicitly marks only a tiny sigmoid plus weak eps as weak regularization.
    return spec.get("regime", "weak" if spec["eps"] == 1e-8 else "normal")


def reference_solve(function, data, eps):
    x, weight, bias = data
    prior = F.interpolate(x, scale_factor=2, mode="nearest")
    return function(x, prior, weight, bias, 2, eps)


def prototype(data, eps):
    """Original FP32 disjoint residual expression, diagnostic only."""
    x, weight, bias = data
    predicted = torch.zeros_like(x)
    for a in range(2):
        for b in range(2):
            predicted = predicted + weight[..., a, b, None, None] * x
    regularizer = torch.sigmoid(bias - 9.0) + eps
    residual = (x - predicted) / (weight.square().sum((-2, -1), keepdim=True) + regularizer)
    output = torch.empty((*x.shape[:2], x.shape[-2]*2, x.shape[-1]*2), device=x.device, dtype=x.dtype)
    for a in range(2):
        for b in range(2):
            output[..., 1-a::2, 1-b::2] = x + weight[..., a, b, None, None] * residual
    return output


def output_check(actual, baseline, high, regime, *, dynamic_range=False):
    result = comparison(actual, baseline, high, regime=regime, distribution=True)
    smoke = bool(torch.isclose(actual, baseline, atol=1e-5, rtol=1e-5).all())
    smoke_is_gate = regime == "normal" and not dynamic_range
    result.update(smoke_1e_minus5=smoke, smoke_is_gate=smoke_is_gate,
                  dynamic_range=dynamic_range)
    result["passed"] = result["passed"] and (smoke or not smoke_is_gate)
    return result


def numeric_case(solve, spec, data):
    before = [record(value) for value in data]
    x, weight, bias = data
    pad = spec["padding"]
    regime = regime_for(spec)
    with mode_context(spec["mode"]):
        padded = F.pad(x, (pad,)*4, mode="replicate") if pad else x
        values = padded, weight, bias
        nearest_prior = F.interpolate(padded, scale_factor=2, mode="nearest")
        denominator = denominator_statistics((padded, nearest_prior, weight, bias),
                                             2, spec["eps"], padded.device)
        baseline = reference_solve(frozen_half, values, spec["eps"])
        high = reference_solve(reference64, tuple(v.double() for v in values), spec["eps"])
        actual = solve(*values, spec["eps"])
        dynamic_range = spec["kind"] == "dynamic"
        checks = {"padded_output": output_check(actual, baseline, high, regime,
                                                dynamic_range=dynamic_range)}
        if pad:
            crop = (..., slice(2*pad, -2*pad), slice(2*pad, -2*pad))
            checks["cropped_output"] = output_check(actual[crop], baseline[crop], high[crop], regime,
                                                     dynamic_range=dynamic_range)
        deterministic = record(actual) == record(solve(*values, spec["eps"]))
        diagnostic_equal = record(actual)["sha256"] == record(prototype(values, spec["eps"]))["sha256"]
    unchanged = before == [record(value) for value in data]
    history_matches = True
    if spec.get("historical"):
        old = json.loads((HERE / "historical_fixture.json").read_text(encoding="utf-8"))
        history_matches = before == old["historical_record"]["input_tensors"]
    return {**spec, "regime": regime, "inputs": before, "checks": checks, "denominator_statistics": denominator,
            "padded_lr_shape": list(padded.shape), "output": record(actual),
            "repeatable": deterministic, "inputs_unchanged": unchanged,
            "original_prototype_bytes_equal_diagnostic": diagnostic_equal,
            "historical_inputs_match": history_matches,
            "passed": all(v["passed"] for v in checks.values()) and deterministic and unchanged and history_matches}


def execution_contracts(solve):
    spec = dict(seed=17, shape=(2, 3, 5, 7), kb=1, kc=3, kind="softmax", layout="strided")
    data = materialize(spec, "cuda")
    rejected = {}
    for index in range(3):
        values = tuple(v.detach().requires_grad_(i == index) for i, v in enumerate(data))
        try:
            with torch.enable_grad():
                solve(*values)
        except RuntimeError as exc:
            rejected[f"grad_input_{index}"] = "differentiable GradMode" in str(exc)
        else:
            rejected[f"grad_input_{index}"] = False
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        try:
            with torch.no_grad():
                solve(*(v.to(dtype) for v in data))
        except ValueError as exc:
            rejected[str(dtype)] = "FP32" in str(exc)
        else:
            rejected[str(dtype)] = False
    try:
        with torch.no_grad(), torch.autocast("cuda"):
            solve(*data)
    except RuntimeError as exc:
        rejected["autocast"] = "autocast" in str(exc)
    else:
        rejected["autocast"] = False
    # no_grad/inference_mode must also accept leaves that would otherwise train.
    allowed = {}
    for mode in ("no_grad", "inference_mode"):
        leaves = tuple(v.detach().requires_grad_() for v in data)
        with mode_context(mode):
            out = solve(*leaves)
            expected = reference_solve(frozen_half, leaves, 1e-5)
            high = reference_solve(reference64, tuple(v.double() for v in leaves), 1e-5)
            allowed[mode] = output_check(out, expected, high, "normal")
            allowed[mode]["passed"] &= not out.requires_grad
    # Stream and graph lifetime checks use changed inputs, not a captured constant.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            solve(*data)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
            graph_output = solve(*data)
        with torch.no_grad():
            data[0].add_(.02)
            data[1].mul_(.9)
            data[2].add_(.1)
            graph.replay()
            torch.cuda.synchronize()
            expected = reference_solve(frozen_half, data, 1e-5)
            high = reference_solve(reference64, tuple(v.double() for v in data), 1e-5)
            graph_check = output_check(graph_output, expected, high, "normal")
            graph_check["same_as_eager"] = record(graph_output) == record(solve(*data))
            graph_check["passed"] &= graph_check["same_as_eager"]
    finally:
        graph.reset()
    return dict(rejected=rejected, disabled_grad_inputs=allowed, graph=graph_check,
                passed=all(rejected.values()) and all(v["passed"] for v in allowed.values()) and graph_check["passed"])


def caller(solve, data, spec):
    x, weight, bias = data
    pad = spec["padding"]
    padded = F.pad(x, (pad,)*4, mode="replicate") if pad else x
    out = solve(padded, weight, bias, spec["eps"])
    return out[..., 2*pad:-2*pad, 2*pad:-2*pad] if pad else out


def benchmark(solve, fixture_specs, args):
    # Independent PyBind research module may coexist with one production
    # TORCH_LIBRARY. The production loader must verify an already-built binary.
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    spec = importlib.util.spec_from_file_location("fftfree_release_loader", ROOT / "test/extension_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    loader.load_extension()
    manifest = json.loads((ROOT / ".build/cuda/source_manifest.json").read_text(encoding="utf-8"))
    sources = loader.production_source_hashes()
    def production(x, weight, bias, eps):
        prior = F.interpolate(x, scale_factor=2, mode="nearest")
        return torch.ops.converse2d.forward(x, prior, weight, bias, 2, eps)
    prepared = [(fixture, materialize(fixture, "cuda")) for fixture in fixture_specs]
    # Both sides' exact timing fixtures must pass before any timings start.
    control_checks = [numeric_case(production, fixture, data) for fixture, data in prepared]
    if not all(row["passed"] for row in control_checks):
        return dict(passed=False, status="blocked_by_production_timing_fixture_gate", controls=control_checks)
    results = []
    for fixture, data in prepared:
        routes = {"production_uncached": lambda: caller(production, data, fixture),
                  "fftfree": lambda: caller(solve, data, fixture)}
        rows = {name: [] for name in routes}
        orders = []
        with torch.no_grad():
            for call in routes.values():
                for _ in range(args.warmup):
                    torch.ops.converse2d.clear_cache()
                    call()
            for round_index in range(args.rounds):
                order = list(routes) if round_index % 2 == 0 else list(reversed(routes))
                orders.append(order)
                for name in order:
                    samples = []
                    for _ in range(args.iters):
                        # Cache clearing is untimed; all actual preparation,
                        # pad/prior/crop, allocation and dispatch are timed.
                        torch.ops.converse2d.clear_cache()
                        torch.cuda.synchronize()
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        tick = time.perf_counter()
                        start.record()
                        output = routes[name]()
                        end.record()
                        end.synchronize()
                        samples.append(dict(wall_ms=(time.perf_counter()-tick)*1000,
                                            cuda_ms=start.elapsed_time(end),
                                            peak_extra_allocated_bytes=max(0, torch.cuda.max_memory_allocated()-base)))
                        del output
                    rows[name].append({key: statistics.mean(v[key] for v in samples) for key in samples[0]})
        medians = {name: {key: statistics.median(v[key] for v in values) for key in values[0]}
                   for name, values in rows.items()}
        results.append(dict(fixture=fixture, inputs=[record(v) for v in data], orders=orders,
                            rounds=rows, median=medians,
                            wall_speedup=medians["production_uncached"]["wall_ms"]/medians["fftfree"]["wall_ms"],
                            cuda_speedup=medians["production_uncached"]["cuda_ms"]/medians["fftfree"]["cuda_ms"]))
    if sources != loader.production_source_hashes() or sha(ROOT / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
        raise RuntimeError("Production source or binary changed during timing")
    return dict(passed=True, status="complete", checked_production_manifest=manifest,
                production_source_sha256=sources, controls=control_checks, cases=results,
                scope="Complete explicit nearest caller; fresh preparation per call, no kernel cache reuse. "
                      "GPU resident synthetic inputs; no model, dataset, training or convergence claim.")


def affinity():
    if os.name != "nt":
        return sorted(os.sched_getaffinity(0))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(current), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return hex(current.value)


def source_identity():
    paths = [*HERE.glob("*.py"), *HERE.glob("*.cu"), *HERE.glob("*.cpp"),
             HERE / "historical_fixture.json", HERE / "candidate_origin.json",
             ROOT / "test/numerical_policy.py", ROOT / "test/fp32_baseline.py",
             ROOT / "test/extension_loader.py", ROOT / "models/converse_core.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def self_check():
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU self-check requires a process without initialized CUDA")
    torch.set_num_threads(1)
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    specs = list(case_specs())
    if len(specs) != 3519 or len({s["name"] for s in specs}) != len(specs):
        raise AssertionError("Incomplete or duplicated gate matrix")
    for selected in specs[3459:]:
        expected_regime = "weak" if selected["bias_value"] <= -30 and selected["eps"] == 1e-8 else "normal"
        if selected.get("regime") != expected_regime or regime_for(selected) != expected_regime:
            raise AssertionError("Bias-sweep regularization regime is mislabeled")
    original_fingerprint = hashlib.sha256()
    original_count = 0
    for selected in specs:
        if selected["kind"] not in NEW_KINDS and not selected.get("bias_sweep"):
            original_count += 1
            original_fingerprint.update(json.dumps({"spec": selected,
                "inputs": [record(v) for v in materialize(selected, "cpu")]}, sort_keys=True).encode())
    if original_count != 2163 or original_fingerprint.hexdigest() != ORIGINAL_INPUT_SHA256:
        raise AssertionError("Existing 2163-case input values/strides/specifications changed")
    old = json.loads((HERE / "historical_fixture.json").read_text(encoding="utf-8"))
    origin = json.loads((HERE / "candidate_origin.json").read_text(encoding="utf-8"))
    previous_fingerprint = hashlib.sha256()
    for selected in specs[:3459]:
        if selected.get("bias_sweep"):
            raise AssertionError("Bias sweep must follow the original 3459 cases")
        previous_fingerprint.update(json.dumps({"spec": selected,
            "inputs": [record(v) for v in materialize(selected, "cpu")]}, sort_keys=True).encode())
    if previous_fingerprint.hexdigest() != origin["original_3459_inputs_sha256"]:
        raise AssertionError("Previous 3459-case input/specification identity changed")
    if sha(HERE / "kernel.cu") in (old["original_source_sha256"]["kernel.cu"], origin["source_sha256"]["kernel.cu"]):
        raise AssertionError("Expected a new compensated kernel, not either historical candidate")
    kernel_source = (HERE / "kernel.cu").read_text(encoding="utf-8")
    device_source = kernel_source.split("// Host scalar boundary only:", 1)[0]
    if re.search(r"\b(?:double|__d(?:add|sub|mul|div|fma)\w*|kDouble)\b", device_source):
        raise AssertionError("Non-FP32 device arithmetic detected")
    if '"--fmad=false"' not in (HERE / "loader.py").read_text(encoding="utf-8"):
        raise AssertionError("The conservative CUDA compile flag is missing")
    history = next(spec for spec in specs if spec.get("historical"))
    data = materialize(history, "cpu")
    if [record(v) for v in data] != old["historical_record"]["input_tensors"]:
        raise AssertionError("Historical CPU fixture values/strides do not match original evidence")
    from cpu_compensated import cpu_solve
    names = (history["name"], "matrix/43/1x1/kb1kc1/weak_1e-6/contiguous/no_grad",
             "matrix/43/5x6/kb1kc1/signed/contiguous/no_grad",
             *(f"matrix/17/7x5/kb1kc3/{kind}/contiguous/no_grad" for kind in NEW_KINDS),
             *(s["name"] for s in specs if s.get("bias_sweep") and s["mode"] == "no_grad"))
    diagnostics = []
    with torch.no_grad():
        for selected in (next(s for s in specs if s["name"] == name) for name in names):
            values = materialize(selected, "cpu")
            actual = cpu_solve(values, selected["eps"])
            baseline = reference_solve(frozen_half, values, selected["eps"])
            high = reference_solve(reference64, tuple(v.double() for v in values), selected["eps"])
            diagnostic = output_check(actual, baseline, high, regime_for(selected),
                                      dynamic_range=selected["kind"] == "dynamic")
            nearest_prior = F.interpolate(values[0], scale_factor=2, mode="nearest")
            denominator = denominator_statistics((values[0], nearest_prior, *values[1:]),
                                                 2, selected["eps"], "cpu")
            if selected["kind"] == "dynamic" and diagnostic["smoke_is_gate"]:
                raise AssertionError("Dynamic-range smoke must remain diagnostic")
            if selected["kind"] == "cancellation" and not torch.equal(actual, nearest_prior):
                raise AssertionError("Unit-kernel cancellation fixture must return exact nearest(x)")
            diagnostics.append(dict(name=selected["name"], compensated=diagnostic,
                                    denominator_statistics=denominator))
    # The CPU diagnostic does not grant CUDA admission or require a known
    # historical failure to become a success on a different FFT backend.
    if actual.shape != high.shape or not torch.isfinite(actual).all():
        raise AssertionError("Bad prototype fixture")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-check initialized CUDA")
    print(json.dumps(dict(status="cpu_self_check_passed", cases=len(specs), historical_inputs_match=True,
                          original_2163_inputs_sha256=original_fingerprint.hexdigest(),
                          original_3459_inputs_sha256=previous_fingerprint.hexdigest(),
                          cuda_initialized=False, cpu_diagnostics=diagnostics,
                          gpu_admission=False), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.artifacts is None or args.output is None:
        parser.error("GPU gate requires --artifacts and --output")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output filename; old evidence is immutable")
    if min(args.warmup, args.iters) < 1 or args.rounds < 2:
        parser.error("positive warmup/iters and at least two alternating rounds required")
    if os.environ.get("CONVERSE2D_BACKEND") or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        raise RuntimeError("Unset backend and CPU-only overrides")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    from loader import load_checked, build_identity
    report = dict(kind="v4_explicit_nearest_fftfree_compensated_fused_lambda", status="running", passed=False,
                  candidate="Prior compensated FP32 solver plus local RN sigmoid/eps; no ATen sigmoid bit-identity claim",
                  source_sha256=source_identity(), budgets=BUDGETS, baseline_sha256=BASELINE_SHA256,
                  baseline="Frozen Python FP32 half-spectrum inference on identical quantized FP32 values",
                  reference="Independent Python full-spectrum FP64; no FP64 candidate backend",
                  timing_status="not requested" if not args.timing else "awaiting complete gate",
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        extension, manifest = load_checked(args.artifacts, build=False)
        solve = NearestK2Inference(extension)
        report.update(checked_research_manifest=manifest,
                      environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                       gpu=torch.cuda.get_device_name(), tf32=False, amp=False,
                                       affinity=affinity(), torch_threads=torch.get_num_threads(),
                                       deterministic_algorithms=True, cudnn_benchmark=False))
        specs = list(case_specs())
        for index, spec in enumerate(specs):
            row = numeric_case(solve, spec, materialize(spec, "cuda"))
            report["cases"].append(row)
            if not row["passed"]:
                print("FAIL", row["name"], flush=True)
            if (index+1) % 100 == 0:
                save()
                print(f"Checked {index+1}/{len(specs)}", flush=True)
        report["contracts"] = execution_contracts(solve)
        report["passed"] = all(row["passed"] for row in report["cases"]) and report["contracts"]["passed"]
        if args.timing and report["passed"]:
            report["timing"] = benchmark(solve, [s for s in specs if s.get("timing_fixture")], args)
            report["timing_status"] = report["timing"]["status"]
            report["passed"] &= report["timing"]["passed"]
        elif args.timing:
            report["timing_status"] = "blocked by complete numerical/contract gate"
        if report["source_sha256"] != source_identity() or build_identity() != manifest["identity"]:
            raise RuntimeError("Experiment source/toolchain changed during gate")
        if sha(manifest["binary"]) != manifest["binary_sha256"]:
            raise RuntimeError("Experiment binary changed during gate")
        report["status"] = "complete"
        report["failed_cases"] = [row["name"] for row in report["cases"] if not row["passed"]]
    except Exception as error:
        report.update(status="error", passed=False,
                      error=dict(type=type(error).__name__, message=str(error)))
        save()
        raise
    save()
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]),
                          failed_cases=report["failed_cases"], timing_status=report["timing_status"]), indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
