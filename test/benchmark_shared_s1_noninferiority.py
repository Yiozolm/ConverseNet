"""New numerical noninferiority protocol against original Python FP32.

Same immutable native s1_module/seed17 fixture as earlier probes. Evaluate the
original Python FP32 baseline and every candidate against ONE independent FP64
full-FFT oracle. For each output/dx/dw/db, require both max_abs and relative L2
to be <= Python's corresponding error, with NO added tolerance or percentage.
The old pointwise differences and FP32 representation floor are diagnostic only.

Only individually passing routes may be timed. Production failing does not
block an independently passing candidate. Python FP32 and mathematically
different native dense are local timing baselines; no absent/historical route
is used as a denominator. No training-quality or production-release claim.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL_PATH = ROOT / "experiments/training_shared_s1/python_fp32_noninferiority_protocol.json"
TENSOR_NAMES = ("output", "dx", "dw", "db")
METRIC_NAMES = ("max_abs", "relative_l2")


def diagnostics(actual, expected):
    from diagnose_boundary_precision import comparison
    if set(actual) != set(TENSOR_NAMES) or set(expected) != set(TENSOR_NAMES):
        raise RuntimeError("Numerical comparison omitted a required output or VJP")
    return {name: comparison(actual[name], expected[name],
                            *((3e-5, 3e-5) if name == "output" else (5e-5, 5e-5)))
            for name in TENSOR_NAMES}


def noninferiority(actual_metrics, baseline_metrics):
    tensors = {}
    for name in TENSOR_NAMES:
        actual, baseline = actual_metrics[name], baseline_metrics[name]
        finite = actual["finite"] and baseline["finite"]
        checked = {metric: dict(candidate=actual[metric], python_fp32=baseline[metric],
                                 passed=bool(finite and actual[metric] <= baseline[metric]))
                   for metric in METRIC_NAMES}
        tensors[name] = dict(passed=all(row["passed"] for row in checked.values()),
                             finite=finite, metrics=checked)
    return dict(passed=all(row["passed"] for row in tensors.values()), tensors=tensors,
                added_tolerance=0, relative_margin=0)


def validate(config, tensors, routes):
    import torch
    from benchmark_deconv_target_final import production_method
    from probe_pointwise_training import capture
    oracle = capture(tensors, torch.float64, production_method(config, reference=True))
    python = capture(tensors, torch.float32, routes["python_fp32"][0])
    if not all(bool(torch.isfinite(value).all()) for value in (*oracle.values(), *python.values())):
        raise RuntimeError("Oracle or Python baseline is nonfinite; noninferiority cannot be defined")
    baseline_metrics = diagnostics(python, oracle)
    floor = diagnostics({name: value.float() for name, value in oracle.items()}, oracle)
    result = {}
    for name, (method, family) in routes.items():
        if name == "native_dense":
            continue
        actual = python if name == "python_fp32" else capture(family, torch.float32, method)
        metrics = diagnostics(actual, oracle)
        result[name] = dict(noninferiority=noninferiority(metrics, baseline_metrics),
                            fp64_error=metrics, pointwise_vs_python_diagnostic=diagnostics(actual, python))
        print(json.dumps(dict(route=name, numerical_noninferiority=result[name]["noninferiority"]["passed"])), flush=True)
    return dict(python_fp32_error=baseline_metrics, fp32_representation_floor_diagnostic=floor, routes=result)


def summarize(rounds):
    names = list(rounds[0]["variants"])
    medians = {name: {metric: statistics.median(row["variants"][name][metric] for row in rounds)
                      for metric in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in names}
    ratios = {}
    for name in names:
        pairs = [("python_over_route", "python_fp32", name), ("route_over_native_dense", name, "native_dense")]
        if "production_fft" in names:
            pairs.append(("production_over_route", "production_fft", name))
        ratios[name] = {}
        for metric in ("wall_ms", "cuda_event_ms"):
            ratios[name][metric] = {}
            for label, numerator, denominator in pairs:
                values = [row["variants"][numerator][metric] / row["variants"][denominator][metric] for row in rounds]
                ratios[name][metric][label] = dict(per_round=values, median=statistics.median(values),
                                                  min=min(values), max=max(values))
    return dict(medians=medians, paired_ratios=ratios, timed_routes=names,
                production_denominator_available="production_fft" in names,
                ratio_scope="Only actually timed same-round routes; no historical or excluded denominator. Native compares different mathematics.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/native_deconv_target/native_baseline.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/shared_s1_cuda_noninferiority.json")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with FP32 execution")
    import torch
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    from benchmark_converse_candidates import audit_native_baseline
    from benchmark_deconv_target_final import production_method, native_method
    from benchmark_shared_s1_cuda import shared_method, check_route
    from experiments.training_shared_s1.loader import load
    from probe_pointwise_training import timed_fixture
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    baseline_audit = audit_native_baseline(args.baseline)
    original = json.loads(args.baseline.read_text(encoding="utf-8"))
    config = native.CASES["s1_module"]
    stored = next(row for row in original["cases"] if row["name"] == "s1_module")
    cpu = native.cpu_data(config, seed=17)
    full_hash = worker.tensor_hash(cpu)
    input_hash = worker.tensor_hash({key: cpu[key] for key in ("x", "upstream")})
    if (worker.json_safe(config) != stored["config"] or full_hash != stored["complete_fixture_sha256"]
            or input_hash != stored["input_upstream_sha256"]):
        raise RuntimeError("Original native fixture changed")
    frozen_path = ROOT / "artifacts/native_deconv_target/source_before/manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if worker.file_hash(ROOT / "models/converse_core.py") != frozen["models/converse_core.py"]:
        raise RuntimeError("Original full-FFT reference source changed")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.variant = "current"
    _, current_build = worker.load_backend(args)
    ops, candidate_build = load(verbose=args.verbose_build)
    channels = config["shape"][1]
    tensors = (cpu["x"], cpu["weight"], cpu["bias"].reshape(1, channels, 1, 1), cpu["upstream"])
    native_tensors = (cpu["x"], cpu["dense_weight"], cpu["bias"], cpu["upstream"])
    routes = dict(production_fft=(production_method(config), tensors),
                  cuda_shared_prep64=(shared_method(ops, config, "fp64"), tensors),
                  cuda_shared_prep32=(shared_method(ops, config, "fp32"), tensors),
                  python_fp32=(production_method(config, reference=True), tensors),
                  native_dense=(native_method(config), native_tensors))
    sources = worker.source_hashes()
    for name in ("benchmark_shared_s1_noninferiority.py", "benchmark_shared_s1_cuda.py", "benchmark_native_deconv.py",
                 "benchmark_converse_candidates.py", "benchmark_deconv_target_final.py", "probe_converse_boundaries.py",
                 "probe_pointwise_training.py", "diagnose_boundary_precision.py", "validate_shared_s1_python_fp32.py"):
        sources["test/" + name] = worker.file_hash(ROOT / "test" / name)
    for name in ("loader.py", "bindings.cpp", "kernels.cu", PROTOCOL_PATH.name):
        path = "experiments/training_shared_s1/" + name
        sources[path] = worker.file_hash(ROOT / path)
    report = dict(status="validating_numerical_noninferiority", protocol=protocol, protocol_sha256=worker.file_hash(PROTOCOL_PATH),
                  settings=worker.json_safe(vars(args)), fixed_config=config, fixed_seed=17,
                  complete_fixture_sha256=full_hash, input_upstream_sha256=input_hash,
                  source_sha256=sources, current_build=current_build, candidate_build=candidate_build,
                  baseline_audit=baseline_audit, production_eligible=False, training_quality_verified=False,
                  reference_manifest_sha256=worker.file_hash(frozen_path), scope=__doc__,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True,
                                   deterministic_algorithms=False, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
                  memory_scope="Total PyTorch allocated/reserved peak of one resident fixture, not process/driver VRAM",
                  route_checks={}, validation={}, timed_routes=[], excluded_routes=[], rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        report["validation"] = validate(config, tensors, routes)
        checked = report["validation"]["routes"]
        report["timed_routes"] = [name for name in routes if name == "native_dense" or checked[name]["noninferiority"]["passed"]]
        report["excluded_routes"] = [dict(route=name, reason="Numerical noninferiority failed; diagnostics retained; not timed")
                                      for name in routes if name not in report["timed_routes"]]
        report["all_converse_routes_noninferior"] = all(row["noninferiority"]["passed"] for row in checked.values())
        report["eligible_cuda_candidates"] = [name for name in report["timed_routes"] if name.startswith("cuda_shared_")]
        worker.write_json(args.output, report)
        # Check only numerically eligible routes, so excluded controls cannot block others.
        report["route_checks"] = {name: check_route(name, routes[name][1], routes[name][0], config["shape"])
                                  for name in report["timed_routes"]}
        report["status"] = "timing_selected_noninferior_routes_and_local_baselines"
        names = report["timed_routes"]
        for index in range(args.rounds):
            order = names[index % len(names):] + names[:index % len(names)]
            values = {name: timed_fixture(routes[name][1], routes[name][0], args) for name in order}
            report["rounds"].append(dict(round=index + 1, order=order, variants=values))
            report["summary"] = summarize(report["rounds"])
            worker.write_json(args.output, report)
            print(json.dumps(dict(round=index + 1, variants=values)), flush=True)
        report["source_unchanged"] = all(worker.file_hash(ROOT / name) == digest for name, digest in sources.items())
        if not report["source_unchanged"]:
            raise RuntimeError("Source changed during measurement")
        report["status"] = ("complete_local_numerical_noninferiority_comparison" if report["eligible_cuda_candidates"]
                            else "complete_local_baselines_no_eligible_cuda_candidate")
        worker.write_json(args.output, report)
        return 0 if report["eligible_cuda_candidates"] else 2
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
