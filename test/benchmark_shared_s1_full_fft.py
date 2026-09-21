"""Local ATen full-FFT shared-H timing under Python FP32 noninferiority.

Reuse the original native s1_module seed17 fixture without changing eps, values
or gradient targets. Four probe_shared_s1_full_fft routes retain native circular
pad2/slice crop2 and their original differentiable kernel preparation. They use
ATen full complex FFTs and automatic backward; no new handwritten CUDA kernel,
torch.compile, CUDA Graph, cache across calls or production source patch.

The unchanged noninferiority protocol compares each output/dx/dw/db max_abs and
relative L2 with original Python FP32 against the same FP64 full-FFT oracle.
There is no margin or added tolerance. Original pointwise budgets remain only
diagnostics. Independently exclude failing routes before local timing; retain
all failures. Production control, Python FP32 and native dense are measured in
the same rotated rounds when eligible. Native dense is a different equation.

Default six rounds x thirty complete FWD+three-VJP calls, warm5 per route;
one resident GPU fixture, all pad/crop/preparation/FFT costs included. This is
operator evidence, not full-model training quality, convergence or eligibility.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANDIDATES = ("full_h32", "full_h64_cast", "full_h64_product", "full_h64_lambda64")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/native_deconv_target/native_baseline.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/shared_s1_full_fft_benchmark.json")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with FP32 execution")
    import torch
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    import probe_shared_s1_full_fft as candidate
    from benchmark_converse_candidates import audit_native_baseline
    from benchmark_deconv_target_final import production_method, native_method
    from benchmark_shared_s1_noninferiority import PROTOCOL_PATH, validate, summarize
    from benchmark_shared_s1_cuda import check_route
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
        raise RuntimeError("Original native fixture/configuration changed")
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
    channels = config["shape"][1]
    tensors = (cpu["x"], cpu["weight"], cpu["bias"].reshape(1, channels, 1, 1), cpu["upstream"])
    native_tensors = (cpu["x"], cpu["dense_weight"], cpu["bias"], cpu["upstream"])
    routes = dict(production_fft=(production_method(config), tensors))
    routes.update({name: (candidate.method(name, padding=config["outer_padding"], eps=config["eps"]), tensors)
                   for name in CANDIDATES})
    routes.update(python_fp32=(production_method(config, reference=True), tensors),
                  native_dense=(native_method(config), native_tensors))
    sources = worker.source_hashes()
    for name in ("benchmark_shared_s1_full_fft.py", "probe_shared_s1_full_fft.py", "benchmark_shared_s1_noninferiority.py",
                 "benchmark_shared_s1_cuda.py", "benchmark_native_deconv.py", "benchmark_converse_candidates.py",
                 "benchmark_deconv_target_final.py", "probe_pointwise_training.py", "diagnose_boundary_precision.py"):
        sources["test/" + name] = worker.file_hash(ROOT / "test" / name)
    sources[PROTOCOL_PATH.relative_to(ROOT).as_posix()] = worker.file_hash(PROTOCOL_PATH)
    report = dict(status="validating_numerical_noninferiority", scope=__doc__, protocol=protocol,
                  protocol_sha256=worker.file_hash(PROTOCOL_PATH), settings=worker.json_safe(vars(args)),
                  fixed_config=config, fixed_seed=17, complete_fixture_sha256=full_hash, input_upstream_sha256=input_hash,
                  source_sha256=sources, current_build=current_build, baseline_audit=baseline_audit,
                  reference_manifest_sha256=worker.file_hash(frozen_path), production_eligible=False,
                  training_quality_verified=False, candidate_backend="ATen full C2C FFT and automatic backward",
                  new_handwritten_cuda_kernels=False, compiled=False, cuda_graphs=False,
                  boundary="Native circular pad2 and two-slice crop2 for every Converse route, inside timing",
                  lambda_policy=dict(full_h32="FP32 sigmoid and H", full_h64_cast="FP32 sigmoid then FP64 H, H rounded before product",
                                     full_h64_product="FP32 sigmoid then FP64 H and product, product rounded before IFFT",
                                     full_h64_lambda64="Explicit FP64 sigmoid and H, H rounded before FP32 product; unchanged bias/eps"),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True,
                                   deterministic_algorithms=False, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
                  memory_scope="Total PyTorch allocated/reserved peak of one fixture, not process/driver VRAM",
                  route_checks={}, validation={}, timed_routes=[], excluded_routes=[], rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        # Reuse the declared gate directly; no monkeypatch or relabelled old report.
        report["validation"] = validate(config, tensors, routes)
        checked = report["validation"]["routes"]
        report["timed_routes"] = [name for name in routes if name == "native_dense" or checked[name]["noninferiority"]["passed"]]
        report["excluded_routes"] = [dict(route=name, reason="Numerical noninferiority failed; not timed; diagnostics retained")
                                      for name in routes if name not in report["timed_routes"]]
        report["all_converse_routes_noninferior"] = all(row["noninferiority"]["passed"] for row in checked.values())
        report["eligible_full_fft_candidates"] = [name for name in CANDIDATES if name in report["timed_routes"]]
        worker.write_json(args.output, report)
        report["route_checks"] = {name: check_route(name, routes[name][1], routes[name][0], config["shape"])
                                  for name in report["timed_routes"]}
        report["status"] = "timing_selected_noninferior_full_fft_routes_and_local_baselines"
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
        report["status"] = ("complete_local_full_fft_noninferiority_comparison" if report["eligible_full_fft_candidates"]
                            else "complete_local_baselines_no_eligible_full_fft_candidate")
        worker.write_json(args.output, report)
        return 0 if report["eligible_full_fft_candidates"] else 2
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
