"""Fixed s1 module: production FFT, two shared-input transfer paths, native dense.

Reuse native_baseline's seed17 B4/C128/96x96 s1_module fixture and eps=1e-5.
Converse includes circular pad2, the entire solver and crop2 in forward/VJP.
The transfer paths use probe_shared_s1_transfer.transfer_core('transfer32'),
with native pad/slice or cat-pad/as_strided-crop. The prior is the same padded x.
Native dense ConvTranspose2d is a different equation and zero-padded boundary;
its additive bias and larger weight tensor are not a quality-equivalent solver.

All Converse routes must pass independent full-FFT FP64 output 3e-5/3e-5 and
dx/dw/db 5e-5/5e-5 before ANY timing. Six rotated rounds, thirty complete FWD+
three-VJP calls after five warmups per route. One GPU fixture is resident at a
time. Eager FP32, TF32 off, cuDNN deterministic on, global deterministic off,
no AMP, profiler, CUDA Graph, loss or optimizer. This is not whole-network time.
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


def transfer_method(config, cat=False):
    import torch.nn.functional as F
    from probe_converse_boundaries import pad_cat, crop_view
    from probe_shared_s1_transfer import transfer_core
    padding, eps = config["outer_padding"], config["eps"]

    def forward(x, weight, bias):
        padded = pad_cat(x, padding) if cat else F.pad(x, (padding,) * 4, mode="circular")
        output = transfer_core(padded, weight, bias, eps, "transfer32")
        return crop_view(output, padding) if cat else output[..., padding:-padding, padding:-padding]
    return forward


def inspect_route(name, tensors, method, expected_shape):
    import torch
    from benchmark_python_training import graph_has_spectral
    from probe_pointwise_training import clear_cuda, fixture
    clear_cuda()
    inputs, run = fixture(tensors, torch.float32, method)
    values = run()
    spectral = graph_has_spectral(values["output"])
    finite = bool(torch.stack([torch.isfinite(value).all() for value in values.values()]).all())
    shapes = {key: list(value.shape) for key, value in values.items()}
    if (shapes["output"] != list(expected_shape) or not finite
            or spectral != (name == "production_fft")
            or any(value.dtype != torch.float32 for value in values.values())
            or any(value.grad is not None for value in inputs)
            or any(values[key].shape != value.shape for key, value in zip(("dx", "dw", "db"), inputs))):
        raise RuntimeError(f"{name}: output/VJP shape, FP32, finite or backend route check failed")
    result = dict(passed=True, spectral_solve=spectral, shapes=shapes,
                  gradient_targets=["x", "weight", "bias"], finite=True,
                  parameter_elements=dict(weight=inputs[1].numel(), bias=inputs[2].numel()))
    del values, inputs, run
    clear_cuda()
    return result


def validate(config, tensors, routes):
    import torch
    from benchmark_deconv_target_final import production_method
    from probe_pointwise_training import capture
    from probe_training_s1_shapes import metrics
    expected = capture(tensors, torch.float64, production_method(config, reference=True))
    result = {}
    for name, (method, family) in routes.items():
        if name == "native_dense":
            continue  # A different equation, checked for shape/finite separately.
        actual = capture(family, torch.float32, method)
        checked = {key: metrics(actual[key], value, output=key == "output", weak=False)
                   for key, value in expected.items()}
        result[name] = dict(passed=all(row["passed"] for row in checked.values()), tensors=checked)
    return result


def summarize(rounds, names):
    medians = {name: {key: statistics.median(row["variants"][name][key] for row in rounds)
                      for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in names}
    ratios = {}
    for name in names:
        ratios[name] = {}
        for metric in ("wall_ms", "cuda_event_ms"):
            values = {}
            for label, numerator, denominator in (("production_over_route", "production_fft", name),
                                                  ("route_over_native_dense", name, "native_dense")):
                paired = [row["variants"][numerator][metric] / row["variants"][denominator][metric]
                          for row in rounds]
                values[label] = dict(per_round=paired, median=statistics.median(paired),
                                     min=min(paired), max=max(paired))
            ratios[name][metric] = values
    return dict(medians=medians, paired_ratios=ratios,
                ratio_scope="Same-round named denominators only; native ratios compare different mathematics; no historical ratios multiplied")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/native_deconv_target/native_baseline.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/shared_s1_transfer_benchmark.json")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with the FP32 benchmark")
    import torch
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    from benchmark_converse_candidates import audit_native_baseline
    from benchmark_deconv_target_final import production_method, native_method
    from probe_pointwise_training import timed_fixture
    baseline_audit = audit_native_baseline(args.baseline)
    original = json.loads(args.baseline.read_text(encoding="utf-8"))
    config = native.CASES["s1_module"]
    stored = next(row for row in original["cases"] if row["name"] == "s1_module")
    cpu = native.cpu_data(config, seed=17)
    full_hash = worker.tensor_hash(cpu)
    input_hash = worker.tensor_hash({key: cpu[key] for key in ("x", "upstream")})
    if (worker.json_safe(config) != stored["config"] or full_hash != stored["complete_fixture_sha256"]
            or input_hash != stored["input_upstream_sha256"]):
        raise RuntimeError("Original native fixture changed: do not adjust eps, weights or upstream")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.variant = "current"
    _, build = worker.load_backend(args)
    channels = config["shape"][1]
    tensors = (cpu["x"], cpu["weight"], cpu["bias"].reshape(1, channels, 1, 1), cpu["upstream"])
    native_tensors = (cpu["x"], cpu["dense_weight"], cpu["bias"], cpu["upstream"])
    routes = dict(production_fft=(production_method(config), tensors),
                  transfer32_native=(transfer_method(config), tensors),
                  transfer32_cat_view=(transfer_method(config, cat=True), tensors),
                  native_dense=(native_method(config), native_tensors))
    sources = worker.source_hashes()
    for name in ("benchmark_shared_s1_transfer.py", "benchmark_native_deconv.py", "benchmark_converse_candidates.py",
                 "benchmark_deconv_target_final.py", "probe_shared_s1_transfer.py", "diagnose_boundary_precision.py",
                 "probe_converse_boundaries.py", "probe_pointwise_training.py", "probe_training_s1_shapes.py",
                 "benchmark_python_training.py"):
        sources["test/" + name] = worker.file_hash(ROOT / "test" / name)
    report = dict(status="validating", scope=__doc__, settings=worker.json_safe(vars(args)),
                  fixed_config=config, fixed_seed=17, complete_fixture_sha256=full_hash,
                  input_upstream_sha256=input_hash, source_sha256=sources, current_build=build,
                  baseline_audit=baseline_audit, production_eligible=False,
                  precision_reference="Independent full-FFT FP64 Converse with native circular pad2/crop2; unchanged budgets",
                  native_scope="Different equation: dense ConvTranspose2d with zero padding1 and additive bias; no Converse quality comparison",
                  memory_scope="Total PyTorch allocator peak allocated/reserved for one resident GPU fixture, not driver/process VRAM",
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True,
                                   deterministic_algorithms=False, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
                  native_geometry=native.native_geometry(config), route_checks={}, validation={}, rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        report["route_checks"] = {name: inspect_route(name, family, method, config["shape"])
                                  for name, (method, family) in routes.items()}
        report["validation"] = validate(config, tensors, routes)
        worker.write_json(args.output, report)
        if not all(row["passed"] for row in report["validation"].values()):
            report["status"] = "precision_gate_failed_no_timing"
            worker.write_json(args.output, report)
            return 1
        report["status"] = "timing_local_gate_passed"
        names = list(routes)
        for index in range(args.rounds):
            order = names[index % len(names):] + names[:index % len(names)]
            values = {name: timed_fixture(routes[name][1], routes[name][0], args) for name in order}
            report["rounds"].append(dict(round=index + 1, order=order, variants=values))
            report["summary"] = summarize(report["rounds"], names)
            worker.write_json(args.output, report)
            print(json.dumps(dict(round=index + 1, variants=values)), flush=True)
        report["source_unchanged"] = all(worker.file_hash(ROOT / name) == digest for name, digest in sources.items())
        if not report["source_unchanged"]:
            raise RuntimeError("Source changed during measurement")
        report["status"] = "complete_same_fixture_operator_comparison"
        worker.write_json(args.output, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
