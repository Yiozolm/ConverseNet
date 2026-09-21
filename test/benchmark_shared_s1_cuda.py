"""Shared-s1 CUDA versus the original Python FP32 operator, new evidence only.

Immutable native s1_module/seed17 fixture, eps=1e-5. Compare production FFT,
shared-H CUDA with differentiable FP64 or FP32 kernel preparation and cat/view
boundaries, original full-FFT Python FP32, and native dense ConvTranspose2d.
The shared-H formula assumes observation IS prior; no independent prior is used.

The numerical oracle is models.converse_core.converse2d_reference in FP32,
with native circular pad2/slice crop2. Output atol/rtol=3e-5/3e-5 and complete
dx/dw/db=5e-5/5e-5. Any Converse route failure stops all timing. Native dense
has different mathematics, parameter count and boundary; only its shape,
dtype, finite values and three VJP targets are checked against its contract.

Default warm5, six rotated rounds of thirty full forward+VJP calls per route.
Device-resident eager FP32, TF32 off, global deterministic off, cuDNN
deterministic on; no profiler, Graph, AMP, loss or optimizer. Preparation,
FFT/IFFT and pad/crop remain inside every measured call. One GPU fixture is
resident at a time. No full-model, task-quality or global release claim.
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


def shared_method(ops, config, preparation):
    import torch
    from diagnose_boundary_precision import kernel_fft64
    from validate_shared_s1_python_fp32 import kernel_fft32
    from probe_converse_boundaries import pad_cat, crop_view
    prepare = {"fp64": kernel_fft64, "fp32": kernel_fft32}[preparation]

    def forward(x, weight, bias):
        padded = pad_cat(x, config["outer_padding"])
        height, width = padded.shape[-2:]
        spectrum = prepare(weight, height, width).cfloat().contiguous()
        observation = torch.fft.rfft2(padded)
        regularizer = torch.sigmoid(bias - 9.) + config["eps"]
        solved = ops.shared_s1_transfer(observation, spectrum, regularizer)
        output = torch.fft.irfft2(solved, s=(height, width))
        return crop_view(output, config["outer_padding"])
    return forward


def check_route(name, tensors, method, shape):
    import torch
    from probe_pointwise_training import clear_cuda, fixture
    clear_cuda()
    inputs, run = fixture(tensors, torch.float32, method)
    values = run()
    pending, visited, nodes = [values["output"].grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        nodes.append(node.name())
        pending.extend(child for child, _ in node.next_functions)
    hits = {token: sum(token in node for node in nodes) for token in ("SpectralSolve", "SharedTransfer")}
    expected = dict(SpectralSolve=int(name == "production_fft"), SharedTransfer=int(name.startswith("cuda_shared_")))
    finite = bool(torch.stack([torch.isfinite(value).all() for value in values.values()]).all())
    if (hits != expected or not finite or list(values["output"].shape) != list(shape)
            or any(value.dtype != torch.float32 for value in values.values())
            or any(value.grad is not None for value in inputs)
            or any(values[key].shape != value.shape for key, value in zip(("dx", "dw", "db"), inputs))):
        raise RuntimeError(f"{name}: incorrect backend/FP32/shape/finite/gradient contract: {hits}")
    row = dict(passed=True, graph_nodes=hits, expected_graph_nodes=expected, finite=True,
               tensor_shapes={key: list(value.shape) for key, value in values.items()},
               gradient_targets=["x", "weight", "bias"],
               parameter_elements=dict(weight=inputs[1].numel(), bias=inputs[2].numel()))
    del values, inputs, run, pending, visited, node
    clear_cuda()
    return row


def validate(tensors, routes):
    import torch
    from probe_pointwise_training import capture
    from probe_training_s1_shapes import metrics
    expected = capture(tensors, torch.float32, routes["python_fp32"][0])
    checked = {}
    for name, (method, family) in routes.items():
        if name == "native_dense":
            continue
        actual = capture(family, torch.float32, method)
        rows = {key: metrics(actual[key], value, output=key == "output", weak=False)
                for key, value in expected.items()}
        checked[name] = dict(passed=all(row["passed"] for row in rows.values()), tensors=rows)
    return checked


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/native_deconv_target/native_baseline.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/shared_s1_cuda_python_fp32_benchmark.json")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with required FP32 execution")
    import torch
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    from benchmark_converse_candidates import audit_native_baseline
    from benchmark_deconv_target_final import production_method, native_method
    from benchmark_shared_s1_transfer import summarize
    from experiments.training_shared_s1.loader import load
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
        raise RuntimeError("Original native fixture changed: never adjust eps, weights or upstream")
    frozen = ROOT / "artifacts/native_deconv_target/source_before/manifest.json"
    frozen_manifest = json.loads(frozen.read_text(encoding="utf-8"))
    oracle_source = "models/converse_core.py"
    if worker.file_hash(ROOT / oracle_source) != frozen_manifest[oracle_source]:
        raise RuntimeError("Original Python oracle source changed")
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
    for name in ("benchmark_shared_s1_cuda.py", "benchmark_shared_s1_transfer.py", "benchmark_native_deconv.py",
                 "benchmark_converse_candidates.py", "benchmark_deconv_target_final.py", "probe_converse_boundaries.py",
                 "probe_pointwise_training.py", "probe_training_s1_shapes.py", "diagnose_boundary_precision.py",
                 "validate_shared_s1_python_fp32.py"):
        sources["test/" + name] = worker.file_hash(ROOT / "test" / name)
    for name in ("loader.py", "bindings.cpp", "kernels.cu"):
        path = "experiments/training_shared_s1/" + name
        sources[path] = worker.file_hash(ROOT / path)
    report = dict(status="validating", scope=__doc__, settings=worker.json_safe(vars(args)),
                  fixed_config=config, fixed_seed=17, complete_fixture_sha256=full_hash, input_upstream_sha256=input_hash,
                  source_sha256=sources, current_build=current_build, candidate_build=candidate_build,
                  baseline_audit=baseline_audit, production_eligible=False,
                  oracle=dict(backend="pytorch", dtype="float32", source=oracle_source, sha256=frozen_manifest[oracle_source],
                              formulation="Original full-FFT residual solve with shared prior, native circular pad2 and slice crop2",
                              frozen_manifest_sha256=worker.file_hash(frozen)),
                  limitation="Local synthetic fixture only; pretrained-module pointwise rounding and full-model quality are not resolved here; prior FP64 evidence remains unchanged",
                  native_scope="Different equation, zero boundary, additive bias and dense weights; no quality equivalence",
                  memory_scope="Total PyTorch allocated/reserved peak for one GPU fixture; excludes driver/process VRAM",
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True,
                                   deterministic_algorithms=False, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
                  native_geometry=native.native_geometry(config), route_checks={}, validation={}, rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        report["route_checks"] = {name: check_route(name, family, method, config["shape"])
                                  for name, (method, family) in routes.items()}
        report["validation"] = validate(tensors, routes)
        worker.write_json(args.output, report)
        if not all(row["passed"] for row in report["validation"].values()):
            report["status"] = "python_fp32_gate_failed_no_timing"
            worker.write_json(args.output, report)
            return 1
        report["status"] = "timing_local_python_fp32_gate_passed"
        names = list(routes)
        for index in range(args.rounds):
            order = names[index % len(names):] + names[:index % len(names)]
            values = {name: timed_fixture(routes[name][1], routes[name][0], args) for name in order}
            report["rounds"].append(dict(round=index + 1, order=order, variants=values))
            report["summary"] = summarize(report["rounds"], names)
            report["summary"]["python_over_route"] = {name: {metric: dict(
                per_round=(ratios := [row["variants"]["python_fp32"][metric] / row["variants"][name][metric]
                                      for row in report["rounds"]]), median=statistics.median(ratios),
                min=min(ratios), max=max(ratios)) for metric in ("wall_ms", "cuda_event_ms")} for name in names}
            worker.write_json(args.output, report)
            print(json.dumps(dict(round=index + 1, variants=values)), flush=True)
        report["source_unchanged"] = all(worker.file_hash(ROOT / name) == digest for name, digest in sources.items())
        if not report["source_unchanged"]:
            raise RuntimeError("Source changed during measurement")
        report["status"] = "complete_local_python_fp32_operator_comparison"
        worker.write_json(args.output, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
