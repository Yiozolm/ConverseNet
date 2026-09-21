"""Replay a captured 40-Converse operator workload, not a replacement network.

Every timed repetition executes 40 forwards, then ONE autograd.grad requesting
all 60 distinct targets. Seven prior weight/bias pairs recur five times; the
DataNet bias recurs five times. Inputs and generated kernels are independent
frozen leaves. KernelNet/projection generation, other model layers, inter-layer
dependencies, loss, optimizer, capture, transfers and I/O are excluded.

Routes: original Python FP32, current production, shared-s1 CUDA+FP64 preparation
and cat/view for the 35 prior modules with unchanged current DataNet, plus native
dense dynamic per-sample loop and batch-folded groups=B. Native weights have a
different shape/equation, are fixed once, and preserve the same sharing graph.

An independent full-FFT FP64 oracle is evaluated ONE CALL AT A TIME solely for
validation, then shared VJPs are summed in CPU FP64 in forward-call order. That
finite-precision reduction order is explicit; no timing is inferred from it.
The timed FP32 graphs are never streamed or split after OOM. Per output/VJP,
max_abs and relative L2 must both be <= Python FP32 errors, without margin.
Old pointwise checks are diagnostics. Native loop/folded equivalence has its
separate predeclared 3e-5 absolute/3e-4 relative budget, not a Converse oracle.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ROUTES = ("python_fp32", "current", "shared_prior", "native_loop", "native_folded")


def load_capture(directory):
    import torch
    import train_usrnet_dataset as worker
    from capture_converse_workload40 import validate_payload, payload_hash
    manifest_path = directory / "capture.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["optimizer_steps"] != 0:
        raise RuntimeError("Capture is incomplete or its pretrained state was updated")
    fixture = directory / report["fixture_file"]
    if worker.file_hash(fixture) != report["fixture_file_sha256"]:
        raise RuntimeError("Raw captured fixture bytes changed")
    if any(worker.file_hash(ROOT / name) != digest for name, digest in report["source_sha256"].items()):
        raise RuntimeError("Capture/replay/model sources changed; create new evidence explicitly")
    payload = torch.load(fixture, map_location="cpu", weights_only=True)
    structure = validate_payload(payload)
    if payload_hash(payload) != report["payload_tensor_sha256"] or payload["source_sha256"] != report["source_sha256"]:
        raise RuntimeError("Captured tensor payload/source identity changed")
    return payload, dict(path=str(directory.resolve()), manifest_sha256=worker.file_hash(manifest_path),
                         fixture_file_sha256=report["fixture_file_sha256"], payload_tensor_sha256=report["payload_tensor_sha256"],
                         capture=report, structure=structure)


def native_leaves(payload):
    """Independent dense weights, identical for loop/folded; shared keys persist."""
    import torch
    result = {key: value for key, value in payload["leaves"].items() if key.startswith("input/")}
    for row in payload["calls"]:
        wk, bk = row["weight_key"], row["bias_key"]
        if wk not in result:
            kernel = payload["leaves"][wk]
            batch, channels, kh, kw = kernel.shape
            shape = (channels, channels, kh, kw) if row["kind"] == "prior" else (batch, channels, channels, kh, kw)
            seed = int.from_bytes(hashlib.sha256(("native-workload40-seed17/" + wk).encode()).digest()[:8], "little")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result[wk] = torch.randn(shape, generator=generator) / math.sqrt(channels * kh * kw)
        if bk not in result:
            result[bk] = payload["leaves"][bk].reshape(-1).clone()
    return result


def restore(value, layout, dtype, requires_grad=False):
    import torch
    # Original strides, fresh independent storage with offset zero. Copy is outside timing.
    tensor = torch.empty_strided(tuple(value.shape), tuple(layout["stride"]), dtype=dtype, device="cuda")
    with torch.no_grad():
        tensor.copy_(value)
    return tensor.requires_grad_(requires_grad)


def ordinary_forward(row, route, x, weight, bias):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    scale = row["scale"]
    if route.startswith("native_"):
        kh, kw = weight.shape[-2:]
        options = dict(stride=scale, padding=(kh // 2, kw // 2), output_padding=scale - 1, dilation=1)
        if row["kind"] == "prior":
            return F.conv_transpose2d(x, weight, bias, groups=1, **options)
        batch, channels, h, w = x.shape
        if route == "native_loop":
            return torch.cat([F.conv_transpose2d(x[index:index + 1], weight[index], bias, groups=1, **options)
                              for index in range(batch)], dim=0)
        result = F.conv_transpose2d(x.reshape(1, batch * channels, h, w),
                                   weight.reshape(batch * channels, channels, kh, kw),
                                   bias.repeat(batch), groups=batch, **options)
        return result.reshape(batch, channels, h * scale, w * scale)
    original_dtype = x.dtype
    dtype = torch.promote_types(torch.promote_types(x.dtype, weight.dtype), bias.dtype)
    if any(value.dtype != dtype for value in (x, weight, bias)):
        x, weight, bias = x.to(dtype), weight.to(dtype), bias.to(dtype)
    padding = row["padding"]
    if padding:
        x = F.pad(x, (padding,) * 4, mode=row["padding_mode"])
    prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
    result = (converse2d_reference(x, prior, weight, bias, scale, row["eps"])
              if route in ("python_fp32", "reference_fp64") else
              torch.ops.converse2d.forward(x, prior, weight, bias, scale, row["eps"], row["variant"]))
    crop = padding * scale
    if crop:
        result = result[..., crop:-crop, crop:-crop]
    return result.to(original_dtype)


def fixture(payload, route, shared_ops, native_cpu):
    import torch
    from benchmark_shared_s1_cuda import shared_method
    cpu = native_cpu if route.startswith("native_") else payload["leaves"]
    leaves = {}
    for key in payload["target_keys"]:
        layout = (dict(stride=list(cpu[key].stride())) if route.startswith("native_") and not key.startswith("input/")
                  else payload["layouts"][key])
        leaves[key] = restore(cpu[key], layout, torch.float32, requires_grad=True)
    upstreams = tuple(restore(payload["grad_outputs"][str(row["index"])],
                             payload["grad_output_layouts"][str(row["index"])], torch.float32)
                      for row in payload["calls"])
    functions = []
    for row in payload["calls"]:
        if route == "shared_prior" and row["kind"] == "prior":
            functions.append(shared_method(shared_ops, dict(outer_padding=row["padding"], eps=row["eps"]), "fp64"))
        else:
            functions.append(lambda x, w, b, row=row: ordinary_forward(row, route, x, w, b))
    targets = tuple(leaves[key] for key in payload["target_keys"])
    if len({id(value) for value in targets}) != 60:
        raise RuntimeError("VJP targets lost uniqueness")

    def run():
        outputs = tuple(function(leaves[row["input_key"]], leaves[row["weight_key"]], leaves[row["bias_key"]])
                        for function, row in zip(functions, payload["calls"]))
        gradients = torch.autograd.grad(outputs, targets, upstreams, allow_unused=False)
        return outputs, gradients
    return leaves, run


def graph_counts(outputs):
    pending, visited = [value.grad_fn for value in outputs], set()
    counts = dict(SpectralSolve=0, SharedTransfer=0)
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for token in counts:
            counts[token] += token in node.name()
        pending.extend(child for child, _ in node.next_functions)
    return counts


def validate_values(payload, route, leaves, values):
    import torch
    outputs, gradients = values
    if len(outputs) != 40 or len(gradients) != 60:
        raise RuntimeError("Whole workload lost an output or VJP")
    if any(tuple(value.shape) != tuple(row["output_shape"]) for value, row in zip(outputs, payload["calls"])):
        raise RuntimeError("Whole-module output geometry mismatch")
    if any(gradient.shape != leaves[key].shape for key, gradient in zip(payload["target_keys"], gradients)):
        raise RuntimeError("VJP geometry mismatch")
    if any(value.dtype != torch.float32 for value in (*outputs, *gradients)) or not bool(torch.stack(
            [torch.isfinite(value).all() for value in (*outputs, *gradients)]).all()):
        raise RuntimeError("Whole workload produced nonfinite or non-FP32 values")
    if any(value.grad is not None for value in leaves.values()):
        raise RuntimeError("Leaf .grad unexpectedly accumulated")
    counts = graph_counts(outputs)
    expected = dict(SpectralSolve=40 if route == "current" else 5 if route == "shared_prior" else 0,
                    SharedTransfer=35 if route == "shared_prior" else 0)
    if counts != expected:
        raise RuntimeError(f"Unexpected whole-workload backend coverage {route}: {counts}, expected {expected}")
    return dict(passed=True, outputs=40, vjp_targets=60, graph_nodes=counts,
                expected_graph_nodes=expected, finite=True, leaf_grad_accumulated=False,
                native_deconvolution_calls=55 if route == "native_loop" else 40 if route == "native_folded" else None,
                dynamic_native_groups=4 if route == "native_folded" else 1 if route == "native_loop" else None)


def capture_route(payload, route, ops, native_cpu):
    from probe_pointwise_training import clear_cuda
    clear_cuda()
    leaves, run = fixture(payload, route, ops, native_cpu)
    values = run()
    checked = validate_values(payload, route, leaves, values)
    saved = {f"output/{index:02d}": value.detach().cpu().contiguous().clone() for index, value in enumerate(values[0])}
    saved.update({"gradient/" + key: value.detach().cpu().contiguous().clone()
                  for key, value in zip(payload["target_keys"], values[1])})
    del values, run, leaves
    clear_cuda()
    return saved, checked


def fp64_oracle(payload):
    """Validation only: per-call full FP64; shared VJPs sum in CPU FP64 call order."""
    import torch
    from probe_pointwise_training import clear_cuda
    clear_cuda()
    saved = {}
    for row in payload["calls"]:
        keys = (row["input_key"], row["weight_key"], row["bias_key"])
        values = tuple(restore(payload["leaves"][key], payload["layouts"][key], torch.float64, True) for key in keys)
        output = ordinary_forward(row, "reference_fp64", *values)
        upstream = restore(payload["grad_outputs"][str(row["index"])],
                           payload["grad_output_layouts"][str(row["index"])], torch.float64)
        grads = torch.autograd.grad(output, values, upstream)
        saved[f"output/{row['index']:02d}"] = output.detach().cpu().contiguous().clone()
        for key, grad in zip(keys, grads):
            name = "gradient/" + key
            cpu = grad.detach().cpu().contiguous().clone()
            saved[name] = saved[name] + cpu if name in saved else cpu
        del output, grads, upstream, values, grad, cpu
    if len(saved) != 100 or any(value.dtype != torch.float64 or not bool(torch.isfinite(value).all()) for value in saved.values()):
        raise RuntimeError("Incomplete/nonfinite FP64 workload oracle")
    clear_cuda()
    return saved


def error_metrics(actual, expected, atol, rtol):
    import torch
    if actual.shape != expected.shape:
        raise RuntimeError("Comparison tensor shape mismatch")
    a, e = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return dict(finite=False, pointwise_passed=False, max_abs=None, relative_l2=None)
    delta = (a - e).abs()
    norm = e.norm()
    failures = int((delta > atol + rtol * e.abs()).sum())
    return dict(finite=True, max_abs=delta.max().item(), relative_l2=(delta.norm() / norm.clamp_min(1e-30)).item(),
                reference_norm=norm.item(), pointwise_passed=failures == 0, pointwise_failed_elements=failures,
                diagnostic_atol=atol, diagnostic_rtol=rtol)


def accuracy(actual, python, oracle, baseline_errors=None):
    if set(actual) != set(python) or set(actual) != set(oracle) or len(actual) != 100:
        raise RuntimeError("The workload accuracy report must cover all 100 tensors")
    rows = {}
    for name in oracle:
        tol = (3e-5, 3e-5) if name.startswith("output/") else (5e-5, 5e-5)
        baseline = error_metrics(python[name], oracle[name], *tol) if baseline_errors is None else baseline_errors[name]
        candidate = error_metrics(actual[name], oracle[name], *tol)
        finite = candidate["finite"] and baseline["finite"]
        gates = {metric: bool(finite and candidate[metric] <= baseline[metric]) for metric in ("max_abs", "relative_l2")}
        rows[name] = dict(passed=all(gates.values()), criteria=gates, candidate_vs_fp64=candidate,
                          python_vs_fp64=baseline, vs_python_pointwise_diagnostic=error_metrics(actual[name], python[name], *tol))
    failed = [name for name, row in rows.items() if not row["passed"]]
    return dict(passed=not failed, tensors=rows, tensor_count=100, failed_tensor_count=len(failed),
                failed_tensor_names=failed, added_tolerance=0, relative_margin=0)


def capture_consistency(payload, python):
    import train_usrnet_dataset as worker
    output_hashes = {str(index): worker.tensor_hash(dict(value=python[f"output/{index:02d}"])) == payload["output_hashes"][str(index)]
                     for index in range(40)}
    shared = {key: error_metrics(python["gradient/" + key], value, 3e-5, 3e-4)
              for key, value in payload["captured_parameter_vjps"].items()}
    return dict(passed=all(output_hashes.values()) and all(row["pointwise_passed"] for row in shared.values()),
                same_backend_output_hashes_match=output_hashes, shared_parameter_and_dynamic_kernel_vjps=shared,
                scope="Capture plumbing check only. Local dx cannot be compared to a full-network tensor gradient containing residual/other consumers.",
                shared_vjp_comparison_atol=3e-5, shared_vjp_comparison_rtol=3e-4)


def time_route(payload, route, ops, native_cpu, args):
    import torch
    from probe_pointwise_training import clear_cuda
    clear_cuda()
    leaves, run = fixture(payload, route, ops, native_cpu)
    for _ in range(args.warmup):
        run()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record(); end.record()
    torch.cuda.synchronize()
    initial_allocated, initial_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    start.record()
    for _ in range(args.iters):
        run()  # Discard the complete result before constructing the next graph.
    end.record()
    torch.cuda.synchronize()
    row = dict(wall_ms=(time.perf_counter() - began) * 1000 / args.iters,
               cuda_event_ms=start.elapsed_time(end) / args.iters,
               initial_allocated_bytes=initial_allocated, initial_reserved_bytes=initial_reserved,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               units="milliseconds per complete 40-forward + one 60-target VJP workload")
    check = run()
    row["post_timing_contract"] = validate_values(payload, route, leaves, check)
    del check, run, leaves
    clear_cuda()
    return row


def summarize(rounds):
    names = list(rounds[0]["variants"])
    medians = {name: {key: statistics.median(row["variants"][name][key] for row in rounds)
                      for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in names}
    ratios = {}
    for name in names:
        ratios[name] = {}
        for denominator in ("python_fp32", "current", "native_loop", "native_folded"):
            if denominator not in names:
                continue
            ratios[name][denominator] = {}
            for metric in ("wall_ms", "cuda_event_ms"):
                values = [row["variants"][name][metric] / row["variants"][denominator][metric] for row in rounds]
                ratios[name][denominator][metric] = dict(per_round=values, median=statistics.median(values), min=min(values), max=max(values))
    return dict(medians=medians, paired_route_over_denominator=ratios,
                scope="Whole workload measurements only; not sums of microbench timings, not end-to-end network speed")


def main():
    from capture_converse_workload40 import DEFAULT_CAPTURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training_research/operator_workload40/benchmark.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    import torch
    import train_usrnet_dataset as worker
    from capture_converse_workload40 import configure_cuda, source_hashes
    from experiments.training_shared_s1.loader import load
    payload, capture_report = load_capture(args.capture_dir)
    environment = configure_cuda()
    sources = source_hashes()
    for name in ("loader.py", "bindings.cpp", "kernels.cu"):
        path = "experiments/training_shared_s1/" + name
        sources[path] = worker.file_hash(ROOT / path)
    report = dict(status="setup", stage="build", scope=__doc__, settings=worker.json_safe(vars(args)),
                  capture=capture_report, source_sha256=sources, environment=environment,
                  production_eligible=False, full_model_replacement=False, training_quality_verified=False,
                  independent_of_other_mixed_model_quality_evidence=True,
                  original_module_counts=dict(s3_data=1, s1_data=4, prior=35), unique_vjp_targets=60,
                  reference_protocol="Same FP64 full-FFT equation per call. Shared dw/db summed on CPU in FP64 forward-call order; tiny finite-precision differences versus a joint FP64 reduction remain possible. No margin added.",
                  accuracy_protocol="All 100 output/VJP tensors: max_abs and relative_l2 <= Python FP32 against that same oracle; old pointwise checks diagnostic only",
                  native_equivalence_protocol=dict(atol=3e-5, rtol=3e-4, comparison="All 40 outputs and all 60 VJPs, folded versus loop with identical dense weights and shared biases"),
                  native_semantics="Dense zero-padded additive-bias operator, different from Converse periodic inverse/lambda. Dynamic loop has 55 conv calls total; folded groups=B has 40.",
                  timing_protocol="One backend resident; transfers/hash/validation outside; warm complete graph; 40 forwards then exactly one 60-target VJP, no optimizer/loss. OOM stops, no streamed fallback.",
                  memory_scope="Total allocator allocated/reserved peak for frozen inputs, upstreams, parameters, complete autograd graph and gradients; no process/driver VRAM claim",
                  validation={}, contracts={}, excluded_routes=[], timed_routes=[], rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        args.variant = "current"
        _, report["current_build"] = worker.load_backend(args)
        shared_ops, report["shared_build"] = load(verbose=args.verbose_build)
        native_cpu = native_leaves(payload)
        report["native_leaf_tensor_sha256"] = worker.tensor_hash(native_cpu)
        report["gradient_elements"] = dict(converse={key: value.numel() for key, value in payload["leaves"].items()},
                                           native={key: value.numel() for key, value in native_cpu.items()})
        report["stage"] = "FP64_reference_validation_only"
        worker.write_json(args.output, report)
        oracle = fp64_oracle(payload)
        report["stage"] = "whole_graph_python_fp32_validation"
        worker.write_json(args.output, report)
        python, report["contracts"]["python_fp32"] = capture_route(payload, "python_fp32", shared_ops, native_cpu)
        report["capture_consistency"] = capture_consistency(payload, python)
        report["validation"]["python_fp32"] = accuracy(python, python, oracle)
        worker.write_json(args.output, report)
        if not report["capture_consistency"]["passed"]:
            report["status"] = "capture_replay_integrity_failed_no_timing"
            worker.write_json(args.output, report)
            return 2
        baseline_errors = {key: row["python_vs_fp64"] for key, row in report["validation"]["python_fp32"]["tensors"].items()}
        for route in ("current", "shared_prior"):
            report["stage"] = "whole_graph_validation/" + route
            worker.write_json(args.output, report)
            values, report["contracts"][route] = capture_route(payload, route, shared_ops, native_cpu)
            report["validation"][route] = accuracy(values, python, oracle, baseline_errors)
            del values
            worker.write_json(args.output, report)
        del python, oracle
        gc.collect()
        report["stage"] = "native_loop_folded_equivalence"
        worker.write_json(args.output, report)
        loop, report["contracts"]["native_loop"] = capture_route(payload, "native_loop", shared_ops, native_cpu)
        folded, report["contracts"]["native_folded"] = capture_route(payload, "native_folded", shared_ops, native_cpu)
        if set(loop) != set(folded) or len(loop) != 100:
            raise RuntimeError("Native equivalence omitted outputs or VJPs")
        comparisons = {key: error_metrics(folded[key], value, 3e-5, 3e-4) for key, value in loop.items()}
        native_passed = all(row["pointwise_passed"] for row in comparisons.values())
        report["native_equivalence"] = dict(passed=native_passed, tensor_count=100, tensors=comparisons,
                                            failed_tensor_names=[key for key, row in comparisons.items() if not row["pointwise_passed"]])
        del loop, folded
        gc.collect()
        report["timed_routes"] = [route for route in ROUTES if (native_passed if route.startswith("native_") else report["validation"][route]["passed"])]
        report["excluded_routes"] = [dict(route=route, reason="Native equivalence failed" if route.startswith("native_") else "Local numerical noninferiority failed")
                                      for route in ROUTES if route not in report["timed_routes"]]
        worker.write_json(args.output, report)
        report["status"], report["stage"] = "timing_selected_whole_workloads", "timing"
        names = report["timed_routes"]
        for index in range(args.rounds):
            order = names[index % len(names):] + names[:index % len(names)]
            values = {route: time_route(payload, route, shared_ops, native_cpu, args) for route in order}
            report["rounds"].append(dict(round=index + 1, order=order, variants=values))
            report["summary"] = summarize(report["rounds"])
            worker.write_json(args.output, report)
            print(json.dumps(dict(round=index + 1, variants=values)), flush=True)
        report["source_unchanged"] = all(worker.file_hash(ROOT / name) == digest for name, digest in sources.items())
        if not report["source_unchanged"]:
            raise RuntimeError("Workload sources changed during execution")
        all_requested = len(names) == len(ROUTES)
        report["status"] = "complete_operator_workload_all_routes" if all_requested else "complete_operator_workload_with_exclusions"
        report["stage"] = "complete"
        worker.write_json(args.output, report)
        return 0 if all_requested else 2
    except Exception as error:
        report.update(status="failed_no_schedule_adaptation", error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
