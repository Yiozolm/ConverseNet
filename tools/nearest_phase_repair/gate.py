"""Exact legacy24-case/96-tensor nearest_spectral gate; no performance lane.

CUDA capture/record/specifications and fixture-generation statements are loaded
from pinned Git source. CPU capture changes only the two CUDA device literals.
Historical artifacts are read-only and never receive retroactive tensor hashes.
"""
import argparse
import ast
import base64
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
COMMIT = "cc244e3"
PINNED = {
    "research/run_algorithms.py": "68958153d88b9326a4641d6e9b8f38eee7dbd6a30994f60a7e1968e59c27164d",
    "research/algorithms.py": "2ca5f02ee9581d5693f2c1d361a2fbc199bc7a689573df3f1562aea1d0b82286",
    "research/training_candidates.py": "f601f90bdd907bdc65c785d33a64cd871784164b964e5e2ce5f11a8973bc5df6",
    "models/converse_core.py": "8b31f77ae03fafad69f6e8d3f696fe02166aa0d8fe258a2937de3ab619041ccd",
}
LABELS = ("output0", "dx", "dweight", "dbias")
ROUTES = ("reference_fp64", "python_fp32", "old_candidate", "new_candidate")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def node_hash(node):
    return sha(ast.dump(node, include_attributes=False).encode())


def functions(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def compile_functions(nodes, namespace, filename):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(list(nodes)), type_ignores=[]))
    exec(compile(module, filename, "exec"), namespace)


def load_legacy(torch, historical):
    source, identities = {}, {}
    for name, expected in PINNED.items():
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{COMMIT}:{name}"])
        require(sha(raw) == expected, "Pinned legacy source hash mismatch: " + name)
        source[name] = raw.decode("utf-8")
        identities[name] = dict(pinned_file_sha256=expected)
        local = ROOT / ".build/roadmap-research" / name
        if local.exists():
            identities[name]["current_research_tree_file_sha256"] = file_sha(local)
            if name != "models/converse_core.py":
                require(local.read_bytes() == raw, "Current research copy differs from pinned source: " + name)
    for name, expected in historical["source_sha256"].items():
        require(PINNED["research/" + name] == expected, "Historical recorded source hash differs: " + name)
    runner = functions(source["research/run_algorithms.py"])
    algorithm = functions(source["research/algorithms.py"])
    reference = functions(source["models/converse_core.py"])
    reference_names = ("alias_mean", "validate_inputs", "converse2d_reference")
    current_reference = ROOT / ".build/roadmap-research/models/converse_core.py"
    if current_reference.exists():
        local = functions(current_reference.read_text(encoding="utf-8"))
        require(all(node_hash(local[name]) == node_hash(reference[name]) for name in reference_names),
                "Required independent reference AST changed")
    namespace = dict(torch=torch, contextlib=contextlib, math=math, _geometry={})
    compile_functions([runner[name] for name in ("specifications", "record", "capture")], namespace,
                      f"git:{COMMIT}:research/run_algorithms.py")
    namespace["capture_cuda"] = namespace["capture"]
    class DeviceOnly(ast.NodeTransformer):
        changed = 0
        def visit_Constant(self, node):
            if node.value == "cuda":
                self.changed += 1
                return ast.copy_location(ast.Constant(value="cpu"), node)
            return node
    adapter = DeviceOnly()
    cpu_capture = adapter.visit(copy.deepcopy(runner["capture"]))
    require(adapter.changed == 2, "CPU adapter did not change exactly two capture device literals")
    cpu_capture.name = "capture_cpu"
    compile_functions([cpu_capture], namespace, "legacy_capture_device_only_cpu_adapter")
    # Extract the original loop statements through the weak transform verbatim.
    outer = next(node for node in runner["main"].body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "name")
    inner = next(node for node in outer.body if isinstance(node, ast.For))
    first_capture = next(index for index, node in enumerate(inner.body) if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "high")
    fixture_statements = copy.deepcopy(inner.body[:first_capture])
    require(len(fixture_statements) == 5 and isinstance(fixture_statements[-1], ast.If),
            "Unexpected legacy fixture generation layout")
    fixture = ast.FunctionDef(name="build_fixture", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg="index"), ast.arg(arg="spec")], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[*fixture_statements, ast.Return(value=ast.Tuple(elts=[ast.Name(id="raw", ctx=ast.Load()),
            ast.Name(id="up", ctx=ast.Load())], ctx=ast.Load()))], decorator_list=[])
    compile_functions([fixture], namespace, "verbatim_legacy_fixture_statements")
    compile_functions([algorithm[name] for name in ("check", "aliases", "kernel_fft", "spectral", "nearest_spectral")],
                      namespace, f"git:{COMMIT}:research/algorithms.py")
    compile_functions([reference[name] for name in reference_names], namespace, f"git:{COMMIT}:models/converse_core.py")
    scopes = next(node for node in ast.parse(source["research/training_candidates.py"]).body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CANDIDATE_SCOPES" for target in node.targets))
    require(isinstance(scopes.value, ast.Dict) and "nearest_spectral" not in [key.value for key in scopes.value.keys],
            "Legacy nearest_spectral unexpectedly has an extra scope")
    specifications = namespace["specifications"]("nearest_spectral")
    old = historical["candidates"]["nearest_spectral"]
    require(len(specifications) == len(old["cases"]) == 24 and old["tensor_count"] == 96 and old["failures"] == 45,
            "Historical matrix cardinality/failure count changed")
    require(specifications == [case["spec"] for case in old["cases"]], "Legacy specifications/order differ from historical matrix")
    require(all(tuple(case["tensors"]) == LABELS for case in old["cases"]), "Historical VJP/output labels changed")
    identity = dict(commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", COMMIT], text=True).strip(),
        files=identities, exact_runner_function_ast_sha256={name:node_hash(runner[name]) for name in ("specifications", "capture", "record")},
        exact_fixture_statement_ast_sha256=sha("\n".join(ast.dump(node, include_attributes=False) for node in fixture_statements).encode()),
        reference_function_ast_sha256={name:node_hash(reference[name]) for name in reference_names},
        cpu_adapter="Exactly two capture device='cuda' literals become device='cpu'; all expressions/order unchanged.",
        nearest_scope=None, cases=24, tensors=96,
        historical_hash_limit="Historical matrix recorded only three research-file hashes and numeric errors; it did not record input, output, VJP, or reference tensor hashes, nor a reference-file hash. All tensor hashes here are newly measured.")
    return namespace, specifications, identity, algorithm


def load_candidate(path, algorithm, serial):
    nodes = functions(path.read_text(encoding="utf-8"))
    class RemoveRepair(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "z"
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "apply_exact_coefficients"):
                require(not node.value.keywords and [getattr(value, "id", None) for value in node.value.args] == ["z", "n", "s"],
                        "Repair insertion arguments changed")
                self.count += 1
                return None
            return self.generic_visit(node)
    strip = RemoveRepair()
    restored = strip.visit(copy.deepcopy(nodes["nearest_spectral"]))
    require(strip.count == 1 and node_hash(restored) == node_hash(algorithm["nearest_spectral"]),
            "New nearest_spectral differs beyond the single authorized exact-coefficient insertion")
    spec = importlib.util.spec_from_file_location(f"nearest_phase_gate_candidate_{serial}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    helper = Path(inspect.getsourcefile(module.spectral)).resolve()
    helper_nodes = functions(helper.read_text(encoding="utf-8"))
    require(all(node_hash(helper_nodes[name]) == node_hash(algorithm[name]) for name in ("check", "aliases", "kernel_fft", "spectral")),
            "Candidate shared solver/helper arithmetic differs from the frozen implementation")
    return module, dict(path=str(path.resolve()), sha256=file_sha(path),
        nearest_body_equal_after_removing_exact_insertion=True, helper_path=str(helper), helper_sha256=file_sha(helper),
        geometry_scope="Exact-coefficient mathematics reviewed separately; this gate verifies unchanged surrounding source and per-tensor numerics.")


def safe(value):
    if isinstance(value, dict):
        return {key:safe(item) for key,item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def tensor_packet(value, packet, key):
    cpu = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    raw = cpu.numpy().tobytes()
    record = dict(shape=list(cpu.shape), dtype=str(cpu.dtype), bytes=len(raw), sha256=sha(raw),
                  provenance="new measurement; not present in historical matrix", packet_key=key)
    packet["tensors"][key] = dict(record, base64_little_endian=base64.b64encode(raw).decode("ascii"))
    return record


def decode_minimal(torch, path):
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data["format"] == "route_b_nearest_f32_v1" and data["scale"] == 2 and data["eps"] == 1e-5,
            "Wrong minimal input packet contract")
    values = {}
    for label, item in data["cases"]["minimal_identity_impulse"].items():
        raw = base64.b64decode(item["base64_little_endian"], validate=True)
        require(item["dtype"] == "torch.float32" and sha(raw) == item["sha256"] and len(raw) == item["bytes"], "Minimal input packet hash/dtype mismatch")
        value = torch.tensor(struct.unpack("<" + "f" * (len(raw) // 4), raw), dtype=torch.float32).reshape(item["shape"])
        require(value.numpy().tobytes() == raw, "Minimal input reconstruction changed bytes")
        values[label] = value
    require(values["x"].shape == (1, 1, 2, 2) and values["weight"].shape == (1, 1, 3, 3)
            and values["bias"].shape == (1, 1, 1, 1), "Unexpected minimal shapes")
    require(values["x"].flatten().tolist() == [1., 0., 0., 0.] and values["weight"].flatten().tolist() == [0.,0.,0.,0.,1.,0.,0.,0.,0.]
            and values["bias"].item() == 0, "Minimal counterexample values changed")
    return values, data


def evaluate_gate(record, actual, control, high):
    a, p = record(actual, high), record(control, high)
    # EXACT original admission expression: finite actual and <= BOTH metrics.
    return dict(candidate=a, python_fp32=p,
        passed=a["finite"] and a["max_abs"] <= p["max_abs"] and a["relative_l2"] <= p["relative_l2"])


def minimal_forward(torch, device, legacy, candidate, values, packet):
    outputs = {}
    for name, function, dtype in (("reference_fp64", legacy["converse2d_reference"], torch.float64),
         ("python_fp32", legacy["converse2d_reference"], torch.float32),
         ("old_candidate", legacy["nearest_spectral"], torch.float32),
         ("new_candidate", candidate.nearest_spectral, torch.float32)):
        x, k, b = [values[label].to(device=device, dtype=dtype).detach().requires_grad_() for label in ("x", "weight", "bias")]
        prior = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        outputs[name] = function(x, prior, k, b, 2, 1e-5)
    rows = {name:tensor_packet(output, packet, f"minimal/{device}/{name}/output0") for name,output in outputs.items()}
    old_gate = evaluate_gate(legacy["record"], outputs["old_candidate"], outputs["python_fp32"], outputs["reference_fp64"])
    new_gate = evaluate_gate(legacy["record"], outputs["new_candidate"], outputs["python_fp32"], outputs["reference_fp64"])
    equal = rows["new_candidate"]["sha256"] == rows["python_fp32"]["sha256"]
    return dict(device=device, scope="Same documented minimal forward counterexample; no invented historical VJP/upstream.",
        tensors=rows, old_gate=old_gate, new_gate=new_gate, new_matches_python_bytes=equal,
        passed=(not old_gate["passed"]) and new_gate["passed"] and equal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("source", "minimal", "matrix"), default="source")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "tools/nearest_phase_repair/candidate.py")
    parser.add_argument("--historical", type=Path, default=ROOT / "artifacts/fp32_roadmap/research_numeric.json")
    parser.add_argument("--minimal-inputs", type=Path, default=ROOT / "docs/training_followup_route_b_inputs.json")
    args = parser.parse_args()
    byte_path = args.output.with_name(args.output.stem + "_bytes.json")
    if args.output.exists() or byte_path.exists():
        parser.error("Fresh output and byte-packet paths required; old evidence is immutable")
    require(sys.byteorder == "little", "Byte packets require little-endian host")
    require(args.stage != "matrix" or args.device == "cuda", "Historical matrix is CUDA; CPU is source/minimal diagnostics only")
    require(args.stage != "source" or args.device == "cpu", "Source fidelity stage is CPU-only")
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    legacy, specifications, identity, algorithm = load_legacy(torch, historical)
    candidate, candidate_identity = load_candidate(args.candidate.resolve(), algorithm, "preflight")
    minimal, original_packet = decode_minimal(torch, args.minimal_inputs)
    watched = {str(path.resolve()):file_sha(path) for path in (Path(__file__), args.candidate, args.historical,
        args.minimal_inputs, Path(candidate_identity["helper_path"]))}
    packet = dict(kind="new_nearest_phase_gate_tensor_bytes", version=1, tensors={},
        provenance="All bytes newly captured by this runner. Historical96-tensor evidence did not include hashes; none are retroactively attributed to it.")
    report = dict(kind="nearest_phase_repair_zero_margin_gate", stage=args.stage, status="running",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), device=args.device,
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"], legacy_source_fidelity=identity,
        candidate_identity=candidate_identity, source_sha256=watched, specifications=specifications,
        minimal_input_packet=str(args.minimal_inputs.resolve()), minimal_input_packet_sha256=file_sha(args.minimal_inputs),
        historical_report=str(args.historical.resolve()), historical_report_sha256=file_sha(args.historical),
        prior_policy="Nearest prior constructed from each fresh dtype-converted x inside original capture; dx includes its interpolation branch; no separate dprior.",
        reference_policy="Independent pinned Python full-spectrum FP64 reference; only exact FP32 input values are lifted to FP64, never resampled.",
        performance_allowed=False, production_admitted=False, cases=[], minimal_checks=[])
    try:
        for index, spec in enumerate(specifications):
            raw, upstream = legacy["build_fixture"](index, spec)
            named = dict(zip(("x", "raw_p_unused", "weight", "bias"), raw))
            named["upstream"] = upstream
            named["prior_fp32_derived"] = torch.nn.functional.interpolate(raw[0], scale_factor=spec["scale"], mode="nearest")
            inputs = {name:tensor_packet(value, packet, f"case{index}/input/{name}") for name,value in named.items()}
            if index == 0:
                require(all(inputs[label]["sha256"] == original_packet["cases"]["historical_seed41191"][label]["sha256"]
                    for label in ("x", "weight", "bias")), "Legacy case-zero RNG bytes differ from prior reproduction packet")
            report["cases"].append(dict(index=index, seed=41191 + index, spec=spec, inputs=inputs,
                raw_p_rng_consumed=True, raw_p_requested_for_grad=False))
        report["source_fidelity_passed"] = True
        if args.stage == "source":
            require(not torch.cuda.is_initialized(), "CPU source stage initialized CUDA")
            report.update(status="source_fidelity_passed", cuda_initialized=False, historical_cuda_reproduction="not_run")
        else:
            for device in (("cpu", "cuda") if args.stage == "matrix" else (args.device,)):
                check = minimal_forward(torch, device, legacy, candidate, minimal, packet)
                report["minimal_checks"].append(check)
                require(check["passed"], "Minimal forward zero-margin/byte gate failed on " + device)
            if args.stage == "minimal":
                report.update(status="minimal_gate_passed", historical_cuda_reproduction="not_run")
            else:
                require(str(torch.__version__) == historical["torch"] and torch.cuda.get_device_name() == historical["gpu"],
                        "Historical CUDA environment identity differs")
                # Fresh namespaces restore the original empty geometry-cache state.
                legacy, _, _, algorithm = load_legacy(torch, historical)
                candidate, fresh_identity = load_candidate(args.candidate.resolve(), algorithm, "matrix")
                require(fresh_identity == candidate_identity, "Candidate/helper changed after minimal checks")
                mismatches, transitions, old_failures, new_failures = [], [], 0, 0
                for case in report["cases"]:
                    index, spec = case["index"], case["spec"]
                    raw, upstream = legacy["build_fixture"](index, spec)
                    outputs = {"reference_fp64":legacy["capture_cuda"](legacy["converse2d_reference"], raw, upstream, spec, torch.float64),
                        "python_fp32":legacy["capture_cuda"](legacy["converse2d_reference"], raw, upstream, spec, torch.float32),
                        "old_candidate":legacy["capture_cuda"](legacy["nearest_spectral"], raw, upstream, spec, torch.float32)}
                    require(all(tuple(values) == LABELS for values in outputs.values()), "Original capture labels/order changed")
                    old_rows = {label:evaluate_gate(legacy["record"], outputs["old_candidate"][label], outputs["python_fp32"][label],
                                                  outputs["reference_fp64"][label]) for label in LABELS}
                    for label in LABELS:
                        if old_rows[label] != historical["candidates"]["nearest_spectral"]["cases"][index]["tensors"][label]:
                            mismatches.append(dict(case=index, tensor=label, replay=old_rows[label],
                                historical=historical["candidates"]["nearest_spectral"]["cases"][index]["tensors"][label]))
                    case["old_replay"] = old_rows
                    case["old_replay_matches_historical"] = not mismatches
                    report["historical_metric_mismatches"] = mismatches
                    require(not mismatches, "Legacy per-tensor numeric evidence did not exactly replay; new admission invalid")
                    outputs["new_candidate"] = legacy["capture_cuda"](candidate.nearest_spectral, raw, upstream, spec, torch.float32)
                    require(tuple(outputs["new_candidate"]) == LABELS, "New capture labels/order differ")
                    case["tensors"] = {}
                    for label in LABELS:
                        gate = evaluate_gate(legacy["record"], outputs["new_candidate"][label], outputs["python_fp32"][label], outputs["reference_fp64"][label])
                        snapshots = {route:tensor_packet(values[label], packet, f"case{index}/{route}/{label}") for route,values in outputs.items()}
                        old_passed, new_passed = old_rows[label]["passed"], gate["passed"]
                        old_failures += not old_passed
                        new_failures += not new_passed
                        transition = ("pass" if old_passed else "fail") + "_to_" + ("pass" if new_passed else "fail")
                        case["tensors"][label] = dict(old_gate=old_rows[label], new_gate=gate,
                            transition=transition, newly_measured_tensors=snapshots,
                            new_matches_python_bytes=snapshots["new_candidate"]["sha256"] == snapshots["python_fp32"]["sha256"])
                        transitions.append(dict(case=index, seed=41191 + index, tensor=label, transition=transition))
                    print(json.dumps(dict(event="case_complete", case=index, old_failures_so_far=old_failures, new_failures_so_far=new_failures)), flush=True)
                    del outputs
                require(old_failures == 45 and len(transitions) == 96, "Legacy45/96 reproduction failed")
                report.update(status="candidate_failed_numeric_gate" if new_failures else "numeric_gate_passed_quality_pending",
                    historical_cuda_reproduction="all96 recorded error dictionaries and pass flags exactly matched;45 failures retained",
                    old_failures=old_failures, new_failures=new_failures, tensor_count=96, transitions=transitions,
                    transition_counts={name:sum(row["transition"] == name for row in transitions)
                                       for name in ("pass_to_pass", "pass_to_fail", "fail_to_pass", "fail_to_fail")},
                    residual_failure_policy="Any remaining tensor failure blocks performance and production; no relaxed margin, averaging, or historical relabeling.")
    except Exception as error:
        report.update(status="error", error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        report["sources_unchanged"] = watched == {path:file_sha(path) for path in watched}
        if not report["sources_unchanged"]:
            report.update(status="error", integrity_error="Candidate, runner, helper, input or historical evidence changed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with byte_path.open("x", encoding="utf-8") as stream:
            json.dump(safe(packet), stream, indent=2, allow_nan=False)
        report["new_byte_packet"] = dict(path=str(byte_path.resolve()), sha256=file_sha(byte_path), tensor_records=len(packet["tensors"]))
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(safe(report), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report["status"], old_failures=report.get("old_failures"),
        new_failures=report.get("new_failures"), output=str(args.output.resolve()))))
    return 0 if report["status"] in ("source_fidelity_passed", "minimal_gate_passed", "numeric_gate_passed_quality_pending") else 4


if __name__ == "__main__":
    raise SystemExit(main())
