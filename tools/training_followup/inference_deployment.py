"""Exclusive-GPU deployment coverage of EXISTING production LayerNorm.

No production edits, build, new fusion policy, or training claim. Historical
statistics and production LayerNorm share one current model/checkpoint/solver.
"""
import argparse
from contextlib import contextmanager
import datetime
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
import types

from inference_cases import ROOT, MODES, case_seed, declared_shapes, make_fixture, layout_tensor, select_cases

sys.path.insert(0, str(ROOT / "tools"))
import benchmark_production_layernorm as base
import benchmark_fp32_p0 as common

NAMES = base.NAMES
MARKER = "deployment_layernorm_ablation"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def context(torch, mode):
    return torch.no_grad() if mode == "no_grad" else torch.inference_mode()


def snapshot(torch, function):
    output = function()
    value = common.tensor_record(output)
    require(value["finite"] and value["dtype"] == "torch.float32", "Nonfinite or non-FP32 output")
    return value


def capture_control(torch, model, x, kernel, scale):
    # Exactly the runner's inference tensor flags and contiguous clone policy.
    with torch.inference_mode():
        sx = x.clone(memory_format=torch.contiguous_format)
        sk = kernel.clone(memory_format=torch.contiguous_format)
        torch.ops.converse2d.clear_cache()
        record = snapshot(torch, lambda: model(sx, sk, scale))
        return dict(output=record, flags=dict(x=base.tensor_flags(sx), kernel=base.tensor_flags(sk)),
                    inputs=common.records(dict(x=sx, kernel=sk)))


def profile_model_calls(torch, model, x, kernel, case):
    """Untimed CPU dispatcher trace of a GPU forward; production dispatch unchanged."""
    from models.util_converse import LayerNorm
    feature = {}
    layer = next(module for module in model.modules() if type(module) is LayerNorm)
    def remember(_, inputs):
        if "value" not in feature:
            feature["original_input_flags"] = base.tensor_flags(inputs[0])
            feature["value"] = inputs[0].detach().clone()
    handle = layer.register_forward_pre_hook(remember)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as trace:
            model(x, kernel, case["scale"])
        torch.cuda.synchronize()
    finally:
        handle.remove()
    calls = [dict(name=event.name, input_shapes=event.input_shapes)
             for event in trace.events() if event.name in ("converse2d::forward", "converse2d::_channel_layernorm")]
    solvers = [row for row in calls if row["name"] == "converse2d::forward"]
    norms = [row for row in calls if row["name"] == "converse2d::_channel_layernorm"]
    expected = declared_shapes(case)
    require(len(solvers) == 40, "Expected five DataNet and35 prior operator calls")
    prior = [row for row in solvers if row["input_shapes"][0][1] == 128]
    data = [row for row in solvers if row["input_shapes"][0][1] == 64]
    require(len(prior) == 35 and len(data) == 5, "Unexpected real-model operator call mix")
    require(all(row["input_shapes"][0][-2:] == expected["prior_padded_fft"] for row in prior),
            "Actual post-padding prior FFT input differs from declared shape")
    require(all(row["input_shapes"][1][-2:] == expected["datanet_output_fft"] for row in data),
            "Actual DataNet output/prior FFT shape differs from declaration")
    return dict(scope="Untimed CPU dispatcher events; actual CUDA operator input shapes after Python padding.",
                calls=calls, datanet_count=len(data), prior_count=len(prior),
                production_layernorm_dispatch_count=len(norms),
                first_layernorm_input_flags=feature["original_input_flags"],
                sampled_feature_clone_flags=base.tensor_flags(feature["value"])), feature["value"]


def run(torch, args, report, sources):
    from models.converse_usrnet import ConverseUSRNet
    from models.cuda_graph import USRNetCUDAGraph
    from models import util_converse as util
    namespace = dict(vars(util))
    exec(compile(sources["old_forward_source"], "pinned_historical_layernorm", "exec"), namespace)
    historical = namespace["forward"]
    model = ConverseUSRNet(backend="cuda").cuda().eval()
    checkpoint = ROOT / "model_zoo/converse_usrnet.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    modules = [module for module in model.modules() if type(module) is util.LayerNorm]
    originals = [(module, "forward" in vars(module), vars(module).get("forward")) for module in modules]
    initial = common.records(dict(model.state_dict()))
    versions = {name: value._version for name, value in model.state_dict(keep_vars=True).items()}
    require(not hasattr(model, MARKER), "Unexpected pre-existing ablation marker")
    setattr(model, MARKER, True)
    report.update(checkpoint_sha256=common.file_sha(checkpoint), norm_module_count=len(modules), cases={})

    @contextmanager
    def route(name):
        previous = getattr(model, MARKER)
        setattr(model, MARKER, name == NAMES[1])
        if name == NAMES[0]:
            for module, _, _ in originals:
                module.forward = types.MethodType(historical, module)
        try:
            yield
        finally:
            for module, had, original in originals:
                if had:
                    module.forward = original
                elif "forward" in vars(module):
                    del module.forward
            setattr(model, MARKER, previous)

    try:
        for case in select_cases(args.cases):
            seed = case_seed(case, args.seed)
            x, kernel = make_fixture(torch, case, seed, "cuda")
            alternate = dict(case, lr=(case["lr"][0] + 1, case["lr"][1] + 2))
            ax, ak = make_fixture(torch, alternate, seed + 1000, "cuda")
            scale = case["scale"]
            for mode in args.modes.split(","):
                key = case["name"] + "/" + mode
                row = dict(case=case, seed=seed, mode=mode, declared=declared_shapes(case),
                    caller_inputs=common.records(dict(x=x, kernel=kernel)),
                    caller_flags=dict(x=base.tensor_flags(x), kernel=base.tensor_flags(kernel)),
                    alternate_shape=alternate["lr"], eager_outputs={}, capture_controls={},
                    first_capture={}, lru_miss={}, lifecycle={}, formal_timings={})
                report["cases"][key] = row
                runners, disabled = {}, {}
                try:
                    with context(torch, mode), route(NAMES[1]):
                        row["actual_operator_shapes"], feature = profile_model_calls(torch, model, x, kernel, case)
                        # Real first-LN feature values, expressed in three layouts.
                        # These are numerical/dispatch probes, never latency claims.
                        fallback = {}
                        for layout in ("contiguous", "channels_last", "strided"):
                            value = layout_tensor(torch, feature, layout)
                            observations = {}
                            for name in NAMES:
                                with route(name):
                                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                                        observations[name] = snapshot(torch, lambda: modules[0](value))
                                    observations[name]["fused_dispatch_count"] = sum(event.name == "converse2d::_channel_layernorm" for event in trace.events())
                            require(observations[NAMES[0]]["sha256"] == observations[NAMES[1]]["sha256"],
                                    "LayerNorm feature-layout numerical gate failed")
                            require(observations[NAMES[1]]["fused_dispatch_count"] == (1 if layout == "contiguous" else 0),
                                    "Unexpected existing LayerNorm layout dispatch/fallback")
                            fallback[layout] = dict(flags=base.tensor_flags(value), routes=observations)
                        row["layernorm_layout_probes"] = fallback
                        del feature, value
                    for name in NAMES:
                        with context(torch, mode), route(name):
                            torch.ops.converse2d.clear_cache()
                            output, timing = base.one_call(torch, lambda: model(x, kernel, scale))
                            row["eager_outputs"][name] = dict(output=common.tensor_record(output), empty_eager_cache_timing=timing)
                            del output
                            row["capture_controls"][name] = dict(primary=capture_control(torch, model, x, kernel, scale),
                                                                alternate=capture_control(torch, model, ax, ak, scale))
                            disabled[name] = USRNetCUDAGraph(model, enabled=False, max_graphs=1, warmup=args.graph_warmup)
                            fallback_output = snapshot(torch, lambda: disabled[name](x, kernel, scale))
                            row.setdefault("disabled_runner", {})[name] = dict(output=fallback_output,
                                byte_equal=fallback_output["sha256"] == row["eager_outputs"][name]["output"]["sha256"],
                                captures=disabled[name].captures)
                    require(all(row["eager_outputs"][name]["output"]["finite"] for name in NAMES), "Nonfinite eager output")
                    require(row["eager_outputs"][NAMES[0]]["output"]["sha256"] == row["eager_outputs"][NAMES[1]]["output"]["sha256"],
                            "Original-caller eager old/production byte gate failed")
                    require(all(row["disabled_runner"][name]["byte_equal"] and row["disabled_runner"][name]["captures"] == 0 for name in NAMES),
                            "Explicit disabled-runner eager fallback changed behavior")
                    for shape in ("primary", "alternate"):
                        require(row["capture_controls"][NAMES[0]][shape]["output"]["sha256"] ==
                                row["capture_controls"][NAMES[1]][shape]["output"]["sha256"], "Matched capture-context old/production byte gate failed")
                    for name in NAMES:
                        with context(torch, mode), route(name):
                            torch.ops.converse2d.clear_cache()
                            def first():
                                runners[name] = USRNetCUDAGraph(model, max_graphs=1, warmup=args.graph_warmup)
                                return runners[name](x, kernel, scale)
                            output, timing = base.one_call(torch, first)
                            record = common.tensor_record(output)
                            row["first_capture"][name] = dict(output=record, timing=timing, captures=runners[name].captures,
                                matched_control_equal=record["sha256"] == row["capture_controls"][name]["primary"]["output"]["sha256"],
                                original_caller_eager_equal=record["sha256"] == row["eager_outputs"][name]["output"]["sha256"])
                            del output
                            require(row["first_capture"][name]["matched_control_equal"] and runners[name].captures == 1,
                                    "First capture differs from its matched input/mode/cache control")
                            before = runners[name].captures
                            output, timing = base.one_call(torch, lambda: runners[name](ax, ak, scale))
                            record = common.tensor_record(output)
                            row["lru_miss"][name] = dict(output=record, timing=timing, capture_delta=runners[name].captures - before,
                                matched_control_equal=record["sha256"] == row["capture_controls"][name]["alternate"]["output"]["sha256"])
                            del output
                            require(row["lru_miss"][name]["capture_delta"] == 1 and row["lru_miss"][name]["matched_control_equal"],
                                    "LRU recapture numerical/lifecycle gate failed")
                            primary = runners[name](x, kernel, scale)
                            saved = primary.clone()
                            shifted = x * .99 + .001
                            shifted_kernel = kernel * .98 + .02 / 49
                            moved = runners[name](shifted, shifted_kernel, scale)
                            control = capture_control(torch, model, shifted, shifted_kernel, scale)
                            require(primary.data_ptr() != moved.data_ptr() and torch.equal(primary, saved), "Graph output storage was reused")
                            require(common.tensor_record(moved)["sha256"] == control["output"]["sha256"], "Dynamic input/kernel copy byte gate failed")
                            # Same shape, different caller strides: capture key intentionally omits layout.
                            before = runners[name].captures
                            layout_x = layout_tensor(torch, x, "strided" if x.is_contiguous() else "contiguous")
                            layout_k = layout_tensor(torch, kernel, "strided" if kernel.is_contiguous() else "contiguous")
                            require(layout_x.stride() != x.stride() and layout_k.stride() != kernel.stride(),
                                    "Changed-layout lifecycle probe did not actually change strides")
                            out = snapshot(torch, lambda: runners[name](layout_x, layout_k, scale))
                            layout_control = capture_control(torch, model, layout_x, layout_k, scale)
                            require(runners[name].captures == before and out["sha256"] == layout_control["output"]["sha256"],
                                    "Same-shape changed-layout copy/cache contract failed")
                            row["lifecycle"][name] = dict(outputs_independent=True, changed_input_and_kernel_match_control=True,
                                same_shape_layout_reused_graph=True, changed_layout_flags=dict(x=base.tensor_flags(layout_x), kernel=base.tensor_flags(layout_k)))
                            del primary, saved, moved, shifted, shifted_kernel, layout_x, layout_k
                    signatures = []
                    for name in NAMES:
                        with context(torch, mode), route(name):
                            signatures.append(runners[name]._model_signature(x.device)[0])
                    require(signatures[0] != signatures[1], "Route marker did not separate Graph signatures")
                    row["public_marker_separates_graph_signatures"] = True
                    row["strict_numerical_lifecycle_gates_passed"] = True
                    kinds = ("eager_warm", "disabled_runner_warm", "graph_runner_hit")
                    timings = {kind: {name: [] for name in NAMES} for kind in kinds}
                    single = argparse.Namespace(warmup=0, rounds=1, iters=args.iters)
                    for name in NAMES:
                        with context(torch, mode), route(name):
                            for _ in range(args.warmup):
                                model(x, kernel, scale)
                                disabled[name](x, kernel, scale)
                                runners[name](x, kernel, scale)
                    capture_counts = {name: runners[name].captures for name in NAMES}
                    for index in range(args.rounds):
                        order = NAMES if index % 2 == 0 else tuple(reversed(NAMES))
                        for name in order:
                            with context(torch, mode), route(name):
                                functions = dict(eager_warm=lambda: model(x, kernel, scale),
                                    disabled_runner_warm=lambda: disabled[name](x, kernel, scale),
                                    graph_runner_hit=lambda: runners[name](x, kernel, scale))
                                for kind in kinds:
                                    timing = common.measure(torch, functions[kind], single)["rounds"][0]
                                    timing.update(paired_round=index, order=list(order), affinity=base.affinity())
                                    timings[kind][name].append(timing)
                    require(all(runners[name].captures == capture_counts[name] for name in NAMES), "Warm Graph timing recaptured")
                    row["formal_timings"] = {kind: {name: base.summarize(values) for name, values in routes.items()} for kind, routes in timings.items()}
                    row["paired_ratios"] = {kind: {metric: [values[NAMES[0]][i][metric] / values[NAMES[1]][i][metric]
                        for i in range(args.rounds)] for metric in ("wall_ms", "cuda_event_ms")} for kind, values in timings.items()}
                    for name in NAMES:
                        with context(torch, mode), route(name):
                            before = runners[name].captures
                            runners[name].clear()
                            require(runners[name].cached_graphs == 0, "clear() retained graphs")
                            output, timing = base.one_call(torch, lambda: runners[name](x, kernel, scale))
                            record = common.tensor_record(output)
                            del output
                            require(runners[name].captures == before + 1 and record["sha256"] == row["first_capture"][name]["output"]["sha256"],
                                    "clear/recapture changed output or lifecycle")
                            row["lifecycle"][name].update(clear_recaptured_once=True, clear_recapture_timing=timing)
                    row["status"] = "complete"
                    print(json.dumps(dict(event="case_complete", case=key, wall_ratios={kind: statistics.median(values["wall_ms"])
                        for kind, values in row["paired_ratios"].items()})), flush=True)
                finally:
                    for runner in [*runners.values(), *disabled.values()]:
                        runner.clear()
                    torch.ops.converse2d.clear_cache()
            del x, kernel, ax, ak
    finally:
        for module, had, original in originals:
            if had:
                module.forward = original
            elif "forward" in vars(module):
                del module.forward
        delattr(model, MARKER)
        report["model_state_unchanged"] = initial == common.records(dict(model.state_dict()))
        report["tensor_versions_unchanged"] = versions == {name: value._version for name, value in model.state_dict(keep_vars=True).items()}
        report["bindings_restored"] = all(("forward" in vars(module)) == had and
            (not had or vars(module)["forward"] is original) for module, had, original in originals)
        require(all(report[field] for field in ("model_state_unchanged", "tensor_versions_unchanged", "bindings_restored")),
                "Benchmark mutated model or left temporary bindings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seed", type=int, default=8011)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--graph-warmup", type=int, default=2)
    parser.add_argument("--validate-source-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh evidence output")
    cases = select_cases(args.cases)
    require(args.warmup >= 1 and min(args.rounds, args.iters, args.graph_warmup) >= 1, "Invalid repetitions")
    require(set(args.modes.split(",")) <= set(MODES) and len(args.modes.split(",")) == len(set(args.modes.split(","))), "Invalid/duplicate modes")
    sources = base.verified_sources(ROOT)
    if args.validate_source_only:
        print(json.dumps(dict(status="passed", gpu_execution=False, cases=cases, modes=args.modes,
                              production_source_sha256=sources["production_normalized_source_sha256"]), indent=2))
        return
    require(os.environ.get("CONVERSE_MSVC_VERSION") == "14.44" and os.environ.get("TORCH_CUDA_ARCH_LIST") == "12.0",
            "Use the checked manifest environment: CONVERSE_MSVC_VERSION=14.44, TORCH_CUDA_ARCH_LIST=12.0")
    affinity = base.affinity()
    require(affinity.get("process_mask_hex", "").lower() == "0xc03c03", "Run through the approved0xC03C03 affinity wrapper")
    require(not os.environ.get("CONVERSE2D_BACKEND") and os.environ.get("CONVERSE2D_CPU_ONLY") != "1", "Unset backend overrides")
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT))
    import torch
    require(torch.cuda.is_available(), "CUDA required after exclusive scheduling")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    require(not torch.is_autocast_enabled("cuda"), "Autocast must remain disabled")
    torch.manual_seed(args.seed)
    spec = importlib.util.spec_from_file_location("inference_deployment_checked_loader", ROOT / "test/extension_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    loader.load_extension()
    manifest_path = ROOT / ".build/cuda/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources_before = loader.production_source_hashes()
    files = [Path(__file__), Path(base.__file__), Path(common.__file__), ROOT / "tools/training_followup/inference_cases.py",
             ROOT / "models/converse_usrnet.py", ROOT / "models/util_converse.py", ROOT / "models/converse_core.py",
             ROOT / "models/cuda_graph.py", ROOT / "test/extension_loader.py", ROOT / "model_zoo/converse_usrnet.pth", manifest_path]
    identities = {str(path): common.file_sha(path) for path in files}
    report = dict(kind="existing_production_layernorm_deployment_coverage", status="running",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), historical_control=sources,
        checked_build_manifest=manifest, source_sha256=identities, production_source_sha256=sources_before,
        settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            tf32=False, amp=False, deterministic_algorithms=True, cudnn_deterministic=True, affinity=affinity,
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            msvc=os.environ["CONVERSE_MSVC_VERSION"], cuda_arch=os.environ["TORCH_CUDA_ARCH_LIST"]),
        scope="Existing production LayerNorm only. Same full pretrained model/solver/runner; claims restricted to measured case/mode/layout. No new fusion support, production dispatch change, training speedup or image-quality claim.",
        cold_scope="Empty eager spectrum cache or empty runner in an initialized process, NOT process-cold CUDA/library plans. First call includes warmup+capture+copies+output clone; warm hits and LRU/clear misses remain separate.",
        timing_scope="Complete public eager, explicitly disabled runner, and enabled runner calls measured separately. Graph timings include signature/lock/copies/replay/output clone. Fixtures/H2D/profiling/route installation/numeric CPU snapshots are outside formal timing. AB/BA paired rounds.",
        graph_gate_scope="Compare identical inference-mode contiguous clones with capture tensor/cache eligibility. Original-caller eager differences remain recorded and are never relabeled. Layout conversion/copy, changed kernel, output ownership, clear and LRU recapture are checked.",
        memory_scope="Per-call/batch additional PyTorch allocated peak above live state, not total process VRAM or library allocations.")
    try:
        run(torch, args, report, sources)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["source_integrity_passed"] = identities == {path: common.file_sha(path) for path in identities} and sources_before == loader.production_source_hashes()
        report["checked_binary_integrity_passed"] = common.file_sha(ROOT / ".build/cuda" / manifest["library"]) == manifest["binary_sha256"]
        report["affinity_after"] = base.affinity()
        report["all_round_affinities_match"] = all(timing["affinity"] == affinity
            for case in report.get("cases", {}).values() for routes in case.get("formal_timings", {}).values()
            for record in routes.values() for timing in record["rounds"])
        if (not report["source_integrity_passed"] or not report["checked_binary_integrity_passed"]
                or report["affinity_after"] != affinity or not report["all_round_affinities_match"]):
            report.update(status="failed", integrity_error="Source/binary/affinity changed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        require(report["status"] == "complete", "Deployment coverage incomplete; preserve failure report")


if __name__ == "__main__":
    main()
