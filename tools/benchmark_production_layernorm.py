"""Same-model LayerNorm ablation: 06a9673 statistics versus ff7f8ba production.

Uses one current model/checkpoint and the current checked CUDA binary for both
routes. Only LayerNorm.forward changes. Historical source is read from a pinned
Git object, checked against fixed SHA256 values, and included in the output.
No build option is provided. Run only in an exclusive GPU measurement window.
"""
import argparse
import ast
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import time
import types

import benchmark_fp32_p0 as common

ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = "06a96735e65bd7c5187517f737a3a25a2944f8ae"
PRODUCTION_COMMIT = "ff7f8ba53f102bde716eefb8c83602d3ed9de1c5"
OLD_SOURCE_SHA = "9371e8b41b16a6778d91655c0c31140d0524ec0566e760d94ea3c16840af0619"
OLD_FORWARD_SHA = "48285b577fe2a9c8c2d09276d059f80e185627bdfaab4ffcf4bd8b7dc3f6b0ec"
PRODUCTION_SOURCE_SHA = "b98320320a82d64859add34befc4a705ba23ff99bb858241bbaa416b0da9e635"
PRODUCTION_FORWARD_SHA = "b39983cd09b5e859dc00d4b7d77c9a8475b025d111f75874c8118be1e6ffa63e"
AFFINE_SHA = "3f6e5e2ea388fea1cbdb510b74f3ce9e9452482ba9c50e43a9ce04bd62a0b38c"
NAMES = ("old_statistics", "production_full_layernorm")
MARKER = "production_layernorm_ablation"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def function_source(source, name, class_name=None):
    nodes = ast.parse(source).body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    return textwrap.dedent(ast.get_source_segment(source, node)) + "\n"


def verified_sources(root):
    """CPU-only; full historical bytes and exact method text have fixed hashes."""
    old_bytes = subprocess.check_output(["git", "-C", str(root), "show", f"{OLD_COMMIT}:models/util_converse.py"])
    if digest(old_bytes) != OLD_SOURCE_SHA:
        raise RuntimeError("Pinned historical util_converse.py SHA256 mismatch")
    old = old_bytes.decode("utf-8")
    current = (root / "models/util_converse.py").read_text(encoding="utf-8")
    if digest(current.encode()) != PRODUCTION_SOURCE_SHA:
        raise RuntimeError("Current util_converse.py must match frozen ff7f8ba (normalized newlines)")
    old_forward = function_source(old, "forward", "LayerNorm")
    current_forward = function_source(current, "forward", "LayerNorm")
    if digest(old_forward.encode()) != OLD_FORWARD_SHA or digest(current_forward.encode()) != PRODUCTION_FORWARD_SHA:
        raise RuntimeError("LayerNorm.forward source hash mismatch")
    for source in (old, current):
        if digest(function_source(source, "_channel_affine").encode()) != AFFINE_SHA:
            raise RuntimeError("Historical/current large affine policy dependency changed")
    return dict(old_commit=OLD_COMMIT, production_commit=PRODUCTION_COMMIT,
                old_complete_source=old, old_complete_source_sha256=OLD_SOURCE_SHA,
                old_forward_source=old_forward, old_forward_sha256=OLD_FORWARD_SHA,
                production_normalized_source_sha256=PRODUCTION_SOURCE_SHA,
                production_forward_sha256=PRODUCTION_FORWARD_SHA, shared_affine_source_sha256=AFFINE_SHA)


def affinity():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.GetProcessAffinityMask.restype = ctypes.c_int
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(cpus=[index for index in range(8 * ctypes.sizeof(process)) if process.value & (1 << index)],
                    process_mask_hex=hex(process.value), system_mask_hex=hex(system.value))
    return dict(cpus=sorted(os.sched_getaffinity(0)))


def one_call(torch, function):
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    tick = time.perf_counter()
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return output, dict(wall_ms=(time.perf_counter() - tick) * 1000, cuda_event_ms=start.elapsed_time(end),
                        peak_extra_allocated_bytes=max(0, torch.cuda.max_memory_allocated() - initial))


def summarize(rows):
    return dict(rounds=rows, median={key: statistics.median(row[key] for row in rows)
                                    for key in ("wall_ms", "cuda_event_ms", "peak_extra_allocated_bytes")})


def tensor_flags(tensor):
    return dict(is_inference=tensor.is_inference(), requires_grad=tensor.requires_grad,
                is_neg=tensor.is_neg(), is_conj=tensor.is_conj(), is_contiguous=tensor.is_contiguous(),
                stride=list(tensor.stride()), storage_offset=tensor.storage_offset())


def run(torch, args, report, sources):
    from models.converse_usrnet import ConverseUSRNet
    from models.cuda_graph import USRNetCUDAGraph
    from models import util_converse

    historical_globals = dict(vars(util_converse))
    exec(compile(sources["old_forward_source"], f"git:{OLD_COMMIT}:LayerNorm.forward", "exec"), historical_globals)
    old_forward = historical_globals["forward"]
    model = ConverseUSRNet(backend="cuda").cuda().eval()
    checkpoint = args.root / "model_zoo/converse_usrnet.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    initial_state = common.records(dict(model.state_dict()))
    initial_versions = {name: tensor._version for name, tensor in model.state_dict(keep_vars=True).items()}
    modules = [module for module in model.modules()
               if type(module) is util_converse.LayerNorm and module.data_format == "channels_first"]
    if not modules or any(module.forward.__func__ is not util_converse.LayerNorm.forward for module in modules):
        raise RuntimeError("Expected unmodified production LayerNorm instance methods")
    originals = [(module, "forward" in vars(module), vars(module).get("forward")) for module in modules]
    had_marker, previous_marker = MARKER in vars(model), vars(model).get(MARKER)
    setattr(model, MARKER, True)

    @contextmanager
    def route(name):
        previous = getattr(model, MARKER)
        is_old = name == NAMES[0]
        setattr(model, MARKER, not is_old)
        if is_old:
            for module, _, _ in originals:
                module.forward = types.MethodType(old_forward, module)
        try:
            yield
        finally:
            if is_old:
                for module, had_override, original in originals:
                    if had_override:
                        module.forward = original
                    else:
                        del module.forward
            setattr(model, MARKER, previous)

    def mode_context(mode):
        return torch.no_grad() if mode == "no_grad" else torch.inference_mode()

    fixtures = {}
    for batch in (1, 4):
        generator = torch.Generator().manual_seed(args.seed + batch)
        x = torch.rand(batch, 3, 32, 32, generator=generator).cuda()
        alternate = torch.rand(batch, 3, 31, 33, generator=generator).cuda()
        kernel = torch.rand(batch, 1, 7, 7, generator=generator)
        kernel = (kernel / kernel.sum((-2, -1), keepdim=True)).cuda()
        fixtures[batch] = (x, alternate, kernel)
    report.update(checkpoint_sha256=common.file_sha(checkpoint), initial_model_state=initial_state,
                  initial_tensor_versions=initial_versions, norm_module_count=len(modules), cases={})
    expected = {}
    try:
        for batch, (x, alternate, kernel) in fixtures.items():
            for mode in ("no_grad", "inference_mode"):
                row = dict(fixture=common.records(dict(x=x, alternate_x=alternate, kernel=kernel)),
                           fixture_flags={name: tensor_flags(value) for name, value in
                                          (("x", x), ("alternate_x", alternate), ("kernel", kernel))},
                           eager_cold={}, eager_numerical={}, graph_eager_capture_context_control={})
                report["cases"][f"B{batch}/{mode}"] = row
                for shape_name, value in (("primary", x), ("alternate", alternate)):
                    outputs = {}
                    for name in NAMES:
                        with mode_context(mode), route(name):
                            torch.ops.converse2d.clear_cache()
                            output, timing = one_call(torch, lambda: model(value, kernel, 3))
                            outputs[name] = common.tensor_record(output)
                            del output
                        if shape_name == "primary":
                            row["eager_cold"][name] = timing
                    equal = outputs[NAMES[0]]["sha256"] == outputs[NAMES[1]]["sha256"]
                    finite = all(value["finite"] for value in outputs.values())
                    row["eager_numerical"][shape_name] = dict(outputs=outputs, byte_equal=equal, all_outputs_finite=finite)
                    expected[(batch, mode, shape_name)] = {name: value["sha256"] for name, value in outputs.items()}
                    controls = {}
                    for name in NAMES:
                        # _capture creates both static inputs inside inference_mode.
                        # Match these flags as well as mode: normal leaf kernels can
                        # take a different eager preparation/cache path.
                        with torch.inference_mode(), route(name):
                            static_x = value.clone(memory_format=torch.contiguous_format)
                            static_kernel = kernel.clone(memory_format=torch.contiguous_format)
                            torch.ops.converse2d.clear_cache()
                            control = model(static_x, static_kernel, 3)
                            controls[name] = dict(output=common.tensor_record(control),
                                input_flags=dict(x=tensor_flags(static_x), kernel=tensor_flags(static_kernel)),
                                input_snapshots=common.records(dict(x=static_x, kernel=static_kernel)))
                            del control, static_x, static_kernel
                    row["graph_eager_capture_context_control"][shape_name] = controls
        report["all_eager_outputs_byte_equal"] = all(item["byte_equal"] for row in report["cases"].values()
                                                        for item in row["eager_numerical"].values())
        report["all_eager_outputs_finite"] = all(item["all_outputs_finite"] for row in report["cases"].values()
                                                  for item in row["eager_numerical"].values())
        if not report["all_eager_outputs_byte_equal"] or not report["all_eager_outputs_finite"]:
            raise RuntimeError("Full-model eager byte gate failed; formal timing blocked")
        report["all_capture_context_eager_routes_byte_equal"] = all(
            controls[NAMES[0]]["output"]["sha256"] == controls[NAMES[1]]["output"]["sha256"]
            and all(control["output"]["finite"] for control in controls.values())
            for row in report["cases"].values() for controls in row["graph_eager_capture_context_control"].values())
        if not report["all_capture_context_eager_routes_byte_equal"]:
            raise RuntimeError("Capture-context cloned-input eager byte gate failed; formal timing blocked")

        for batch, (x, alternate, kernel) in fixtures.items():
            for mode in ("no_grad", "inference_mode"):
                row = report["cases"][f"B{batch}/{mode}"]
                runners = {}
                row.update(graph_first_call={}, graph_lru_misses={name: [] for name in NAMES}, lifecycle={})
                try:
                    for name in NAMES:
                        with mode_context(mode), route(name):
                            def first_call():
                                runners[name] = USRNetCUDAGraph(model, max_graphs=1, warmup=args.graph_warmup)
                                return runners[name](x, kernel, 3)
                            output, timing = one_call(torch, first_call)
                            snapshot = common.tensor_record(output)
                            row["graph_first_call"][name] = dict(timing=timing, output=snapshot, captures=runners[name].captures,
                                eager_byte_equal=snapshot["sha256"] == expected[(batch, mode, "primary")][name],
                                capture_context_eager_byte_equal=snapshot["sha256"] ==
                                    row["graph_eager_capture_context_control"]["primary"][name]["output"]["sha256"])
                            del output
                    signatures = []
                    for name in NAMES:
                        with mode_context(mode), route(name):
                            signatures.append(runners[name]._model_signature(x.device)[0])
                    row["public_marker_separates_signatures"] = signatures[0] != signatures[1]
                    row["graph_first_routes_byte_equal"] = (
                        row["graph_first_call"][NAMES[0]]["output"]["sha256"] ==
                        row["graph_first_call"][NAMES[1]]["output"]["sha256"])
                    if not row["public_marker_separates_signatures"] or not all(
                            v["capture_context_eager_byte_equal"] and v["captures"] == 1 for v in row["graph_first_call"].values()
                            ) or not row["graph_first_routes_byte_equal"]:
                        raise RuntimeError("Graph context signature or first-output byte gate failed")

                    for index in range(args.graph_misses):
                        value, shape = (alternate, "alternate") if index % 2 == 0 else (x, "primary")
                        for name in NAMES if index % 2 == 0 else tuple(reversed(NAMES)):
                            runner = runners[name]
                            with mode_context(mode), route(name):
                                before = runner.captures
                                output, timing = one_call(torch, lambda: runner(value, kernel, 3))
                                snapshot = common.tensor_record(output)
                                miss = dict(paired_miss=index, shape=shape, timing=timing, output=snapshot,
                                            capture_delta=runner.captures - before,
                                            eager_byte_equal=snapshot["sha256"] == expected[(batch, mode, shape)][name],
                                            capture_context_eager_byte_equal=snapshot["sha256"] ==
                                                row["graph_eager_capture_context_control"][shape][name]["output"]["sha256"])
                                row["graph_lru_misses"][name].append(miss)
                                del output
                                if miss["capture_delta"] != 1 or not miss["capture_context_eager_byte_equal"]:
                                    raise RuntimeError("Graph LRU miss capture or output byte gate failed")
                        same = (row["graph_lru_misses"][NAMES[0]][index]["output"]["sha256"] ==
                                row["graph_lru_misses"][NAMES[1]][index]["output"]["sha256"])
                        if not same:
                            raise RuntimeError("Graph LRU miss cross-route byte gate failed")

                    for name, runner in runners.items():
                        with mode_context(mode), route(name):
                            output = runner(x, kernel, 3)
                            saved = output.clone()
                            shifted = x * .99 + .001
                            another = runner(shifted, kernel, 3)
                            shifted_eager = model(shifted, kernel, 3)
                            with torch.inference_mode():
                                static_shifted = shifted.clone(memory_format=torch.contiguous_format)
                                static_kernel = kernel.clone(memory_format=torch.contiguous_format)
                                shifted_capture_eager = model(static_shifted, static_kernel, 3)
                            independent = output.data_ptr() != another.data_ptr() and torch.equal(output, saved)
                            snapshot, shifted_snapshot = common.tensor_record(output), common.tensor_record(another)
                            eager_shifted_snapshot = common.tensor_record(shifted_eager)
                            capture_eager_snapshot = common.tensor_record(shifted_capture_eager)
                            lifecycle = dict(returned_outputs_independent=independent, primary_output=snapshot,
                                eager_byte_equal=snapshot["sha256"] == expected[(batch, mode, "primary")][name],
                                first_graph_byte_equal=snapshot["sha256"] == row["graph_first_call"][name]["output"]["sha256"],
                                shifted_output=shifted_snapshot, shifted_eager_output=eager_shifted_snapshot,
                                input_copy_byte_equal=shifted_snapshot["sha256"] == eager_shifted_snapshot["sha256"],
                                shifted_capture_context_eager_output=capture_eager_snapshot,
                                shifted_capture_context_input_flags=dict(x=tensor_flags(static_shifted), kernel=tensor_flags(static_kernel)),
                                shifted_capture_context_input_snapshots=common.records(dict(x=static_shifted, kernel=static_kernel)),
                                input_copy_capture_context_byte_equal=shifted_snapshot["sha256"] == capture_eager_snapshot["sha256"])
                            row["lifecycle"][name] = lifecycle
                            if not independent or not lifecycle["first_graph_byte_equal"] or not lifecycle["input_copy_capture_context_byte_equal"]:
                                raise RuntimeError("Graph output ownership/input-copy byte contract failed")
                            runner(x, kernel, 3)
                            del output, saved, another, shifted, shifted_eager, shifted_capture_eager, static_shifted, static_kernel
                    row["shifted_routes_byte_equal"] = (
                        row["lifecycle"][NAMES[0]]["shifted_output"]["sha256"] ==
                        row["lifecycle"][NAMES[1]]["shifted_output"]["sha256"])
                    if not row["shifted_routes_byte_equal"]:
                        raise RuntimeError("Shifted-input cross-route byte gate failed")

                    timings = {kind: {name: [] for name in NAMES} for kind in ("eager_warm", "graph_hit")}
                    single = argparse.Namespace(warmup=0, rounds=1, iters=args.iters)
                    for name in NAMES:
                        with mode_context(mode), route(name):
                            for _ in range(args.warmup):
                                model(x, kernel, 3)
                                runners[name](x, kernel, 3)
                    counts = {name: runner.captures for name, runner in runners.items()}
                    for index in range(args.rounds):
                        order = NAMES if index % 2 == 0 else tuple(reversed(NAMES))
                        for name in order:
                            with mode_context(mode), route(name):
                                for kind, function in (("eager_warm", lambda: model(x, kernel, 3)),
                                                       ("graph_hit", lambda: runners[name](x, kernel, 3))):
                                    timing = common.measure(torch, function, single)["rounds"][0]
                                    timing.update(paired_round=index, order=list(order), affinity=affinity())
                                    timings[kind][name].append(timing)
                    if any(runners[name].captures != count for name, count in counts.items()):
                        raise RuntimeError("Warm Graph measurement unexpectedly recaptured")
                    row["graph_hit_capture_counts_unchanged"] = True
                    row["formal_timings"] = {kind: {name: summarize(values) for name, values in routes.items()}
                                             for kind, routes in timings.items()}
                    row["paired_speedups"] = {kind: {metric: [routes[NAMES[0]][i][metric] / routes[NAMES[1]][i][metric]
                                                               for i in range(args.rounds)]
                                                        for metric in ("wall_ms", "cuda_event_ms")}
                                              for kind, routes in timings.items()}
                    for name, runner in runners.items():
                        with mode_context(mode), route(name):
                            before = runner.captures
                            runner.clear()
                            empty = runner.cached_graphs == 0
                            output, timing = one_call(torch, lambda: runner(x, kernel, 3))
                            snapshot = common.tensor_record(output)
                            row["lifecycle"][name].update(clear_emptied_cache=empty, recapture_delta=runner.captures - before,
                                recapture_timing=timing, recapture_output=snapshot,
                                recapture_byte_equal=snapshot["sha256"] == row["graph_first_call"][name]["output"]["sha256"])
                            del output
                            if not empty or runner.captures - before != 1 or not row["lifecycle"][name]["recapture_byte_equal"]:
                                raise RuntimeError("Graph clear/recapture contract failed")
                    print(f"B{batch}/{mode}", {kind: statistics.median(value["wall_ms"])
                                              for kind, value in row["paired_speedups"].items()}, flush=True)
                finally:
                    for runner in runners.values():
                        runner.clear()
                    torch.ops.converse2d.clear_cache()
        report["formal_timing_status"] = "complete"
    finally:
        for module, had_override, original in originals:
            if had_override:
                module.forward = original
            elif "forward" in vars(module):
                del module.forward
        if had_marker:
            setattr(model, MARKER, previous_marker)
        else:
            delattr(model, MARKER)
        report["model_state_unchanged"] = initial_state == common.records(dict(model.state_dict()))
        report["tensor_versions_unchanged"] = initial_versions == {name: tensor._version for name, tensor in model.state_dict(keep_vars=True).items()}
        report["all_instance_bindings_restored"] = all(("forward" in vars(module)) == had and
            (not had or vars(module)["forward"] is original) for module, had, original in originals)
        report["public_marker_restored"] = (MARKER in vars(model)) == had_marker and (not had_marker or vars(model)[MARKER] == previous_marker)
        if not all(report[name] for name in ("model_state_unchanged", "tensor_versions_unchanged", "all_instance_bindings_restored", "public_marker_restored")):
            raise RuntimeError("Ablation changed model state, tensor versions, or temporary bindings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1109)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--graph-misses", type=int, default=3)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--expected-affinity", help="Optional comma-separated CPU IDs; affinity is set externally")
    parser.add_argument("--validate-source-only", action="store_true", help="CPU-only source/AST/affinity validation; never imports torch")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Choose a new evidence output")
    if args.warmup < 0 or min(args.rounds, args.iters, args.graph_misses, args.graph_warmup) < 1:
        parser.error("Invalid repetition counts")
    sources = verified_sources(args.root)
    initial_affinity = affinity()
    if args.expected_affinity and initial_affinity["cpus"] != sorted(set(map(int, args.expected_affinity.split(",")))):
        raise RuntimeError("Process affinity does not match externally requested CPU IDs")
    if args.validate_source_only:
        print(json.dumps(dict(source_validation="passed", old_source_sha256=OLD_SOURCE_SHA,
                              production_forward_sha256=PRODUCTION_FORWARD_SHA, affinity=initial_affinity), indent=2))
        return
    if os.environ.get("CONVERSE2D_BACKEND") or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        raise RuntimeError("Unset backend overrides and CPU-only mode")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    sys.path.insert(0, str(args.root))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    specification = importlib.util.spec_from_file_location("production_ln_checked_loader", args.root / "test/extension_loader.py")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    loader.load_extension()
    if not hasattr(torch.ops.converse2d, "_channel_layernorm"):
        raise RuntimeError("Checked production build lacks channel LayerNorm")
    manifest_path = args.root / ".build/cuda/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    production_sources = loader.production_source_hashes()
    files = {str(args.root / name): common.file_sha(args.root / name) for name in (
        "models/converse_usrnet.py", "models/util_converse.py", "models/converse_core.py", "models/cuda_graph.py", "test/extension_loader.py")}
    files.update({str(Path(__file__).resolve()): common.file_sha(__file__), str(Path(common.__file__).resolve()): common.file_sha(common.__file__),
                  str(manifest_path): common.file_sha(manifest_path)})
    report = dict(kind="same_model_production_layernorm_ablation", status="running", root=str(args.root),
        historical_control=sources, checked_build_manifest=manifest, production_source_sha256=production_sources,
        file_sha256=files, settings={name: getattr(args, name) for name in ("seed", "warmup", "rounds", "iters", "graph_misses", "graph_warmup", "expected_affinity")},
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         tf32=False, amp=False, cudnn_deterministic=True, deterministic_algorithms=True,
                         torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                         affinity=initial_affinity, pid=os.getpid(), cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
        source_model_object_shared=True, graph_signature_public_marker=MARKER,
        scope="Same ff7f8ba model, weights, solver and Graph runner; only historical LayerNorm.forward versus true production LayerNorm.forward. B1/B4 LR32 s3: HR96, prior FFT100x100. Alternate LR31x33 for LRU misses.",
        cold_definition="Empty Converse eager spectrum cache or first empty Graph runner in an initialized process; CUDA/cuFFT/cuDNN plans may already be warm. Setup observations are not process-cold speedup claims.",
        timing_inclusions="Full public model/runner calls, including Python checks, signature, locks, input/kernel copies, replay, output clone and events. Context installation, CPU snapshots/source checks and fixture H2D setup excluded. Alternating AB/BA rounds; one shared model avoids competing eager caches.",
        graph_control_policy="Runner capture executes internally in inference_mode and clones x/kernel there. Same-route controls clone inputs inside inference_mode with identical contiguous_format, preserving preparation/cache eligibility; original and control flags/SHA are recorded. Graph old/current and same-route cloned-input eager controls are strict byte gates. Original normal-leaf/outer-mode eager comparisons remain recorded even if false. Clear/recapture and hit outputs must match the same route's first Graph output.",
        memory_scope="Additional PyTorch allocated peak over live state; excludes driver/library allocations and process VRAM.")
    try:
        run(torch, args, report, sources)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        failures = []
        if files != {path: common.file_sha(path) for path in files} or production_sources != loader.production_source_hashes():
            failures.append("Sources or checked manifest changed")
        if common.file_sha(args.root / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
            failures.append("Checked binary changed")
        report["affinity_after"] = affinity()
        if report["affinity_after"] != initial_affinity:
            failures.append("Process affinity changed")
        for row in report.get("cases", {}).values():
            for routes in row.get("formal_timings", {}).values():
                for route in routes.values():
                    if any(value["affinity"] != initial_affinity for value in route["rounds"]):
                        failures.append("Per-round process affinity changed")
        report["integrity_failures"] = failures
        if failures:
            report.update(status="failed", integrity_error="; ".join(failures))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if failures:
            raise RuntimeError(report["integrity_error"])
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
