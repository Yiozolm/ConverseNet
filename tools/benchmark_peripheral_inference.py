"""Paired inference-only peripheral and same-model Graph-runner comparisons.

Every peripheral pair uses identical tensors and alternating AB/BA rounds.
The fused route includes the actual Python helper/module overhead. Graph A/B
loads the frozen old runner with imports bound to the selected current model;
both capture that SAME model, so model/kernel changes are not Graph savings.
No backward, optimizer, FP64 production backend or model-quality claim.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import benchmark_fp32_p0 as common

TOOL_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("wall_ms", "cuda_event_ms", "peak_extra_allocated_bytes")


def layernorm_reference(torch, x, weight, bias, eps=1e-5):
    mean = x.mean(1, keepdim=True)
    variance = (x - mean).pow(2).mean(1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return weight[:, None, None] * normalized + bias[:, None, None]


def paired_measure(torch, functions, args):
    names = list(functions)
    for function in functions.values():
        for _ in range(args.warmup):
            function()
    torch.cuda.synchronize()
    rows = {name: [] for name in names}
    single = argparse.Namespace(warmup=0, rounds=1, iters=args.iters)
    for index in range(args.rounds):
        order = names if index % 2 == 0 else list(reversed(names))
        for position, name in enumerate(order):
            row = common.measure(torch, functions[name], single)["rounds"][0]
            row.update(paired_round=index, execution_position=position)
            rows[name].append(row)
    timings = {name: {"rounds": values, "median": {
        metric: statistics.median(value[metric] for value in values) for metric in METRICS}}
        for name, values in rows.items()}
    paired = {metric: [rows[names[0]][i][metric] / rows[names[1]][i][metric]
                       for i in range(args.rounds)] for metric in ("wall_ms", "cuda_event_ms")}
    ratios = {metric: {"paired_speedups": values, "median": statistics.median(values),
                       "range": [min(values), max(values)]} for metric, values in paired.items()}
    return timings, ratios


def inference_call(torch, mode, function):
    context = torch.no_grad if mode == "no_grad" else torch.inference_mode
    def call():
        with context():
            return function()
    return call


def check_output(actual, control):
    metrics = {name: actual["fp64_error"][name] <= control["fp64_error"][name]
               for name in ("max_abs", "relative_l2")}
    return {"byte_equal_to_python_fp32": actual["sha256"] == control["sha256"],
            "metric_noninferiority": metrics, "passed": actual["finite"] and control["finite"] and all(metrics.values())}


def run_peripherals(torch, args):
    import models.util_converse as util

    class PlainLayerNorm(torch.nn.Module):
        """Literal original module expression, without the new affine helper."""
        def __init__(self, weight, bias):
            super().__init__()
            self.weight, self.bias = weight, bias
            self.eps, self.data_format = 1e-5, "channels_first"
            self.normalized_shape = (weight.numel(),)

        def forward(self, x):
            if self.data_format == "channels_last":
                return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                mean = x.mean(1, keepdim=True)
                variance = (x - mean).pow(2).mean(1, keepdim=True)
                normalized = (x - mean) / torch.sqrt(variance + self.eps)
                return self.weight[:, None, None] * normalized + self.bias[:, None, None]

    cases = {}
    for index, shape in enumerate(((1, 64, 20, 24), (1, 64, 96, 96), (4, 64, 96, 96))):
        generator = torch.Generator().manual_seed(args.seed + index)
        x = torch.randn(shape, generator=generator).cuda()
        branch = torch.randn(shape, generator=generator).cuda()
        weight = (1 + torch.randn(shape[1], generator=generator) * .1).cuda()
        bias = (torch.randn(shape[1], generator=generator) * .1).cuda()
        alpha = weight.reshape(1, shape[1], 1, 1)
        layer = util.LayerNorm(shape[1], eps=1e-5, data_format="channels_first").cuda().eval()
        if hasattr(layer, "backend"):
            layer.backend = "cuda"
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        plain_layer = PlainLayerNorm(layer.weight, layer.bias).eval()
        for kind in ("alpha", "layernorm"):
            if kind == "alpha":
                plain = lambda: alpha * branch + x
                helper = getattr(util, "_alpha_residual", None)
                current = (lambda: helper(alpha, branch, x, backend="cuda")) if helper is not None else plain
                with torch.no_grad():
                    reference64 = alpha.double() * branch.double() + x.double()
                fixture = common.records({"alpha": alpha, "branch": branch, "residual": x})
                has_helper = helper is not None and hasattr(torch.ops.converse2d, "_alpha_residual")
            else:
                plain, current = lambda: plain_layer(x), lambda: layer(x)
                with torch.no_grad():
                    reference64 = layernorm_reference(torch, x.double(), weight.double(), bias.double())
                fixture = common.records({"x": x, "weight": weight, "bias": bias})
                has_helper = hasattr(util, "_channel_affine") and hasattr(torch.ops.converse2d, "_channel_affine")
            for mode in ("no_grad", "inference_mode"):
                functions = {"python_fp32": inference_call(torch, mode, plain),
                             "selected_helper": inference_call(torch, mode, current)}
                snapshots = {name: common.tensor_record(function(), reference64) for name, function in functions.items()}
                timings, ratios = paired_measure(torch, functions, args)
                key = f"{kind}/{'x'.join(map(str, shape))}/{mode}"
                cases[key] = {
                    "scope": "Forward only; both context-manager calls included; alpha uses actual helper, complete LN uses nn.Module on both sides and preserves original statistics",
                    "shape": shape, "mode": mode, "extension_helper_available": has_helper, "fixture": fixture,
                    "reference_fp64": common.tensor_record(reference64),
                    "routes": {name: {"output": snapshots[name], "timing": timings[name]} for name in functions},
                    "precision": check_output(snapshots["selected_helper"], snapshots["python_fp32"]),
                    "selected_helper_speedup": ratios,
                }
                print(key, "wall", ratios["wall_ms"]["median"], "CUDA", ratios["cuda_event_ms"]["median"], flush=True)
            del reference64
    return cases


def load_old_runner(path):
    name = "roadmap_frozen_graph_runner_same_model"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    # Dataclasses resolves its module while class definitions execute.
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module.USRNetCUDAGraph


def run_graph_pair(torch, args):
    from models.converse_usrnet import ConverseUSRNet
    from models.cuda_graph import USRNetCUDAGraph
    old_type = load_old_runner(args.old_runner)
    model = ConverseUSRNet(backend="cuda").cuda().eval()
    checkpoint = args.root / "model_zoo/converse_usrnet.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    generator = torch.Generator().manual_seed(args.seed + 100)
    x = torch.rand((1, 3, 32, 32), generator=generator).cuda()
    k = torch.rand((1, 1, 7, 7), generator=generator)
    k = (k / k.sum()).cuda()
    entries = {}
    for mode in ("no_grad", "inference_mode"):
        runners = {"frozen_old_runner": old_type(model, max_graphs=1, warmup=args.graph_warmup),
                   "selected_runner": USRNetCUDAGraph(model, max_graphs=1, warmup=args.graph_warmup)}
        try:
            eager = common.tensor_record(inference_call(torch, mode, lambda: model(x, k, 3))())
            functions = {name: inference_call(torch, mode, lambda runner=runner: runner(x, k, 3))
                         for name, runner in runners.items()}
            snapshots, captures = {}, {}
            for name, function in functions.items():
                torch.cuda.synchronize()
                tick = time.perf_counter()
                output = function()
                torch.cuda.synchronize()
                captures[name] = {"first_capture_and_call_wall_ms": (time.perf_counter() - tick) * 1000,
                                  "captures": runners[name].captures}
                snapshots[name] = common.tensor_record(output)
                del output
            before = {name: runner.captures for name, runner in runners.items()}
            timings, ratios = paired_measure(torch, functions, args)
            if any(runners[name].captures != count for name, count in before.items()):
                raise RuntimeError("A Graph-hit measurement unexpectedly recaptured")
            entries[mode] = {
                "scope": "One shared selected-root model object/checkpoint for both runner implementations; full __call__ includes lock, validation, signature, copies, replay, output clone and event",
                "fixture": common.records({"x": x, "kernel": k}), "checkpoint_sha256": common.file_sha(checkpoint),
                "eager_output": eager, "setup_not_used_for_speedup": captures,
                "routes": {name: {"output": snapshots[name], "timing": timings[name],
                                   "eager_byte_equal": snapshots[name]["sha256"] == eager["sha256"]} for name in functions},
                "runner_outputs_byte_equal": snapshots["frozen_old_runner"]["sha256"] == snapshots["selected_runner"]["sha256"],
                "selected_runner_speedup": ratios,
            }
            print("Graph", mode, "same-model wall", ratios["wall_ms"]["median"], "CUDA", ratios["cuda_event_ms"]["median"], flush=True)
        finally:
            for runner in runners.values():
                runner.clear()
            torch.ops.converse2d.clear_cache()
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TOOL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-runner", type=Path, default=TOOL_ROOT / "artifacts/fp32_roadmap/baseline/models/cuda_graph.py")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename; existing evidence is immutable")
    if args.warmup < 0 or min(args.rounds, args.iters, args.graph_warmup) < 1:
        parser.error("warmup must be nonnegative; rounds/iters/graph-warmup must be positive")
    if os.environ.get("CONVERSE2D_BACKEND") or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        raise RuntimeError("Unset backend overrides and CPU-only mode for checked CUDA measurements")
    args.root, args.old_runner = args.root.resolve(), args.old_runner.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["CONVERSE2D_SKIP_BUILD"] = "0" if args.build else "1"
    sys.path.insert(0, str(args.root))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(args.deterministic_algorithms)
    torch.manual_seed(args.seed)
    specification = importlib.util.spec_from_file_location("inference_checked_loader", args.root / "test/extension_loader.py")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    loader.load_extension()
    manifest = json.loads((args.root / ".build/cuda/source_manifest.json").read_text(encoding="utf-8"))
    sources = loader.production_source_hashes()
    files = {str(args.root / name): common.file_sha(args.root / name) for name in
             ("models/util_converse.py", "models/converse_usrnet.py", "models/converse_core.py", "models/cuda_graph.py", "test/extension_loader.py")}
    if not args.skip_graph:
        files[str(args.old_runner)] = common.file_sha(args.old_runner)
    harness, helper = common.file_sha(__file__), common.file_sha(common.__file__)
    result = {
        "kind": "peripheral_inference_and_same_model_graph", "root": str(args.root),
        "harness_sha256": harness, "helper_sha256": helper, "source_sha256": sources,
        "python_file_sha256": files, "checked_build_manifest": manifest,
        "settings": {name: getattr(args, name) for name in ("seed", "warmup", "rounds", "iters", "graph_warmup", "skip_graph", "deterministic_algorithms")},
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                        "tf32": False, "amp": False, "cudnn_deterministic": True,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "peripheral_cases": run_peripherals(torch, args),
        "timing_scope": "Alternating paired rounds of complete Python callables, GPU-resident inputs. Wall includes end synchronization. CUDA events measure stream spans, not isolated kernel instruction time.",
        "memory_scope": "Additional PyTorch allocated peak over live state; not driver/library memory or process VRAM.",
    }
    if not args.skip_graph:
        result["graph_runner_pair"] = run_graph_pair(torch, args)
        result["graph_runner_sources_differ"] = common.file_sha(args.old_runner) != common.file_sha(args.root / "models/cuda_graph.py")
        result["graph_comparison_note"] = "Both runners capture the exact same selected-root model, so any shared model-side CUDA inference changes affect both routes. First captures are separate setup observations, not Graph-hit speedup evidence."
    if sources != loader.production_source_hashes() or files != {path: common.file_sha(path) for path in files}:
        raise RuntimeError("Checked production or model/Graph sources changed during measurement")
    if common.file_sha(args.root / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
        raise RuntimeError("Checked binary changed during measurement")
    if harness != common.file_sha(__file__) or helper != common.file_sha(common.__file__):
        raise RuntimeError("Benchmark or frozen helper changed during measurement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
