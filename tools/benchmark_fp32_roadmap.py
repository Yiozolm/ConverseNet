"""Checked, separate-process roadmap benchmark with complete caller overhead.

The two old P0/PSF harnesses are imported without modification. Default work:
alpha residual and complete LayerNorm FWD/VJPs, full pretrained USRNet LR32/s3
Adam steps, eager inference and Graph first-call/LRU-miss/hit latency. Graph
measurements always invoke the public runner: checks, static input/kernel
copies, replay and independent output clone remain inside the timing interval.
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
import benchmark_fp32_psf as psf


def layernorm_reference(torch, data, eps=1e-5):
    x, weight, bias = data
    mean = x.mean(1, keepdim=True)
    variance = (x - mean).pow(2).mean(1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return weight[:, None, None] * normalized + bias[:, None, None]


def forward_vjp(torch, function, data, upstream, names):
    output = function(data)
    requested = [(name, value) for name, value in zip(names, data) if value.requires_grad]
    gradients = torch.autograd.grad(output, [value for _, value in requested], upstream)
    return {"output": output.detach(), **{name: value.detach() for (name, _), value in zip(requested, gradients)}}


def run_peripherals(torch, args):
    from models.util_converse import LayerNorm
    cases = {}
    for index, shape in enumerate(((1, 64, 20, 24), (4, 64, 96, 96))):
        generator = torch.Generator().manual_seed(args.seed + index)
        x = torch.randn(shape, generator=generator)
        weight = 1 + torch.randn(shape[1], generator=generator) * .1
        bias = torch.randn(shape[1], generator=generator) * .1
        branch = torch.randn(shape, generator=generator)
        upstream = torch.randn(shape, generator=generator).cuda() / x.numel() ** .5
        for kind in ("alpha", "layernorm"):
            raw = [weight.reshape(1, shape[1], 1, 1), branch, x] if kind == "alpha" else [x, weight, bias]
            names = ("dalpha", "dbranch", "dresidual") if kind == "alpha" else ("dx", "dweight", "dbias")
            masks = {"all": (True,) * 3,
                     "input_only": (False, True, True) if kind == "alpha" else (True, False, False),
                     "params_only": (True, False, False) if kind == "alpha" else (False, True, True)}
            for mask, needs in masks.items():
                data = tuple(value.cuda().detach().requires_grad_(need) for value, need in zip(raw, needs))
                double_data = tuple(value.to(device="cuda", dtype=torch.float64).detach().requires_grad_(need)
                                    for value, need in zip(raw, needs))
                reference = (lambda values: values[0] * values[1] + values[2]) if kind == "alpha" else lambda values: layernorm_reference(torch, values)
                layer = None
                if kind == "alpha":
                    available = hasattr(torch.ops.converse2d, "_alpha_residual")
                    function = (lambda values: torch.ops.converse2d._alpha_residual(*values)) if available else reference
                else:
                    layer = LayerNorm(shape[1], eps=1e-5, data_format="channels_first").cuda()
                    if hasattr(layer, "backend"):
                        layer.backend = "cuda"
                    with torch.no_grad():
                        layer.weight.copy_(data[1])
                        layer.bias.copy_(data[2])
                    layer.weight.requires_grad_(needs[1])
                    layer.bias.requires_grad_(needs[2])
                    data = (data[0], layer.weight, layer.bias)
                    function = lambda values: layer(values[0])
                    available = hasattr(torch.ops.converse2d, "_channel_affine")
                expected = forward_vjp(torch, reference, double_data, upstream.double(), names)
                control = forward_vjp(torch, reference, data, upstream, names)
                actual = forward_vjp(torch, function, data, upstream, names)
                snapshot, python_snapshot = common.records(actual, expected), common.records(control, expected)
                del expected, control, actual, double_data
                name = f"peripheral/{kind}/{'x'.join(map(str, shape))}/{mask}"
                item = {
                    "scope": "Full expression forward and selected VJPs; LayerNorm includes original mean, centered-square mean, sqrt and division",
                    "shape": shape, "needs_grad": dict(zip(names, needs)), "extension_helper_available": available,
                    "fixture": common.records(dict(zip(("arg0", "arg1", "arg2", "upstream"), (*data, upstream)))),
                    "snapshot": snapshot, "python_fp32": python_snapshot,
                    "timing": common.measure(torch, lambda: forward_vjp(torch, function, data, upstream, names), args),
                }
                item["python_fp32_noninferiority"] = common.noninferiority(item)
                cases[name] = item
                print(name, item["timing"]["median"], flush=True)
                del data, layer
    return cases


def measure_once(torch, function):
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    tick = time.perf_counter()
    start.record()
    output = function()
    end.record()
    end.synchronize()
    row = {"wall_ms": (time.perf_counter() - tick) * 1000, "cuda_event_ms": start.elapsed_time(end),
           "peak_extra_allocated_bytes": max(0, torch.cuda.max_memory_allocated() - initial)}
    return output, row


def timing_rows(rows, policy):
    return {"rounds": rows, "median": {key: statistics.median(row[key] for row in rows) for key in rows[0]},
            "timing_policy": policy}


def run_model_and_graph(torch, args):
    from models.converse_usrnet import ConverseUSRNet
    from models.cuda_graph import USRNetCUDAGraph
    checkpoint = args.root / "model_zoo/converse_usrnet.pth"
    model = ConverseUSRNet(backend="cuda").cuda()
    initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(initial, strict=True)
    generator = torch.Generator().manual_seed(args.seed + 100)
    size, scale = args.lr_size, args.scale
    x = torch.rand((1, 3, size, size), generator=generator).cuda()
    k = torch.rand((1, 1, 7, 7), generator=generator)
    k = (k / k.sum()).cuda()
    target = torch.rand((1, 3, size * scale, size * scale), generator=generator).cuda()
    input_records = common.records({"x": x, "kernel": k, "target": target})
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)
    def step(snapshot=False):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, k, scale)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        values = common.records({"output": output, "loss": loss}) if snapshot else None
        if snapshot:
            values.update({f"gradient/{name}": common.tensor_record(parameter.grad)
                           for name, parameter in model.named_parameters() if parameter.grad is not None})
        optimizer.step()
        if snapshot:
            for name, parameter in model.named_parameters():
                values[f"parameter/{name}"] = common.tensor_record(parameter)
                for state_name, value in optimizer.state[parameter].items():
                    if torch.is_tensor(value):
                        values[f"adam/{name}/{state_name}"] = common.tensor_record(value)
        return values
    training_snapshot = step(snapshot=True)
    def restore():
        model.load_state_dict(initial, strict=True)
        optimizer.state.clear()
        step()  # Identical state allocation and one warm Adam step, outside timing.
        torch.cuda.synchronize()
    model_name = f"usrnet/lr{size}_s{scale}"
    cases = {f"{model_name}/adam_training": {
        "scope": "Full pretrained 5-iteration/7-block model: zero_grad, forward, MSE, all backward and Adam; GPU-resident synthetic data",
        "checkpoint_sha256": common.file_sha(checkpoint), "fixture": input_records,
        "snapshot": training_snapshot, "fft_shapes": {"DataNet": [size * scale] * 2, "35_prior_calls": [size * scale + 4] * 2},
        "timing": common.measure(torch, step, args, before_round=restore)}}
    print(f"{model_name}/adam_training", cases[f"{model_name}/adam_training"]["timing"]["median"], flush=True)
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    model.load_state_dict(initial, strict=True)
    model.eval()
    torch.ops.converse2d.clear_cache()
    def eager():
        with torch.no_grad():
            return model(x, k, scale)
    eager_output = eager()
    eager_snapshot = common.records({"output": eager_output})
    del eager_output
    cases[f"{model_name}/eager_inference"] = {
        "scope": "Whole model eager no_grad, all Python and operator checks; warmed fixed-kernel cache",
        "checkpoint_sha256": common.file_sha(checkpoint), "fixture": input_records,
        "snapshot": eager_snapshot, "timing": common.measure(torch, eager, args)}
    if args.no_graph:
        return cases
    alternate = torch.rand((1, 3, size - 1, size + 1), generator=generator).cuda()
    holder = {}
    def first_call():
        holder["runner"] = USRNetCUDAGraph(model, max_graphs=1, warmup=args.graph_warmup)
        with torch.no_grad():
            return holder["runner"](x, k, scale)
    torch.ops.converse2d.clear_cache()
    cold, cold_row = measure_once(torch, first_call)
    runner = holder["runner"]
    cold_snapshot = common.records({"output": cold})
    cases[f"{model_name}/graph_first_call"] = {
        "scope": "First empty runner plus cleared Converse eager cache, in an already initialized GPU process; constructor, validation, signature, warmup, capture, copies, replay and output clone included",
        "fixture": input_records, "snapshot": cold_snapshot,
        "eager_output_sha256": eager_snapshot["output"]["sha256"], "captures": runner.captures,
        "eager_bitwise_equal": cold_snapshot["output"]["sha256"] == eager_snapshot["output"]["sha256"],
        "timing": timing_rows([cold_row], "one synchronized first runner call; not a fresh process startup")}
    del cold
    miss_rows, miss_snapshot = [], {}
    captures_before = runner.captures
    try:
        for rep in range(args.graph_misses):
            value = alternate if rep % 2 == 0 else x
            def miss():
                with torch.no_grad():
                    return runner(value, k, scale)
            output, row = measure_once(torch, miss)
            miss_rows.append(row)
            miss_snapshot[f"output_{rep}"] = common.tensor_record(output)
            del output
        if runner.captures - captures_before != args.graph_misses:
            raise RuntimeError("The LRU-miss workload unexpectedly reused a captured graph")
        cases[f"{model_name}/graph_lru_miss"] = {
            "scope": "Alternate two input shapes with max_graphs=1; includes eviction synchronization, validation, capture warmup, copies, replay and clone",
            "fixture": {**input_records, "alternate_x": common.tensor_record(alternate)}, "snapshot": miss_snapshot,
            "capture_delta": runner.captures - captures_before,
            "timing": timing_rows(miss_rows, "each complete miss synchronized; CPU snapshots outside timing")}
        def hit():
            with torch.no_grad():
                return runner(x, k, scale)
        output = hit()  # Ensure the primary shape is resident, outside timing.
        hit_snapshot = common.records({"output": output})
        del output
        captures_before = runner.captures
        hit_timing = common.measure(torch, hit, args)
        if runner.captures != captures_before:
            raise RuntimeError("The Graph-hit workload unexpectedly captured another graph")
        cases[f"{model_name}/graph_hit"] = {
            "scope": "Public runner __call__: lock, input/model/backend checks, signature, two static-buffer copies, replay, independent output clone, completion event",
            "fixture": input_records, "snapshot": hit_snapshot, "capture_delta": 0,
            "eager_bitwise_equal": hit_snapshot["output"]["sha256"] == eager_snapshot["output"]["sha256"],
            "eager_output_sha256": eager_snapshot["output"]["sha256"], "timing": hit_timing}
        for suffix in ("graph_first_call", "graph_lru_miss", "graph_hit"):
            print(f"{model_name}/{suffix}", cases[f"{model_name}/{suffix}"]["timing"]["median"], flush=True)
    finally:
        runner.clear()
        torch.ops.converse2d.clear_cache()
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--lr-size", type=int, default=32)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--graph-misses", type=int, default=3)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--include-operators", action="store_true", help="Also replay the frozen PSF harness's 41 complete operator/VJP cases")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename; benchmark evidence is immutable")
    if args.compare:
        result = common.compare(args.compare)
        a, b = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
        result.update(kind="roadmap_comparison", harness_match=a["harness_sha256"] == b["harness_sha256"],
                      helpers_match=a["helper_sha256"] == b["helper_sha256"],
                      note="All timings are complete callable spans. Graph first/miss spans include capture and CPU submission gaps; they are not kernel-only timings. No dataset quality or convergence conclusion.")
    else:
        if min(args.rounds, args.iters, args.graph_misses, args.graph_warmup, args.scale) < 1 or args.warmup < 0 or args.lr_size < 4:
            parser.error("Positive repeat/scale counts, lr-size>=4 and warmup>=0 are required")
        if os.environ.get("CONVERSE2D_BACKEND") or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
            raise RuntimeError("Unset backend and CPU-only overrides for checked CUDA comparisons")
        args.root = args.root.resolve()
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
        loader_spec = importlib.util.spec_from_file_location("roadmap_checked_loader", args.root / "test/extension_loader.py")
        loader = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(loader)
        loader.load_extension()
        from models.converse_core import converse2d_reference
        manifest = json.loads((args.root / ".build/cuda/source_manifest.json").read_text(encoding="utf-8"))
        source_hashes = loader.production_source_hashes()
        model_paths = ("models/converse_core.py", "models/util_converse.py", "models/converse_usrnet.py", "models/cuda_graph.py", "test/extension_loader.py")
        model_hashes = {name: common.file_sha(args.root / name) for name in model_paths}
        helper_hashes = {"p0": common.file_sha(common.__file__), "psf": common.file_sha(psf.__file__)}
        harness_hash = common.file_sha(__file__)
        settings = {name: getattr(args, name) for name in ("warmup", "rounds", "iters", "seed", "lr_size", "scale", "graph_misses", "graph_warmup", "no_graph", "include_operators", "deterministic_algorithms")}
        result = {"kind": "roadmap_benchmark", "root": str(args.root), "harness_sha256": harness_hash,
                  "helper_sha256": helper_hashes, "source_sha256": source_hashes, "python_sha256": model_hashes,
                  "checked_build_manifest": manifest, "settings": settings,
                  "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                                  "tf32": False, "amp": False, "cudnn_deterministic": True,
                                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                  "memory_scope": "Additional PyTorch allocated peak above live state before each timed call/round; not whole-process VRAM. Graph capture pools are included in first/miss allocation peaks.",
                  "cases": run_peripherals(torch, args)}
        result["cases"].update(run_model_and_graph(torch, args))
        if args.include_operators:
            args.profile = False
            result["cases"].update(psf.run_operators(torch, converse2d_reference, args))
        if source_hashes != loader.production_source_hashes() or model_hashes != {name: common.file_sha(args.root / name) for name in model_paths}:
            raise RuntimeError("Source/model/Graph file changed during benchmark")
        if common.file_sha(args.root / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
            raise RuntimeError("Checked binary changed during benchmark")
        if harness_hash != common.file_sha(__file__) or helper_hashes != {"p0": common.file_sha(common.__file__), "psf": common.file_sha(psf.__file__)}:
            raise RuntimeError("Harness or frozen helper changed during benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
