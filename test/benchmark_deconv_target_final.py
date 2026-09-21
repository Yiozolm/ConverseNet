"""Final same-fixture FFT/spatial/wrapper/native-deconvolution comparison.

Defaults to native.cpu_data(seed17), s3_large_batch and its ORIGINAL eps=1e-5.
All routes use deterministic FP32, TF32 off, CUBLAS=:4096:8, no CUDA Graphs,
and donated_buffer=False set before compilation/first execution. Both native
dense and spatial candidates get fresh Inductor callables with identical options.

Every first complete FWD+dx/dw/db setup is reported separately. Original FP64
budgets gate Converse routes; compiled native is checked against eager native,
not against the mathematically different Converse equation. Hot timing rejects
recompilation and rotates routes, with one GPU fixture resident at a time.
Optional s1_module/s3_dynamic cases explicitly exclude inapplicable k3/s3 spatial
routes. No production change, hidden eager fallback or inherited speed ratio.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {"triton.cudagraphs": False}


def production_method(config, reference=False):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference

    def forward(x, weight, bias):
        padding, scale = config.get("outer_padding", 0), config["scale"]
        value = F.pad(x, (padding,) * 4, mode="circular") if padding else x
        prior = value if scale == 1 else F.interpolate(value, scale_factor=scale, mode="nearest")
        output = (converse2d_reference(value, prior, weight, bias, scale, config["eps"]) if reference else
                  torch.ops.converse2d.forward(value, prior, weight, bias, scale, config["eps"], "v7"))
        crop = padding * scale
        return output[..., crop:-crop, crop:-crop] if crop else output
    return forward


def native_method(config):
    import torch
    import torch.nn.functional as F
    from benchmark_native_deconv import native_geometry
    geometry = native_geometry(config)
    kwargs = {key: geometry[key] for key in ("stride", "padding", "output_padding", "dilation")}

    def forward(x, weight, bias):
        if config["dynamic"]:
            return torch.cat([F.conv_transpose2d(x[index:index + 1], weight[index], bias, groups=1, **kwargs)
                              for index in range(config["shape"][0])], dim=0)
        return F.conv_transpose2d(x, weight, bias, groups=1, **kwargs)
    return forward


def spatial_method(config):
    import probe_spatial_nonoverlap_fast as fast
    eps = config["eps"]
    return lambda x, weight, bias: fast.nearest_route(x, weight, bias, 3, eps, "lrpred_tensor")


def graph_count():
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def first_execution(tensors, method):
    import torch
    from probe_pointwise_training import clear_cuda, fixture
    clear_cuda()
    inputs, run = fixture(tensors, torch.float32, method)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record(); end.record()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = graph_count()
    began = time.perf_counter()
    start.record()
    values = run()  # Forces both forward and first-order backward setup.
    end.record()
    torch.cuda.synchronize()
    result = dict(first_fwd_vjp_wall_ms=(time.perf_counter() - began) * 1000,
                  first_fwd_vjp_cuda_span_ms=start.elapsed_time(end),
                  unique_graphs_before=before, unique_graphs_after=graph_count(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  output_shape=list(values["output"].shape),
                  tensor_dtypes={key: str(value.dtype) for key, value in values.items()},
                  gradient_shapes={key: list(values[key].shape) for key in ("dx", "dw", "db")},
                  scope="Setup/compile/cache lookup plus first complete FWD+3VJP; event span includes host gaps, not pure kernels")
    if (any(value.dtype != torch.float32 for value in values.values())
            or any(value.grad is not None for value in inputs)
            or not all(bool(torch.isfinite(value).all()) for value in values.values())):
        raise RuntimeError("First execution violated FP32/finite/no-leaf-accumulation contract")
    del values, inputs, run
    clear_cuda()
    return result


def prepare_routes(config, tensors, native_tensors, wrapper_factory):
    import torch
    routes, setup, exclusions = {}, {}, []

    def add(name, factory, family):
        began = time.perf_counter()
        method = factory()
        factory_ms = (time.perf_counter() - began) * 1000
        setup[name] = dict(callable_factory_wall_ms=factory_ms, **first_execution(family, method))
        routes[name] = (method, family)

    add("production_fft", lambda: production_method(config), tensors)
    eligible = config["scale"] == 3 and config["kernel"] == 3 and not config.get("outer_padding", 0)
    if eligible:
        # Separate fresh compile objects for raw and proxy-view wrapper execution.
        add("compiled_spatial_raw", lambda: torch.compile(spatial_method(config), backend="inductor",
                                                           fullgraph=True, dynamic=False, options=OPTIONS), tensors)

        def make_wrapped():
            eager = spatial_method(config)
            compiled = torch.compile(eager, backend="inductor", fullgraph=True, dynamic=False, options=OPTIONS)
            return wrapper_factory(eager, compiled)
        add("compiled_spatial_wrapped", make_wrapped, tensors)
    else:
        exclusions.append("Spatial compilation is only applicable to k3/s3 with no external module pad; no FFT/eager substitute is labelled compiled spatial")
    add("native_dense_eager", lambda: native_method(config), native_tensors)
    add("native_dense_compiled", lambda: torch.compile(native_method(config), backend="inductor",
                                                        fullgraph=True, dynamic=False, options=OPTIONS), native_tensors)
    return routes, setup, exclusions


def validate_routes(config, tensors, routes):
    import torch
    from probe_pointwise_training import capture
    from probe_training_s1_shapes import metrics
    reference = capture(tensors, torch.float64, production_method(config, reference=True))
    checks = {}
    for name, (method, family) in routes.items():
        if name.startswith("native_"):
            continue
        actual = capture(family, torch.float32, method)
        comparison = {key: metrics(actual[key], expected, output=key == "output", weak=False)
                      for key, expected in reference.items()}
        checks[name] = dict(reference="Independent full-FFT FP64 Converse", passed=all(row["passed"] for row in comparison.values()),
                            tensors=comparison)
    eager, native_tensors = routes["native_dense_eager"]
    expected = capture(native_tensors, torch.float32, eager)
    actual = capture(native_tensors, torch.float32, routes["native_dense_compiled"][0])
    comparison = {key: metrics(actual[key], value, output=key == "output", weak=False) for key, value in expected.items()}
    checks["native_dense_compiled"] = dict(reference="Native dense eager FP32, not Converse", tensors=comparison,
                                           passed=all(row["passed"] for row in comparison.values()))
    return checks


def benchmark(routes, args):
    import torch._dynamo
    from probe_pointwise_training import timed_fixture
    names, rounds = list(routes), []
    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(args.rounds):
            order = names[index % len(names):] + names[:index % len(names)]
            before = graph_count()
            results = {name: timed_fixture(routes[name][1], routes[name][0], args) for name in order}
            after = graph_count()
            if after != before:
                raise RuntimeError("New graph compilation occurred during hot timing")
            rounds.append(dict(round=index + 1, order=order, variants=results,
                               unique_graphs_before=before, unique_graphs_after=after))
    medians = {name: {key: statistics.median(row["variants"][name][key] for row in rounds)
                      for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in names}
    ratios = {}
    for name in names:
        ratios[name] = {}
        for metric in ("wall_ms", "cuda_event_ms"):
            denominators = {}
            for denominator in ("production_fft", "native_dense_eager", "native_dense_compiled"):
                values = [row["variants"][name][metric] / row["variants"][denominator][metric] for row in rounds]
                denominators[denominator] = dict(per_round=values, median=statistics.median(values), min=min(values), max=max(values))
            ratios[name][metric] = denominators
    return dict(rounds=rounds, medians=medians, paired_route_over_denominator=ratios,
                ratio_scope="Each ratio uses its explicitly named same-round denominator; no historical ratios or different mathematical operators are conflated")


def evidence(path):
    import train_usrnet_dataset as worker
    if not path.is_file():
        return dict(path=str(path), available=False)
    value = json.loads(path.read_text(encoding="utf-8"))
    weak = [row for row in value.get("cases", []) if row.get("weak")]
    return dict(path=str(path), available=True, sha256=worker.file_hash(path), status=value.get("status"),
                passed=value.get("passed"), weak_cases=[dict(name=row.get("name"), eps=row.get("eps"),
                                                            validation=row.get("validation"), repeatability=row.get("repeatability")) for row in weak],
                scope="Related weak/high-order evidence only; different fixtures/configuration and donation policy do not replace this final fixture's own gates")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", action="append", choices=("s3_large_batch", "s1_module", "s3_dynamic"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/native_deconv_target/native_baseline.json")
    parser.add_argument("--wrapper-factory", default="experiments.training_nonoverlap.compiled_autograd:wrap_compiled")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/final_comparison.json")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        parser.error("TF32 override conflicts with required FP32 protocol")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cache_state = {}
    for name, folder in (("TORCHINDUCTOR_CACHE_DIR", "inductor_deconv_final_nodonate"),
                         ("TRITON_CACHE_DIR", "triton_deconv_final_nodonate")):
        path = ROOT / ".build" / folder
        cache_state[name] = dict(path=str(path), existed=path.exists(), had_entries=path.exists() and any(path.iterdir()))
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.path.insert(0, str(ROOT))
    import torch
    import torch._dynamo
    import torch._functorch.config as functorch_config
    functorch_config.donated_buffer = False  # Before ANY compile or first execution, for every route.
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.error_on_recompile = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    import train_usrnet_dataset as worker
    import benchmark_native_deconv as native
    from benchmark_converse_candidates import audit_native_baseline
    from probe_compiled_nonoverlap import repeatability
    baseline_audit = audit_native_baseline(args.baseline)
    original = json.loads(args.baseline.read_text(encoding="utf-8"))
    args.variant = "current"
    _, current_build = worker.load_backend(args)
    module_name, function_name = args.wrapper_factory.split(":", 1)
    wrapper_module = importlib.import_module(module_name)
    wrapper_factory = getattr(wrapper_module, function_name)
    wrapper_path = Path(wrapper_module.__file__).resolve()
    sources = worker.source_hashes()
    for name in ("benchmark_deconv_target_final.py", "benchmark_native_deconv.py", "benchmark_converse_candidates.py",
                 "probe_spatial_nonoverlap_fast.py", "probe_spatial_nonoverlap.py", "probe_pointwise_training.py",
                 "probe_training_s1_shapes.py", "probe_compiled_nonoverlap.py"):
        sources["test/" + name] = worker.file_hash(ROOT / "test" / name)
    sources[str(wrapper_path.relative_to(ROOT))] = worker.file_hash(wrapper_path)
    report = dict(status="preparing", settings=worker.json_safe(vars(args)), source_sha256=sources,
                  current_build=current_build, baseline_audit=baseline_audit, cache_directories=cache_state,
                  compile_settings=dict(backend="inductor", fullgraph=True, dynamic=False, options=OPTIONS,
                                        suppress_errors=False, donated_buffer=False, hot_error_on_recompile=True,
                                        fresh_raw_wrapped_and_native_callables=True),
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, deterministic_algorithms=True, cudnn_deterministic=True,
                                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"], donated_buffer=False),
                  related_evidence=dict(weak_reference=evidence(ROOT / "artifacts/native_deconv_target/compiled_nonoverlap.json"),
                                        wrapper_contract=evidence(ROOT / "artifacts/native_deconv_target/compiled_autograd_contract_nodonation.json"),
                                        prior_donation_failure=evidence(ROOT / "artifacts/native_deconv_target/compiled_autograd_contract.json")),
                  scope=__doc__, production_eligible=False, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    worker.write_json(args.output, report)
    try:
        for name in args.case or ["s3_large_batch"]:
            config = native.CASES[name]
            stored = next(row for row in original["cases"] if row["name"] == name)
            cpu = native.cpu_data(config, seed=17)
            full_hash, pair_hash = worker.tensor_hash(cpu), worker.tensor_hash({key: cpu[key] for key in ("x", "upstream")})
            if worker.json_safe(config) != stored["config"] or full_hash != stored["complete_fixture_sha256"] or pair_hash != stored["input_upstream_sha256"]:
                raise RuntimeError("Original native fixture changed; never adjust eps, weights or upstream")
            channels = config["shape"][1]
            tensors = (cpu["x"], cpu["weight"], cpu["bias"].reshape(1, channels, 1, 1), cpu["upstream"])
            native_tensors = (cpu["x"], cpu["dense_weight"], cpu["bias"], cpu["upstream"])
            routes, setup, exclusions = prepare_routes(config, tensors, native_tensors, wrapper_factory)
            row = dict(name=name, config=config, complete_fixture_sha256=full_hash, input_upstream_sha256=pair_hash,
                       native_geometry=native.native_geometry(config), setup=setup, excluded_routes=exclusions,
                       parameter_elements=dict(converse_weight=cpu["weight"].numel(), native_dense_weight=cpu["dense_weight"].numel(), bias=channels),
                       gradient_targets=["x", "weight", "bias"], validation=validate_routes(config, tensors, routes),
                       repeatability={}, timing=None)
            report["cases"].append(row)
            # Fixture family differs for native, but the same three-gradient contract applies.
            for route, (method, family) in routes.items():
                row["repeatability"][route] = repeatability(dict(tensors=family), {route: method})[route]
            prepared.append((row, routes))
            worker.write_json(args.output, report)
        if not all(all(value["passed"] for value in row["validation"].values())
                   and all(value["passed"] for value in row["repeatability"].values()) for row, _ in prepared):
            report["status"] = "numerical_or_repeatability_gate_failed_no_timing"
            worker.write_json(args.output, report)
            return 1
        report["status"] = "hot_timing"
        for row, routes in prepared:
            row["timing"] = benchmark(routes, args)
            worker.write_json(args.output, report)
            print(json.dumps(dict(case=row["name"], medians=row["timing"]["medians"])), flush=True)
        report["status"] = "complete_same_fixture_final_comparison"
        worker.write_json(args.output, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
