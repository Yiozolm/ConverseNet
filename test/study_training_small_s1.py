"""Refactor-era s1 study: frozen control, isolated candidates, actual production.

GPU jobs are serial. Operator timings include pad/crop and all VJPs; model
timings include zero_grad/forward/MSE/backward/Adam, with real-image batches
already resident on GPU. Numerical snapshots and profiling are outside timing.
Unchanged bit patterns establish compatibility, not universal Python-FP32
noninferiority: the latter is reported independently against the same FP64.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(args):
    manifest = json.loads((args.snapshot / "before_manifest.json").read_text())
    supplement = args.snapshot / "adapter_manifest.json"
    if supplement.exists():
        manifest = dict(manifest, files={**manifest["files"], **json.loads(supplement.read_text())})
    for name, expected in manifest["files"].items():
        if sha(args.snapshot / "before" / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    path = args.snapshot / "before/Converse2D/build_config.py"
    spec = importlib.util.spec_from_file_location("small_s1_frozen_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.legacy_sources(), manifest


def load_ops(args):
    import torch
    import probe_training_s1_shapes as probe
    texts, manifest = frozen(args)
    if args.warm_builds:
        saved = json.loads(args.warm_builds.read_text())
        if saved["environment"]["torch"] != str(torch.__version__) or saved["environment"]["cuda"] != torch.version.cuda:
            raise RuntimeError("Warm binary runtime mismatch")
        result = {}
        for name, build in saved["identity"]["builds"].items():
            original = {n: hashlib.sha256(t.encode()).hexdigest() for n, t in texts.items()}
            if original != build["source_sha256"]:
                raise RuntimeError("Warm binary source mismatch")
            for n, expected in build["namespaced_source_sha256"].items():
                # The established research loader records UTF-8 text hashes;
                # Windows write_text emits CRLF. Do not compare that identity
                # to raw disk bytes. Binary identity below is always raw bytes.
                text = (Path(build["build_directory"]) / n).read_text(encoding="utf-8")
                if hashlib.sha256(text.encode()).hexdigest() != expected:
                    raise RuntimeError("Warm derived source changed")
            if sha(Path(build["library"])) != build["binary_sha256"]:
                raise RuntimeError("Warm binary changed")
            torch.ops.load_library(build["library"])
            result[name] = getattr(torch.ops, build["namespace"])
        return result, saved["identity"]
    result, builds = {}, {}
    # Reuse the checked research loader, but derive both builds from the same
    # immutable refactor snapshot. No production source string is rewritten.
    with patch.object(probe, "legacy_source_texts", return_value=texts):
        with patch.object(probe, "FORCED_SELECTOR", probe.SELECTOR):
            result["before"], builds["before"] = probe.load_forced()
        result["candidate"], builds["candidate"] = probe.load_forced()
    for name, build in builds.items():
        if "build_directory" in build:
            library = Path(build["build_directory"]) / (build["extension"] + (".pyd" if os.name == "nt" else ".so"))
            build.update(library=str(library.resolve()), binary_sha256=sha(library))
    return result, dict(builds=builds, frozen=manifest)


def same(a, b):
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def errors(actual, reference):
    import torch
    a, r = actual.double(), reference.double()
    e = a - r
    return dict(finite=bool(torch.isfinite(a).all()), max_abs=e.abs().max().item(),
                relative_l2=(e.norm() / r.norm().clamp_min(1e-30)).item())


def operator_study(args, report, save, ops):
    import torch
    import torch.nn.functional as F
    from models.converse_training import circular_pad, crop_view
    from models.converse_core import converse2d_reference
    from probe_pointwise_training import capture, timed_fixture, tensor_hash
    state = torch.load(ROOT / "model_zoo/converse_usrnet.pth", map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(9214)
    report["cases"] = []
    for batch in (1, 4, 8, 32):
        x = torch.randn(batch, 128, 96, 96, generator=generator)
        w = state["p.m_body.0.conv1.3.weight"].clone()
        b = state["p.m_body.0.conv1.3.bias"].clone()
        upstream = torch.randn(x.shape, generator=generator) / x.numel() ** .5
        tensors = (x, w, b, upstream)

        def method(core, boundary):
            def run(x, w, b):
                padded = circular_pad(x, 2) if boundary else F.pad(x, (2,) * 4, mode="circular")
                out = core(padded, w, b)
                return crop_view(out, 2) if boundary else out[..., 2:-2, 2:-2]
            return run

        def core(name):
            return lambda x, w, b: ops[name].forward(x, x, w, b, 1, 1e-5, "v7")

        routes = dict(before=method(core("before"), False),
                      dispatch=method(core("candidate"), False),
                      boundary=method(core("before"), True),
                      combined=method(core("candidate"), True))
        reference = method(lambda x, w, b: converse2d_reference(x, x, w, b, 1, 1e-5), False)
        oracle = capture(tensors, torch.float64, reference)
        python = capture(tensors, torch.float32, reference)
        python_error = {k: errors(v, oracle[k]) for k, v in python.items()}
        before = capture(tensors, torch.float32, routes["before"])
        row = dict(batch=batch, shape=list(x.shape), fixture_sha256=tensor_hash(tensors),
                   python_fp32_error=python_error, validation={}, rounds=[])
        report["cases"].append(row)
        for name, run in routes.items():
            actual = capture(tensors, torch.float32, run)
            metrics = {k: errors(v, oracle[k]) for k, v in actual.items()}
            row["validation"][name] = dict(bitwise_before={k: same(v, before[k]) for k, v in actual.items()},
                error=metrics, python_noninferior={k: all(metrics[k][m] <= python_error[k][m]
                for m in ("max_abs", "relative_l2")) for k in metrics})
            save()
        if not all(all(v["bitwise_before"].values()) for v in row["validation"].values()):
            raise RuntimeError("Changed operator bit patterns; inspect before timing")
        for index in range(args.rounds):
            names = list(routes)
            order = names[index % 4:] + names[:index % 4]
            values = {name: timed_fixture(tensors, routes[name], args) for name in order}
            row["rounds"].append(dict(order=order, values=values))
            save()
        row["median_ms"] = {name: statistics.median(r["values"][name]["wall_ms"] for r in row["rounds"])
                            for name in routes}
        row["paired_speedup"] = {name: statistics.median(r["values"]["before"]["wall_ms"] /
                                  r["values"][name]["wall_ms"] for r in row["rounds"]) for name in routes}
        print(json.dumps(dict(batch=batch, median_ms=row["median_ms"], speedup=row["paired_speedup"])), flush=True)
        save()


@contextmanager
def model_route(args, ops, route):
    import torch
    from models import util_converse
    from models.converse_training import circular_pad, crop_view
    path = args.snapshot / "before/models/util_converse.py"
    spec = importlib.util.spec_from_file_location("small_s1_old_util", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    original = old.Converse2D.forward

    def combined(layer, x):
        if (torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.float32
                and x.shape[1:] == (128, 96, 96) and layer.scale == 1
                and layer.padding == 2 and layer.padding_mode == "circular"
                and layer.variant == "v7" and (x.requires_grad or layer.weight.requires_grad or layer.bias.requires_grad)):
            p = circular_pad(x, 2)
            return crop_view(torch.ops.converse2d.forward(p, p, layer.weight, layer.bias, 1, layer.eps, "v7"), 2)
        return original(layer, x)

    selected = ops["before" if route == "before" else "candidate"]
    with patch.object(torch.ops, "converse2d", selected):
        with patch.object(util_converse.Converse2D, "forward", original if route == "before" else combined):
            yield


def model_study(args, report, save, ops):
    import torch
    from models.converse_usrnet import ConverseUSRNet
    from usrnet_training_data import DatasetProtocol
    from benchmark_python_training import adam, snapshot
    from probe_pointwise_training import clear_cuda
    initial = torch.load(ROOT / "model_zoo/converse_usrnet.pth", map_location="cpu", weights_only=True)
    protocol = DatasetProtocol(ROOT / "artifacts/dataset_training/split_900_100.json",
                               patch_size=96, scale=3, seed=17, noise_std=.01)
    cpu_batches = [protocol.train_batch(i, args.batch) for i in range(max(args.iters, args.check_steps))]
    from train_usrnet_dataset import tensor_hash
    report["batch_hashes"] = [tensor_hash(dict(lr=b[0], kernel=b[1], hr=b[2])) for b in cpu_batches]
    report["checkpoint_sha256"] = sha(ROOT / "model_zoo/converse_usrnet.pth")
    report["validation"] = {}
    report["rounds"] = []

    def execute(route, validate=False, profile=False):
        clear_cuda()
        model = ConverseUSRNet(num_iterations=5, num_blocks=7, in_channels=64, backend="cuda",
                               reuse_training_spectra=False).float().cuda().train()
        model.load_state_dict(initial, strict=True)
        optimizer = adam(model)
        batches = [tuple(t.cuda() for t in b) for b in cpu_batches]
        outputs = []

        def step(index, keep=False):
            lr, kernel, hr = batches[index % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            out = model(lr, kernel, 3)
            torch.nn.functional.mse_loss(out, hr).backward()
            optimizer.step()
            if keep:
                outputs.append(out.detach().cpu())

        with model_route(args, ops, route):
            if validate:
                for i in range(args.check_steps):
                    step(i, True)
                result = snapshot(model, optimizer, outputs)
            else:
                for i in range(args.warmup):
                    step(i)
                from benchmark_python_training import restore_initial
                restore_initial(model, optimizer, initial)
                torch.cuda.reset_peak_memory_stats()
                if profile:
                    torch.cuda.cudart().cudaProfilerStart()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                began = time.perf_counter()
                for i in range(args.iters):
                    if profile:
                        with torch.cuda.nvtx.range(f"small_s1/{route}/step{i}"):
                            step(i)
                    else:
                        step(i)
                end.record()
                torch.cuda.synchronize()
                result = dict(wall_ms=(time.perf_counter()-began)*1000/args.iters,
                    event_ms=start.elapsed_time(end)/args.iters, allocated=torch.cuda.max_memory_allocated(),
                    reserved=torch.cuda.max_memory_reserved())
                if profile:
                    torch.cuda.cudart().cudaProfilerStop()
        del model, optimizer, batches
        clear_cuda()
        return result

    if args.phase == "profile":
        execute(args.route, profile=True)
        report["profile_only"] = True
        save()
        return
    control = execute("before", validate=True)
    candidate = execute("combined", validate=True)
    for group in control:
        report["validation"][group] = {key: same(value, candidate[group][key])
                                       for key, value in control[group].items()}
    save()
    if not all(all(v.values()) for v in report["validation"].values()):
        raise RuntimeError("Full model changed output/gradient/Adam/parameter bits")
    del control, candidate
    if args.phase == "validation":
        return
    for index in range(args.rounds):
        order = ["before", "combined"] if index % 2 == 0 else ["combined", "before"]
        values = {route: execute(route) for route in order}
        report["rounds"].append(dict(order=order, values=values))
        print(json.dumps(dict(round=index, batch=args.batch, values=values)), flush=True)
        save()
    report["paired_speedup"] = statistics.median(row["values"]["before"]["wall_ms"] /
                                                row["values"]["combined"]["wall_ms"] for row in report["rounds"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/small_s1_20260921")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("operators", "model", "validation", "profile", "build"), required=True)
    parser.add_argument("--warm-builds", type=Path, help="Verified study build report; load libraries without compiler probes")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--check-steps", type=int, default=3)
    parser.add_argument("--route", choices=("before", "combined"), default="before")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evidence")
    if min(args.batch, args.warmup, args.iters, args.rounds, args.check_steps) < 1:
        parser.error("Counts must be positive")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ops, identity = load_ops(args)
    from extension_loader import production_source_hashes
    report = dict(status="running", settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        identity=identity, source=production_source_hashes(), script_sha256=sha(Path(__file__)),
        model_sha256={p.name: sha(p) for p in (ROOT / "models").glob("*.py")},
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         tf32=False, cudnn_deterministic=True, amp=False))
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        archive = args.output.parent / ("study_source_" + report["script_sha256"] + ".py")
        if not archive.exists():
            archive.write_bytes(Path(__file__).read_bytes())
    save()
    try:
        if args.phase == "operators":
            operator_study(args, report, save, ops)
        elif args.phase != "build":
            model_study(args, report, save, ops)
        if production_source_hashes() != report["source"]:
            raise RuntimeError("Production source changed during run")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
