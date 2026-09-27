"""Paired deterministic NaN-fill experiment; never changes production defaults.

Uses a separately checked checkout and the frozen real-image training worker.
Snapshots are outside timing. Timing includes the worker's H2D, finite/norm
checks, synchronization and unfused Adam, but excludes data generation/eval/I/O.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import unittest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def affinity(mask):
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.GetCurrentProcess.restype = ctypes.c_void_p
    lib.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    handle = lib.GetCurrentProcess()
    if not lib.SetProcessAffinityMask(handle, mask):
        raise ctypes.WinError(ctypes.get_last_error())
    actual, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not lib.GetProcessAffinityMask(handle, ctypes.byref(actual), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if actual.value != mask:
        raise RuntimeError("Affinity did not match requested mask")
    return hex(actual.value)


def configure(torch, fill):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def tensor_record(value):
    value = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    if not bool(value.isfinite().all()):
        raise RuntimeError("Nonfinite tensor in snapshot")
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def snapshot(model, optimizer, output, row):
    tensors = {"output": tensor_record(output)}
    tensors.update({"parameter/" + name: tensor_record(value) for name, value in model.state_dict().items()})
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError("Missing gradient: " + name)
        tensors["gradient/" + name] = tensor_record(parameter.grad)
        for key, value in optimizer.state[parameter].items():
            tensors["adam/" + name + "/" + key] = tensor_record(value)
    return dict(tensors=tensors, scalars={key: row[key] for key in
                ("loss", "grad_l2_norm", "loss_and_grad_finite", "optimizer_applied", "gradient_tensor_count")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("tests", "numeric", "timing"), required=True)
    parser.add_argument("--fill", choices=("on", "off"), default="on")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--affinity", type=lambda s: int(s, 0), default=0xC03C03)
    args = parser.parse_args()
    args.root, args.data_root, args.output = (p.resolve() for p in (args.root, args.data_root, args.output))
    if args.output.exists():
        parser.error("Preserve prior evidence: choose a new output path")
    args.output.mkdir(parents=True)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    actual_affinity = affinity(args.affinity)
    sys.path[:0] = [str(args.root / "test"), str(args.root), str(args.root / "tools/roadmap_quality")]
    import torch
    import extension_loader
    configure(torch, args.fill == "on")
    extension_loader.load_extension()
    manifest = json.loads((args.root / ".build/cuda/source_manifest.json").read_text())
    result = dict(status="running", mode=args.mode, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  commit=subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
                  harness_sha256=sha(__file__), checked_manifest=manifest,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                     deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False, tf32=False, amp=False,
                     affinity=actual_affinity, torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads()),
                  source_hashes={name: sha(args.root / name) for name in (
                      "models/converse_usrnet.py", "models/util_converse.py", "models/converse_core.py",
                      "tools/roadmap_quality/train_usrnet_dataset.py", "tools/roadmap_quality/usrnet_training_data.py")})
    report_path = args.output / "result.json"
    write(report_path, result)
    try:
        if args.mode == "tests":
            torch.manual_seed(17)
            suite = unittest.defaultTestLoader.discover(str(args.root / "test"), pattern="test_*.py")
            outcome = unittest.TextTestRunner(verbosity=2).run(suite)
            result["tests"] = dict(run=outcome.testsRun, skipped=[(str(t), reason) for t, reason in outcome.skipped],
                                   failures=[(str(t), detail) for t, detail in outcome.failures],
                                   errors=[(str(t), detail) for t, detail in outcome.errors],
                                   successful=outcome.wasSuccessful())
            result["fill_after"] = torch.utils.deterministic.fill_uninitialized_memory
            result["deterministic_after"] = torch.are_deterministic_algorithms_enabled()
            fp64 = args.root / "artifacts/fp32_release/fp64_errors.json"
            if fp64.exists():
                (args.output / "fp64_errors.json").write_bytes(fp64.read_bytes())
            result["status"] = "passed" if outcome.wasSuccessful() else "failed"
            return 0 if outcome.wasSuccessful() else 1

        from models.converse_usrnet import ConverseUSRNet
        import usrnet_training_data as data
        import train_usrnet_dataset as worker
        data.ROOT = args.data_root
        checkpoint = args.root / "model_zoo/converse_usrnet.pth"
        initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
        result["checkpoint_sha256"] = sha(checkpoint)
        result["scope"] = "Full pretrained USRNet, LR32 to HR96, scale3, real images; frozen train_step with H2D, finite/norm checks and Adam. Excludes data preparation, evaluation, checkpoint I/O. No convergence claim."
        result["cases"] = []
        def make_model(seed):
            torch.manual_seed(seed)
            model = ConverseUSRNet(backend="cuda").cuda().train()
            model.load_state_dict(initial, strict=True)
            return model, torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)

        for seed in args.seeds:
            protocol = data.DatasetProtocol(args.data_root / "artifacts/dataset_training/split_900_100.json",
                                           patch_size=96, scale=3, seed=seed, noise_std=.01)
            for batch_size in args.batches:
                recipe = argparse.Namespace(batch_size=batch_size, microbatch_size=batch_size, patch_size=96, scale=3, loss="mse")
                count = args.steps if args.mode == "numeric" else args.warmup + args.iters
                batches = [protocol.train_batch(i, batch_size) for i in range(count)]
                for batch in batches:
                    worker.check_cpu_batch(batch, recipe, batch_size)
                case = dict(seed=seed, batch=batch_size, dataset=protocol.metadata,
                            batches=[{name: tensor_record(value) for name, value in zip(("lr", "kernel", "hr"), batch)} for batch in batches])
                if args.mode == "numeric":
                    histories = {}
                    # Reverse arm order for the middle seed.
                    order = [False, True] if seed == 29 else [True, False]
                    case["order"] = order
                    for fill in order:
                        configure(torch, fill)
                        model, optimizer = make_model(seed)
                        captured = []
                        hook = model.register_forward_hook(lambda _m, _i, out: captured.__setitem__(slice(None), [out.detach()]))
                        rows = []
                        try:
                            for index, batch in enumerate(batches):
                                row = worker.train_step(model, optimizer, batch, recipe)
                                rows.append(snapshot(model, optimizer, captured[0], row))
                                captured.clear()
                                print(f"numeric seed={seed} B{batch_size} fill={fill} step={index+1} loss={row['loss']:.9g}", flush=True)
                        finally:
                            hook.remove()
                        histories[str(fill)] = rows
                        write(args.output / f"numeric_seed{seed}_b{batch_size}_fill{int(fill)}.json", rows)
                        del model, optimizer, rows
                        gc.collect()
                        torch.cuda.empty_cache()
                    mismatches = []
                    comparisons = 0
                    for index, (left, right) in enumerate(zip(histories["True"], histories["False"])):
                        for name in left["tensors"]:
                            comparisons += 1
                            if left["tensors"][name] != right["tensors"].get(name):
                                mismatches.append(dict(step=index + 1, tensor=name))
                        if left["scalars"] != right["scalars"]:
                            mismatches.append(dict(step=index + 1, scalars=True))
                    case.update(tensor_comparisons=comparisons, mismatches=mismatches)
                    if mismatches:
                        result["cases"].append(case)
                        raise RuntimeError(f"NaN-fill numerical mismatch: {mismatches[:5]}")
                else:
                    case["pairs"] = []
                    # One model object per case; restore checkpoint and empty
                    # optimizer state before each arm. Warmup creates Adam state.
                    model, optimizer = make_model(seed)
                    for round_index in range(args.rounds):
                        pair = dict(round=round_index, order=[True, False] if round_index % 2 == 0 else [False, True], arms={})
                        for fill in pair["order"]:
                            configure(torch, fill)
                            model.load_state_dict(initial, strict=True)
                            optimizer.zero_grad(set_to_none=True)
                            optimizer.state.clear()
                            torch.manual_seed(seed)
                            for batch in batches[:args.warmup]:
                                row = worker.train_step(model, optimizer, batch, recipe)
                                if not row["optimizer_applied"]:
                                    raise RuntimeError("Nonfinite warmup")
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            rows = [worker.train_step(model, optimizer, batch, recipe) for batch in batches[args.warmup:]]
                            if not all(row["optimizer_applied"] for row in rows):
                                raise RuntimeError("Nonfinite timed step")
                            terminal = {name: tensor_record(value) for name, value in model.state_dict().items()}
                            arm = dict(steps=rows, wall_ms=statistics.mean(row["training_step_wall_ms"] for row in rows),
                                       cuda_span_ms=statistics.mean(row["training_step_cuda_span_ms"] for row in rows), terminal_parameters=terminal)
                            pair["arms"][str(fill)] = arm
                            print(f"timing seed={seed} B{batch_size} round={round_index} fill={fill} wall={arm['wall_ms']:.3f}ms", flush=True)
                        on, off = pair["arms"]["True"], pair["arms"]["False"]
                        if on["terminal_parameters"] != off["terminal_parameters"]:
                            raise RuntimeError("Timed arms reached different parameters")
                        if [(r["loss"], r["grad_l2_norm"]) for r in on["steps"]] != [(r["loss"], r["grad_l2_norm"]) for r in off["steps"]]:
                            raise RuntimeError("Timed trajectories differ")
                        pair["wall_speedup"] = on["wall_ms"] / off["wall_ms"]
                        pair["cuda_span_speedup"] = on["cuda_span_ms"] / off["cuda_span_ms"]
                        case["pairs"].append(pair)
                        write(args.output / f"timing_seed{seed}_b{batch_size}.json", case)
                    case["summary"] = dict(
                        on_wall_ms=statistics.median(p["arms"]["True"]["wall_ms"] for p in case["pairs"]),
                        off_wall_ms=statistics.median(p["arms"]["False"]["wall_ms"] for p in case["pairs"]),
                        paired_speedup_median=statistics.median(p["wall_speedup"] for p in case["pairs"]),
                        paired_speedup_range=[min(p["wall_speedup"] for p in case["pairs"]), max(p["wall_speedup"] for p in case["pairs"])])
                    print(json.dumps(case["summary"]), flush=True)
                    del model, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                result["cases"].append(case)
                write(report_path, result)
        result["status"] = "passed"
        return 0
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        write(report_path, result)


if __name__ == "__main__":
    raise SystemExit(main())
