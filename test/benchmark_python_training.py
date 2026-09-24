"""Alternate full real-data Adam steps: Python/ATen versus current production.

Reuses train_usrnet_dataset.train_step unchanged. Each round creates one GPU
fixture, warms it, then restores the exact pretrained parameters and zeros the
already allocated Adam states. Timed steps therefore start from identical model
and optimizer states and consume the same cached real-image CPU batches.

The two-step numerical comparison reports differences without inventing a new
tolerance or replacing the independent FP64 and task-quality gates. No profiler.
"""
import argparse
import copy
import datetime
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import train_usrnet_dataset as worker

ROOT = Path(__file__).resolve().parents[1]


def clear_cuda(ops):
    import torch
    clear = getattr(ops, "clear_cache", None)
    if clear is not None:
        clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def adam(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0, amsgrad=False, foreach=False, fused=False)


def restore_initial(model, optimizer, initial):
    """Keep warmed Adam allocations; reset all mathematical optimizer state."""
    import torch
    with torch.no_grad():
        model.load_state_dict(initial, strict=True)
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            state = optimizer.state.get(parameter)
            if state is None or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise RuntimeError(f"Warmup did not initialize Adam buffers for {name}")
            state["step"].zero_()
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
        torch.cuda.synchronize()


def adam_tensors(model, optimizer):
    result = {}
    for name, parameter in model.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if hasattr(value, "detach"):
                result[f"{name}/{key}"] = value.detach().cpu().clone()
    return result


def snapshot(model, optimizer, outputs):
    return dict(parameters={name: value.detach().cpu().clone() for name, value in model.named_parameters()},
                gradients={name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()
                           if value.grad is not None},
                adam=adam_tensors(model, optimizer),
                outputs={str(index): value for index, value in enumerate(outputs)})


def graph_has_spectral(output):
    pending, visited = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        if any(kind in node.name() for kind in ("SpectralSolve", "FullSolve")):
            return True
        pending.extend(child for child, _ in node.next_functions)
    return False


def difference(actual, expected):
    import torch
    if actual.keys() != expected.keys():
        raise RuntimeError("Paired numerical comparison has different tensor/gradient keys")
    if not actual:
        return dict(tensors=0, elements=0, max_abs=0.0, relative_l2=0.0, finite=True)
    absolute, squared_error, squared_reference, elements = 0.0, 0.0, 0.0, 0
    per_tensor_relative = 0.0
    for name in actual:
        value, reference = actual[name].double(), expected[name].double()
        if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise FloatingPointError(f"Nonfinite numerical snapshot: {name}")
        error = value - reference
        absolute = max(absolute, error.abs().max().item())
        error_l2, reference_l2 = error.norm().item(), reference.norm().item()
        squared_error += error_l2 ** 2
        squared_reference += reference_l2 ** 2
        per_tensor_relative = max(per_tensor_relative, error_l2 / max(reference_l2, 1e-30))
        elements += value.numel()
    return dict(tensors=len(actual), elements=elements, max_abs=absolute,
                relative_l2=squared_error ** 0.5 / max(squared_reference ** 0.5, 1e-30),
                max_per_tensor_relative_l2=per_tensor_relative, finite=True)


def numerical_check(template, batches, args, ops):
    from evaluate_usrnet_quality import backend_scope
    states, trajectories, routes = {}, {}, {}
    for backend in ("pytorch", "current"):
        clear_cuda(ops)
        model = copy.deepcopy(template).cuda().train()
        optimizer = adam(model)
        outputs, observed_route = [], []

        def capture(_model, _inputs, output):
            outputs.append(output.detach().cpu().clone())
            observed_route.append(graph_has_spectral(output))

        # Output copies and graph inspection occur only in this untimed check.
        hook = model.register_forward_hook(capture)
        with backend_scope(model, backend):
            states[backend], trajectories[backend] = [], []
            for step in range(2):
                outputs.clear()
                row = worker.train_step(model, optimizer, batches[step], args)
                if not row["loss_and_grad_finite"] or not row["optimizer_applied"]:
                    raise FloatingPointError(f"{backend}: nonfinite numerical-check step {step + 1}")
                states[backend].append(snapshot(model, optimizer, outputs))
                trajectories[backend].append(dict(loss=row["loss"], grad_l2_norm=row["grad_l2_norm"]))
        hook.remove()
        expected_route = backend == "current"
        if not observed_route or any(value != expected_route for value in observed_route):
            raise RuntimeError(f"{backend}: unexpected SpectralSolve graph routing {observed_route}")
        routes[backend] = dict(spectral_solve=expected_route, forward_calls=len(observed_route))
        del optimizer, model, hook
        clear_cuda(ops)
    comparisons = []
    for step in range(2):
        before, current = states["pytorch"][step], states["current"][step]
        comparisons.append(dict(step=step + 1,
                                **{name: difference(current[name], before[name])
                                   for name in ("outputs", "parameters", "gradients", "adam")},
                                loss_absolute_difference=abs(trajectories["current"][step]["loss"] -
                                                             trajectories["pytorch"][step]["loss"])))
    return dict(scope="Two fresh Adam updates from the same pretrained state; reported differences only, no newly selected tolerances",
                routes=routes, trajectories=trajectories, current_vs_pytorch=comparisons)


def timed_fixture(template, batches, batch_hashes, args, backend, initial_hash, ops):
    import torch
    from evaluate_usrnet_quality import backend_scope
    clear_cuda(ops)
    model = copy.deepcopy(template).cuda().train()
    optimizer = adam(model)
    with backend_scope(model, backend):
        began = time.perf_counter()
        for index in range(args.warmup):
            row = worker.train_step(model, optimizer, batches[index], args)
            if not row["loss_and_grad_finite"] or not row["optimizer_applied"]:
                raise FloatingPointError(f"{backend}: nonfinite warmup step {index + 1}")
        warmup_wall_s = time.perf_counter() - began
        restore_initial(model, optimizer, template.state_dict())
        restored_hash = worker.tensor_hash(model.state_dict())
        if restored_hash != initial_hash:
            raise RuntimeError(f"{backend}: initial parameters were not restored exactly")
        initial_adam = adam_tensors(model, optimizer)
        if any(torch.count_nonzero(value).item() for value in initial_adam.values()):
            raise RuntimeError(f"{backend}: initial Adam state is not zero")
        initial_adam_hash = worker.tensor_hash(initial_adam)
        del initial_adam
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for index in range(args.steps):
            row = worker.train_step(model, optimizer, batches[index], args)
            row.update(step=index + 1, batch_sha256=batch_hashes[index])
            if not row["loss_and_grad_finite"] or not row["optimizer_applied"]:
                raise FloatingPointError(f"{backend}: nonfinite timed step {index + 1}")
            rows.append(row)
        parameter_check = worker.check_parameters(model)
        # No output/gradient/optimizer snapshot remains on the GPU after return.
        final_state_hash = worker.tensor_hash(model.state_dict())
        result = dict(backend=backend, warmup_steps=args.warmup, warmup_wall_s=warmup_wall_s,
                      initial_state_tensor_sha256=restored_hash, initial_adam_state_sha256=initial_adam_hash,
                      final_state_tensor_sha256=final_state_hash, parameters_finite=parameter_check,
                      rows=rows,
                      medians={key: statistics.median(row[key] for row in rows) for key in
                               ("training_step_wall_ms", "training_step_cuda_span_ms", "h2d_wall_ms",
                                "forward_backward_wall_ms", "forward_backward_cuda_event_ms")},
                      total_training_step_wall_s=sum(row["training_step_wall_ms"] for row in rows) / 1000,
                      peak_memory={key: max(row["peak_memory"][key] for row in rows)
                                   for key in ("allocated_bytes", "reserved_bytes")})
    del optimizer, model
    clear_cuda(ops)
    return result


def summarize_rounds(rounds):
    result = {}
    keys = ("training_step_wall_ms", "training_step_cuda_span_ms", "forward_backward_wall_ms",
            "forward_backward_cuda_event_ms")
    for key in keys:
        ratios = [row["variants"]["pytorch"]["medians"][key] /
                  row["variants"]["current"]["medians"][key] for row in rounds]
        result[key] = dict(paired_round_ratios=ratios, median=statistics.median(ratios),
                           min=min(ratios), max=max(ratios))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset_training/split_900_100.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_zoo/converse_usrnet.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, help="Same for both backends; defaults to batch-size")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/python_training/paired_timing.json")
    args = parser.parse_args()
    args.microbatch_size = args.batch_size if args.microbatch_size is None else args.microbatch_size
    if min(args.batch_size, args.microbatch_size, args.steps, args.rounds, args.warmup) < 1:
        parser.error("Batch sizes, steps, rounds and warmup must be positive")
    if args.microbatch_size > args.batch_size or args.batch_size % args.microbatch_size:
        parser.error("microbatch-size must divide batch-size and apply identically to both backends")
    if args.output.exists():
        parser.error(f"Refusing to overwrite an existing timing report: {args.output}")
    # These match the declared fine-tuning recipe; no auto-adjustment after OOM.
    args.seed, args.patch_size, args.scale, args.noise_std = 17, 96, 3, 0.01
    args.loss, args.lr, args.variant = "mse", 1e-5, "current"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from evaluate_usrnet_quality import checkpoint_layout
    from usrnet_training_data import DatasetProtocol
    ops, backend_manifest = worker.load_backend(args)
    from models.converse_usrnet import ConverseUSRNet
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    architecture = checkpoint_layout(state)
    template = ConverseUSRNet(num_iterations=5, num_blocks=7, in_channels=64,
                              backend="cuda", reuse_training_spectra=False).float().train()
    template.load_state_dict(state, strict=True)
    initial_hash = worker.tensor_hash(template.state_dict())
    protocol = DatasetProtocol(args.manifest, patch_size=args.patch_size, scale=args.scale,
                               seed=args.seed, noise_std=args.noise_std)
    batches, batch_hashes = [], []
    for step in range(max(args.steps, args.warmup, 2)):
        batch = protocol.train_batch(step, args.batch_size)
        worker.check_cpu_batch(batch, args, args.batch_size)
        batches.append(batch)
        batch_hashes.append(worker.tensor_hash(dict(lr=batch[0], kernel=batch[1], hr=batch[2])))
    source_hashes = worker.source_hashes()
    source_hashes[Path(__file__).relative_to(ROOT).as_posix()] = worker.file_hash(Path(__file__))
    report = dict(status="running", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  config=worker.json_safe(vars(args)), architecture=architecture,
                  input_checkpoint_sha256=worker.file_hash(args.checkpoint), initial_state_tensor_sha256=initial_hash,
                  source_sha256=source_hashes, current_backend=backend_manifest, dataset=protocol.metadata,
                  cached_batch_sha256=batch_hashes, timed_batch_sequence_sha256=worker.hash_json(batch_hashes[:args.steps]),
                  environment=dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(), tf32=False, cudnn_deterministic=True,
                                   cudnn_benchmark=False, amp=False),
                  protocol=dict(pytorch="Production Python converse2d_reference / ATen full FFT, backend='pytorch'",
                                current="Production CUDA spectral training backend, reuse_training_spectra=False",
                                optimizer="Adam MSE lr=1e-5; betas .9/.999, eps1e-8, weight_decay0, foreach/fused false",
                                reset="Same strict pretrained parameters; Adam step/exp_avg/exp_avg_sq reset to zero after warmup while retaining allocations",
                                timing="Unmodified train_usrnet_dataset.train_step: synchronized full step includes H2D, microbatch accumulation, loss/backward, finite/norm checks and Adam. Phase synchronization retained for both variants.",
                                excluded="Compilation, data decoding/degradation/hash, model/optimizer creation, warmup/reset, numerical snapshots, serialization and profiling",
                                memory="One GPU fixture at a time; total PyTorch peak allocated/reserved during warm timed steps. Reserved includes allocated; driver/library memory excluded.",
                                interpretation="Paired workload timing only; no convergence claim and no replacement or relaxation of independent FP64/task-quality gates"),
                  rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        report["numerical_check"] = numerical_check(template, batches, args, ops)
        worker.write_json(args.output, report)
        for index in range(args.rounds):
            order = ("pytorch", "current") if index % 2 == 0 else ("current", "pytorch")
            row = dict(round=index + 1, order=list(order), variants={})
            for backend in order:
                row["variants"][backend] = timed_fixture(template, batches, batch_hashes, args, backend, initial_hash, ops)
                print(json.dumps(dict(round=index + 1, backend=backend,
                                      medians=row["variants"][backend]["medians"],
                                      peak_memory=row["variants"][backend]["peak_memory"])), flush=True)
            if row["variants"]["pytorch"]["initial_adam_state_sha256"] != row["variants"]["current"]["initial_adam_state_sha256"]:
                raise RuntimeError("Paired round starts from different Adam states")
            row["speedup_pytorch_over_current"] = {
                key: row["variants"]["pytorch"]["medians"][key] / row["variants"]["current"]["medians"][key]
                for key in ("training_step_wall_ms", "training_step_cuda_span_ms", "forward_backward_wall_ms",
                            "forward_backward_cuda_event_ms")}
            report["rounds"].append(row)
            report["paired_speedup_summary"] = summarize_rounds(report["rounds"])
            worker.write_json(args.output, report)
        report["status"] = "complete"
        report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        worker.write_json(args.output, report)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
