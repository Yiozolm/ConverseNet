"""Continuous real-data FP32 training throughput, current versus Python/ATen.

Default: full pretrained USRNet 5/7/64, HR96/s3, batch4/micro4, Adam1e-5/MSE,
two alternating rounds of 100 updates per backend. Real image decoding,
degradation, CPU batch hashing, H2D and update-before finite checking are inside
the loop timer. There is one aggregated finite/loss D2H read per optimizer step,
no phase synchronization, and no validation, logging or checkpoint write inside
the measured loop. The loop drains CUDA once at the end.

Twenty-step blocks record host submission intervals and CUDA stream spans.
They do not independently drain CUDA, so they are not standalone step latency.
The final drain is recorded separately and included in overall loop wall time.
This measures throughput; it does not establish task quality or convergence.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import train_usrnet_dataset as worker
from benchmark_python_training import adam, adam_tensors, clear_cuda, difference, graph_has_spectral, restore_initial

ROOT = Path(__file__).resolve().parents[1]


def sustained_step(model, optimizer, batch, *, scale=3, microbatch_size=4,
                   expected_gradients=133, verify_backend=None):
    """One full Adam update, reusable by a profiler outside this benchmark.

    `batch` is an already prepared CPU (LR, kernel, HR) tuple. Dataset work is
    intentionally owned by the caller. The only explicit host/device rendezvous
    is one combined loss/finite transfer before Adam. Pageable H2D may itself
    block; no asynchronous-transfer performance is claimed.
    """
    import torch
    import torch.nn.functional as F
    batch_size = batch[0].shape[0]
    if len(batch) != 3 or microbatch_size < 1 or batch_size % microbatch_size:
        raise ValueError("Expected LR/kernel/HR and a dividing positive microbatch size")
    optimizer.zero_grad(set_to_none=True)
    losses, route = [], None
    for offset in range(0, batch_size, microbatch_size):
        gpu = tuple(value[offset:offset + microbatch_size].cuda() for value in batch)
        output = model(gpu[0], gpu[1], scale)
        if output.dtype != torch.float32 or output.shape != gpu[2].shape:
            raise RuntimeError("Require FP32 output matching the HR target")
        if verify_backend is not None:
            found = graph_has_spectral(output)
            expected = verify_backend == "current"
            if found != expected:
                raise RuntimeError(f"{verify_backend}: unexpected SpectralSolve graph route")
            route = found
        loss = F.mse_loss(output, gpu[2]) * (microbatch_size / batch_size)
        loss.backward()
        losses.append(loss.detach())
        del output, loss, gpu
    named_parameters = list(model.named_parameters())
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    if len(named_parameters) != expected_gradients or missing:
        raise RuntimeError(f"Expected all {expected_gradients} parameter gradients; got {len(named_parameters)}, missing={missing}")
    total_loss = torch.stack(losses).sum()
    finite = torch.stack([torch.isfinite(total_loss), *[
        torch.isfinite(parameter.grad).all() for _, parameter in named_parameters
    ]]).all()
    # A single CPU read, rather than .item() for every parameter. No norm is
    # computed here: it neither changes the update nor determines task quality.
    loss_value, finite_value = torch.stack((total_loss, finite.to(total_loss.dtype))).detach().cpu().tolist()
    if not bool(finite_value):
        raise FloatingPointError("Nonfinite loss/gradient; Adam update was not applied")
    optimizer.step()
    return dict(loss=loss_value, loss_and_grad_finite=True, optimizer_applied=True,
                gradient_tensor_count=len(named_parameters), all_gradients_present=True,
                spectral_solve_observed=route)


def state_snapshot(model, optimizer):
    return dict(parameters={name: value.detach().cpu().clone() for name, value in model.named_parameters()},
                gradients={name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()
                           if value.grad is not None}, adam=adam_tensors(model, optimizer))


def loop_equivalence(template, protocol, args, ops):
    """Two untimed updates of each loop from identical fresh Adam/model state."""
    import torch
    from evaluate_usrnet_quality import backend_scope
    batches = [protocol.train_batch(index, args.batch_size) for index in range(2)]
    backends = ("current", "pytorch") if args.equivalence_backends == "both" else ("current",)
    report = dict(criterion="Exact same-backend tensor/loss equality for a measurement-only loop rewrite; not an FP64/task-quality gate",
                  backends={}, all_identical=True)
    for backend in backends:
        saved = {}
        for loop in ("old_worker", "sustained"):
            clear_cuda(ops)
            model = copy.deepcopy(template).cuda().train()
            optimizer = adam(model)
            values = []
            with backend_scope(model, backend):
                for index, batch in enumerate(batches):
                    if loop == "old_worker":
                        result = worker.train_step(model, optimizer, batch, args)
                    else:
                        result = sustained_step(model, optimizer, batch, scale=args.scale,
                                                microbatch_size=args.microbatch_size,
                                                verify_backend=backend if index == 0 else None)
                    if not result["loss_and_grad_finite"] or not result["optimizer_applied"]:
                        raise FloatingPointError("Nonfinite loop-equivalence update")
                    values.append(dict(loss=result["loss"], state=state_snapshot(model, optimizer)))
            saved[loop] = values
            del optimizer, model
            clear_cuda(ops)
        checks = []
        for index in range(2):
            old, new = saved["old_worker"][index], saved["sustained"][index]
            details, identical = {}, old["loss"] == new["loss"]
            for group in ("parameters", "gradients", "adam"):
                details[group] = difference(new["state"][group], old["state"][group])
                equal = all(torch.equal(value, old["state"][group][name]) for name, value in new["state"][group].items())
                details[group]["exactly_equal"] = equal
                identical &= equal
            checks.append(dict(step=index + 1, old_loss=old["loss"], sustained_loss=new["loss"],
                               loss_abs_difference=abs(old["loss"] - new["loss"]),
                               identical=identical, **details))
            report["all_identical"] &= identical
        report["backends"][backend] = checks
    return report


def fixture(template, protocol, args, ops, backend, round_index, initial_hash, state_directory):
    import torch
    from evaluate_usrnet_quality import backend_scope
    clear_cuda(ops)
    model = copy.deepcopy(template).cuda().train()
    optimizer = adam(model)
    with backend_scope(model, backend):
        warm_started = time.perf_counter()
        for step in range(args.warmup):
            batch = protocol.train_batch(step, args.batch_size)
            sustained_step(model, optimizer, batch, scale=args.scale, microbatch_size=args.microbatch_size,
                           verify_backend=backend if step == 0 else None)
            del batch
        torch.cuda.synchronize()
        warmup_wall_s = time.perf_counter() - warm_started
        restore_initial(model, optimizer, template.state_dict())
        actual_initial = worker.tensor_hash(model.state_dict())
        if actual_initial != initial_hash:
            raise RuntimeError("Model reset did not recover the exact pretrained state")
        initial_adam = adam_tensors(model, optimizer)
        if bool(torch.cat([value.reshape(-1) for value in initial_adam.values()]).count_nonzero()):
            raise RuntimeError("Adam states were not zeroed after warmup")
        initial_adam_hash = worker.tensor_hash(initial_adam)
        del initial_adam
        # Pre-create event objects outside the loop. CUDA records do not impose
        # host waits; all elapsed-time reads happen after the one final drain.
        block_bounds = [(start, min(start + args.block_steps, args.steps))
                        for start in range(0, args.steps, args.block_steps)]
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in block_bounds]
        # Event handles initialize lazily on first record, so initialize those
        # handles before timing as well as allocating the Python objects.
        for start_event, end_event in events:
            start_event.record()
            end_event.record()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rows, blocks, data_wall_s = [], [], 0.0
        loop_started = time.perf_counter()
        for block_index, (first, end) in enumerate(block_bounds):
            block_started = time.perf_counter()
            events[block_index][0].record()
            block_data_s = 0.0
            for step in range(first, end):
                data_started = time.perf_counter()
                batch = protocol.train_batch(step, args.batch_size)
                worker.check_cpu_batch(batch, args, args.batch_size)
                batch_hash = worker.tensor_hash(dict(lr=batch[0], kernel=batch[1], hr=batch[2]))
                data_s = time.perf_counter() - data_started
                result = sustained_step(model, optimizer, batch, scale=args.scale,
                                        microbatch_size=args.microbatch_size)
                rows.append(dict(step=step + 1, batch_sha256=batch_hash,
                                 data_prepare_validate_hash_wall_ms=data_s * 1000, **result))
                data_wall_s += data_s
                block_data_s += data_s
                del batch
            events[block_index][1].record()
            blocks.append(dict(first_step=first + 1, last_step=end, updates=end - first,
                               host_interval_wall_ms=(time.perf_counter() - block_started) * 1000,
                               cpu_data_wall_ms=block_data_s * 1000))
        drain_started = time.perf_counter()
        torch.cuda.synchronize()
        loop_wall_s = time.perf_counter() - loop_started
        final_drain_s = time.perf_counter() - drain_started
        # Everything below, including serialization, is outside loop_wall_s.
        peak_memory = worker.cuda_peaks()
        for block, (start_event, end_event) in zip(blocks, events):
            block["cuda_stream_span_ms"] = start_event.elapsed_time(end_event)
        parameter_check = worker.check_parameters(model)
        final_parameters = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        final_gradients = {name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()}
        final_adam = adam_tensors(model, optimizer)
        state_path = state_directory / f"round{round_index + 1}_{backend}.pth"
        torch.save(dict(state_dict=final_parameters, adam_tensors=final_adam, optimizer_steps=args.steps,
                        backend=backend, config=worker.json_safe(vars(args))), state_path)
        result = dict(backend=backend, round=round_index + 1, warmup_steps=args.warmup,
                      warmup_wall_s=warmup_wall_s, optimizer_steps=args.steps,
                      samples=args.steps * args.batch_size, all_loss_and_grad_finite=True,
                      initial_state_tensor_sha256=actual_initial, initial_adam_state_sha256=initial_adam_hash,
                      final_state_tensor_sha256=worker.tensor_hash(final_parameters),
                      final_gradient_tensor_sha256=worker.tensor_hash(final_gradients),
                      final_adam_state_sha256=worker.tensor_hash(final_adam),
                      final_state_artifact=dict(path=str(state_path), sha256=worker.file_hash(state_path)),
                      parameters_finite=parameter_check,
                      loop_wall_s=loop_wall_s, updates_per_second=args.steps / loop_wall_s,
                      samples_per_second=args.steps * args.batch_size / loop_wall_s,
                      mean_loop_wall_ms_per_update=loop_wall_s * 1000 / args.steps,
                      cpu_data_prepare_validate_hash_wall_s=data_wall_s,
                      final_drain_wall_s=final_drain_s, peak_memory=peak_memory,
                      batch_sequence_sha256=worker.hash_json([row["batch_sha256"] for row in rows]),
                      blocks=blocks, steps=rows)
    del optimizer, model
    clear_cuda(ops)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset_training/split_900_100.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_zoo/converse_usrnet.pth")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--block-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--check-loop-equivalence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--equivalence-backends", choices=("current", "both"), default="current")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sustained_training/benchmark.json")
    args = parser.parse_args()
    if min(args.steps, args.rounds, args.warmup, args.block_steps, args.batch_size, args.microbatch_size) < 1:
        parser.error("Step, round, warmup, block and batch counts must be positive")
    if args.microbatch_size > args.batch_size or args.batch_size % args.microbatch_size:
        parser.error("microbatch-size must divide the common effective batch size")
    args.output = args.output.resolve()
    state_directory = args.output.parent / (args.output.stem + "_states")
    if args.output.exists() or state_directory.exists():
        parser.error("Output report/state directory already exists; choose a new output path")
    args.variant, args.patch_size, args.scale, args.noise_std, args.loss, args.lr = "current", 96, 3, 0.01, "mse", 1e-5
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
    state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    architecture = checkpoint_layout(state)
    template = ConverseUSRNet(num_iterations=5, num_blocks=7, in_channels=64,
                              backend="cuda", reuse_training_spectra=False).float().train()
    template.load_state_dict(state, strict=True)
    initial_hash = worker.tensor_hash(template.state_dict())
    protocol = DatasetProtocol(args.manifest, patch_size=96, scale=3, seed=args.seed, noise_std=0.01)
    if args.output.is_relative_to(protocol.root):
        parser.error("Outputs must remain outside the read-only source image directory")
    hashes = worker.source_hashes()
    for name in ("benchmark_python_training.py", "benchmark_sustained_training.py"):
        hashes["test/" + name] = worker.file_hash(ROOT / "test" / name)
    report = dict(status="ready", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  config=worker.json_safe(vars(args)), architecture=architecture, source_sha256=hashes,
                  checkpoint_sha256=worker.file_hash(args.checkpoint), initial_state_tensor_sha256=initial_hash,
                  dataset=protocol.metadata, current_backend=backend_manifest,
                  environment=dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(), tf32=False, amp=False, cudnn_benchmark=False,
                                   cudnn_deterministic=True), rounds=[],
                  protocol=dict(optimizer="Adam1e-5, betas=.9/.999, eps1e-8, weight_decay=0, foreach/fused=False",
                                model="Strict pretrained full5/7/64 FP32; gates/lambda unchanged; reuse disabled",
                                loop="Decode/degrade/CPU validate/hash + H2D + forward/MSE/backward + all-133-gradient finite reduction + one combined loss/flag D2H + Adam",
                                synchronization="One aggregate finite/loss host read per step; no per-phase waits. Pageable H2D may block. One final CUDA drain inside overall wall timer.",
                                blocks="Host boundary intervals have no independent end-of-block GPU drain; CUDA stream event spans read only after final synchronization. Neither is kernel-sum time.",
                                excluded="Validation, profiler, stdout, report/checkpoint writes, warmup/reset and final state hashing are outside the measured loop",
                                cache="Warm FFT/allocator state per fixture; data decoding repeated. OS file caching uncontrolled; image manifest hashing occurs before timing.",
                                memory="One GPU model/optimizer fixture at a time; PyTorch total allocated/reserved peaks; driver/library allocations excluded; reserved includes allocated",
                                interpretation="Continuous throughput experiment only, not a task-quality or convergence gate"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_directory.mkdir(exist_ok=False)
    worker.write_json(args.output, report)
    try:
        if args.check_loop_equivalence:
            report["loop_equivalence"] = loop_equivalence(template, protocol, args, ops)
            worker.write_json(args.output, report)
            if not report["loop_equivalence"]["all_identical"]:
                raise RuntimeError("Measurement-only loop rewrite changed a same-backend two-step result; inspect untimed diagnostic differences before timing")
        else:
            report["loop_equivalence"] = dict(skipped=True, reason="Explicit --no-check-loop-equivalence")
        report["status"] = "running"
        initial_adam_hashes, sequence_hashes = set(), set()
        for index in range(args.rounds):
            order = ["current", "pytorch"] if index % 2 == 0 else ["pytorch", "current"]
            row = dict(round=index + 1, order=order, variants={})
            for backend in order:
                value = fixture(template, protocol, args, ops, backend, index, initial_hash, state_directory)
                row["variants"][backend] = value
                initial_adam_hashes.add(value["initial_adam_state_sha256"])
                sequence_hashes.add(value["batch_sequence_sha256"])
            if len(initial_adam_hashes) != 1 or len(sequence_hashes) != 1:
                raise RuntimeError("Fixtures differ in initial Adam state or real input sequence")
            row["speedup_pytorch_over_current"] = row["variants"]["pytorch"]["loop_wall_s"] / row["variants"]["current"]["loop_wall_s"]
            report["rounds"].append(row)
            ratios = [item["speedup_pytorch_over_current"] for item in report["rounds"]]
            report["paired_speedup"] = dict(per_round=ratios, median=statistics.median(ratios), min=min(ratios), max=max(ratios))
            worker.write_json(args.output, report)
            print(json.dumps(dict(round=index + 1, order=order,
                                  loop_wall_s={name: value["loop_wall_s"] for name, value in row["variants"].items()},
                                  speedup_pytorch_over_current=row["speedup_pytorch_over_current"])), flush=True)
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
