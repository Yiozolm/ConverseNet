"""Audited full USRNet FP32 fine-tuning on the declared real-photo protocol.

Default recipe: unchanged pretrained 5-iteration/7-block/64-channel model,
unclipped RGB MSE, Adam(lr=1e-5, betas=.9/.999, eps=1e-8, weight_decay=0),
effective batch 4, HR crop 96, scale 3, 250 optimizer steps. No AMP, scheduler,
gradient clipping, gate/lambda reinitialization or automatic recipe adjustment.
Every evaluation covers all 100 held-out fixed crops in both RGB and Y using
evaluate_usrnet_quality.quality. A short fine-tuning run is not convergence proof.

Example capacity pilot (choose a NEW output directory):
  python tools/roadmap_quality/train_usrnet_dataset.py --purpose pilot --steps 5 \
    --eval-every 5 --run-dir artifacts/roadmap_quality/pilot_current

Sampling is deterministic. --deterministic-algorithms selects a separate,
explicit CUDA algorithm lane; it never changes production defaults. That lane
no longer NaN-fills uninitialized memory (byte-identical, measured 1.11x B4
training); --fill-uninitialized-memory restores the legacy fill. Configs and
environments record the setting only when the fill is off, so legacy runs keep
their recipe hashes and resume only into a matching lane.
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import traceback

import run_state

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]


class BudgetStop(Exception):
    pass


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(value), allow_nan=False) + "\n")
        stream.flush()


def tensor_hash(tensors):
    """Canonical tensor content hash, independent of torch.save archive metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def production_namespace(ops):
    import torch
    previous = torch.ops.converse2d
    torch.ops.converse2d = ops
    try:
        yield
    finally:
        torch.ops.converse2d = previous


def load_backend(args):
    import torch
    if os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        raise RuntimeError("Unset CONVERSE2D_CPU_ONLY for CUDA training")
    if os.environ.get("CONVERSE2D_BACKEND", "").lower() not in ("", "auto", "cuda"):
        raise RuntimeError("CONVERSE2D_BACKEND must not override this run with the PyTorch backend")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training worker")
    # Each source checkout runs in its own process. The variant is only a
    # comparison label, never a route to an old half-spectrum backend.
    loader_path = ROOT / "test/extension_loader.py"
    spec = importlib.util.spec_from_file_location("quality_checked_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    os.environ["CONVERSE2D_SKIP_BUILD"] = "0" if args.build else "1"
    loader.load_extension(verbose=args.verbose_build)
    manifest = ROOT / ".build/cuda/source_manifest.json"
    return torch.ops.converse2d, dict(kind="checked production checkout", root=str(ROOT),
                                    build_manifest=json.loads(manifest.read_text(encoding="utf-8")))


def source_hashes():
    helpers = {"tools/roadmap_quality/" + name: file_hash(TOOLS / name) for name in
               ("train_usrnet_dataset.py", "usrnet_training_data.py", "evaluate_usrnet_quality.py", "run_state.py")}
    paths = [ROOT / "test/extension_loader.py", ROOT / "utils/utils_image.py"]
    paths += sorted((ROOT / "models").rglob("*.py"))
    paths += [path for path in sorted((ROOT / "Converse2D/torch_converse2d").rglob("*"))
              if path.suffix in (".cpp", ".cu", ".h", ".cuh")]
    paths += [ROOT/"Converse2D/build_config.py",ROOT/"Converse2D/setup.py"]
    return {**helpers, **{path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}}


def cuda_timed(fn):
    import torch
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    began = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, dict(wall_ms=(time.perf_counter() - began) * 1000,
                       cuda_event_ms=start.elapsed_time(end))


def cuda_peaks():
    import torch
    return dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.max_memory_reserved())


def update_peaks(destination, values):
    for key, value in values.items():
        destination[key] = max(destination.get(key, 0), value)


def check_parameters(model):
    import torch
    values = [value.detach() for value in (*model.parameters(), *model.buffers())]
    finite = bool(torch.stack([torch.isfinite(value).all() for value in values]).all().item())
    if not finite:
        raise FloatingPointError("Nonfinite model parameter or buffer")
    return dict(finite=True, tensor_count=len(values))


def check_cpu_batch(batch, args, expected_batch):
    import torch
    expected = [(expected_batch, 3, args.patch_size // args.scale, args.patch_size // args.scale),
                (expected_batch, 1, 7, 7),
                (expected_batch, 3, args.patch_size, args.patch_size)]
    if len(batch) != 3:
        raise ValueError("Dataset must return LR, kernel, HR")
    for tensor, shape in zip(batch, expected):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError(f"Expected finite CPU FP32 tensor {shape}; got {tensor.device}, {tensor.dtype}, {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite dataset tensor")


def train_step(model, optimizer, batch, args):
    import torch
    import torch.nn.functional as F
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    began = time.perf_counter()
    start.record()
    optimizer.zero_grad(set_to_none=True)
    h2d, compute = [], []
    losses = []
    for offset in range(0, args.batch_size, args.microbatch_size):
        size = min(args.microbatch_size, args.batch_size - offset)
        gpu, transfer_time = cuda_timed(lambda: tuple(value[offset:offset + size].cuda()
                                                     for value in batch))
        h2d.append(transfer_time)

        def forward_backward():
            output = model(gpu[0], gpu[1], args.scale)
            if output.dtype != torch.float32 or output.shape != gpu[2].shape:
                raise RuntimeError("Training output must match the FP32 HR target")
            # Loss is applied directly to unclipped, unrounded RGB output.
            loss = (F.mse_loss(output, gpu[2]) if args.loss == "mse" else F.l1_loss(output, gpu[2]))
            weighted = loss * (size / args.batch_size)
            weighted.backward()
            return weighted.detach()

        loss, timing = cuda_timed(forward_backward)
        losses.append(loss)
        compute.append(timing)
        del gpu, loss
    grads = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
    if len(grads) != sum(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Every differentiable model parameter must receive a gradient")

    def finite_and_norm():
        loss = torch.stack(losses).sum()
        finite = torch.stack([torch.isfinite(loss), *[torch.isfinite(grad).all() for grad in grads]]).all()
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        norm = torch.linalg.vector_norm(flat)
        finite = finite & torch.isfinite(norm)
        # One host transfer for loss, norm and the aggregated finite flag.
        return torch.stack((loss, norm, finite.to(loss.dtype))).cpu().tolist()

    (loss_value, grad_norm, finite_value), finite_time = cuda_timed(finite_and_norm)
    finite = bool(finite_value)
    optimizer_time = dict(wall_ms=0.0, cuda_event_ms=0.0)
    if finite:
        _, optimizer_time = cuda_timed(optimizer.step)
    end.record()
    end.synchronize()
    return dict(loss=loss_value, grad_l2_norm=grad_norm, loss_and_grad_finite=finite,
                gradient_tensor_count=len(grads), optimizer_applied=finite,
                microbatches=len(h2d), samples=args.batch_size,
                training_step_wall_ms=(time.perf_counter() - began) * 1000,
                training_step_cuda_span_ms=start.elapsed_time(end),
                h2d_wall_ms=sum(row["wall_ms"] for row in h2d),
                h2d_cuda_event_ms=sum(row["cuda_event_ms"] for row in h2d),
                forward_backward_wall_ms=sum(row["wall_ms"] for row in compute),
                forward_backward_cuda_event_ms=sum(row["cuda_event_ms"] for row in compute),
                finite_check=finite_time, optimizer=optimizer_time, peak_memory=cuda_peaks())


def evaluate(model, optimizer, protocol, ops, args, optimizer_steps, expected_ids, stop_reason=None):
    import numpy as np
    import torch
    from evaluate_usrnet_quality import quality
    parameter_check = check_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    clear_cache = getattr(ops, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    was_training = model.training
    model.eval()
    began = time.perf_counter()
    rows, seen, payload_hash = [], set(), hashlib.sha256()
    transfer_ms = 0.0
    try:
        with torch.inference_mode():
            for ids, lr, kernel, hr in protocol.validation_batches(args.eval_batch_size):
                if stop_reason is not None:
                    reason = stop_reason()
                    if reason:
                        raise BudgetStop(reason)
                check_cpu_batch((lr, kernel, hr), args, len(ids))
                if len(set(ids)) != len(ids) or seen.intersection(ids):
                    raise ValueError("Repeated validation IDs")
                seen.update(ids)
                payload_hash.update(json.dumps(ids, ensure_ascii=True).encode())
                payload_hash.update(tensor_hash(dict(lr=lr, kernel=kernel, hr=hr)).encode())
                gpu, timing = cuda_timed(lambda: (lr.cuda(), kernel.cuda()))
                transfer_ms += timing["wall_ms"]
                output = model(gpu[0], gpu[1], args.scale)
                if output.dtype != torch.float32 or not torch.isfinite(output).all().item():
                    raise FloatingPointError("Nonfinite or non-FP32 validation output")
                raw = output.cpu()
                for index, image_id in enumerate(ids):
                    target = np.rint(np.clip(hr[index].permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
                    rows.append(dict(id=image_id,
                                     rgb=quality(raw[index:index + 1], target, "rgb", args.crop_border),
                                     y=quality(raw[index:index + 1], target, "y", args.crop_border)))
                del gpu, output, raw
        if seen != expected_ids or len(rows) != 100:
            raise ValueError(f"Evaluation must cover the exact 100 held-out images; got {len(rows)}")
        torch.cuda.synchronize()
        result = dict(optimizer_steps=optimizer_steps, images=len(rows), all_outputs_finite=True,
                      parameter_check=parameter_check, validation_payload_sha256=payload_hash.hexdigest(),
                      wall_s=time.perf_counter() - began, h2d_wall_ms=transfer_ms,
                      peak_memory=cuda_peaks(), per_image=rows)
        for space in ("rgb", "y"):
            result[space] = {metric: statistics.mean(row[space][metric] for row in rows)
                             for metric in ("psnr_db", "ssim")}
        return result
    finally:
        model.train(was_training)
        if clear_cache is not None:
            clear_cache()
        torch.cuda.synchronize()


def save_checkpoint(path, model, optimizer, optimizer_steps, report, metrics=None):
    import torch
    began = time.perf_counter()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu_copy(item) for item in value)
        return value

    state_digest = tensor_hash(state)
    payload = dict(state_dict=state, optimizer_state_dict=cpu_copy(optimizer.state_dict()),
                   optimizer_steps=optimizer_steps, config=report["config"],
                   config_sha256=report["config_sha256"], split_sha256=report["split_sha256"],
                   source_sha256=report["source_sha256"], metrics=json_safe(metrics),
                   resume_state_version=run_state.CHECKPOINT_VERSION,
                   resume_recipe_sha256=report["resume_recipe_sha256"],
                   dataset=report["dataset"], input_checkpoint_sha256=report["input_checkpoint_sha256"],
                   environment=report["environment"], next_data_step=report["next_data_step"],
                   backend_manifest=report["backend"]["build_manifest"],
                   rng_state=run_state.capture_rng(), metric_history=json_safe(report["metric_history"]),
                   origin_initial_state_tensor_sha256=report.get("origin_initial_state_tensor_sha256",
                       report.get("initial_state_tensor_sha256", state_digest)),
                   run_status=report["status"], stop_reason=report.get("stop_reason"),
                   parent_run_dir=str(Path(report["config"]["run_dir"])),
                   session_start_step=report.get("session_start_step", 0))
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return dict(path=str(path), file_sha256=file_hash(path), state_tensor_sha256=state_digest,
                optimizer_steps=optimizer_steps, wall_s=time.perf_counter() - began)


def execute(args, protocol, report):
    import numpy as np
    import torch
    from evaluate_usrnet_quality import checkpoint_layout
    process_started = report["_process_started"]
    deadline = run_state.parse_deadline(args.deadline_utc)

    def stopping_reason():
        return run_state.budget_reason(process_started, args.max_wall_seconds, deadline)

    resume = None
    resume_checksum = None
    if args.resume:
        checkpoint_bytes = args.resume.read_bytes()
        resume_checksum = hashlib.sha256(checkpoint_bytes).hexdigest()
        resume = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        del checkpoint_bytes
        run_state.validate_resume(resume, report)
        if args.steps <= resume["optimizer_steps"]:
            raise ValueError("--steps must exceed the checkpoint's completed update count")
    ops, backend_manifest = load_backend(args)
    report["backend"] = backend_manifest
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(args.deterministic_algorithms)
    if args.deterministic_algorithms:
        torch.utils.deterministic.fill_uninitialized_memory = args.fill_uninitialized_memory
    report["environment"] = dict(torch=str(torch.__version__), numpy=np.__version__,
                                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True,
                                deterministic_algorithms=args.deterministic_algorithms,
                                python=platform.python_version(),
                                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""))
    if args.deterministic_algorithms and not args.fill_uninitialized_memory:
        report["environment"]["fill_uninitialized_memory"] = False
    from importlib.metadata import version
    report["environment"].update(scipy=version("scipy"), pillow=version("pillow"))
    if resume is not None:
        run_state.validate_resume(resume, report)
    # Frozen imports and ALL forwards/evaluations share the same namespace scope.
    with production_namespace(ops):
        from models.converse_usrnet import ConverseUSRNet
        model_class = ConverseUSRNet
        model = model_class(num_iterations=5, num_blocks=7, in_channels=64, backend="cuda")
        if args.init == "pretrained":
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            report["architecture"] = checkpoint_layout(state)
            model.load_state_dict(state, strict=True)
        else:
            report["architecture"] = checkpoint_layout(model.state_dict())
        model = model.float().cuda().train()
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise RuntimeError("All training parameters must be FP32")
        report["parameter_count"] = sum(value.numel() for value in model.parameters())
        report["parameter_tensor_count"] = len(list(model.parameters()))
        check_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0, amsgrad=False, foreach=False, fused=False)
        start_step = 0
        if resume is not None:
            if any(value.is_floating_point() and value.dtype != torch.float32
                   for value in resume["state_dict"].values()):
                raise ValueError("Resume model tensors must be FP32")
            model.load_state_dict(resume["state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            for group in optimizer.param_groups:
                expected = dict(lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                amsgrad=False, foreach=False, fused=False)
                if any(group.get(key) != value for key, value in expected.items()):
                    raise ValueError("Resume optimizer hyperparameters differ from the declared recipe")
            start_step = resume["next_data_step"]
            report["optimizer_steps"] = report["next_data_step"] = start_step
            report["samples_seen"] = start_step * args.batch_size
            report["metric_history"] = resume["metric_history"]
            if args.stop_when_stable:
                report["stability"] = run_state.stability(report["metric_history"], start_step)
            report["origin_initial_state_tensor_sha256"] = resume["origin_initial_state_tensor_sha256"]
            report["resume_parent"] = dict(path=str(args.resume), file_sha256=resume_checksum,
                                           run_dir=resume["parent_run_dir"], optimizer_steps=start_step,
                                           previous_status=resume["run_status"])
            run_state.restore_rng(resume["rng_state"])
            check_parameters(model)
        report["session_start_step"] = start_step
        report["checkpoints"]["initial"] = save_checkpoint(args.run_dir / "initial.pth", model, optimizer, start_step, report)
        report["initial_state_tensor_sha256"] = report["checkpoints"]["initial"]["state_tensor_sha256"]
        report.setdefault("origin_initial_state_tensor_sha256", report["initial_state_tensor_sha256"])
        report["timing"]["checkpoint_wall_s"] += report["checkpoints"]["initial"]["wall_s"]
        expected_ids = {row["relative_path"] for row in protocol.validation}
        best_y = max((float(row["y"]["psnr_db"]) for row in report["metric_history"]), default=-math.inf)
        validation_hash = None
        loop_started = time.perf_counter()
        report["timing"]["setup_wall_s"] = loop_started - report.pop("_process_started")

        def evaluation(step):
            nonlocal best_y, validation_hash
            evaluation_started = time.perf_counter()
            try:
                value = evaluate(model, optimizer, protocol, ops, args, step, expected_ids, stopping_reason)
            except BudgetStop as error:
                elapsed = time.perf_counter() - evaluation_started
                report["interrupted_evaluation"] = dict(optimizer_steps=step, reason=str(error), wall_s=elapsed)
                report["timing"]["evaluation_wall_s"] += elapsed
                return str(error)
            if validation_hash is None:
                validation_hash = value["validation_payload_sha256"]
                report["validation_payload_sha256"] = validation_hash
            elif value["validation_payload_sha256"] != validation_hash:
                raise RuntimeError("Validation tensors changed during the run")
            report["timing"]["evaluation_wall_s"] += value["wall_s"]
            update_peaks(report["evaluation_peak_memory"], value["peak_memory"])
            update_peaks(report["overall_peak_memory"], value["peak_memory"])
            report["metric_history"].append(dict(optimizer_steps=step, rgb=value["rgb"], y=value["y"]))
            if args.stop_when_stable:
                report["stability"] = run_state.stability(report["metric_history"], step)
            value["best_saved"] = value["y"]["psnr_db"] > best_y
            if value["best_saved"]:
                best_y = value["y"]["psnr_db"]
                checkpoint = save_checkpoint(args.run_dir / "best.pth", model, optimizer, step, report,
                                             {space: value[space] for space in ("rgb", "y")})
                report["checkpoints"]["best"] = checkpoint
                report["timing"]["checkpoint_wall_s"] += checkpoint["wall_s"]
                value["saved_checkpoint_sha256"] = checkpoint["file_sha256"]
            append_jsonl(args.run_dir / "evaluations.jsonl", value)
            report["last_evaluation"] = {key: item for key, item in value.items() if key != "per_image"}
            report["evaluation_steps"].append(step)
            if args.stop_when_stable or args.max_wall_seconds is not None or args.deadline_utc or args.resume:
                checkpoint = save_checkpoint(args.run_dir / "latest.pth", model, optimizer, step, report,
                                             {space: value[space] for space in ("rgb", "y")})
                report["checkpoints"]["latest"] = checkpoint
                report["timing"]["checkpoint_wall_s"] += checkpoint["wall_s"]
            print(json.dumps(dict(event="evaluation", optimizer_steps=step, images=value["images"],
                                  rgb=value["rgb"], y=value["y"], wall_s=value["wall_s"])), flush=True)
            torch.cuda.reset_peak_memory_stats()
            write_json(args.run_dir / "run.json", report)
            return None

        reason = stopping_reason()
        if reason is None and args.stop_when_stable and report["stability"]["satisfied"]:
            reason = "stability_window"
        last_evaluated = report["metric_history"][-1]["optimizer_steps"] if report["metric_history"] else -1
        # A deadline may interrupt validation after its optimizer update has
        # completed. Finish that scheduled evaluation on resume before taking
        # another update; do not invent an extra off-cadence evaluation.
        pending_evaluation = start_step % args.eval_every == 0 and last_evaluated < start_step
        if reason is None and pending_evaluation:
            reason = evaluation(start_step)
            if reason is None and args.stop_when_stable and report["stability"]["satisfied"]:
                reason = "stability_window"
        torch.cuda.reset_peak_memory_stats()
        for step in range(start_step, args.steps):
            reason = reason or stopping_reason()
            if reason:
                break
            began = time.perf_counter()
            batch = protocol.train_batch(step, args.batch_size)
            check_cpu_batch(batch, args, args.batch_size)
            batch_hash = tensor_hash(dict(lr=batch[0], kernel=batch[1], hr=batch[2]))
            data_ms = (time.perf_counter() - began) * 1000
            row = train_step(model, optimizer, batch, args)
            row.update(step=step + 1, data_step=step, data_prepare_and_hash_wall_ms=data_ms,
                       batch_sha256=batch_hash)
            if row["optimizer_applied"]:
                report["optimizer_steps"] += 1
                report["samples_seen"] += args.batch_size
                report["next_data_step"] = step + 1
            row["optimizer_steps"] = report["optimizer_steps"]
            row["samples_seen"] = report["samples_seen"]
            append_jsonl(args.run_dir / "training.jsonl", row)
            for key in ("training_step_wall_ms", "h2d_wall_ms", "forward_backward_wall_ms"):
                report["timing"]["total_" + key] += row[key]
            report["timing"]["data_prepare_and_hash_wall_ms"] += data_ms
            report["timing"]["finite_check_wall_ms"] += row["finite_check"]["wall_ms"]
            report["timing"]["optimizer_wall_ms"] += row["optimizer"]["wall_ms"]
            update_peaks(report["training_peak_memory"], row["peak_memory"])
            update_peaks(report["overall_peak_memory"], row["peak_memory"])
            report["last_training_step"] = row
            report["all_loss_and_grad_finite"] &= row["loss_and_grad_finite"]
            print(json.dumps(json_safe(dict(event="training", optimizer_steps=report["optimizer_steps"],
                                           loss=row["loss"], grad_l2_norm=row["grad_l2_norm"],
                                           finite=row["loss_and_grad_finite"], samples_seen=report["samples_seen"],
                                           step_wall_ms=row["training_step_wall_ms"])), allow_nan=False), flush=True)
            if not row["loss_and_grad_finite"]:
                raise FloatingPointError(f"Step {step + 1}: nonfinite loss/gradient; optimizer update was skipped")
            del batch
            reason = stopping_reason()
            if reason:
                break
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                reason = evaluation(step + 1)
                if reason:
                    break
                if args.stop_when_stable and report["stability"]["satisfied"]:
                    reason = "stability_window"
                    break
            report["timing"]["training_loop_end_to_end_wall_s"] = time.perf_counter() - loop_started
            write_json(args.run_dir / "run.json", report)
        report["final_parameter_check"] = check_parameters(model)
        if source_hashes() != report["source_sha256"]:
            raise RuntimeError("Source changed during this quality run; result is not admissible")
        manifest = backend_manifest["build_manifest"]
        if file_hash(ROOT / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
            raise RuntimeError("Extension binary changed during this quality run")
        if reason in ("max_wall_seconds", "deadline_utc"):
            report["status"] = "budget_stopped"
        elif reason == "stability_window":
            report["status"] = "stable"
        elif args.stop_when_stable:
            report["status"] = "max_steps_reached"
            reason = "max_steps_without_stability"
        else:
            report["status"] = "complete"
            reason = "max_steps"
        report["stop_reason"] = reason
        report["session_optimizer_steps"] = report["optimizer_steps"] - start_step
        report["final_evaluation_step"] = report["metric_history"][-1]["optimizer_steps"] if report["metric_history"] else None
        for name in (("latest", "final") if args.stop_when_stable or args.max_wall_seconds is not None or args.deadline_utc or args.resume else ("final",)):
            checkpoint = save_checkpoint(args.run_dir / f"{name}.pth", model, optimizer,
                                         report["optimizer_steps"], report, report.get("last_evaluation"))
            report["checkpoints"][name] = checkpoint
            report["timing"]["checkpoint_wall_s"] += checkpoint["wall_s"]
        report["timing"]["training_loop_end_to_end_wall_s"] = time.perf_counter() - loop_started
        report["process_elapsed_wall_s"] = time.perf_counter() - process_started
        report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.run_dir / "run.json", report)
        print(json.dumps(dict(event="stopped", status=report["status"], reason=report["stop_reason"],
                              optimizer_steps=report["optimizer_steps"],
                              stability_satisfied=bool(report["stability"] and report["stability"]["satisfied"]))), flush=True)


def main():
    global ROOT
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset_training/split_900_100.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--variant", choices=("current", "before"), default="current",
                        help="Comparison label only; --root selects the actual source checkout")
    parser.add_argument("--root", type=Path, default=ROOT, help="Checked source checkout to execute")
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Repository containing images and kernels")
    parser.add_argument("--init", choices=("pretrained", "scratch"), default="pretrained")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_zoo/converse_usrnet.pth")
    parser.add_argument("--purpose", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-every", type=int, help="Default 125, or 250 in stability mode")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, help="Defaults to batch-size; must divide it")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--scale", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--crop-border", type=int, help="Metric border crop; defaults to scale")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--build", action="store_true", help="Allow a checked build; default verifies an existing binary")
    parser.add_argument("--deterministic-algorithms", action="store_true",
                        help="Separate opt-in algorithm lane, applied equally to both variants")
    parser.add_argument("--fill-uninitialized-memory", action="store_true",
                        help="Deterministic lane only: restore the legacy NaN fill of new allocations")
    parser.add_argument("--stop-when-stable", action="store_true",
                        help="After at least 1000 updates, stop on the declared five-evaluation stability window")
    parser.add_argument("--max-wall-seconds", type=float, help="Per-process wall budget including setup")
    parser.add_argument("--deadline-utc", help="Timezone-aware ISO deadline; stop at a safe optimizer/evaluation boundary")
    parser.add_argument("--resume", type=Path, help="Explicit roadmap checkpoint to resume into a new run directory")
    args = parser.parse_args()
    if args.fill_uninitialized_memory and not args.deterministic_algorithms:
        parser.error("--fill-uninitialized-memory only applies with --deterministic-algorithms")
    if args.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ROOT = args.root = args.root.resolve()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.resume = args.resume.resolve() if args.resume is not None else None
    args.run_dir = args.run_dir.resolve()
    args.eval_every = (250 if args.stop_when_stable else 125) if args.eval_every is None else args.eval_every
    args.microbatch_size = args.batch_size if args.microbatch_size is None else args.microbatch_size
    args.crop_border = args.scale if args.crop_border is None else args.crop_border
    if min(args.steps, args.eval_every, args.batch_size, args.microbatch_size, args.eval_batch_size) < 1:
        parser.error("Steps, evaluation interval and batch sizes must be positive")
    if args.microbatch_size > args.batch_size or args.batch_size % args.microbatch_size:
        parser.error("microbatch-size must be <= batch-size and divide it exactly")
    if args.patch_size < 1 or args.patch_size % args.scale or args.crop_border < 0 or args.patch_size - 2 * args.crop_border < 11:
        parser.error("Crop must be scale-divisible and leave at least 11 pixels for SSIM")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("lr must be finite and positive; noise-std finite and nonnegative")
    if args.stop_when_stable and (args.steps < 1000 or args.eval_every != 250):
        parser.error("Stability mode requires --steps >= 1000 and evaluations every 250 updates")
    if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
        parser.error("max-wall-seconds must be finite and positive")
    try:
        deadline = run_state.parse_deadline(args.deadline_utc)
        args.deadline_utc = deadline.isoformat() if deadline is not None else None
    except ValueError as error:
        parser.error(str(error))
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume}")
    if args.run_dir.exists():
        parser.error(f"Run directory already exists; refusing overwrite: {args.run_dir}")
    if args.init == "pretrained" and not args.checkpoint.is_file():
        parser.error(f"Pretrained checkpoint does not exist: {args.checkpoint}")
    sys.path.insert(0, str(ROOT))
    import usrnet_training_data
    usrnet_training_data.ROOT = args.data_root
    from usrnet_training_data import DatasetProtocol
    protocol = DatasetProtocol(args.manifest, patch_size=args.patch_size, scale=args.scale,
                               seed=args.seed, noise_std=args.noise_std)
    if protocol.metadata["validation_images"] != 100:
        parser.error("This experiment requires exactly 100 held-out validation images")
    if args.run_dir.is_relative_to(protocol.root):
        parser.error("Run output must be outside the read-only image directory")
    config = json_safe(vars(args))
    # Absent means the legacy lane (fill on, or irrelevant without determinism).
    if config.pop("fill_uninitialized_memory") is False and args.deterministic_algorithms:
        config["fill_uninitialized_memory"] = False
    split = {name: sorted(row["relative_path"] for row in getattr(protocol, name))
             for name in ("train", "validation")}
    report = dict(status="initializing", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  config=config, config_sha256=hash_json(config), comparison_recipe_sha256=run_state.recipe_hash(config),
                  resume_recipe_sha256=run_state.recipe_hash(config, resume=True),
                  dataset=protocol.metadata, split_sha256=hash_json(split), source_sha256=source_hashes(),
                  input_checkpoint_sha256=file_hash(args.checkpoint) if args.init == "pretrained" else None,
                  optimizer_steps=0, next_data_step=0, samples_seen=0, all_loss_and_grad_finite=True, checkpoints={},
                  session_start_step=0, resume_parent=None, metric_history=[],
                  stability=run_state.stability([], 0) if args.stop_when_stable else None,
                  long_run=dict(enabled=bool(args.stop_when_stable or args.resume or args.max_wall_seconds is not None or args.deadline_utc),
                                maximum_updates=args.steps, stop_when_stable=args.stop_when_stable,
                                timing_scope="Per-session timings; resume parent identifies earlier sessions. Metric history retains the complete trajectory.",
                                interpretation="A satisfied stability window is not a convergence or independent-test-quality proof."),
                  evaluation_steps=[], training_peak_memory={}, evaluation_peak_memory={}, overall_peak_memory={},
                  timing=dict(checkpoint_wall_s=0.0, evaluation_wall_s=0.0, total_training_step_wall_ms=0.0,
                              total_h2d_wall_ms=0.0, total_forward_backward_wall_ms=0.0,
                              data_prepare_and_hash_wall_ms=0.0, finite_check_wall_ms=0.0, optimizer_wall_ms=0.0),
                  _process_started=process_started,
                  protocol=dict(optimizer="Adam", betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
                                loss="unclipped/unrounded RGB " + args.loss, amp=False, scheduler=False,
                                gradient_clipping=False, gates_and_lambda="unchanged constructor/checkpoint values",
                                diagnostic_norm_dtype="float32",
                                best_checkpoint_metric="mean held-out Y PSNR", metric_border=args.crop_border,
                                quality="evaluate_usrnet_quality.quality: clip/round prediction to uint8, existing RGB/Y PSNR/SSIM",
                                timing="End-to-end loop includes eval, I/O and logging. Step wall includes microbatch H2D, forward/backward, finite/norm checks and Adam; phase timings are synchronized and exclude data preparation. No profiler.",
                                memory="Total PyTorch allocated/reserved high-water marks; includes optimizer/data/checks/evaluation by labelled phase; excludes driver/library allocations. Reserved includes allocated.",
                                scope=("Extended or resumed training; a satisfied metric window does not prove convergence or external quality"
                                       if args.stop_when_stable or args.resume else
                                       "Short pretrained fine-tuning on real photos with declared synthetic degradation; not convergence or an external benchmark"),
                                preregistered_paired_final_quality_gate=dict(max_psnr_drop_db=0.05, max_ssim_drop=0.001,
                                                                            spaces=["rgb", "y"], each_seed=True, all_steps_finite=True)))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir / "run.json", report)
    try:
        report["status"] = "running"
        execute(args, protocol, report)
    except BaseException as error:
        report.pop("_process_started", None)
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        report["process_elapsed_wall_s"] = time.perf_counter() - process_started
        write_json(args.run_dir / "run.json", report)
        raise
    print("Saved", args.run_dir, flush=True)


if __name__ == "__main__":
    main()
