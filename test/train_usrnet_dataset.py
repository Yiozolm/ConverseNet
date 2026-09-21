"""Deterministic full USRNet FP32 fine-tuning on the declared real-photo protocol.

Default recipe: unchanged pretrained 5-iteration/7-block/64-channel model,
unclipped RGB MSE, Adam(lr=1e-5, betas=.9/.999, eps=1e-8, weight_decay=0),
effective batch 4, HR crop 96, scale 3, 250 optimizer steps. No AMP, scheduler,
gradient clipping, gate/lambda reinitialization or automatic recipe adjustment.
Every evaluation covers all 100 held-out fixed crops in both RGB and Y using
evaluate_usrnet_quality.quality. A short fine-tuning run is not convergence proof.

Example capacity pilot (choose a NEW output directory):
  python test/train_usrnet_dataset.py --purpose pilot --steps 5 --eval-every 5 \
    --batch-size 4 --microbatch-size 4 --run-dir artifacts/dataset_training/pilot_current
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


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
    if args.variant == "before":
        from training_refinement_baseline import load_baseline
        return load_baseline(args.snapshot, verbose=args.verbose_build)
    from extension_loader import load_extension
    skip = os.environ.pop("CONVERSE2D_SKIP_BUILD", None)
    try:
        load_extension(verbose=args.verbose_build)
    finally:
        if skip is not None:
            os.environ["CONVERSE2D_SKIP_BUILD"] = skip
    manifest = ROOT / ".build/cuda/source_manifest.json"
    return torch.ops.converse2d, dict(kind="current production", build_manifest=json.loads(manifest.read_text(encoding="utf-8")))


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "test/usrnet_training_data.py",
             ROOT / "test/evaluate_usrnet_quality.py", ROOT / "test/extension_loader.py",
             ROOT / "test/training_refinement_baseline.py", ROOT / "utils/utils_image.py",
             ROOT / "models/converse_usrnet.py", ROOT / "models/util_converse.py",
             ROOT / "models/converse_core.py"]
    paths += [path for path in sorted((ROOT / "Converse2D/torch_converse2d").iterdir())
              if path.suffix in (".cpp", ".cu", ".h")]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


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
    if not grads:
        raise RuntimeError("No parameter gradients were produced")

    def finite_and_norm():
        loss = torch.stack(losses).sum()
        finite = torch.stack([torch.isfinite(loss), *[torch.isfinite(grad).all() for grad in grads]]).all()
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        norm = torch.linalg.vector_norm(flat, dtype=torch.float64)
        finite = finite & torch.isfinite(norm)
        # One host transfer for loss, norm and the aggregated finite flag.
        return torch.stack((loss.double(), norm, finite.double())).cpu().tolist()

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


def evaluate(model, optimizer, protocol, ops, args, optimizer_steps, expected_ids):
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

    payload = dict(state_dict=state, optimizer_state_dict=cpu_copy(optimizer.state_dict()),
                   optimizer_steps=optimizer_steps, config=report["config"],
                   config_sha256=report["config_sha256"], split_sha256=report["split_sha256"],
                   source_sha256=report["source_sha256"], metrics=json_safe(metrics))
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return dict(path=str(path), file_sha256=file_hash(path), state_tensor_sha256=tensor_hash(state),
                optimizer_steps=optimizer_steps, wall_s=time.perf_counter() - began)


def execute(args, protocol, report):
    import numpy as np
    import torch
    from evaluate_usrnet_quality import checkpoint_layout
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
    report["environment"] = dict(torch=str(torch.__version__), numpy=np.__version__,
                                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True)
    # Frozen imports and ALL forwards/evaluations share the same namespace scope.
    with production_namespace(ops):
        if args.variant == "before":
            from training_refinement_baseline import load_frozen_models
            model_class = load_frozen_models(args.snapshot).converse_usrnet.ConverseUSRNet
        else:
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
        if args.variant == "reuse":
            if not hasattr(model, "reuse_training_spectra"):
                raise RuntimeError("Current model does not expose the explicit reuse candidate")
            model.reuse_training_spectra = True
        elif hasattr(model, "reuse_training_spectra"):
            model.reuse_training_spectra = False
        model = model.float().cuda().train()
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise RuntimeError("All training parameters must be FP32")
        report["parameter_count"] = sum(value.numel() for value in model.parameters())
        report["parameter_tensor_count"] = len(list(model.parameters()))
        check_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0, amsgrad=False, foreach=False, fused=False)
        report["checkpoints"]["initial"] = save_checkpoint(args.run_dir / "initial.pth", model, optimizer, 0, report)
        report["initial_state_tensor_sha256"] = report["checkpoints"]["initial"]["state_tensor_sha256"]
        report["timing"]["checkpoint_wall_s"] += report["checkpoints"]["initial"]["wall_s"]
        expected_ids = {row["relative_path"] for row in protocol.validation}
        best_y, validation_hash = -math.inf, None
        loop_started = time.perf_counter()
        report["timing"]["setup_wall_s"] = loop_started - report.pop("_process_started")

        def evaluation(step):
            nonlocal best_y, validation_hash
            value = evaluate(model, optimizer, protocol, ops, args, step, expected_ids)
            if validation_hash is None:
                validation_hash = value["validation_payload_sha256"]
                report["validation_payload_sha256"] = validation_hash
            elif value["validation_payload_sha256"] != validation_hash:
                raise RuntimeError("Validation tensors changed during the run")
            report["timing"]["evaluation_wall_s"] += value["wall_s"]
            update_peaks(report["evaluation_peak_memory"], value["peak_memory"])
            update_peaks(report["overall_peak_memory"], value["peak_memory"])
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
            print(json.dumps(dict(event="evaluation", optimizer_steps=step, images=value["images"],
                                  rgb=value["rgb"], y=value["y"], wall_s=value["wall_s"])), flush=True)
            torch.cuda.reset_peak_memory_stats()
            write_json(args.run_dir / "run.json", report)

        evaluation(0)
        for step in range(args.steps):
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
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                evaluation(step + 1)
            report["timing"]["training_loop_end_to_end_wall_s"] = time.perf_counter() - loop_started
            write_json(args.run_dir / "run.json", report)
        report["final_parameter_check"] = check_parameters(model)
        report["checkpoints"]["final"] = save_checkpoint(args.run_dir / "final.pth", model, optimizer,
                                                          report["optimizer_steps"], report,
                                                          report["last_evaluation"])
        report["timing"]["checkpoint_wall_s"] += report["checkpoints"]["final"]["wall_s"]
        report["timing"]["training_loop_end_to_end_wall_s"] = time.perf_counter() - loop_started
        report["status"] = "complete"
        report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.run_dir / "run.json", report)


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset_training/split_900_100.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--variant", choices=("current", "before", "reuse"), default="current")
    parser.add_argument("--init", choices=("pretrained", "scratch"), default="pretrained")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_zoo/converse_usrnet.pth")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/training_refinements/source_before")
    parser.add_argument("--purpose", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=125)
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
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
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
    if args.run_dir.exists():
        parser.error(f"Run directory already exists; refusing overwrite: {args.run_dir}")
    if args.init == "pretrained" and not args.checkpoint.is_file():
        parser.error(f"Pretrained checkpoint does not exist: {args.checkpoint}")
    sys.path.insert(0, str(ROOT))
    from usrnet_training_data import DatasetProtocol
    protocol = DatasetProtocol(args.manifest, patch_size=args.patch_size, scale=args.scale,
                               seed=args.seed, noise_std=args.noise_std)
    if protocol.metadata["validation_images"] != 100:
        parser.error("This experiment requires exactly 100 held-out validation images")
    if args.run_dir.is_relative_to(protocol.root):
        parser.error("Run output must be outside the read-only image directory")
    config = json_safe(vars(args))
    recipe = {key: value for key, value in config.items() if key not in
              ("run_dir", "variant", "verbose_build", "snapshot", "purpose")}
    split = {name: sorted(row["relative_path"] for row in getattr(protocol, name))
             for name in ("train", "validation")}
    report = dict(status="initializing", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  config=config, config_sha256=hash_json(config), comparison_recipe_sha256=hash_json(recipe),
                  dataset=protocol.metadata, split_sha256=hash_json(split), source_sha256=source_hashes(),
                  input_checkpoint_sha256=file_hash(args.checkpoint) if args.init == "pretrained" else None,
                  optimizer_steps=0, samples_seen=0, all_loss_and_grad_finite=True, checkpoints={},
                  evaluation_steps=[], training_peak_memory={}, evaluation_peak_memory={}, overall_peak_memory={},
                  timing=dict(checkpoint_wall_s=0.0, evaluation_wall_s=0.0, total_training_step_wall_ms=0.0,
                              total_h2d_wall_ms=0.0, total_forward_backward_wall_ms=0.0,
                              data_prepare_and_hash_wall_ms=0.0, finite_check_wall_ms=0.0, optimizer_wall_ms=0.0),
                  _process_started=process_started,
                  protocol=dict(optimizer="Adam", betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
                                loss="unclipped/unrounded RGB " + args.loss, amp=False, scheduler=False,
                                gradient_clipping=False, gates_and_lambda="unchanged constructor/checkpoint values",
                                best_checkpoint_metric="mean held-out Y PSNR", metric_border=args.crop_border,
                                quality="evaluate_usrnet_quality.quality: clip/round prediction to uint8, existing RGB/Y PSNR/SSIM",
                                timing="End-to-end loop includes eval, I/O and logging. Step wall includes microbatch H2D, forward/backward, finite/norm checks and Adam; phase timings are synchronized and exclude data preparation. No profiler.",
                                memory="Total PyTorch allocated/reserved high-water marks; includes optimizer/data/checks/evaluation by labelled phase; excludes driver/library allocations. Reserved includes allocated.",
                                scope="Short pretrained fine-tuning on real photos with declared synthetic degradation; not convergence or an external benchmark",
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
