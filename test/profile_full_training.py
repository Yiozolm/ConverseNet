"""Diagnostic full-model real-photo training traces; never use as speed results.

The historical train_step stays unchanged. Module hooks add CPU scopes only;
GPU ownership must be established using launch correlation, not host durations.
"""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def capture(args):
    import numpy as np
    import torch
    import train_usrnet_dataset as worker
    from benchmark_python_training import adam, restore_initial, graph_has_spectral
    from evaluate_usrnet_quality import backend_scope
    from models.converse_usrnet import ConverseUSRNet
    from models.util_converse import Converse2D
    from usrnet_training_data import DatasetProtocol

    args.variant, args.verbose_build = "current", False
    args.batch_size = args.microbatch_size = 4
    args.patch_size, args.scale, args.loss = 96, 3, "mse"
    ops, build = worker.load_backend(args)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol = DatasetProtocol(args.manifest, patch_size=96, scale=3, seed=17, noise_std=.01)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in initial:
        initial = initial["state_dict"]
    model = ConverseUSRNet(num_iterations=5, num_blocks=7, in_channels=64,
                           backend="cuda", reuse_training_spectra=False)
    model.load_state_dict(initial, strict=True)
    model = model.float().cuda().train()
    optimizer = adam(model)
    rows, call_map, handles = [], [], []
    context = dict(step=-1, iteration=-1, solver=0)

    @contextlib.contextmanager
    def scope(name):
        if args.tool == "torch":
            with torch.profiler.record_function(name):
                yield
        else:
            with torch.cuda.nvtx.range(name):
                yield

    def hooks(name, module):
        stack = []

        def before(_module, inputs):
            if name == "d":
                context["iteration"] += 1
            label = f"Module/I{context['iteration']}/{name}"
            if name == "d" or isinstance(module, Converse2D):
                label = f"Solver/L{context['solver']}/I{context['iteration']}/{name}"
                call_map.append(dict(step=context["step"], index=context["solver"],
                                     iteration=context["iteration"], module=name,
                                     input_shape=list(inputs[0].shape),
                                     scale=int(inputs[2]) if name == "d" else int(module.scale),
                                     label=label))
                context["solver"] += 1
            entered = scope(label)
            entered.__enter__()
            stack.append(entered)

        def after(_module, inputs, output):
            stack.pop().__exit__(None, None, None)

        handles.extend([module.register_forward_pre_hook(before),
                        module.register_forward_hook(after, always_call=True)])

    with backend_scope(model, args.backend):
        warm = protocol.train_batch(0, 4)
        for _ in range(args.warmup):
            worker.train_step(model, optimizer, warm, args)
        restore_initial(model, optimizer, initial)
        probe = tuple(value.cuda() for value in warm)
        output = model(probe[0], probe[1], 3)
        route = graph_has_spectral(output)
        if route != (args.backend == "current"):
            raise RuntimeError("Unexpected backend graph")
        del output, probe, warm
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        # Detailed module hooks are installed only after warmup and route probe.
        for name, module in model.named_modules():
            if name:
                hooks(name, module)
        original_timed = worker.cuda_timed
        phase_index = 0

        def marked_timed(fn):
            nonlocal phase_index
            name = ("h2d", "model_forward_backward", "finite_and_norm", "optimizer")[phase_index % 4]
            phase_index += 1
            with scope("Phase/" + name):
                return original_timed(fn)

        worker.cuda_timed = marked_timed

        def run_steps():
            for step in range(args.steps):
                context.update(step=step, iteration=-1, solver=0)
                with scope(f"NsightStep/{step}"):
                    with scope("Phase/data_prepare_and_hash"):
                        batch = protocol.train_batch(step, 4)
                        worker.check_cpu_batch(batch, args, 4)
                        batch_hash = worker.tensor_hash(dict(lr=batch[0], kernel=batch[1], hr=batch[2]))
                    row = worker.train_step(model, optimizer, batch, args)
                    if not row["loss_and_grad_finite"] or context["solver"] != 40:
                        raise RuntimeError("Invalid full-model capture")
                    rows.append(dict(step=step, batch_sha256=batch_hash, diagnostic=row))
                    del batch
            torch.cuda.synchronize()

        try:
            if args.tool == "torch":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, with_stack=False,
                                            profile_memory=False) as prof:
                    run_steps()
                prof.export_chrome_trace(str(args.output / "full.trace.json"))
                averages = prof.key_averages()
                (args.output / "operators.txt").write_text(
                    averages.table(sort_by="self_cuda_time_total", row_limit=80), encoding="utf-8")
                from profile_training_refinements import profile_summary
                worker.write_json(args.output / "torch_summary.json", profile_summary(prof.events(), args.steps))
            else:
                with torch.autograd.profiler.emit_nvtx(record_shapes=False):
                    torch.cuda.cudart().cudaProfilerStart()
                    try:
                        run_steps()
                    finally:
                        torch.cuda.cudart().cudaProfilerStop()
        finally:
            worker.cuda_timed = original_timed
            for handle in handles:
                handle.remove()
    sources = worker.source_hashes()
    sources[Path(__file__).relative_to(ROOT).as_posix()] = worker.file_hash(__file__)
    worker.write_json(args.output / "metadata.json", dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config=vars(args), build=build, source_sha256=sources,
        input_checkpoint_sha256=worker.file_hash(args.checkpoint),
        split_file_sha256=worker.file_hash(args.manifest),
        initial_tensor_sha256=worker.tensor_hash(initial),
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
        route_has_spectral_solve=route, call_map=call_map, rows=rows,
        warning="Diagnostic capture with original synchronized audit train_step and module hooks. "
                "Host scopes are not GPU boundaries. No profiler durations are benchmark speed results.",
        recipe="Full5/7/64; FP32/TF32 off/reuse off; pretrained; HR96/s3/B4; MSE/Adam1e-5; "
               "cuDNN deterministic on, benchmark off; 5 warmup then restored pretrained and zero Adam; "
               "actual data preparation/hash included, validation and checkpoint I/O excluded."))
    print("Saved", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tool", choices=("nsys", "torch"), required=True)
    parser.add_argument("--backend", choices=("current", "pytorch"), default="current")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset_training/split_900_100.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_zoo/converse_usrnet.pth")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.steps, args.warmup) < 1:
        parser.error("Positive steps/warmup required")
    args.output = args.output.resolve()
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.tool == "torch" or args.worker:
        capture(args)
        return
    from profile_nsight_training import locate
    command = [locate("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=false",
               "--export=sqlite", "-o", str(args.output / "full"), sys.executable, "-u", __file__,
               "--worker", "--tool", args.tool, "--backend", args.backend,
               "--output", str(args.output), "--manifest", str(args.manifest),
               "--checkpoint", str(args.checkpoint), "--steps", str(args.steps), "--warmup", str(args.warmup)]
    (args.output / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (args.output / "capture.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    print("\n".join((args.output / "capture.log").read_text(encoding="utf-8").splitlines()[-12:]))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
