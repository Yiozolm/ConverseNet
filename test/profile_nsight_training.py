"""Capture current FP32 training with Nsight; diagnostic timings are not benchmarks.

Run via experiments/training_speed/run.ps1 to inherit the validated CUDA loader
environment. The launcher starts a fresh Python process under the selected tool.
Compilation, verification, fixture creation and warmup are outside the capture.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "op-b1-256-s1": ("operator", (1, 32, 256, 256), 1),
    "op-b4-256-s1": ("operator", (4, 32, 256, 256), 1),
    "op-b1-256-s3": ("operator", (1, 32, 256, 256), 3),
    "op-b4-256-s3": ("operator", (4, 32, 256, 256), 3),
    "op-b32-64x80-s3": ("operator", (32, 32, 64, 80), 3),
    "usrnet-b16-s3": ("usrnet", (16, 3, 16, 20), 3),
}


def locate(tool):
    found = shutil.which(tool)
    # Launch the executable directly: a .bat wrapper reparses regex pipes.
    if found and Path(found).suffix.lower() == ".bat":
        candidates = sorted(Path(found).parent.glob("target/*/ncu.exe"))
        found = str(candidates[0]) if candidates else None
    if found:
        return found
    base = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation"
    pattern = ("Nsight Systems */target-windows-x64/nsys.exe" if tool == "nsys"
               else "Nsight Compute */target/*/ncu.exe")
    matches = sorted(base.glob(pattern), reverse=True)
    if not matches:
        raise FileNotFoundError(f"Cannot locate {tool}; add it to PATH")
    return str(matches[0])


def worker(args):
    import torch
    import torch.nn.functional as F
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    import benchmark_fp32_training as common

    if os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        raise RuntimeError("CUDA profiling cannot use CONVERSE2D_CPU_ONLY")
    if os.environ.get("CONVERSE2D_BACKEND", "").lower() not in ("", "auto", "cuda"):
        raise RuntimeError("CONVERSE2D_BACKEND must select CUDA")
    load_extension()
    dispatch = common.verify_fused_dispatch(torch.ops.converse2d)
    torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    kind, shape, scale = CASES[args.case]

    @contextlib.contextmanager
    def phase(name):
        if args.tool != "none":
            with torch.cuda.nvtx.range(name):
                yield
        else:
            yield

    if kind == "operator":
        batch, channels, height, width = shape
        x = torch.randn(shape, device="cuda", requires_grad=True)
        weight = torch.randn(1, channels, 3, 3, device="cuda").flatten(2).softmax(-1)
        weight = weight.reshape(1, channels, 3, 3).detach().requires_grad_()
        bias = torch.zeros(1, channels, 1, 1, device="cuda", requires_grad=True)
        upstream = torch.randn(batch, channels, height * scale, width * scale,
                               device="cuda") * 0.001

        def step():
            with phase("prior"):
                prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
            with phase("solver_forward"):
                output = torch.ops.converse2d.forward(x, prior, weight, bias, scale, 1e-3, "v7")
            with phase("solver_backward"):
                grads = torch.autograd.grad(output, (x, weight, bias), upstream)
            return (output, *grads)

        reset = lambda: None
        scope = "Operator forward + x/kernel/bias VJP; shared x prior at s1, nearest prior at s3; no loss/optimizer"
    else:
        sys.path.insert(0, str(ROOT))
        template, batches, targets = common.workload(kind, shape, scale, 1)
        runner = common.TrainingRunner(template, batches, targets)

        def step():
            with phase("zero_grad"):
                runner.optimizer.zero_grad(set_to_none=True)
            with phase("model_forward"):
                output = runner.model(*runner.batches[0])
            with phase("loss"):
                loss = (output - runner.targets[0]).square().mean()
            with phase("model_backward"):
                loss.backward()
            with phase("optimizer"):
                runner.optimizer.step()
            return (loss.detach(),)

        reset = runner.reset
        scope = "Reduced USRNet 2 iterations/1 block; FP32 eager full SGD step, lr1e-4 momentum0.9 alpha0.1; reuse disabled"
    for _ in range(args.warmup):
        step()
    reset()
    torch.cuda.synchronize()
    if args.tool == "none":
        timing = common.sample(step, args.steps)
        result = step()
    else:
        timing = None
        nvtx_context = (torch.autograd.profiler.emit_nvtx(record_shapes=False)
                        if args.tool == "nsys" else contextlib.nullcontext())
        with nvtx_context:
            torch.cuda.cudart().cudaProfilerStart()
            try:
                for index in range(args.steps):
                    # Match the unprofiled runner's lifetime: previous outputs
                    # and VJPs must not survive into the next operator call.
                    result = None
                    with phase(f"NsightStep/{index}"):
                        result = step()
                torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
    if not all(torch.isfinite(value).all().item() for value in result):
        raise AssertionError("Nonfinite captured result")
    report = dict(case=args.case, kind=kind, shape=shape, scale=scale, scope=scope,
                  steps=args.steps, warmup=args.warmup, tool=args.tool,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  source_sha256=current_manifest(), dispatch=dispatch, timing=timing,
                  fixture_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (ROOT / "test/benchmark_fp32_training.py",
                                               ROOT / "models/converse_core.py")},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  capture="Warm current production FP32; no reuse, AMP or CUDA Graph. Profiler time is diagnostic only.")
    (args.output_dir / f"{args.case}.{args.tool}.metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tool", choices=("nsys", "ncu", "none"), default="nsys")
    parser.add_argument("--case", choices=tuple(CASES), default="op-b1-256-s3")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/nsight_training")
    parser.add_argument("--kernel", default="regex:.*(forward_scale1|backward_scale1|filter_scale1|solve_alias|solve_output|adjoint_q|adjoint_inputs|adjoint_filter).*")
    parser.add_argument("--launch-count", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.launch_count) < 1:
        parser.error("steps, warmup and launch-count must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker or args.tool == "none":
        worker(args)
        return
    executable = locate(args.tool)
    prefix = args.output_dir / f"{args.case}.{args.tool}"
    if args.tool == "nsys":
        options = ["profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=false",
                   "--export=sqlite", "-o", str(prefix)]
    else:
        options = ["--profile-from-start", "off", "--target-processes", "application-only",
                   "--kernel-name-base", "demangled", "--kernel-name", args.kernel,
                   "--launch-count", str(args.launch_count), "--clock-control", "none",
                   "--cache-control", "none", "--section", "SpeedOfLight",
                   "--section", "LaunchStats", "--section", "Occupancy",
                   "--section", "MemoryWorkloadAnalysis", "--section", "SchedulerStats",
                   "--section", "WarpStateStats", "-o", str(prefix)]
    command = [executable, *options, sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", "--tool", args.tool, "--case", args.case, "--steps", str(args.steps),
               "--warmup", str(args.warmup), "--output-dir", str(args.output_dir)]
    (args.output_dir / f"{args.case}.{args.tool}.command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8")
    print("Launching", json.dumps(command), flush=True)
    with (args.output_dir / f"{args.case}.{args.tool}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    lines = (args.output_dir / f"{args.case}.{args.tool}.log").read_text(encoding="utf-8").splitlines()
    print("\n".join(lines[-25:]), flush=True)
    print("Full log:", args.output_dir / f"{args.case}.{args.tool}.log", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
