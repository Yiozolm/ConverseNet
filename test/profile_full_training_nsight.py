"""Launch full-training Nsight captures using a source-verified warm build.

Build once with test/extension_loader.py before use. A stale source/header/
PyTorch/binary manifest fails rather than loading unchecked code. The generated
bootstrap keeps target arguments out of the Windows profiler command parser.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from profile_nsight_training import locate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tool", choices=("nsys", "ncu"), default="nsys")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--kernel", default="regex:.*wgrad2d_grouped_direct_kernel.*")
    parser.add_argument("--launch-skip", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.launch_skip < 0:
        parser.error("Positive steps and nonnegative launch skip required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    argv = [str(ROOT / "test/profile_full_training.py"), "--worker", "--tool", "nsys",
            "--output", str(out), "--steps", str(args.steps)]
    bootstrap = out / "capture_worker.py"
    bootstrap.write_text(
        "import os, runpy, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'test')!r})\n"
        "from extension_loader import load_extension\n"
        "previous = os.environ.get('CONVERSE2D_SKIP_BUILD')\n"
        "os.environ['CONVERSE2D_SKIP_BUILD'] = '1'\n"
        "try:\n    load_extension()\n"
        "finally:\n"
        "    if previous is None:\n        os.environ.pop('CONVERSE2D_SKIP_BUILD', None)\n"
        "    else:\n        os.environ['CONVERSE2D_SKIP_BUILD'] = previous\n"
        f"sys.argv = {argv!r}\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n", encoding="utf-8")
    if args.tool == "nsys":
        options = ["profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=false",
                   "--wait=primary", "--show-output=true", "--export=sqlite", "-o", str(out / "full")]
        expected = out / "full.nsys-rep"
    else:
        options = ["--profile-from-start", "off", "--target-processes", "application-only",
                   "--kernel-name-base", "demangled", "--kernel-name", args.kernel,
                   "--launch-skip", str(args.launch_skip), "--launch-count", "1",
                   "--clock-control", "none", "--cache-control", "none"]
        for section in ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
                        "SchedulerStats", "WarpStateStats"):
            options += ["--section", section]
        options += ["-o", str(out / "hotspot")]
        expected = out / "hotspot.ncu-rep"
    command = [locate(args.tool), *options, sys.executable, "-u", str(bootstrap)]
    manifest = dict(command=command, actual_tool=args.tool, expected_report=str(expected),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    bootstrap_sha256=hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
                    build="Existing source/header/torch/binary SHA-validated library; no compiler probe",
                    scope="Diagnostic full-model capture, never a speed benchmark")
    (out / "launcher.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out / "capture.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    manifest["returncode"] = result.returncode
    manifest["report_exists"] = expected.is_file()
    manifest["training_metadata_exists"] = (out / "metadata.json").is_file()
    manifest["status"] = ("complete" if result.returncode == 0 and manifest["report_exists"]
                          and manifest["training_metadata_exists"] else "failed")
    (out / "launcher.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    if manifest["status"] != "complete":
        print("\n".join((out / "capture.log").read_text(encoding="utf-8").splitlines()[-20:]))
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
