"""Serial warm-step Nsight Systems capture for the refactored small-s1 study."""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("before", "combined"), required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warm-builds", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    argv = [str(ROOT / "test/study_training_small_s1.py"), "--phase", "profile",
            "--route", args.route, "--batch", str(args.batch), "--iters", "2",
            "--output", str(out / "metadata.json"), "--warm-builds", str(args.warm_builds.resolve())]
    bootstrap = out / "worker.py"
    bootstrap.write_text("import sys, runpy, traceback\nfrom pathlib import Path\n" +
                        "print('Small-s1 capture worker entered', flush=True)\n" +
                        f"sys.path[:0] = {[str(ROOT / 'test'), str(ROOT)]!r}\n" +
                        f"sys.argv = {argv!r}\n" +
                        "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\n" +
                        f"except BaseException:\n    Path({str(out / 'worker_error.txt')!r}).write_text(traceback.format_exc())\n    raise\n",
                        encoding="utf-8")
    command = [locate("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--kill=false",
               "--wait=primary", "--show-output=true", "--export=sqlite", "-o", str(out / "full"),
               sys.executable, "-u", str(bootstrap)]
    manifest = dict(command=command, scope="Diagnostic only; not formal timing", route=args.route,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    bootstrap_sha256=hashlib.sha256(bootstrap.read_bytes()).hexdigest())
    path = out / "launcher.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out / "capture.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    manifest.update(returncode=result.returncode, report_exists=(out / "full.nsys-rep").is_file(),
                    metadata_exists=(out / "metadata.json").is_file())
    metadata = json.loads((out / "metadata.json").read_text()) if manifest["metadata_exists"] else {}
    manifest["passed"] = (result.returncode == 0 and manifest["report_exists"] and metadata.get("status") == "complete")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
