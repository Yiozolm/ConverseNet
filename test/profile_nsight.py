"""Profile a prebuilt extension with Nsight Systems or Nsight Compute."""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from extension_loader import ROOT, load_extension


def find_tool(kind, explicit):
    if explicit:
        path = pathlib.Path(explicit).resolve()
        if os.name == "nt" and path.suffix.lower() in (".bat", ".cmd"):
            raise ValueError("Pass the profiler .exe, not a batch wrapper")
        return str(path)
    name = "nsys" if kind == "systems" else "ncu"
    found = shutil.which(name)
    if found and not found.lower().endswith((".bat", ".cmd")):
        return found
    if os.name == "nt":
        vendor = pathlib.Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation"
        pattern = ("Nsight Systems */target-windows-x64/nsys.exe" if kind == "systems" else
                   "Nsight Compute */target/windows-desktop-win7-x64/ncu.exe")
        matches = sorted(vendor.glob(pattern))
        if matches:
            return str(matches[-1])
    raise RuntimeError(f"Cannot find {name}; supply --tool /path/to/{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("systems", "compute"), default="systems")
    parser.add_argument("--tool", help="Explicit profiler executable")
    parser.add_argument("--variant", choices=("v2", "v6", "v7"), default="v7")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--C", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--set", dest="compute_set", choices=("basic", "detailed", "full"), default="basic",
                        help="Nsight Compute section set")
    parser.add_argument("--clock-control", choices=("none", "base", "boost"), default="none")
    parser.add_argument("--cache-control", choices=("all", "none"), default="all")
    parser.add_argument("--replay-mode", choices=("kernel", "application"), default="kernel")
    parser.add_argument("--output", default="artifacts/profiles")
    args = parser.parse_args()
    load_extension()
    tool = find_tool(args.kind, args.tool)
    folder = (ROOT / args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{args.kind}_{args.variant}_s{args.scale}"
    target = [sys.executable, str(ROOT / "test" / "benchmark.py"), "--profile",
              "--variant",args.variant,"--scale",str(args.scale),"--C",str(args.C),
              "--H",str(args.H),"--W",str(args.W),"--iters",str(args.iters)]
    if args.kind == "systems":
        command = [tool,"profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi","--capture-range-end=stop",
                   "--force-overwrite=true","-o",str(stem),*target]
    else:
        command = [tool,"--target-processes","all","--set",args.compute_set,"--profile-from-start","off",
                   "--clock-control",args.clock_control,"--cache-control",args.cache_control,
                   "--replay-mode",args.replay_mode,"--import-source","yes",
                   "--kernel-name","regex:.*(alias_correction|apply_correction|correction_scale_one).*",
                   "--launch-count","2","--force-overwrite","-o",str(stem),*target]
    env = os.environ.copy()
    # Nsight can intercept compiler subprocess output. Build before starting it.
    env["CONVERSE2D_SKIP_BUILD"] = "1"
    log_path = stem.with_suffix(".log")
    with log_path.open("w",encoding="utf-8") as log:
        result = subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print(log_path.read_text(encoding="utf-8",errors="replace"))
    if result.returncode:
        sys.exit(result.returncode)
    if args.kind == "systems":
        report = stem.with_suffix(".nsys-rep")
        with stem.with_suffix(".summary.txt").open("w",encoding="utf-8") as summary:
            result = subprocess.run([tool,"stats","--report","cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
                                     "--format","csv",str(report)],env=env,cwd=ROOT,
                                     stdout=summary,stderr=subprocess.STDOUT)
        if result.returncode:
            sys.exit(result.returncode)
        print(f"Kernel/API summaries: {stem.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
