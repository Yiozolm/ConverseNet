"""Freeze checked-out baseline bytes for separate-process arithmetic benchmarks.

This standard-library helper never imports Torch, builds, or uses the GPU.
Run before editing production arithmetic. The destination must be new. Build
that snapshot with its own test/extension_loader.py; no binary is copied.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ("Converse2D", "models", "test", "tools", "model_zoo", "AGENTS.md")


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root, output):
    root, output = root.resolve(), output.resolve()
    # The snapshot must represent committed benchmark inputs. Unrelated local
    # artifacts do not enter the snapshot, and are never deleted or rewritten.
    status = git(root, "status", "--porcelain", "--untracked-files=no", "--", *INPUTS)
    if status.strip():
        raise RuntimeError("Commit the baseline inputs before freezing:\n" + status.decode())
    if output.exists():
        raise FileExistsError("Choose a fresh baseline directory: " + str(output))
    names = [name.decode("utf-8") for name in
             git(root, "ls-files", "-z", "--", *INPUTS).split(b"\0") if name]
    for required in ("Converse2D/build_config.py", "test/extension_loader.py",
                     "tools/benchmark_fp32_p0.py", "tools/benchmark_batch_training.py",
                     "model_zoo/converse_usrnet.pth"):
        if required not in names:
            raise RuntimeError("Missing tracked baseline input: " + required)
    head = git(root, "rev-parse", "HEAD").decode().strip()
    hashes = {name: sha(root / name) for name in names}
    output.mkdir(parents=True)
    metadata = {
        "kind": "v4_arithmetic_source_snapshot", "status": "copying",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "git_head": head,
        "source_sha256": hashes, "helper_sha256": sha(Path(__file__)),
        "note": "Actual checkout bytes; fresh separate checked build required. "
                "No binary or Git metadata is copied. Benchmark with --root; "
                "the snapshot's origin is this record, not git discovery from its parent.",
    }
    manifest = root / ".build/cuda/source_manifest.json"
    if manifest.exists():
        metadata["source_checkout_checked_build"] = json.loads(manifest.read_text(encoding="utf-8"))
    record = output / "baseline_snapshot.json"
    record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name in names:
        source, destination = root / name, output / name
        if not destination.resolve().is_relative_to(output):
            raise RuntimeError("Tracked path escapes snapshot: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha(destination) != hashes[name]:
            raise RuntimeError("Source changed during snapshot: " + name)
    if git(root, "rev-parse", "HEAD").decode().strip() != head or any(
            sha(root / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("Baseline checkout changed during snapshot; retain this incomplete directory")
    metadata["status"] = "complete"
    record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.root, args.output)


if __name__ == "__main__":
    main()
