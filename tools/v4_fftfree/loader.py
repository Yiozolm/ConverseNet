"""Explicit independent checked build/load; never registers release operators."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from torch.utils import cpp_extension


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # tools/v4_fftfree -> repository root
SOURCE_NAMES = ("bindings.cpp", "kernel.cu", "inference.py", "loader.py")
_loaded = None
_loaded_identity = None
_loaded_manifest = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_identity():
    spec = importlib.util.spec_from_file_location("fftfree_build_config", ROOT / "Converse2D/build_config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    cxx, cuda = config.compile_flags()
    return {
        "sources": {name: sha256(HERE / name) for name in SOURCE_NAMES},
        "build_config_sha256": sha256(ROOT / "Converse2D/build_config.py"),
        "torch": str(torch.__version__), "torch_git": torch.version.git_version,
        "torch_cuda": torch.version.cuda, "toolchain": config.toolchain_identity(True),
        "python": sys.version, "platform": sys.platform,
        "cxx_flags": cxx, "cuda_flags": [*cuda, "--fmad=false"],
    }


def load_checked(artifacts, *, build=False, verbose=False):
    global _loaded, _loaded_identity, _loaded_manifest
    artifacts = Path(artifacts).resolve()
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG", "1033")
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise RuntimeError("set TORCH_CUDA_ARCH_LIST explicitly for a reproducible research build")
    identity = build_identity()
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = artifacts / "manifest.json"
    if _loaded is not None:
        if _loaded_identity != (str(artifacts), token):
            raise RuntimeError("research build inputs changed; restart Python")
        return _loaded, _loaded_manifest
    if build:
        if artifacts.exists() and any(artifacts.iterdir()):
            raise FileExistsError("explicit builds require a fresh directory; failed/old artifacts are preserved")
        sources, binary_dir = artifacts / "sources", artifacts / "build"
        sources.mkdir(parents=True)
        binary_dir.mkdir()
        for name in SOURCE_NAMES:
            shutil.copyfile(HERE / name, sources / name)
        (artifacts / "build_inputs.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        name = "converse_v4_fftfree_k2_" + token[:12]
        module = cpp_extension.load(name=name,
            sources=[str(sources / "bindings.cpp"), str(sources / "kernel.cu")],
            extra_cflags=identity["cxx_flags"], extra_cuda_cflags=identity["cuda_flags"],
            with_cuda=True, build_directory=str(binary_dir), verbose=verbose)
        binary = Path(module.__file__).resolve()
        manifest = {"identity": identity, "module_name": name,
                    "binary": str(binary), "binary_sha256": sha256(binary)}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise RuntimeError("stale research source/toolchain identity; use a fresh explicit build")
        binary = Path(manifest["binary"])
        if sha256(binary) != manifest["binary_sha256"]:
            raise RuntimeError("research binary hash mismatch")
        for name, digest in identity["sources"].items():
            if sha256(artifacts / "sources" / name) != digest:
                raise RuntimeError("frozen source hash mismatch: " + name)
        spec = importlib.util.spec_from_file_location(manifest["module_name"], binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    _loaded, _loaded_identity, _loaded_manifest = module, (str(artifacts), token), manifest
    return module, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _, manifest = load_checked(args.artifacts, build=args.build, verbose=args.verbose)
    print(json.dumps({"binary": manifest["binary"], "binary_sha256": manifest["binary_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
