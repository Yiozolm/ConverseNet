"""Explicit, source-verified loader for the isolated full-spectrum candidate."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback

import torch
import torch.utils.cpp_extension as cpp_extension

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("CONVERSE_FULL_TRAIN_ARTIFACTS", ROOT / "artifacts" / "training_full_spectrum_fusion")).expanduser().resolve()
CORE = ROOT / "Converse2D" / "torch_converse2d" / "training" / "full_spectrum"
SOURCES = (CORE / "full_fusion.cpp", CORE / "full_fusion.cu")
# Quoted CUDA dependencies must be frozen and identity-checked alongside the
# translation units, but headers must not be passed to the compiler as sources.
DEPENDENCIES = SOURCES + (CORE / "scale1.cuh", CORE / "scale2.cuh")
NAMESPACE = "converse_full_training"
CXX_FLAGS = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
CUDA_FLAGS = ["-O3", "-lineinfo"]
_LOADED_FINGERPRINT = None

if os.name == "nt":
    # Non-English MSVC diagnostics must not turn a compiler error into UnicodeDecodeError.
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in DEPENDENCIES}


def architecture():
    explicit = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if explicit:
        return explicit
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set a supported TORCH_CUDA_ARCH_LIST for an explicit offline build")
    capabilities = sorted({torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())})
    return ";".join(f"{major}.{minor}" for major, minor in capabilities)


@contextmanager
def selected_architecture(value):
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def runtime_identity():
    arch = architecture()
    with selected_architecture(arch):
        arch_flags = sorted(cpp_extension._get_cuda_arch_flags())
    return {
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "architecture": arch, "cuda_arch_flags": arch_flags,
        "python_cache_tag": sys.implementation.cache_tag,
        "platform": sys.platform, "machine": platform.machine(),
        "cxx11_abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)),
        "cxx_flags": CXX_FLAGS, "cuda_flags": CUDA_FLAGS,
        "current_source_sha256": source_identity(),
    }


def fingerprint(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_path(relative, output):
    path = (output / relative).resolve()
    if not path.is_relative_to(output):
        raise RuntimeError(f"Manifest path escapes the artifact directory: {relative}")
    return path


def ensure_namespace_available(expected_fingerprint=None):
    if hasattr(getattr(torch.ops, NAMESPACE), "spectral"):
        if expected_fingerprint is not None and _LOADED_FINGERPRINT == expected_fingerprint:
            return False
        raise RuntimeError("The full-spectrum namespace is already registered by another load; use a fresh Python process")
    return True


def adopt_existing(output):
    """Read-only adoption of the research loader's previously measured manifest."""
    global _LOADED_FINGERPRINT
    manifest = json.loads((output / "build.json").read_text(encoding="utf-8"))
    if "schema" in manifest or manifest.get("namespace") != NAMESPACE:
        raise RuntimeError("--adopt expects the original research build.json and final namespace")
    identity = runtime_identity()
    if manifest.get("torch") != identity["torch"] or manifest.get("cuda") != identity["torch_cuda"]:
        raise RuntimeError("Research build Torch/CUDA identity mismatch")
    expected_sources = {path.name: sha(path) for path in DEPENDENCIES}
    observed_sources = {}
    for filename, expected in manifest["sources"].items():
        source = Path(filename)
        source = source if source.is_absolute() else output / source
        if source.name in observed_sources or sha(source) != expected:
            raise RuntimeError(f"Duplicate or changed frozen source: {source}")
        observed_sources[source.name] = expected
    if observed_sources != expected_sources:
        raise RuntimeError("Research frozen sources do not match the current core bytes")
    library = Path(manifest["library"])
    library = library if library.is_absolute() else output / library
    if sha(library) != manifest["binary_sha256"]:
        raise RuntimeError("Research binary hash mismatch")
    # The old manifest did not record architecture; require its retained build evidence.
    ninja = (library.parent / "build.ninja").read_text(encoding="utf-8")
    compiled_arch = sorted(set(re.findall(r"-gencode=arch=[^\s]+", ninja)))
    if compiled_arch != identity["cuda_arch_flags"]:
        raise RuntimeError(f"Research build architecture mismatch: {compiled_arch} != {identity['cuda_arch_flags']}")
    digest = fingerprint({"runtime": identity, "binary_sha256": manifest["binary_sha256"]})
    if ensure_namespace_available(digest):
        torch.ops.load_library(str(library))
        _LOADED_FINGERPRINT = digest
    return getattr(torch.ops, NAMESPACE)


def load_fusion(build=False, *, artifacts=None, adopt=False):
    """Build only when explicitly requested; otherwise verify and load the manifest.

    Building requires a fresh/empty artifact directory. Warm loading checks the
    current core, frozen copies, binary, Torch/CUDA, architecture and ABI identity.
    """
    global _LOADED_FINGERPRINT
    output = Path(artifacts).expanduser().resolve() if artifacts is not None else OUT
    if adopt:
        if build:
            raise ValueError("build and adopt are mutually exclusive")
        return adopt_existing(output)
    if build:
        # Refuse even partial previous runs: retain failed builds and measurements.
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Build requires an empty artifact directory: {output}. Use a new CONVERSE_FULL_TRAIN_ARTIFACTS or warm load.")
        ensure_namespace_available()
        identity = runtime_identity()
        digest = fingerprint(identity)
        name = "converse_full_training_" + digest[:16]
        output.mkdir(parents=True, exist_ok=True)
        request = {"schema": 1, "namespace": NAMESPACE, "fingerprint": digest, "identity": identity}
        with (output / "build_request.json").open("x", encoding="utf-8") as stream:
            json.dump(request, stream, indent=2)
        try:
            folder = output / "build" / digest[:16]
            folder.mkdir(parents=True, exist_ok=False)
            frozen = []
            frozen_hashes = {}
            for source in DEPENDENCIES:
                destination = folder / source.name
                with destination.open("xb") as stream:
                    stream.write(source.read_bytes())
                original_hash = identity["current_source_sha256"][source.relative_to(ROOT).as_posix()]
                if sha(destination) != original_hash:
                    raise RuntimeError(f"Source changed while freezing: {source}")
                if source in SOURCES:
                    frozen.append(destination)
                frozen_hashes[destination.relative_to(output).as_posix()] = original_hash
            with selected_architecture(identity["architecture"]):
                cpp_extension.load(
                    name=name, sources=[str(path) for path in frozen],
                    extra_cflags=CXX_FLAGS, extra_cuda_cflags=CUDA_FLAGS,
                    with_cuda=True, is_python_module=False,
                    build_directory=str(folder), verbose=True,
                )
            if source_identity() != identity["current_source_sha256"]:
                raise RuntimeError("Current core sources changed during compilation; this run remains unaccepted")
            library = folder / (name + cpp_extension.LIB_EXT)
            manifest = {
                **request, "library": library.relative_to(output).as_posix(),
                "binary_sha256": sha(library), "frozen_sources": frozen_hashes,
            }
            with (output / "build.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
            _LOADED_FINGERPRINT = digest
            return getattr(torch.ops, NAMESPACE)
        except BaseException:
            with (output / "build_failure.txt").open("x", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
            raise

    manifest_path = output / "build.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No accepted build manifest: {manifest_path}. Run loader.py --build explicitly in a fresh directory.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("namespace") != NAMESPACE:
        raise RuntimeError("Unsupported build manifest schema or namespace")
    identity = runtime_identity()
    if manifest["identity"] != identity or manifest["fingerprint"] != fingerprint(identity):
        differences = [key for key in identity if manifest["identity"].get(key) != identity[key]]
        raise RuntimeError(f"Warm-load identity mismatch ({', '.join(differences)}); retain this run and build in a fresh directory")
    for relative, expected in manifest["frozen_sources"].items():
        if sha(artifact_path(relative, output)) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    if sorted(manifest["frozen_sources"].values()) != sorted(identity["current_source_sha256"].values()):
        raise RuntimeError("Frozen/current source hash sets differ")
    library = artifact_path(manifest["library"], output)
    if sha(library) != manifest["binary_sha256"]:
        raise RuntimeError(f"Binary hash mismatch: {library}")
    if ensure_namespace_available(manifest["fingerprint"]):
        torch.ops.load_library(str(library))
        _LOADED_FINGERPRINT = manifest["fingerprint"]
    return getattr(torch.ops, NAMESPACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", action="store_true", help="Compile in a fresh artifact directory")
    group.add_argument("--warm", action="store_true", help="Verify/load an existing build (default)")
    group.add_argument("--adopt", action="store_true", help="Read-only load of a verified research build.json")
    parser.add_argument("--artifacts", type=Path, help="Explicit artifact directory; overrides the environment/default")
    args = parser.parse_args()
    load_fusion(build=args.build, artifacts=args.artifacts, adopt=args.adopt)
    selected = args.artifacts.expanduser().resolve() if args.artifacts else OUT
    print(json.dumps({"loaded": NAMESPACE, "artifacts": str(selected), "built": args.build, "adopted": args.adopt}, indent=2))


if __name__ == "__main__":
    main()
