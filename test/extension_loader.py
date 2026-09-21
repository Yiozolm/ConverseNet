"""Build/load the corrected extension in the checkout, without global installs."""
import hashlib
import json
import os
import pathlib

import torch
from torch.utils.cpp_extension import CUDA_HOME, load
from torch.utils import cpp_extension

ROOT = pathlib.Path(__file__).resolve().parents[1]
_loaded_inputs = None
_loaded_extension = None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_extension(cpu_only=False, verbose=False):
    global _loaded_inputs, _loaded_extension
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG", "1033")
        # Headless Windows shells may not expose a usable OEM code page.
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    cuda = (not cpu_only and os.environ.get("CONVERSE2D_CPU_ONLY") != "1" and
            torch.version.cuda is not None and CUDA_HOME is not None)
    build = ROOT / ".build" / ("cuda" if cuda else "cpu")
    build.mkdir(parents=True, exist_ok=True)
    source = ROOT / "Converse2D" / "torch_converse2d"
    names = ["converse2d.cpp"]
    if cuda:
        names.extend(("converse2d_kernels.cu", "converse2d_training.cu"))
    inputs = {"sources": {name: _sha256(source / name) for name in
                          [*names, "converse2d_training.h"]},
              "torch": str(torch.__version__), "cuda": torch.version.cuda if cuda else None}
    # TORCH_LIBRARY cannot be registered twice in one process. A new source
    # revision must be tested in a fresh process instead of silently staying old.
    if _loaded_inputs is not None:
        if _loaded_inputs != inputs:
            raise RuntimeError("Converse2D build inputs changed; restart Python before loading again")
        return _loaded_extension
    manifest_path = build / "source_manifest.json"
    if os.environ.get("CONVERSE2D_SKIP_BUILD") == "1":
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            library = build / manifest["library"]
            valid = manifest["inputs"] == inputs and _sha256(library) == manifest["binary_sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise RuntimeError("Missing or stale Converse2D build; run test/extension_loader.py "
                               "without CONVERSE2D_SKIP_BUILD before reusing its binary")
        torch.ops.load_library(str(library))
        _loaded_inputs = inputs
        return
    sources = [str(source / name) for name in names]
    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    # PyTorch's JIT versioner hashes source files and flags, but not headers.
    flags.append("-DCONVERSE2D_TRAINING_HEADER_REV=0x" +
                 inputs["sources"]["converse2d_training.h"][:12])
    if cuda:
        flags.append("-DCONVERSE2D_WITH_CUDA=1")
    extension = load(name="converse2d_checked_ext", sources=sources,
                     extra_cflags=flags, extra_cuda_cflags=["-O3", "-lineinfo"],
                     with_cuda=cuda, build_directory=str(build), verbose=verbose)
    library = pathlib.Path(extension.__file__)
    manifest_path.write_text(json.dumps({"inputs": inputs, "library": library.name,
                                        "binary_sha256": _sha256(library)}, indent=2), encoding="utf-8")
    _loaded_inputs = inputs
    _loaded_extension = extension
    return extension


if __name__ == "__main__":
    load_extension(verbose=True)
    print("Loaded", torch.ops.converse2d.forward)
