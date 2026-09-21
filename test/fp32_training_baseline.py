"""Build the exact pre-integration dev operator in an isolated namespace."""
import hashlib
import os
from pathlib import Path
import subprocess

import torch
from torch.utils import cpp_extension


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REF = "b850e3885d566ad70cd90cd2213497215370980e"
SOURCE_DIR = "Converse2D/torch_converse2d"
BASELINE_SOURCES = ("converse2d.cpp", "converse2d_kernels.cu")
_loaded = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_manifest():
    """Record source identity, including headers and the production build recipe."""
    files = sorted((ROOT / SOURCE_DIR).iterdir())
    files = [p for p in files if p.suffix in (".cpp", ".cu", ".h")]
    files += [ROOT / "Converse2D/setup.py", ROOT / "test/extension_loader.py",
              ROOT / "models/util_converse.py", ROOT / "models/converse_usrnet.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def load_baseline(verbose=False):
    """Compile git blobs, never load a binary copied from an old .build directory."""
    global _loaded
    if _loaded is not None:
        return _loaded
    originals = {
        name: subprocess.check_output(
            ["git", "show", f"{BASELINE_REF}:{SOURCE_DIR}/{name}"], cwd=ROOT
        ) for name in BASELINE_SOURCES
    }
    hashes = {name: hashlib.sha256(value).hexdigest()
              for name, value in originals.items()}
    fingerprint = hashlib.sha256(
        "".join(f"{name}:{hashes[name]}\n" for name in BASELINE_SOURCES).encode()
    ).hexdigest()[:16]
    namespace = "dev_fp32_converse2d"
    build = ROOT / ".build/fp32_training_baseline" / fingerprint
    build.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, original in originals.items():
        # Rename both the dispatcher namespace and all externally visible C++
        # symbols, so loading this library cannot replace current production.
        content = original.decode("utf-8").replace("\r\n", "\n")
        content = content.replace("converse", "dev_fp32_converse")
        path = build / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        sources.append(str(path))
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG", "1033")
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    cflags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    cflags += ["-DCONVERSE2D_WITH_CUDA=1"]
    cuda_flags = ["-O3", "-lineinfo"]
    name = f"converse2d_dev_fp32_{fingerprint}"
    cpp_extension.load(
        name=name, sources=sources, extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags, with_cuda=True,
        is_python_module=False, build_directory=str(build), verbose=verbose,
    )
    manifest = dict(ref=BASELINE_REF, source_sha256=hashes,
                    namespace=namespace, extension_name=name,
                    build_directory=str(build), cflags=cflags, cuda_flags=cuda_flags)
    _loaded = (getattr(torch.ops, namespace), manifest)
    return _loaded
