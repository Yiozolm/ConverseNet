"""Build/load the corrected extension in the checkout, without global installs."""
import os
import pathlib

import torch
from torch.utils.cpp_extension import CUDA_HOME, load
from torch.utils import cpp_extension

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_extension(cpu_only=False, verbose=False):
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
    if os.environ.get("CONVERSE2D_SKIP_BUILD") == "1":
        library = build / ("converse2d_checked_ext.pyd" if os.name == "nt" else "converse2d_checked_ext.so")
        if not library.exists():
            raise RuntimeError("Build test/extension_loader.py before using CONVERSE2D_SKIP_BUILD=1")
        torch.ops.load_library(str(library))
        return
    source = ROOT / "Converse2D" / "torch_converse2d"
    sources = [str(source / "converse2d.cpp")]
    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    if cuda:
        sources.append(str(source / "converse2d_kernels.cu"))
        flags.append("-DCONVERSE2D_WITH_CUDA=1")
    return load(name="converse2d_checked_ext", sources=sources,
                extra_cflags=flags, extra_cuda_cflags=["-O3", "-lineinfo"],
                with_cuda=cuda, build_directory=str(build), verbose=verbose)


if __name__ == "__main__":
    load_extension(verbose=True)
    print("Loaded", torch.ops.converse2d.forward)
