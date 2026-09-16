"""Frozen pre-mixed-precision implementation, in an isolated namespace."""
import os
import subprocess

import torch
from torch.utils import cpp_extension
from extension_loader import ROOT

BASELINE_REF = "85b9f80fa40dceedd88a5f543fd727617c7cf7bc"


def load_baseline():
    build = ROOT / ".build" / "low_precision_baseline" / BASELINE_REF[:12]
    build.mkdir(parents=True,exist_ok=True)
    sources = []
    for name in ("converse2d.cpp","converse2d_kernels.cu","converse2d_fft.h"):
        original = subprocess.check_output([
            "git","show",f"{BASELINE_REF}:Converse2D/torch_converse2d/{name}"
        ],cwd=ROOT).decode("utf-8")
        content = original.replace("converse","lp_baseline_converse")
        path = build / name.replace("converse","lp_baseline_converse")
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content,encoding="utf-8")
        if path.suffix != ".h":
            sources.append(str(path))
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8","replace")
    flags = ["/O2","/std:c++17"] if os.name == "nt" else ["-O3","-std=c++17"]
    cpp_extension.load(name="converse2d_low_precision_baseline",sources=sources,
        extra_cflags=flags+["-DCONVERSE2D_WITH_CUDA=1"],
        extra_cuda_cflags=["-O3","-lineinfo"],
        extra_ldflags=["cufft.lib" if os.name=="nt" else "-lcufft"],
        with_cuda=True,build_directory=str(build),verbose=False)
    return torch.ops.lp_baseline_converse2d
