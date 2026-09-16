"""Freeze the completed mixed-precision stage before training fusion."""
import os
import subprocess

import torch
from torch.utils import cpp_extension
from extension_loader import ROOT

BASELINE_REF="c4b950908645e616fedd24c84ea3c845d051126c"


def load_baseline():
    build=ROOT/".build"/"training_baseline"/BASELINE_REF[:7]
    build.mkdir(parents=True,exist_ok=True)
    sources=[]
    for name in ("converse2d.cpp","converse2d_fp32.h","converse2d_low_precision.h",
                 "converse2d_fft.h","converse2d_kernels.cu","converse2d_low_precision.cu"):
        original=subprocess.check_output(["git","show",f"{BASELINE_REF}:Converse2D/torch_converse2d/{name}"],cwd=ROOT).decode("utf-8")
        content=original.replace("converse","train_baseline_converse")
        path=build/name.replace("converse","train_baseline_converse")
        if not path.exists() or path.read_text(encoding="utf-8")!=content:
            path.write_text(content,encoding="utf-8")
        if path.suffix!=".h":
            sources.append(str(path))
    if os.name=="nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"]=os.environ["CONVERSE2D_BUILD_PATH"]
        cpp_extension.SUBPROCESS_DECODE_ARGS=("utf-8","replace")
    flags=["/O2","/std:c++17"] if os.name=="nt" else ["-O3","-std=c++17"]
    cpp_extension.load(name="converse2d_training_baseline",sources=sources,
        extra_cflags=flags+["-DCONVERSE2D_WITH_CUDA=1"],extra_cuda_cflags=["-O3","-lineinfo"],
        extra_ldflags=["cufft.lib" if os.name=="nt" else "-lcufft"],with_cuda=True,
        build_directory=str(build),verbose=False)
    return torch.ops.train_baseline_converse2d
