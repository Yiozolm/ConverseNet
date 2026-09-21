"""Isolated shared-s1 spectral candidate; load() never edits production sources."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_LOADED = {}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load(verbose=False):
    import torch
    from torch.utils import cpp_extension
    originals = {name:(HERE/name).read_bytes() for name in ("bindings.cpp","kernels.cu")}
    hashes = {name:digest(value) for name,value in originals.items()}
    hashes["loader.py"] = digest(Path(__file__).read_bytes())
    identity = dict(source_sha256=hashes,torch=str(torch.__version__),cuda=torch.version.cuda,
                    cuda_arch_list=os.environ.get("TORCH_CUDA_ARCH_LIST"))
    fingerprint = digest(json.dumps(identity,sort_keys=True).encode())[:16]
    namespace = "training_shared_s1_"+fingerprint
    if fingerprint in _LOADED:
        return _LOADED[fingerprint]
    build = ROOT/".build/training_shared_s1"/fingerprint
    build.mkdir(parents=True,exist_ok=True)
    sources,generated = [],{}
    for name,value in originals.items():
        text = value.decode("utf-8").replace("\r\n","\n")
        text = text.replace("shared_s1_experiment",namespace)
        text = text.replace("shared_s1_forward_cuda",namespace+"_forward_cuda")
        text = text.replace("shared_s1_backward_cuda",namespace+"_backward_cuda")
        target = build/name
        if not target.exists() or target.read_text(encoding="utf-8")!=text:
            target.write_text(text,encoding="utf-8")
        sources.append(str(target))
        generated[name] = digest(target.read_bytes())
    if os.name=="nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG","1033")
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8","replace")
    flags = ["/O2","/std:c++17"] if os.name=="nt" else ["-O3","-std=c++17"]
    flags += ["-DSHARED_S1_SOURCE_REV=0x"+fingerprint[:12]]
    wrapper_cl = "/Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX"
    normalize_cl = os.name=="nt" and os.environ.get("CL")==wrapper_cl
    if normalize_cl:
        flags += wrapper_cl.split()
    try:
        if normalize_cl:
            os.environ.pop("CL")
        binary = cpp_extension.load(name=namespace,sources=sources,extra_cflags=flags,
            extra_cuda_cflags=["-O3","-lineinfo"],with_cuda=True,is_python_module=False,
            build_directory=str(build),verbose=verbose)
    finally:
        if normalize_cl:
            os.environ["CL"] = wrapper_cl
    binary = Path(binary)
    if not binary.is_file():
        raise RuntimeError(f"Loaded extension did not return a binary file: {binary}")
    metadata = dict(**identity,generated_source_sha256=generated,namespace=namespace,
        build_directory=str(build),binary_path=str(binary),binary_sha256=digest(binary.read_bytes()),
        cflags=flags,cuda_flags=["-O3","-lineinfo"],known_wrapper_cl_moved_to_cflags=normalize_cl,
        nvcc_prepend_flags=os.environ.get("NVCC_PREPEND_FLAGS"),
        arithmetic="H/D and pointwise products use input dtype; B/C and lambda reductions accumulate FP64; no floating atomics")
    _LOADED[fingerprint] = (getattr(torch.ops,namespace),metadata)
    return _LOADED[fingerprint]
