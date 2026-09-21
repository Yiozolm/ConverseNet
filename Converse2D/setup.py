from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME
from torch.utils import cpp_extension
import hashlib, json, os
import torch
import build_config as config

if os.name == "nt":
    os.environ.setdefault("VSLANG", "1033")
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

has_cu = (os.environ.get("CONVERSE2D_CPU_ONLY") != "1" and
          torch.version.cuda is not None and CUDA_HOME is not None)
if has_cu and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}+PTX"
    else:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9+PTX")
cxx, nvcc = config.compile_flags()
fingerprint = hashlib.sha256(json.dumps(config.source_hashes(has_cu),sort_keys=True).encode()).hexdigest()
revision = "-DCONVERSE2D_SOURCE_REV=0x" + fingerprint[:12]
cxx.append(revision)
nvcc.append(revision)
sources = ["torch_converse2d/" + name for name in config.source_names(has_cu)]
depends = ["torch_converse2d/" + name for name in config.dependency_names(has_cu)]
extension = (CUDAExtension if has_cu else CppExtension)(
    name="converse2d_ext", sources=sources, depends=depends,
    extra_compile_args={"cxx":cxx,"nvcc":nvcc} if has_cu else cxx,
    define_macros=[("CONVERSE2D_WITH_CUDA","1")] if has_cu else [],
)
setup(name="torch_converse2d",version="0.3.0",
      description="Converse2D CUDA extension for PyTorch",packages=["torch_converse2d"],
      package_data={"torch_converse2d":list(config.dependency_names(True))},
      ext_modules=[extension],cmdclass={"build_ext":BuildExtension},zip_safe=False)
