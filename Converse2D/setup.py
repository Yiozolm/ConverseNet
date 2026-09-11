from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os, pathlib
import torch
from torch.utils.cpp_extension import CUDA_HOME

PKG_DIR = pathlib.Path(__file__).resolve().parent / "torch_converse2d"


CPP = str(PKG_DIR / f"converse2d.cpp")
CU  = str(PKG_DIR / "converse2d_kernels.cu")
has_cu = (os.environ.get("CONVERSE2D_CPU_ONLY") != "1" and
          torch.version.cuda is not None and CUDA_HOME is not None)

extra_cflags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
extra_cuda = ["-O3", "-lineinfo"]
macros = [("CONVERSE2D_WITH_CUDA", "1")] if has_cu else []

if has_cu and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    try:
        import torch
        if torch.cuda.is_available():
            maj, min = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{min}+PTX"
    except Exception:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9+PTX")

if has_cu:
    ext = CUDAExtension(
        name="converse2d_ext",
        sources=[CPP, CU],
        extra_compile_args={"cxx": extra_cflags, "nvcc": extra_cuda},
        define_macros=macros,
    )
else:
    ext = CppExtension(
        name="converse2d_ext",
        sources=[CPP],
        extra_compile_args={"cxx": extra_cflags},
    )

print(f"[setup.py] building sources={[p for p in ([CPP] + ([CU] if has_cu else []))]}")
print(f"[setup.py] TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST','<unset>')}")

setup(
    name="torch_converse2d",
    version="0.3.0",
    description="Converse2D CUDA extension for PyTorch",
    packages=["torch_converse2d"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
