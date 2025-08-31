# setup.py — selectable variants: v1 | v2 | v3 | v4
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os, sys, pathlib

PKG_DIR = pathlib.Path(__file__).resolve().parent / "torch_converse2d"

# ---------------------------
# parse custom args
# ---------------------------
variant = os.environ.get("CONVERSE2D_VARIANT", "").lower()
to_remove = []
for i, a in enumerate(list(sys.argv)):
    aa = a.lower()
    if aa.startswith("--variant="):
        variant = a.split("=", 1)[1].lower(); to_remove.append(i)
    elif aa in ("--v1","--v2","--v3","--v4"):
        variant = aa[2:]; to_remove.append(i)
# scrub custom flags so setuptools doesn't see them
for idx in reversed(to_remove):
    sys.argv.pop(idx)

if variant not in {"", "v1","v2","v3","v4"}:
    raise SystemExit(f"[setup.py] invalid --variant={variant!r}; pick from v1|v2|v3|v4")

if not variant:
    variant = "v1"  # default

# ---------------------------
# pick sources per variant
# ---------------------------
CPP = str(PKG_DIR / f"converse2d_{variant}.cpp")
CU  = str(PKG_DIR / f"converse2d_{variant}.cu")
has_cu = os.path.exists(CU)  # v3,v4 have .cu; v1,v2 usually not

# ---------------------------
# CUDA arch (auto if not set)
# ---------------------------
extra_cflags = ["-O3"]
extra_cuda   = ["-O3"]

# Respect TORCH_CUDA_ARCH_LIST if user already set it; otherwise auto-detect.
if has_cu and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    try:
        import torch
        if torch.cuda.is_available():
            maj, min = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{min}+PTX"
    except Exception:
        # Fallback: a safe default that covers Ampere/Lovelace widely.
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9+PTX")

# ---------------------------
# Extension definition
# ---------------------------
if has_cu:
    ext = CUDAExtension(
        name="converse2d_ext",
        sources=[CPP, CU],
        extra_compile_args={"cxx": extra_cflags, "nvcc": extra_cuda},
    )
else:
    ext = CppExtension(
        name="converse2d_ext",
        sources=[CPP],
        extra_compile_args={"cxx": extra_cflags},
    )

print(f"[setup.py] building variant={variant}  sources={[p for p in ([CPP] + ([CU] if has_cu else []))]}")
print(f"[setup.py] TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST','<unset>')}")

setup(
    name="torch_converse2d",
    version="0.1",
    description="Converse2D CUDA extension for PyTorch",
    packages=["torch_converse2d"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
