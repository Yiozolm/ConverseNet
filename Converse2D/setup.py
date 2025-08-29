from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="torch_converse2d",
    version="0.1",
    description="Converse2D CUDA extension for PyTorch",
    packages=["torch_converse2d"],
    ext_modules=[
        CppExtension(
            name="converse2d_ext",
            sources=["torch_converse2d/converse2d_ext.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
