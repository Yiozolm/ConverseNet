"""Build the research cuFFT-callback extension under .build/cufft_callbacks.

The callbacks are compiled to an LTO-IR fatbin for the current GPU and handed to
cuFFT at plan creation; cuFFT links them with nvJitLink. The extension links the
toolkit's cufft import library, which resolves to the cufft64 DLL torch loads.
"""
import hashlib
import os
from pathlib import Path
import subprocess

import torch
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / '.build' / 'cufft_callbacks'


def build(verbose=False):
    BUILD.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f'{major}{minor}'
    nvcc = Path(cpp_extension.CUDA_HOME) / 'bin' / ('nvcc.exe' if os.name == 'nt' else 'nvcc')
    fatbin = BUILD / f'callbacks_lto_{arch}.fatbin'
    command = [str(nvcc), '-std=c++17', '--expt-relaxed-constexpr', '-rdc=true', '-fatbin', f'-gencode=arch=compute_{arch},code=lto_{arch}',
               *[f'-I{path}' for path in cpp_extension.include_paths(device_type='cuda')],
               str(HERE / 'callbacks.cu'), '-o', str(fatbin)]
    result = subprocess.run(command, capture_output=True, text=True, errors='replace')
    if result.returncode:
        raise RuntimeError('LTO callback compile failed:\n' + result.stdout + result.stderr)
    if os.name == 'nt':
        cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
        ldflags = [f'/LIBPATH:{Path(cpp_extension.CUDA_HOME) / "lib" / "x64"}', 'cufft.lib']
        cflags = ['/O2', '/std:c++17']
    else:
        ldflags = [f'-L{Path(cpp_extension.CUDA_HOME) / "lib64"}', '-lcufft']
        cflags = ['-O3', '-std=c++17']
    ext = cpp_extension.load(name='cufft_callbacks_research', sources=[str(HERE / 'ext.cpp')],
                             extra_cflags=cflags, extra_ldflags=ldflags, with_cuda=True,
                             build_directory=str(BUILD), verbose=verbose)
    data = fatbin.read_bytes()
    ext.set_fatbin(data)
    identity = dict(fatbin=str(fatbin), fatbin_sha256=hashlib.sha256(data).hexdigest(), arch=arch,
                    sources_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                    for name in ('callbacks.cu', 'ext.cpp', 'loader.py')},
                    nvcc=str(nvcc))
    return ext, identity


if __name__ == '__main__':
    print(build(verbose=True)[1])
