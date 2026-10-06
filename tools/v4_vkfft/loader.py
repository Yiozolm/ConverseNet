"""Fetch a pinned VkFFT into .build/VkFFT and build the research wrapper under .build/vkfft."""
import hashlib
import os
from pathlib import Path
import subprocess

import torch
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / '.build' / 'VkFFT'
BUILD = ROOT / '.build' / 'vkfft'
URL = 'https://github.com/DTolm/VkFFT.git'
COMMIT = '066a17c17068c0f11c9298d848c2976c71fad1c1'  # v1.3.4


def fetch():
    if not (SOURCE / '.git').exists():
        subprocess.run(['git', 'clone', '--quiet', URL, str(SOURCE)], check=True)
    head = subprocess.run(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--quiet', 'origin', COMMIT], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--quiet', COMMIT], check=True)
    return SOURCE / 'vkFFT'


def build(verbose=False):
    include = fetch()
    BUILD.mkdir(parents=True, exist_ok=True)
    cuda = Path(cpp_extension.CUDA_HOME)
    if os.name == 'nt':
        cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
        ldflags = [f'/LIBPATH:{cuda / "lib" / "x64"}', 'cuda.lib', 'nvrtc.lib']
        cflags = ['/O2', '/std:c++17', '/DNOMINMAX', '/wd4244', '/wd4267', '/wd4996']
    else:
        ldflags = [f'-L{cuda / "lib64"}', '-lcuda', '-lnvrtc']
        cflags = ['-O3', '-std=c++17', '-w']
    ext = cpp_extension.load(name='vkfft_research', sources=[str(HERE / 'ext.cpp')],
                             extra_include_paths=[str(include)], extra_cflags=cflags, extra_ldflags=ldflags,
                             with_cuda=True, build_directory=str(BUILD), verbose=verbose)
    identity = dict(vkfft_commit=COMMIT, cuda_home=str(cuda), torch=torch.__version__,
                    sources_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                    for name in ('ext.cpp', 'loader.py')})
    return ext, identity


if __name__ == '__main__':
    print(build(verbose=True)[1])
