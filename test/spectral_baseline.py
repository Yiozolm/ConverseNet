"""Build a frozen pre-optimization kernel under a separate operator namespace."""
import os
import subprocess

import torch
from torch.utils import cpp_extension
from extension_loader import ROOT

BASELINE_REF = '19c1bfc7c98fbf8c6c4f9e07393d581431008103'


class _FrozenBaseline:
    """Adapt the historical ABI to the current six-argument test interface."""
    def forward(self, x, x0, weight, bias, scale, eps=1e-5):
        return torch.ops.baseline_converse2d.forward(x, x0, weight, bias, scale, eps, 'v7')

    def __getattr__(self, name):
        return getattr(torch.ops.baseline_converse2d, name)


def load_baseline():
    build = ROOT / '.build' / 'spectral_baseline' / BASELINE_REF[:12]
    build.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CONVERSE2D_SKIP_BUILD') == '1':
        library = build / ('converse2d_spectral_baseline.pyd' if os.name == 'nt' else 'converse2d_spectral_baseline.so')
        if not library.exists():
            raise RuntimeError('Build test/spectral_baseline.py before profiling with CONVERSE2D_SKIP_BUILD=1')
        torch.ops.load_library(str(library))
        return _FrozenBaseline()
    sources = []
    for name in ('converse2d.cpp', 'converse2d_kernels.cu'):
        original = subprocess.check_output([
            'git', 'show', f'{BASELINE_REF}:Converse2D/torch_converse2d/{name}'
        ], cwd=ROOT).decode('utf-8')
        # Rename registration and exported Converse symbols, not the algorithm.
        content = original.replace('converse', 'baseline_converse')
        path = build / name
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            path.write_text(content, encoding='utf-8')
        sources.append(str(path))
    if os.name == 'nt':
        if os.environ.get('CONVERSE2D_BUILD_PATH'):
            os.environ['PATH'] = os.environ['CONVERSE2D_BUILD_PATH']
        cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
    flags = ['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17']
    cpp_extension.load(name='converse2d_spectral_baseline', sources=sources,
                       extra_cflags=flags + ['-DCONVERSE2D_WITH_CUDA=1'],
                       extra_cuda_cflags=['-O3', '-lineinfo'], with_cuda=True,
                       build_directory=str(build), verbose=False)
    return _FrozenBaseline()


if __name__ == '__main__':
    print(load_baseline().forward)
