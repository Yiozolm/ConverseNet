"""Build source-identified isolated training variants, never an installed binary."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'Converse2D/torch_converse2d'
BUILD = ROOT / '.build/training_speed'
_loaded = None


def _write(path, content):
    if not path.exists() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8')


def _build(name, folder, sources, headers=(), verbose=False):
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in [*sources, *headers]:
        digest.update(path.read_bytes())
    flags = ['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17']
    flags += ['-DCONVERSE2D_WITH_CUDA=1', '-DTRAINING_SOURCE_REV=0x'+digest.hexdigest()[:12]]
    cpp_extension.load(name=name, sources=[str(p) for p in sources],
        extra_include_paths=[str(folder)], extra_cflags=flags,
        extra_cuda_cflags=['-O3','-lineinfo'], with_cuda=True, is_python_module=False,
        build_directory=str(folder), verbose=verbose)


def load(include_checkout=False, verbose=False):
    global _loaded
    if os.name == 'nt':
        cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8','replace')
    if _loaded is None:
        folder = BUILD / 'fused'
        folder.mkdir(parents=True, exist_ok=True)
        original = (SOURCE/'converse2d_training.h').read_text(encoding='utf-8')
        optimized = original.replace('namespace converse2d::training', 'namespace converse2d::training_optimized')
        optimized = optimized.replace('converse_training_forward_cuda','training_speed_forward_cuda')
        optimized = optimized.replace('converse_training_backward_cuda','training_speed_backward_cuda')
        headers = [folder/'converse2d_training.h', folder/'optimized_training.h']
        _write(headers[0], original)
        _write(headers[1], optimized)
        sources = [HERE/'bindings.cpp', SOURCE/'converse2d_training.cu', HERE/'scale1.cu']
        _build('training_speed_fused', folder, sources, headers, verbose)
        paths = [*sources, SOURCE/'converse2d_training.h', HERE/'extension.py']
        manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        _loaded = (torch.ops.training_speed, manifest)
    if include_checkout and 'checkout_sources' not in _loaded[1]:
        folder = BUILD/'checkout'
        folder.mkdir(parents=True, exist_ok=True)
        sources = []
        hashes = {}
        for name in ('converse2d.cpp','converse2d_kernels.cu'):
            source = SOURCE/name
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            content = source.read_text(encoding='utf-8').replace('converse','training_checkout_converse')
            dest = folder/name
            _write(dest, content)
            sources.append(dest)
        _build('training_speed_checkout', folder, sources, verbose=verbose)
        _loaded[1]['checkout_sources'] = hashes
    return _loaded


if __name__ == '__main__':
    ops, manifest = load(include_checkout=True, verbose=True)
    print(json.dumps(manifest, indent=2))
