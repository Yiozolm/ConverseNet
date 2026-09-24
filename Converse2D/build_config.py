"""Explicit production build inputs; no torch import or candidate discovery.

Only production FP32 sources are compiled. Historical experiments live in Git history.
"""
from pathlib import Path
import hashlib
import os
import re
import shutil

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'torch_converse2d'
HOST_SOURCES = (
    'converse2d.cpp', 'operator.cpp', 'reference/reference.cpp',
    'inference/cache.cpp', 'inference/inference_preparation.cpp',
    'training/full_spectrum/production.cpp',
)
CUDA_SOURCES = (
    'inference/inference_prepare.cu', 'inference/inference_dispatch.cu',
    'inference/inference_scale1.cu', 'inference/inference_scale2.cu',
    'inference/inference_scale3.cu', 'inference/inference_generic.cu',
    'training/full_spectrum/full_fusion.cu',
)
_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.M)

def source_names(cuda=True):
    return HOST_SOURCES + (CUDA_SOURCES if cuda else ())

def dependency_names(cuda=True):
    """Conservative quoted-include closure, including conditionally used headers."""
    found = set()
    def visit(name):
        if name in found:
            return
        found.add(name)
        path = PACKAGE / name
        for target in _INCLUDE.findall(path.read_text(encoding='utf-8')):
            child = (path.parent / target).resolve()
            visit(child.relative_to(PACKAGE.resolve()).as_posix())
    for name in source_names(cuda):
        visit(name)
    return tuple(sorted(found))

def source_hashes(cuda=True):
    paths = [(name, PACKAGE / name) for name in dependency_names(cuda)]
    paths += [('../build_config.py', Path(__file__)), ('../setup.py', ROOT / 'setup.py')]
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths}

def compile_flags():
    cxx = ['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17']
    return cxx, ['-O3', '-lineinfo']

def toolchain_identity(cuda):
    names = ['cl' if os.name == 'nt' else os.environ.get('CXX', 'c++')]
    if cuda:
        names.append('nvcc')
    result = {}
    for name in names:
        executable = shutil.which(name)
        if executable:
            path = Path(executable).resolve()
            stat = path.stat()
            result[name] = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        else:
            result[name] = None
    result['environment'] = {key: os.environ.get(key, '') for key in (
        'CUDA_HOME', 'CUDA_PATH', 'TORCH_CUDA_ARCH_LIST', 'CXX', 'CC',
        'CL', '_CL_', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'INCLUDE', 'LIB')}
    return result
