"""Shared metadata/setup for isolated 1x1-wgrad follow-up tools."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_fp32_p0 as tensors


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def affinity():
    if os.name != 'nt':
        return sorted(os.sched_getaffinity(0))
    lib = ctypes.WinDLL('kernel32', use_last_error=True)
    lib.GetCurrentProcess.restype = ctypes.c_void_p
    lib.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not lib.GetProcessAffinityMask(lib.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return [i for i in range(8 * ctypes.sizeof(process)) if process.value & (1 << i)]


def setup(root, *, checked=True):
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend and CPU-only overrides')
    os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(root))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    loader = None
    if checked:
        spec = importlib.util.spec_from_file_location('wgrad_checked_loader', root / 'test/extension_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.load_extension()
    return torch, loader


def metadata(root, loader):
    import torch
    files = ('models/converse_usrnet.py', 'models/util_converse.py', 'models/converse_core.py',
             'test/extension_loader.py', 'model_zoo/converse_usrnet.pth')
    return dict(source_sha256={name: sha(root / name) for name in files},
        production_sources=loader.production_source_hashes() if loader else None,
        checked_manifest=json.loads((root / '.build/cuda/source_manifest.json').read_text()) if loader else None,
        tool_sha256={path.name: sha(path) for path in Path(__file__).parent.glob('wgrad*.py')},
        tensor_helper_sha256=sha(tensors.__file__),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
                         tf32=False, amp=False, affinity=affinity(),
                         torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads()))


def write_report(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
