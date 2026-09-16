"""Opt-in native FP16/BF16 activation FFTs for the unified Converse2D entry."""
from contextlib import contextmanager

import torch


@contextmanager
def native_fft(enabled=True):
    """Thread-local, nestable experimental FFT policy; restores state on exit.

    Only FP16/BF16 CUDA activations select native FFTs. Kernel FFTs and the
    spectral solve stay FP32. Unsupported sizes/devices and graph capture use
    FP32. The forward's FFT choice is retained for backward after scope exit.
    """
    from models.util_converse import _try_import_converse2d_ext
    _try_import_converse2d_ext()
    if not hasattr(torch.ops.converse2d,"set_native_fft"):
        raise RuntimeError("Rebuild the Converse2D extension to enable native FFTs")
    previous=torch.ops.converse2d.set_native_fft(bool(enabled))
    try:
        yield
    finally:
        torch.ops.converse2d.set_native_fft(previous)
