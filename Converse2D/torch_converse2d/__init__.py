import os
try:
    import converse2d_ext
except Exception as e:
    print("[torch_converse2d] extension import failed:", e)

__all__ = ["converse2d_ext"]
