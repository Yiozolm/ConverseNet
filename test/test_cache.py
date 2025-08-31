#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, pathlib, torch
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.util_converse import Converse2D

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backend = "cuda" if device == "cuda" else "pytorch"
    m = Converse2D(3,3,5, scale=2, padding=2, padding_mode="circular", eps=1e-5, backend=backend).to(device)
    m.eval()

    x = torch.randn(1,3,64,64, device=device)
    with torch.no_grad():
        _ = m(x)

    if hasattr(torch.ops, "converse2d") and hasattr(torch.ops.converse2d, "clear_cache"):
        torch.ops.converse2d.clear_cache()
        print("[INFO] cleared converse2d FB cache")
    else:
        print("[WARN] converse2d extension not available; nothing to clear")

if __name__ == "__main__":
    main()
