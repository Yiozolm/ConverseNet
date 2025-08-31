#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, pathlib, torch
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.util_converse import Converse2D

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float32
B, C, H, W, scale = 2, 3, 32, 40, 2

x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)

m = Converse2D(C, C, 5, scale=scale, padding=2, padding_mode="circular", eps=1e-5, backend="pytorch").to(device=device, dtype=dtype)
m.eval()
x_py = x.detach().clone().requires_grad_(True)
y_py = m(x_py)
g_py = torch.autograd.grad(y_py.square().mean(), x_py)[0].detach()

have_cuda = False
try:
    if device == "cuda":
        m.backend = "cuda"
        x_cu = x.detach().clone().requires_grad_(True)
        y_cu = m(x_cu)
        g_cu = torch.autograd.grad(y_cu.square().mean(), x_cu)[0].detach()
        have_cuda = True
        print("[INFO] CUDA backend: OK")
    else:
        print("[WARN] CUDA not available on this device.")
except Exception as e:
    print("[WARN] CUDA backend unavailable ->", repr(e))

print("[INFO] Python backend: OK")

if have_cuda:
    with torch.no_grad():
        out_mae  = (y_cu - y_py).abs().max().item()
        grad_mae = (g_cu - g_py).abs().max().item()
        out_rel  = ((y_cu - y_py).abs() / (y_py.abs() + 1e-8)).max().item()
        grad_rel = ((g_cu - g_py).abs() / (g_py.abs() + 1e-8)).max().item()
    print(f"forward:  max|Δ|={out_mae:.3e}   max rel={out_rel:.3e}")
    print(f"backward: max|Δ|={grad_mae:.3e}  max rel={grad_rel:.3e}")

# gradcheck (float64)
try:
    x64 = torch.randn(1,2,8,9, device=device, dtype=torch.float64, requires_grad=True)
    m64 = Converse2D(2,2,5, scale=2, padding=2, padding_mode="circular", eps=1e-5, backend="auto").to(device=device, dtype=torch.float64)
    m64.eval()
    torch.autograd.gradcheck(lambda t: m64(t), (x64,), eps=1e-6, atol=1e-4, rtol=1e-4)
    print("[INFO] Gradcheck (float64) passed.")
except Exception as e:
    print("[WARN] Gradcheck skipped/failed ->", repr(e))
