import os
import torch
from models.util_converse import Converse2D

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float32
B, C, H, W, scale = 2, 3, 32, 40, 2

x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)

m = Converse2D(
    in_channels=C, out_channels=C, kernel_size=5, scale=scale,
    padding=2, padding_mode="circular", eps=1e-5, backend="pytorch" 
).to(device=device, dtype=dtype)
m.eval()

x_py = x.detach().clone().requires_grad_(True)
m.backend = "python"
y_py = m(x_py)
loss_py = y_py.square().mean()
g_py = torch.autograd.grad(loss_py, x_py)[0].detach()


have_cuda = False
try:
    if device == "cuda":
        m.backend = "cuda"  
        x_cuda = x.detach().clone().requires_grad_(True)
        y_cuda = m(x_cuda)
        loss_cuda = y_cuda.square().mean()
        g_cuda = torch.autograd.grad(loss_cuda, x_cuda)[0].detach()
        have_cuda = True
        print("[INFO] CUDA backend: OK")
    else:
        print("[WARN] CUDA not available on this device.")
except Exception as e:
    print("[WARN] CUDA backend unavailable ->", repr(e))

print("[INFO] Python backend: OK")

if have_cuda:
    with torch.no_grad():
        out_abs  = (y_cuda - y_py).abs()
        grad_abs = (g_cuda - g_py).abs()
        out_mae  = out_abs.max().item()
        grad_mae = grad_abs.max().item()
        out_rel  = (out_abs  / (y_py.abs() + 1e-8)).max().item()
        grad_rel = (grad_abs / (g_py.abs() + 1e-8)).max().item()

    print(f"forward:  max|Δ|={out_mae:.3e}   max rel={out_rel:.3e}")
    print(f"backward: max|Δ|={grad_mae:.3e}  max rel={grad_rel:.3e}")

try:
    torch.manual_seed(0)
    B2, C2, H2, W2, s2 = 1, 2, 8, 9, 2
    x64 = torch.randn(B2, C2, H2, W2, device=device, dtype=torch.float64, requires_grad=True)
    m64 = Converse2D(C2, C2, kernel_size=5, scale=s2, padding=2,
                     padding_mode="circular", eps=1e-5, backend="auto").to(device=device, dtype=torch.float64)
    m64.eval()
    def f(inp): return m64(inp)
    torch.autograd.gradcheck(f, (x64,), eps=1e-6, atol=1e-4, rtol=1e-4)
    print("[INFO] Gradcheck (float64) passed.")
except Exception as e:
    print("[WARN] Gradcheck skipped/failed ->", repr(e))
