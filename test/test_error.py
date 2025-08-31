#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converse2D accuracy test:
- Forward/Backward numerical consistency: CUDA backend vs PyTorch backend
- Autograd gradcheck in float64
- Multiple shapes / batch sizes / scales
"""

import sys, pathlib, argparse, math, itertools, time
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.util_converse import Converse2D  # noqa: E402


def max_rel_err(a, b, eps=1e-8):
    with torch.no_grad():
        return ((a - b).abs() / (b.abs() + eps)).max().item()


def run_one_case(device, dtype, B, C, H, W, k, scale, eps, atol_fwd, rtol_fwd, atol_bwd, rtol_bwd, seed=0):
    torch.manual_seed(seed)

    # Build inputs
    x  = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
    x0 = torch.randn(B, C, H*scale, W*scale, device=device, dtype=dtype)

    # Two identical modules (same init), different backends
    m_py = Converse2D(C, C, k, scale=scale, padding=k//2, padding_mode="circular",
                      eps=eps, backend="pytorch").to(device=device, dtype=dtype).eval()
    m_cu = Converse2D(C, C, k, scale=scale, padding=k//2, padding_mode="circular",
                      eps=eps, backend="cuda").to(device=device, dtype=dtype).eval()

    # Copy weights/bias so two modules are identical
    with torch.no_grad():
        for (pn, p_py), (_, p_cu) in zip(m_py.named_parameters(), m_cu.named_parameters()):
            p_cu.copy_(p_py)

    # Forward
    x_py = x.detach().clone().requires_grad_(True)
    x_cu = x.detach().clone().requires_grad_(True)

    y_py = m_py(x_py)
    y_cu = m_cu(x_cu)

    # Backward (use a simple scalar loss so both sides comparable)
    loss_py = (y_py.square()).mean()
    loss_cu = (y_cu.square()).mean()
    g_py = torch.autograd.grad(loss_py, x_py, retain_graph=False, create_graph=False)[0].detach()
    g_cu = torch.autograd.grad(loss_cu, x_cu, retain_graph=False, create_graph=False)[0].detach()

    # Errors
    with torch.no_grad():
        f_mae  = (y_cu - y_py).abs().max().item()
        f_mre  = max_rel_err(y_cu, y_py)
        b_mae  = (g_cu - g_py).abs().max().item()
        b_mre  = max_rel_err(g_cu, g_py)

    print(f"[Case] B{B} C{C} {H}x{W} s{scale} k{k}  dtype={str(dtype).split('.')[-1]}")
    print(f"  forward:  max|Δ|={f_mae:.3e}   max rel={f_mre:.3e}")
    print(f"  backward: max|Δ|={b_mae:.3e}   max rel={b_mre:.3e}")

    # Assertions
    assert f_mae <= atol_fwd + rtol_fwd * y_py.abs().max().item() + 1e-12, \
        f"FWD max abs error too large: {f_mae:.3e}"
    assert f_mre <= rtol_fwd or f_mae <= atol_fwd, \
        f"FWD max rel error too large: {f_mre:.3e}"

    assert b_mae <= atol_bwd + rtol_bwd * g_py.abs().max().item() + 1e-12, \
        f"BWD max abs error too large: {b_mae:.3e}"
    assert b_mre <= rtol_bwd or b_mae <= atol_bwd, \
        f"BWD max rel error too large: {b_mre:.3e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",  default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--eps", type=float, default=1e-5)

    # tolerances (forward/backward)
    parser.add_argument("--atol-fwd", type=float, default=5e-4)
    parser.add_argument("--rtol-fwd", type=float, default=5e-3)
    parser.add_argument("--atol-bwd", type=float, default=8e-4)
    parser.add_argument("--rtol-bwd", type=float, default=8e-3)

    # shapes
    parser.add_argument("--B",  type=int, nargs="*", default=[1, 2])
    parser.add_argument("--C",  type=int, nargs="*", default=[2, 3])
    parser.add_argument("--H",  type=int, nargs="*", default=[16, 32])
    parser.add_argument("--W",  type=int, nargs="*", default=[18, 40])
    parser.add_argument("--k",  type=int, nargs="*", default=[3, 5])
    parser.add_argument("--s",  type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--gradcheck", action="store_true", help="run float64 gradcheck")
    args = parser.parse_args()

    device = args.device
    if device == "cpu":
        print("[WARN] CUDA not available, only PyTorch backend will run; comparisons will be skipped.")

    # dtype
    map_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = map_dtype[args.dtype]

    # Run a grid
    any_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if any_cuda:
        print("[INFO] CUDA backend: will be tested")
    print("[INFO] Python backend: will be tested")

    for (B, C, H, W, k, s) in itertools.product(args.B, args.C, args.H, args.W, args.k, args.s):
        if H * s > 2048 or W * s > 2048:
            # keep test practical
            continue
        if not any_cuda:
            # Just dry run to ensure Python path healthy
            torch.manual_seed(args.seed)
            m_py = Converse2D(C, C, k, scale=s, padding=k//2, padding_mode="circular",
                              eps=args.eps, backend="pytorch").to(device=device, dtype=dtype).eval()
            x  = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
            x0 = torch.randn(B, C, H*s, W*s, device=device, dtype=dtype)
            y  = m_py(x)
            _  = torch.autograd.grad((y.square()).mean(), x)[0]
            print(f"[CPU dry-run] B{B} C{C} {H}x{W} s{s} k{k} OK")
        else:
            run_one_case(device, dtype, B, C, H, W, k, s, args.eps,
                         args.atol_fwd, args.rtol_fwd, args.atol_bwd, args.rtol_bwd, seed=args.seed)

    # ====== gradcheck (float64, stricter) ======
    if args.gradcheck and any_cuda:
        print("[INFO] Running gradcheck (float64)…")
        torch.manual_seed(0)
        x64 = torch.randn(1, 2, 8, 9, device=device, dtype=torch.float64, requires_grad=True)
        x0  = torch.randn(1, 2, 16, 18, device=device, dtype=torch.float64)
        m64 = Converse2D(2, 2, 5, scale=2, padding=2, padding_mode="circular",
                         eps=args.eps, backend="cuda").to(device=device, dtype=torch.float64).eval()

        # Wrap a function of a single tensor for gradcheck; x0 is closed-over constant
        def f(t):
            return m64(t)

        ok = torch.autograd.gradcheck(f, (x64,), eps=1e-6, atol=1e-4, rtol=1e-4)
        print("[INFO] Gradcheck (float64) passed." if ok else "[WARN] Gradcheck failed.")

if __name__ == "__main__":
    main()
