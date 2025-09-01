#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, pathlib, statistics, subprocess, csv, re
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PKG = PROJECT_ROOT / "Converse2D/torch_converse2d"

# ------------ utils ------------
def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, warmup, iters):
    # warmup
    for _ in range(warmup):
        fn()
    synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p90_ms": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else statistics.median_high(times),
    }

def tp_gpix_per_s(B, H, W, s, mean_ms):
    if not mean_ms or mean_ms <= 0:
        return None
    # pixels processed per example = (H*s)*(W*s)
    return (B * (H * s) * (W * s) / (mean_ms / 1e3)) / 1e9

def to_dtype(name):
    name = name.lower()
    if name in ("fp16", "half", "float16"):   return torch.float16
    if name in ("bf16", "bfloat16"):          return torch.bfloat16
    if name in ("fp32", "float32", "float"):  return torch.float32
    raise ValueError(name)

# ------------ parent <-> child plumbing ------------
def _parse_last_json_from_text(txt: str):
    m = re.findall(r"\{.*\}", txt, flags=re.S)
    if not m:
        tail = txt[-2000:] if len(txt) > 2000 else txt
        raise RuntimeError("Child produced no JSON. Tail:\n" + tail)
    return json.loads(m[-1])

def run_variant_subprocess(variant, case_args, cache_root):
    cmd = [
        sys.executable, __file__, "--worker",
        "--variant", variant,
        "--B", str(case_args["B"]), "--C", str(case_args["C"]),
        "--H", str(case_args["H"]), "--W", str(case_args["W"]),
        "--scale", str(case_args["scale"]), "--ksize", str(case_args["ksize"]),
        "--warmup", str(case_args["warmup"]), "--iters", str(case_args["iters"]),
        "--dtype", case_args["dtype"], "--device", case_args["device"],
        "--eps",  str(case_args["eps"]),
    ]
    env = os.environ.copy()
    env["TORCH_EXTENSIONS_DIR"] = str(pathlib.Path(cache_root) / variant)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed (variant={variant}). Output:\n{out}")
    return _parse_last_json_from_text(out)

# ------------ worker ------------
def worker_main(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype  = to_dtype(args.dtype)
    B, C, H, W, s, k = args.B, args.C, args.H, args.W, args.scale, args.ksize
    eps = float(args.eps)

    if args.variant == "pytorch":
        from models.util_converse import Converse2D   # 只在 baseline 导入，避免注册 v1 扩展冲突
        torch.manual_seed(0)
        m = Converse2D(C, C, kernel_size=k, scale=s, padding=k//2,
                       padding_mode="circular", eps=eps, backend="pytorch").to(device=device, dtype=dtype).eval()

        x_fwd = torch.randn(B, C, H, W, device=device, dtype=dtype)
        x_bwd = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)

        def call_fwd():
            with torch.no_grad():
                _ = m(x_fwd)
        fstat = timed_run(call_fwd, args.warmup, args.iters)

        def call_bwd():
            y = m(x_bwd)
            (y.square().mean()).backward()
            assert x_bwd.grad is not None, "x.grad is None"
            x_bwd.grad.zero_()
        bstat = timed_run(call_bwd, args.warmup, args.iters)

        out = {
            "variant": "pytorch",
            "fwd_mean_ms": fstat["mean_ms"], "fwd_p50_ms": fstat["p50_ms"], "fwd_p90_ms": fstat["p90_ms"], "fwd_tp": tp_gpix_per_s(B,H,W,s,fstat["mean_ms"]),
            "bwd_mean_ms": bstat["mean_ms"], "bwd_p50_ms": bstat["p50_ms"], "bwd_p90_ms": bstat["p90_ms"], "bwd_tp": tp_gpix_per_s(B,H,W,s,bstat["mean_ms"]),
            "grad_ok": True,
            "warmup": args.warmup, "iters": args.iters, "dtype": args.dtype, "device": device
        }
        print(json.dumps(out))
        return

    from torch.utils.cpp_extension import load
    vnum = int(args.variant.split("_v")[1])
    cpp = PKG / f"converse2d_v{vnum}.cpp"
    cu  = PKG / f"converse2d_v{vnum}.cu"
    sources = [str(cpp)]
    if cu.exists(): sources.append(str(cu))

    maj, minr = torch.cuda.get_device_capability(0) if device == "cuda" else (0, 0)
    arch_num = f"{maj}{minr}"     
    arch_str = f"{maj}.{minr}"     
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{arch_str}+PTX")
    extra_cuda = ["-O3", f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}"] if (cu.exists() and device=="cuda") else []

    ext_name = f"converse2d_v{vnum}_sm{arch_num}_ext"
    print(f"[build] compiling {ext_name} (variant={args.variant}) ...", flush=True)
    load(name=ext_name, sources=sources, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=extra_cuda)

    converse2d_forward = torch.ops.converse2d.forward

    torch.manual_seed(0)
    x_fwd = torch.randn(B, C, H, W, device=device, dtype=dtype)
    x_bwd = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
    x0_fwd = x_fwd if s == 1 else F.interpolate(x_fwd, scale_factor=s, mode="nearest")
    x0_bwd = x_bwd if s == 1 else F.interpolate(x_bwd, scale_factor=s, mode="nearest")
    weight = torch.randn(1, C, k, k, device=device, dtype=dtype)
    weight = torch.softmax(weight.view(1, C, -1), dim=-1).view(1, C, k, k).contiguous()
    bias = torch.zeros(1, C, 1, 1, device=device, dtype=dtype)

    def call_fwd():
        with torch.no_grad():
            _ = converse2d_forward(x_fwd, x0_fwd, weight, bias, int(s), float(eps))
    fstat = timed_run(call_fwd, args.warmup, args.iters)

    def call_bwd():
        y = converse2d_forward(x_bwd, x0_bwd, weight, bias, int(s), float(eps))
        (y.square().mean()).backward()
        assert x_bwd.grad is not None, "x.grad is None"
        x_bwd.grad.zero_()

    grad_ok = True
    try:
        bstat = timed_run(call_bwd, args.warmup, args.iters)
    except Exception:
        grad_ok = False
        bstat = {"mean_ms": None, "p50_ms": None, "p90_ms": None}

    try:
        torch.ops.converse2d.clear_cache()
    except Exception:
        pass

    out = {
        "variant": args.variant,
        "fwd_mean_ms": fstat["mean_ms"], "fwd_p50_ms": fstat["p50_ms"], "fwd_p90_ms": fstat["p90_ms"],
        "fwd_tp": tp_gpix_per_s(B, H, W, s, fstat["mean_ms"]),
        "bwd_mean_ms": bstat["mean_ms"], "bwd_p50_ms": bstat["p50_ms"], "bwd_p90_ms": bstat["p90_ms"],
        "bwd_tp": tp_gpix_per_s(B, H, W, s, bstat["mean_ms"]) if bstat["mean_ms"] else None,
        "grad_ok": grad_ok,
        "warmup": args.warmup, "iters": args.iters, "dtype": args.dtype, "device": device
    }
    print(json.dumps(out))

# ------------ parent orchestrator ------------
def parent_main(args):
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    Bs  = [int(x) for x in args.B_list.split(",")]
    Cs  = [int(x) for x in args.C_list.split(",")]
    Hs  = [int(x) for x in args.H_list.split(",")]
    Ws  = [int(x) for x in args.W_list.split(",")]
    Ss  = [int(x) for x in args.scale_list.split(",")]
    Ks  = [int(x) for x in args.ksize_list.split(",")]

    print(f"[Env] device={device}, torch={torch.__version__}, cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}")
    print(f"[Cfg] dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"[Grid] B={Bs} C={Cs} H={Hs} W={Ws} scale={Ss} ksize={Ks}\n")

    variants = ["pytorch", "cuda"]

    results = []
    cache_root = PROJECT_ROOT / ".torch_ext_cache_grid"
    cache_root.mkdir(exist_ok=True)

    for B in Bs:
        for C in Cs:
            for H in Hs:
                for W in Ws:
                    for s in Ss:
                        for k in Ks:
                            case = dict(B=B, C=C, H=H, W=W, scale=s, ksize=k,
                                        warmup=args.warmup, iters=args.iters,
                                        dtype=args.dtype, device=device, eps=args.eps)

                            base = run_variant_subprocess("pytorch", case, cache_root)
                            results.append({**case, **base})
                            print(f"[Case] B{B} C{C} {H}x{W} s{s} k{k}")
                            print(f"  PyTorch : fwd {base['fwd_mean_ms']:.3f} ms  |  bwd {base['bwd_mean_ms']:.3f} ms  | grad_ok={base['grad_ok']}")

                            base_fwd = base["fwd_mean_ms"]
                            base_bwd = base["bwd_mean_ms"]

                            for v in variants[1:]:
                                r = run_variant_subprocess(v, case, cache_root)
                                r["fwd_speedup_vs_pytorch"] = (base_fwd / r["fwd_mean_ms"]) if (r["fwd_mean_ms"] and base_fwd) else None
                                r["bwd_speedup_vs_pytorch"] = (base_bwd / r["bwd_mean_ms"]) if (r["bwd_mean_ms"] and base_bwd) else None
                                results.append({**case, **r})
                                fsp = f"{r['fwd_speedup_vs_pytorch']:.2f}x" if r["fwd_speedup_vs_pytorch"] else "n/a"
                                if r["bwd_mean_ms"]:
                                    bsp = f"{r['bwd_speedup_vs_pytorch']:.2f}x" if r["bwd_speedup_vs_pytorch"] else "n/a"
                                    bwd_repr = f"{r['bwd_mean_ms']:.3f} ms ({bsp})"
                                else:
                                    bwd_repr = "n/a"
                                print(f"  {v:8s}: fwd {r['fwd_mean_ms']:.3f} ms ({fsp}) | bwd {bwd_repr} | grad_ok={r['grad_ok']}")
                            print("")

    header = [
        "variant","B","C","H","W","scale","ksize",
        "fwd_mean_ms","fwd_p50_ms","fwd_p90_ms","fwd_tp","fwd_speedup_vs_pytorch",
        "bwd_mean_ms","bwd_p50_ms","bwd_p90_ms","bwd_tp","bwd_speedup_vs_pytorch",
        "grad_ok","eps","warmup","dtype","device","iters"
    ]
    print("\n=== Summary (speed vs PyTorch) ===")
    print(" | ".join(h.rjust(14) for h in header))
    print("-"*160)
    for r in results:
        row=[]
        for h in header:
            v = r.get(h,"")
            if isinstance(v, float):
                row.append(f"{v:14.3f}")
            else:
                row.append(str(v).rjust(14))
        print(" | ".join(row))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(results)
        print(f"\n[Saved] {args.csv}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true", help="internal")
    p.add_argument("--variant", default="")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--C", type=int, default=16)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", default="float32", choices=["float16","bfloat16","float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--eps", default=1e-5, type=float)
    # grid
    p.add_argument("--B_list", default="8")
    p.add_argument("--C_list", default="8")
    p.add_argument("--H_list", default="256")
    p.add_argument("--W_list", default="256")
    p.add_argument("--scale_list", default="1,2,3")
    p.add_argument("--ksize_list", default="3,5,7")
    p.add_argument("--csv", default="")
    args = p.parse_args()
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)

if __name__ == "__main__":
    main()
