#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, pathlib, statistics, subprocess, csv, re
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PKG = PROJECT_ROOT / "Converse2D/torch_converse2d" 

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, warmup, iters):
    for _ in range(warmup): fn()
    synchronize()
    times=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); synchronize()
        times.append((time.perf_counter()-t0)*1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p90_ms": statistics.quantiles(times, n=10)[8] if len(times)>=10 else statistics.median_high(times),
    }

def tp_gpix_per_s(B,H,W,s,mean_ms):
    if mean_ms<=0: return None
    return (B*(H*s)*(W*s)/(mean_ms/1e3))/1e9

def to_dtype(name):
    name=name.lower()
    if name in ("fp16","half","float16"): return torch.float16
    if name in ("bf16","bfloat16"):       return torch.bfloat16
    if name in ("fp32","float32","float"):return torch.float32
    raise ValueError(name)

# -------- parent <-> child plumbing --------
import re
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
    ]
    env = os.environ.copy()
    env["TORCH_EXTENSIONS_DIR"] = str(pathlib.Path(cache_root) / variant)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed (variant={variant}). Output:\n{out}")
    return _parse_last_json_from_text(out)

# -------- worker --------
def worker_main(args):
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    dtype  = to_dtype(args.dtype)
    B,C,H,W,s,k = args.B,args.C,args.H,args.W,args.scale,args.ksize

    if args.variant == "pytorch":
        from models.util_converse import Converse2D
        torch.manual_seed(0)
        x = torch.randn(B,C,H,W, device=device, dtype=dtype)
        m = Converse2D(C, C, kernel_size=k, scale=s, padding=k//2,
                       padding_mode="circular", eps=1e-5, backend="pytorch").to(device=device, dtype=dtype)
        m.eval()
        def call(): 
            with torch.no_grad():
                _ = m(x)
        stat = timed_run(call, args.warmup, args.iters)
        stat["tp"] = tp_gpix_per_s(B,H,W,s,stat["mean_ms"])
        print(json.dumps({"variant":"pytorch", **stat}))
        return

    from torch.utils.cpp_extension import load
    vnum = int(args.variant.split("_v")[1])
    cpp = PKG / f"converse2d_v{vnum}.cpp"
    cu  = PKG / f"converse2d_v{vnum}.cu"
    sources = [str(cpp)]
    if cu.exists(): sources.append(str(cu))

    maj, min = torch.cuda.get_device_capability(0) if device == "cuda" else (0, 0)
    arch_num = f"{maj}{min}"        # e.g. "75", "86", "89"
    arch_str = f"{maj}.{min}"       # e.g. "7.5"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{arch_str}+PTX")

    ext_name = f"converse2d_v{vnum}_sm{arch_num}_ext"

    print(f"[build] compiling {ext_name} for sm_{arch_num} (variant={args.variant}) ...", flush=True)

    extra_cuda = []
    if cu.exists() and device == "cuda":
        extra_cuda = ["-O3", f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}"]

    load(
        name=ext_name,
        sources=sources,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda,
    )

    torch.manual_seed(0)
    x = torch.randn(B,C,H,W, device=device, dtype=dtype)
    x0 = x if s==1 else F.interpolate(x, scale_factor=s, mode="nearest")
    weight = torch.randn(1,C,k,k, device=device, dtype=dtype)
    weight = torch.softmax(weight.view(1,C,-1), dim=-1).view(1,C,k,k).contiguous()
    bias = torch.zeros(1,C,1,1, device=device, dtype=dtype)

    converse2d_forward = torch.ops.converse2d.forward
    def call():
        with torch.no_grad():
            _ = converse2d_forward(x, x0, weight, bias, int(s), float(1e-5))
    stat = timed_run(call, args.warmup, args.iters)
    stat["tp"] = tp_gpix_per_s(B,H,W,s,stat["mean_ms"])
    try: torch.ops.converse2d.clear_cache()
    except Exception: pass
    print(json.dumps({"variant": args.variant, **stat}))

# -------- parent orchestrator --------
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

    variants = ["pytorch", "cuda_v1", "cuda_v2", "cuda_v3", "cuda_v4"]
    results = list()
    cache_root = PROJECT_ROOT / ".torch_ext_cache_grid"
    cache_root.mkdir(exist_ok=True)

    for B in Bs:
        for C in Cs:
            for H in Hs:
                for W in Ws:
                    for s in Ss:
                        for k in Ks:
                            case = dict(B=B,C=C,H=H,W=W,scale=s,ksize=k,
                                        warmup=args.warmup,iters=args.iters,
                                        dtype=args.dtype,device=device)
                            base = run_variant_subprocess("pytorch", case, cache_root)
                            base_mean = base["mean_ms"]
                            results.append({**case,"variant":"pytorch",**base})
                            print(f"[Case] B{B} C{C} {H}x{W} s{s} k{k}")
                            print(f"  PyTorch : {base_mean:.3f} ms")
                            for v in variants[1:]:
                                r = run_variant_subprocess(v, case, cache_root)
                                sp = base_mean / r["mean_ms"] if r["mean_ms"]>0 else None
                                results.append({**case, "variant":v, **r, "speedup_vs_pytorch": sp})
                                print(f"  {v:8s}: {r['mean_ms']:.3f} ms  ({sp:.2f}x vs PyTorch)")
                            print("")

    header = ["variant","B","C","H","W","scale","ksize","mean_ms","p50_ms","p90_ms","tp","speedup_vs_pytorch","warmup","dtype","device","iters"]
    print("\n=== Summary (normalized to PyTorch) ===")
    print(" | ".join(h.rjust(10) for h in header))
    print("-"*120)
    for r in results:
        line=[]
        for h in header:
            v = r.get(h,"")
            if isinstance(v,float):
                line.append(f"{v:10.3f}")
            else:
                line.append(str(v).rjust(10))
        print(" | ".join(line))

    if args.csv:
        with open(args.csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(results)
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
    # grid
    p.add_argument("--B_list", default="1")
    p.add_argument("--C_list", default="3,8")
    p.add_argument("--H_list", default="128,256")
    p.add_argument("--W_list", default="128,256")
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
