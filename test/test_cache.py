
import os, sys, math, subprocess, json
import torch

# Paths to the two C++ sources (no-cache vs cache)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_NOCACHE = os.path.join(ROOT, "models/backend/converse2d_v1.cpp")
SRC_CACHE   = os.path.join(ROOT, "models/backend/converse2d_v3.cpp")

# Benchmark config
CASES = [
    # (B, C, H, W, scale, ksize)
    (1, 3, 128, 128, 2, 5),
    (2, 3, 256, 256, 2, 5),
    (4, 3, 256, 256, 2, 5),
    (2, 8, 256, 256, 2, 5),
    (1, 3, 512, 512, 2, 5),
]
WARMUP = 10
ITERS  = 50
DTYPE  = "float32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_single_bench(src_path, tag):
    # compile & run timings in a clean subprocess to avoid op name collisions
    child = f'''
import os, time, torch, json
from torch.utils.cpp_extension import load

torch.manual_seed(0)
device = "{DEVICE}"
dtype  = torch.{DTYPE}

ext = load(
    name="converse2d_ext",
    sources=[r\"\"\"{src_path}\"\"\"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3","-gencode","arch=compute_89,code=sm_89"],
)

op = torch.ops.converse2d.forward
clear = getattr(torch.ops.converse2d, "clear_cache", None)

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed(fn, warmup={WARMUP}, iters={ITERS}):
    for _ in range(warmup):
        fn()
    synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def bench_case(B,C,H,W,scale,ksize):
    x = torch.randn(B,C,H,W, device=device, dtype=dtype, requires_grad=True)
    x0 = x if scale == 1 else torch.nn.functional.interpolate(x, scale_factor=scale, mode="nearest")
    weight = torch.randn(1,C,ksize,ksize, device=device, dtype=dtype, requires_grad=False)
    weight = torch.nn.functional.softmax(weight.view(1,C,-1), dim=-1).view_as(weight)
    bias = torch.zeros(1,C,1,1, device=device, dtype=dtype, requires_grad=False)

    if clear is not None:
        clear()

    def fwd():
        with torch.no_grad():
            _ = op(x, x0, weight, bias, int(scale), float(1e-5))

    def train():
        x_local = x.detach().clone().requires_grad_(True)
        y = op(x_local, x0, weight, bias, int(scale), float(1e-5))
        loss = y.square().mean()
        loss.backward()

    t_f = timed(fwd)
    t_b = timed(train)
    t_f_hot = timed(fwd)   # hot cache
    t_b_hot = timed(train) # hot cache

    def tp(B,H,W,s,t): return (B*H*s*W*s / t) / 1e9

    return dict(
        shape=(B,C,H,W,scale,ksize),
        fwd_ms=t_f*1e3, bwd_ms=t_b*1e3, fwd_hot_ms=t_f_hot*1e3, bwd_hot_ms=t_b_hot*1e3,
        fwd_tp=tp(B,H,W,scale,t_f), bwd_tp=tp(B,H,W,scale,t_b),
        fwd_tp_hot=tp(B,H,W,scale,t_f_hot), bwd_tp_hot=tp(B,H,W,scale,t_b_hot),
    )

rows = []
for (B,C,H,W,s,k) in {CASES}:
    rows.append(bench_case(B,C,H,W,s,k))

print(json.dumps(dict(tag="{tag}", rows=rows), indent=2))
'''
    out = subprocess.check_output([sys.executable, "-c", child], text=True)
    return json.loads(out)

def main():
    if DEVICE != "cuda":
        print("[WARN] CUDA device not available; this script is intended for RTX 4090 tests.")
    res_nc = run_single_bench(SRC_NOCACHE, "nocache")
    res_cc = run_single_bench(SRC_CACHE,   "cache")

    def fmt_ms(x): return f"{x:7.2f}"
    print("=== Converse2D CUDA Backend: Cache vs No-Cache ===")
    print(f"[Env] torch={torch.__version__}, cuda={torch.version.cuda}, device={DEVICE}")
    print("case                      |  no‑cache fwd  cache fwd |  no‑cache bwd  cache bwd ||  fwd speedup  bwd speedup")
    print("-"*110)
    for r_nc, r_c in zip(res_nc["rows"], res_cc["rows"]):
        B,C,H,W,s,k = r_nc["shape"]
        tag = f"B{B} C{C} {H}x{W} s{s} k{k}"
        tf0, tb0 = r_nc["fwd_hot_ms"], r_nc["bwd_hot_ms"]
        tf1, tb1 = r_c["fwd_hot_ms"],  r_c["bwd_hot_ms"]
        sp_f = tf0 / tf1 if tf1 > 0 else float('nan')
        sp_b = tb0 / tb1 if tb1 > 0 else float('nan')
        print(f"{tag:24s} |   {fmt_ms(tf0)}   {fmt_ms(tf1)} |    {fmt_ms(tb0)}   {fmt_ms(tb1)} ||     {sp_f:5.2f}×       {sp_b:5.2f}×")

    # Geometric mean speedups
    sps_f, sps_b = [], []
    for r_nc, r_c in zip(res_nc["rows"], res_cc["rows"]):
        tf0, tb0 = r_nc["fwd_hot_ms"], r_nc["bwd_hot_ms"]
        tf1, tb1 = r_c["fwd_hot_ms"],  r_c["bwd_hot_ms"]
        sps_f.append(tf0/tf1); sps_b.append(tb0/tb1)
    def gmean(a):
        a = [x for x in a if x>0 and math.isfinite(x)]
        return math.exp(sum(math.log(x) for x in a)/len(a)) if a else float('nan')
    print("-"*110)
    print(f"Geomean speedup: Forward {gmean(sps_f):.2f}×, Backward {gmean(sps_b):.2f}×")

if __name__ == "__main__":
    main()
