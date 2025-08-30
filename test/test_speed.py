import time
import math
import torch
from util_converse import Converse2D

# --------------------------
# Config
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [torch.float32]            
USE_AUTOCast = False                
WARMUP = 10
ITERS  = 50
TRAIN  = True                      
INFER  = True                       
CASES = [
    # (B, C, H, W, scale, ksize, padding, padding_mode)
    (1,   3,  128, 128, 2, 5, 2, "circular"),
    (2,   3,  256, 256, 2, 5, 2, "circular"),
    (4,   3,  256, 256, 2, 5, 2, "circular"),
    (2,   8,  256, 256, 2, 5, 2, "circular"),
    (1,   3,  512, 512, 2, 5, 2, "circular"), # 320
]

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def make_model(C, scale, ksize=5, padding=2, padding_mode="circular", dtype=torch.float32):
    m = Converse2D(
        in_channels=C, out_channels=C, kernel_size=ksize,
        scale=scale, padding=padding, padding_mode=padding_mode,
        eps=1e-5, backend="pytorch" 
    ).to(device=device, dtype=dtype)
    m.eval()
    return m

def clone_as_cuda_backend(m):
    m2 = Converse2D(
        in_channels=m.in_channels, out_channels=m.out_channels,
        kernel_size=m.kernel_size, scale=m.scale, padding=m.padding,
        padding_mode=m.padding_mode, eps=m.eps, backend="cuda"
    ).to(device=device, dtype=next(m.parameters()).dtype)
    m2.load_state_dict(m.state_dict())
    m2.eval()
    return m2

def tp_gpix_per_s(B,H,W,s,t):
    if t is None or t <= 0: return None
    return (B * (H*s) * (W*s) / t) / 1e9

def speedup_and_pct(t_py, t_cu):
    if t_py and t_cu and t_py > 0 and t_cu > 0:
        sp = t_py / t_cu
        pct = (t_py - t_cu) / t_py * 100.0
        return sp, pct
    return None, None

def fmt_ms(t):   return "-" if t is None else f"{t*1e3:7.2f}"
def fmt_tp(x):   return "-" if x is None else f"{x:6.3f}"
def fmt_sp(x):   return "-" if x is None else f"{x:5.2f}×"
def fmt_pct(p):  return "-" if p is None else f"{p:6.1f}%"

def geom_mean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals: return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

def run_case(B,C,H,W,scale,ksize,padding,padding_mode,dtype):
    x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=TRAIN)

    torch.manual_seed(0); m_py = make_model(C, scale, ksize, padding, padding_mode, dtype)
    torch.manual_seed(0); m_cu = clone_as_cuda_backend(m_py)

    fwd_py = fwd_cu = None
    if INFER:
        def fwd_run(m):
            def _call():
                with torch.no_grad():
                    if USE_AUTOCast and dtype is torch.bfloat16 and device == "cuda":
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            _ = m(x)
                    else:
                        _ = m(x)
            return _call
        fwd_py = timed_run(fwd_run(m_py))
        fwd_cu = timed_run(fwd_run(m_cu))

    bwd_py = bwd_cu = None
    if TRAIN:
        def train_run(m):
            def _call():
                x_local = x.detach().clone().requires_grad_(True)
                if USE_AUTOCast and dtype is torch.bfloat16 and device == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        y = m(x_local); loss = y.square().mean()
                else:
                    y = m(x_local); loss = y.square().mean()
                loss.backward()
            return _call
        bwd_py = timed_run(train_run(m_py))
        bwd_cu = timed_run(train_run(m_cu))

    fwd_tp_py = tp_gpix_per_s(B,H,W,scale,fwd_py)
    fwd_tp_cu = tp_gpix_per_s(B,H,W,scale,fwd_cu)
    bwd_tp_py = tp_gpix_per_s(B,H,W,scale,bwd_py)
    bwd_tp_cu = tp_gpix_per_s(B,H,W,scale,bwd_cu)

    fwd_sp, fwd_pct = speedup_and_pct(fwd_py, fwd_cu)
    bwd_sp, bwd_pct = speedup_and_pct(bwd_py, bwd_cu)

    return {
        "shape": (B,C,H,W,scale,ksize,padding,padding_mode,str(dtype).split('.')[-1]),
        "fwd_py": fwd_py, "fwd_cu": fwd_cu, "fwd_tp_py": fwd_tp_py, "fwd_tp_cu": fwd_tp_cu,
        "bwd_py": bwd_py, "bwd_cu": bwd_cu, "bwd_tp_py": bwd_tp_py, "bwd_tp_cu": bwd_tp_cu,
        "fwd_sp": fwd_sp, "fwd_pct": fwd_pct, "bwd_sp": bwd_sp, "bwd_pct": bwd_pct
    }

def main():
    print(f"[Env] device={device}, torch={torch.__version__}, cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}")
    print(f"[Cfg] dtypes={[d.__str__() for d in DTYPES]}, AMP={USE_AUTOCast}, TRAIN={TRAIN}, INFER={INFER}, warmup={WARMUP}, iters={ITERS}\n")

    rows = []
    for dtype in DTYPES:
        for (B,C,H,W,s,ks,pd,pm) in CASES:
            rows.append(run_case(B,C,H,W,s,ks,pd,pm,dtype))

    print("=== Per‑case Comparison (CUDA vs PyTorch) ===")
    for r in rows:
        B,C,H,W,s,ks,pd,pm,dtype = r["shape"]
        tag = f"[{dtype}] B{B} C{C} {H}x{W} s{s} k{ks}"
        # forward
        if INFER:
            print(f"{tag} | Forward : Py {fmt_ms(r['fwd_py'])} ms ({fmt_tp(r['fwd_tp_py'])} Gpix/s) "
                  f"vs CUDA {fmt_ms(r['fwd_cu'])} ms ({fmt_tp(r['fwd_tp_cu'])} Gpix/s) "
                  f"-> CUDA is {fmt_sp(r['fwd_sp'])} faster ({fmt_pct(r['fwd_pct'])} time saved)")
        # backward
        if TRAIN:
            print(f"{tag} | Train   : Py {fmt_ms(r['bwd_py'])} ms ({fmt_tp(r['bwd_tp_py'])} Gpix/s) "
                  f"vs CUDA {fmt_ms(r['bwd_cu'])} ms ({fmt_tp(r['bwd_tp_cu'])} Gpix/s) "
                  f"-> CUDA is {fmt_sp(r['bwd_sp'])} faster ({fmt_pct(r['bwd_pct'])} time saved)")
    print("")

    hdr = ("dtype   B   C     H     W  s   k |  fwd_py(ms)  fwd_cu(ms)  fwd_Gpix/s(py)  fwd_Gpix/s(cu)  fwd_speedup "
           "|  bwd_py(ms)  bwd_cu(ms)  bwd_Gpix/s(py)  bwd_Gpix/s(cu)  bwd_speedup")
    print(hdr); print("-"*len(hdr))
    for r in rows:
        B,C,H,W,s,ks,pd,pm,dtype = r["shape"]
        line = (f"{dtype:6s} {B:3d} {C:3d} {H:5d} {W:5d} {s:2d} {ks:3d} | "
                f"{fmt_ms(r['fwd_py'])}    {fmt_ms(r['fwd_cu'])}      {fmt_tp(r['fwd_tp_py'])}            {fmt_tp(r['fwd_tp_cu'])}        {fmt_sp(r['fwd_sp'])} | "
                f"{fmt_ms(r['bwd_py'])}    {fmt_ms(r['bwd_cu'])}      {fmt_tp(r['bwd_tp_py'])}            {fmt_tp(r['bwd_tp_cu'])}        {fmt_sp(r['bwd_sp'])}")
        print(line)

    fwd_sps = [r["fwd_sp"] for r in rows if r["fwd_sp"]]
    bwd_sps = [r["bwd_sp"] for r in rows if r["bwd_sp"]]
    gm_fwd = geom_mean(fwd_sps)
    gm_bwd = geom_mean(bwd_sps)
    if gm_fwd:
        print(f"\nOverall Forward Geomean Speedup : {gm_fwd:.2f}× (CUDA vs PyTorch)")
    if gm_bwd:
        print(f"Overall Train   Geomean Speedup : {gm_bwd:.2f}× (CUDA vs PyTorch)")

if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
