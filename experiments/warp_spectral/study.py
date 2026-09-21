"""Isolated scale-2 FP32 spectral study; never installs/changes the operator.

Run in a CUDA/MSVC developer environment: python experiments/warp_spectral/study.py
Raw results, source hashes and clock traces are written under artifacts/warp_spectral.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[2]
import sys as _layout_sys
_layout_sys.path.insert(0,str(ROOT/"test"))
from extension_loader import legacy_source_texts, production_source_hashes

HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "warp_spectral"
NAMES = ["two_pass", "thread_fused", "warp_cooperative", "warp_specialized"]


def load():
    build = ROOT / ".build" / "warp_spectral"
    build.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    revision = "-DWARP_PRODUCTION_REV=0x" + hashlib.sha256(
        json.dumps(production_source_hashes(),sort_keys=True).encode()).hexdigest()[:12]
    cpp_extension.load(
        name="warp_spectral_experiment", sources=[str(HERE / "bindings.cpp"),str(HERE / "warp_spectral.cu")],
        extra_cflags=["/O2", "/std:c++17"] if os.name=="nt" else ["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v",revision],
        with_cuda=True, is_python_module=False, build_directory=str(build), verbose=True,
    )


def inputs(B, C, H, W, KB=1, KC=None, lam=1e-3):
    KC = C if KC is None else KC
    y = torch.randn(B, C, H, W, device="cuda")
    p = torch.randn(B, C, 2*H, 2*W, device="cuda")
    k = torch.randn(KB, KC, 2*H, 2*W, device="cuda") / (2*(H*W)**0.5)
    l = torch.full((C,), lam, device="cuda")
    spectra = tuple(torch.fft.rfft2(t).contiguous() for t in (y, p, k))
    return (y, p, k, l), (*spectra, l, H, W)


def reference(real):
    y, p, k, l = real
    H, W = y.shape[-2:]
    Y, P, K = [torch.fft.fft2(x.double()) for x in (y, p, k)]
    prediction = torch.zeros_like(Y)
    power = torch.zeros_like(K[..., :H, :W].real)
    for a in range(2):
        for b in range(2):
            kk = K[..., a*H:(a+1)*H, b*W:(b+1)*W]
            pp = P[..., a*H:(a+1)*H, b*W:(b+1)*W]
            prediction += kk * pp
            power += kk.abs().square()
    q = (Y - prediction / 4) / (power / 4 + l.double()[None, :, None, None])
    return (P + K.conj() * q.repeat(1, 1, 2, 2))[..., :W+1]


def validate():
    rows = []
    # Singleton/odd/even dimensions, partial warps, all broadcast combinations,
    # and two regularization magnitudes. This is an inference-only experiment.
    cases = [(1, 1, 1, 1), (1, 2, 1, 2), (1, 2, 2, 3),
             (1, 3, 5, 7), (2, 3, 7, 8), (2, 3, 8, 9),
             (1, 4, 31, 33), (1, 4, 32, 32)]
    for B, C, H, W in cases:
        for KB, KC in sorted(set([(1, 1), (1, C), (B, 1), (B, C)])):
            for lam in (1e-5, 1e-2):
                real, args = inputs(B, C, H, W, KB, KC, lam)
                ref = reference(real)
                base = torch.ops.warp_spectral.run(*args, 0)
                ref_image = torch.fft.irfft2(ref, s=(2*H, 2*W))
                for mode in range(4):
                    got = torch.ops.warp_spectral.run(*args, mode)
                    torch.testing.assert_close(got, base, rtol=3e-5, atol=3e-4)
                    # Compare relative norm as well as per-element tolerances;
                    # FFT near-zero bins make max relative error misleading.
                    rel = (torch.linalg.vector_norm(got.to(torch.complex128) - ref)
                           / torch.linalg.vector_norm(ref)).item()
                    assert rel < 2e-5, (B,C,H,W,KB,KC,lam,mode,rel)
                    image = torch.fft.irfft2(got, s=(2*H,2*W))
                    torch.testing.assert_close(image.double(), ref_image, rtol=3e-4, atol=3e-5)
                    rows.append(dict(shape=[B,C,H,W], broadcast=[KB,KC], lam=lam,
                                     mode=NAMES[mode], max_vs_baseline=(got-base).abs().max().item(),
                                     relative_l2_vs_fp64=rel,
                                     image_max_abs_vs_fp64=(image-ref_image).abs().max().item()))
    return rows


def timed(fn, repeats):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(repeats):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def capture(fn, batch):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            value = fn()
    return graph, value


def benchmark(rounds, repeats):
    rows = []
    for B,C,H,W,KB,KC in [(1,3,32,40,1,1), (1,64,128,128,1,64),
                          (1,32,127,129,1,32), (8,32,128,128,8,32)]:
        real, args = inputs(B,C,H,W,KB,KC)
        ref = torch.ops.warp_spectral.run(*args,0)
        for mode in range(1,4):
            torch.testing.assert_close(torch.ops.warp_spectral.run(*args,mode),ref,rtol=3e-5,atol=3e-4)
        for scope in ("spectral_only", "fft_solve_ifft"):
            def make(mode):
                if scope == "spectral_only":
                    return lambda: torch.ops.warp_spectral.run(*args,mode)
                def pipeline():
                    y,p,k,l = real
                    fy,fp,fk = [torch.fft.rfft2(x).contiguous() for x in (y,p,k)]
                    solved = torch.ops.warp_spectral.run(fy,fp,fk,l,H,W,mode)
                    return torch.fft.irfft2(solved,s=(2*H,2*W))
                return pipeline
            batch = 16
            held = [capture(make(mode), batch) for mode in range(4)]
            calls = [g.replay for g,_ in held]
            samples = [[] for _ in calls]
            for _ in range(8):
                for fn in calls: fn()
            for rep in range(rounds):
                # Rotate order so the same mode is not always first or last.
                order = [(rep+j)%4 for j in range(4)]
                for i in order:
                    samples[i].append(timed(calls[i],repeats)/batch)
            medians = [statistics.median(s) for s in samples]
            row = dict(shape=[B,C,H,W],broadcast=[KB,KC],scope=scope,
                       graph_replay=True, calls_per_graph=batch, units="us", medians=dict(zip(NAMES,medians)),
                       samples=dict(zip(NAMES,samples)),
                       speedup_vs_two_pass=dict(zip(NAMES,[medians[0]/x for x in medians])))
            rows.append(row)
            print(json.dumps(row),flush=True)
    return rows


def traces():
    # W=65 => 32 interior columns, H=16 => exactly 16 iterations / one CTA.
    _, args = inputs(1,1,16,65,1,1)
    results=[]
    for i in range(5):
        raw=torch.ops.warp_spectral.trace(*args,16).cpu().numpy()
        # Keep uint32 bits; analyzer unwraps relative to a known first event.
        np.savez(OUT/f"trace_{i}.npz",prof=raw,meta=json.dumps(dict(single_cta=True,iterations=16)))
        results.append(f"trace_{i}.npz")
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rounds",type=int,default=7)
    ap.add_argument("--repeats",type=int,default=20)
    ap.add_argument("--check-only",action="store_true")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(916)
    load()
    result=dict(date=time.strftime("%Y-%m-%d %H:%M:%S"),torch=torch.__version__,
                cuda_runtime=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),
                os=platform.platform(),clock_policy="unchanged; desktop WDDM workload",
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [HERE/"study.py",HERE/"bindings.cpp",HERE/"warp_spectral.cu",HERE/"specialized.cuh",
                            ROOT/"Converse2D/torch_converse2d/converse2d_kernels.cu"]},
                nvcc=subprocess.check_output([str(Path(cpp_extension.CUDA_HOME)/"bin"/("nvcc.exe" if os.name=="nt" else "nvcc")),"--version"],text=True))
    result["production_source_sha256"]=production_source_hashes()
    result["correctness"]=validate()
    print(f"Correctness passed: {len(result['correctness'])} comparisons",flush=True)
    if not args.check_only:
        result["benchmark"]=benchmark(args.rounds,args.repeats)
        result["traces"]=traces()
    (OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    with torch.inference_mode():
        main()
