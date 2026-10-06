"""Memory, DRAM traffic and FLOP efficiency of fused vs production training, against hardware peaks.

FlashAttention-style accounting for each case in flash_study.CASES and each eligible path:

Memory (PyTorch allocator, inputs and upstream gradient excluded):
- saved: bytes alive after the forward beyond its output's storage, i.e. what autograd
  keeps for backward;
- peak: maximum allocated over forward + VJP.

FLOPs (5 N log2 N per complex N-point 2-D transform, the FFTW convention; pointwise
arithmetic is not counted):
- model: the standard algorithm's transforms, counted once for every path, so
  model GFLOP/s compares paths by useful work (recomputation is not credited);
- fused_hw: model plus the fused backward's on-chip recomputation of FFT(x) (and FFT(x0)).

Hardware peaks: FP32 = SMs x FP32 lanes/SM x 2 x max SM clock; DRAM = 2 x memory clock x
bus width; `copy` is a measured device-to-device copy (read + write bytes / time).

With --ncu (Nsight Compute, application replay, no cache flush, no clock locking), one
forward+VJP call per case/path in a worker process gives per-kernel DRAM bytes and executed
FP32 FLOPs (fadd + fmul + 2 x ffma, all threads). Rates divide these by the uninstrumented
CUDA-event forward+VJP time: counting SASS instructions instruments the kernels, and the
kernel durations ncu records alongside are 10-40x too long (kept as instrumented_time_s).
--render <dir> recomputes rates and the table from an existing efficiency.json.
Writes <output>/efficiency.json and <output>/efficiency.md. Research only.
"""
import argparse
import csv
import gc
import glob
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import study
import flash_study as fs

FP32_LANES = {(7, 0): 64, (7, 5): 64, (8, 0): 64}  # everything newer listed here has 128
# dram__bytes is named alike on every chip; read/write splits are not (dram__bytes_read on
# GA100, dram__bytes_op_read on GB206), and ncu drops unknown names silently.
METRICS = ['dram__bytes.sum', 'gpu__time_duration.sum',
           'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
           'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
           'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum']
NCU_CASES = ['forward_s1_b4_c128_100', 'circular_s1_b4_c64_96_pad2', 'forward_s2_b4_c64_64',
             'forward_s2_b4_c64_48', 'forward_s3_b2_c32_48', 'forward_s3_b2_c32_32']


def fft_flops(n):
    return 5 * n * math.log2(n)


def model_flops(b, c, h, w, scale, pad):
    if scale == 1:
        n = (h + 2 * pad) * (w + 2 * pad)
        # forward: kernel, input, output IFFT; backward: G, grad_x IFFT, kernel VJP
        standard = (4 * b * c + 2 * c) * fft_flops(n)
        return dict(standard=standard, fused_hw=standard + b * c * fft_flops(n))
    lo, hi = h * w, scale * scale * h * w
    standard = 2 * b * c * fft_flops(lo) + (4 * b * c + 2 * c) * fft_flops(hi)
    return dict(standard=standard, fused_hw=standard + b * c * (fft_flops(lo) + fft_flops(hi)))


def peaks():
    p = torch.cuda.get_device_properties(0)
    lanes = FP32_LANES.get((p.major, p.minor), 128)
    fp32 = p.multi_processor_count * lanes * 2 * p.clock_rate * 1e3
    dram = 2 * p.memory_clock_rate * 1e3 * p.memory_bus_width / 8
    x = torch.empty(256 * 2 ** 20, dtype=torch.float32, device='cuda')  # 1 GiB
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        y.copy_(x)
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) / 1e3)
    copy = 2 * x.numel() * 4 / statistics.median(times)
    del x, y
    torch.cuda.empty_cache()
    return dict(fp32_flops=fp32, dram_bytes_per_s=dram, copy_bytes_per_s=copy, sms=p.multi_processor_count,
                fp32_lanes_per_sm=lanes, sm_clock_hz=p.clock_rate * 1e3, memory_clock_hz=p.memory_clock_rate * 1e3,
                bus_width_bits=p.memory_bus_width, l2_bytes=p.L2_cache_size)


def setup(name):
    b, c, h, w, scale, pad = fs.CASES[name]
    torch.manual_seed(76001)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    x0 = torch.randn(b, c, h * scale, w * scale, device='cuda', requires_grad=scale > 1)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(b, c, h * scale, w * scale, device='cuda')
    inputs = (x, weight, bias) if scale == 1 else (x, x0, weight, bias)
    return (x, x0, weight, bias, scale, pad), inputs, upstream


def path_fn(ext, path):
    return (lambda *a: fs.flash(ext, *a)) if path == 'fused' else fs.production


def memory(fn, args, inputs, upstream):
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*args)
    torch.cuda.synchronize()
    saved = torch.cuda.memory_allocated() - base - out.untyped_storage().nbytes()
    grads = torch.autograd.grad(out, inputs, upstream)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out, grads
    return dict(saved_bytes=saved, peak_bytes=peak)


def worker(name, path):
    """One forward+VJP inside cudaProfilerStart/Stop for ncu."""
    study.load_extension()
    ext = study.build()
    args, inputs, upstream = setup(name)
    fn = path_fn(ext, path)
    for _ in range(3):
        torch.autograd.grad(fn(*args), inputs, upstream)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.autograd.grad(fn(*args), inputs, upstream)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def find_ncu():
    found = shutil.which('ncu')
    if found:
        return found
    patterns = ['/usr/local/cuda/bin/ncu', '/opt/nvidia/nsight-compute/*/ncu',
                'C:/Program Files/NVIDIA Corporation/Nsight Compute */target/windows-desktop-win7-x64/ncu.exe']
    hits = sorted(h for p in patterns for h in glob.glob(p))
    return hits[-1] if hits else None


def ncu_measure(ncu, name, path, out_dir):
    rep = out_dir / f'ncu_{name}_{path}'
    command = [ncu, '--target-processes', 'all', '--profile-from-start', 'off', '--replay-mode', 'application',
               '--app-replay-mode', 'strict', '--app-replay-match', 'grid', '--cache-control', 'none',
               '--clock-control', 'none', '--metrics', ','.join(METRICS), '--export', str(rep), '-f',
               sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name, path]
    with (out_dir / f'ncu_{name}_{path}.log').open('w', encoding='utf-8') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        return dict(error=f'ncu exit {code}; see ncu_{name}_{path}.log')
    reports = sorted(glob.glob(str(rep) + '.ncu-rep*'))  # .ncu-rep, or .ncu-repz since Nsight Compute 2026
    if not reports:
        return dict(error=f'no report; see ncu_{name}_{path}.log')
    raw = subprocess.run([ncu, '--import', reports[0], '--csv', '--page', 'raw', '--print-units', 'base'],
                         capture_output=True, text=True, errors='replace').stdout
    if '"ID"' not in raw:
        return dict(error='ncu --import produced no CSV')
    rows = list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])))[1:]  # skip the units row
    missing = [m for m in METRICS if m not in rows[0]] if rows else METRICS
    if missing:
        return dict(error=f'metrics not collected: {missing}')
    value = lambda row, m: float(row[m].replace(',', '') or 0)
    kernels = [dict(name=r['Kernel Name'][:120], **{m: value(r, m) for m in METRICS}) for r in rows]
    total = {m: sum(k[m] for k in kernels) for m in METRICS}
    flops = (total['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum'] +
             total['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum'] +
             2 * total['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum'])
    return dict(kernels=len(kernels), instrumented_time_s=total['gpu__time_duration.sum'] * 1e-9,
                dram_bytes=total['dram__bytes.sum'],
                fp32_flops=flops, per_kernel=kernels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cases', nargs='*', default=list(fs.CASES))
    parser.add_argument('--ncu', action='store_true', help='also collect DRAM bytes and FP32 FLOPs with Nsight Compute')
    parser.add_argument('--ncu-cases', nargs='*', default=NCU_CASES)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--render', type=Path, help='recompute rates/table of an existing output directory')
    parser.add_argument('--worker', nargs=2, metavar=('CASE', 'PATH'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(*args.worker)
    if args.render:
        return render(args.render)
    if args.output is None or args.output.exists():
        raise SystemExit('give a fresh --output directory')
    args.output.mkdir(parents=True)
    study.load_extension()
    ext = study.build()
    device = torch.cuda.current_device()
    report = dict(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  torch=torch.__version__, cuda=torch.version.cuda, smem_optin_bytes=ext.smem_capacity(device),
                  peaks=peaks(), started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cases={})
    pk = report['peaks']
    print(f"{report['device']}: FP32 peak {pk['fp32_flops'] / 1e12:.1f} TFLOP/s, DRAM peak "
          f"{pk['dram_bytes_per_s'] / 1e9:.0f} GB/s, measured copy {pk['copy_bytes_per_s'] / 1e9:.0f} GB/s", flush=True)
    ncu = find_ncu() if args.ncu else None
    if args.ncu and not ncu:
        print('ncu not found; skipping hardware counters', flush=True)
    report['ncu'] = ncu
    for name in args.cases:
        b, c, h, w, scale, pad = fs.CASES[name]
        call, inputs, upstream = setup(name)
        paths = ['production'] + (['fused'] if ext.supported(h, w, scale, pad, device) else [])
        flops = model_flops(b, c, h, w, scale, pad)
        row = report['cases'][name] = dict(paths=paths, model_flops=flops)
        fns = {p: path_fn(ext, p) for p in paths}
        timing = paired_all(fns, call, inputs, upstream, args.rounds, args.iters)
        for p in paths:
            r = row[p] = dict(memory(fns[p], call, inputs, upstream), **timing[p])
            t = r['fwd_vjp_s']
            r['model_flops_per_s'] = flops['standard'] / t
            r['model_pct_fp32_peak'] = 100 * r['model_flops_per_s'] / pk['fp32_flops']
            if p == 'fused':
                r['fft_hw_flops_per_s'] = flops['fused_hw'] / t
            if ncu and name in args.ncu_cases:
                n = r['ncu'] = ncu_measure(ncu, name, p, args.output)
                derive(r, pk)
        line = ' | '.join(
            f"{p} {row[p]['fwd_vjp_s'] * 1e3:6.2f} ms saved {row[p]['saved_bytes'] / 2 ** 20:6.1f} MiB peak "
            f"{row[p]['peak_bytes'] / 2 ** 20:6.1f} MiB model {row[p]['model_pct_fp32_peak']:4.1f}% FP32"
            + (f" DRAM {row[p]['ncu']['dram_bytes'] / 1e6:6.0f} MB ({row[p]['ncu']['dram_pct_peak']:4.1f}% BW) "
               f"FP32 {row[p]['ncu']['fp32_pct_peak']:4.1f}%" if 'dram_pct_peak' in row[p].get('ncu', {}) else '')
            for p in paths)
        print(f'{name:28s} {line}', flush=True)
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / 'efficiency.json').write_text(json.dumps(report, indent=1))
    (args.output / 'efficiency.md').write_text(markdown(report), encoding='utf-8')
    print(markdown(report))


def derive(r, pk):
    """ncu counts over the uninstrumented forward+VJP time."""
    n = r.get('ncu', {})
    if 'dram_bytes' in n:
        n.pop('kernel_time_s', None)
        n['dram_bytes_per_s'] = n['dram_bytes'] / r['fwd_vjp_s']
        n['dram_pct_peak'] = 100 * n['dram_bytes_per_s'] / pk['dram_bytes_per_s']
        n['fp32_flops_per_s'] = n['fp32_flops'] / r['fwd_vjp_s']
        n['fp32_pct_peak'] = 100 * n['fp32_flops_per_s'] / pk['fp32_flops']


def render(directory):
    path = directory / 'efficiency.json'
    report = json.loads(path.read_text())
    for row in report['cases'].values():
        for p in row['paths']:
            n = row[p].get('ncu', {})
            if 'kernel_time_s' in n:  # reports written before the timing fix
                n['instrumented_time_s'] = n['kernel_time_s']
            derive(row[p], report['peaks'])
    path.write_text(json.dumps(report, indent=1))
    (directory / 'efficiency.md').write_text(markdown(report), encoding='utf-8')
    print(markdown(report))


def paired_all(fns, call, inputs, upstream, rounds, iters):
    def vjp(fn):
        return lambda: torch.autograd.grad(fn(*call), inputs, upstream)

    def fwd(fn):
        return lambda: fn(*call)

    timed = {f'{p}:fwd': fwd(fn) for p, fn in fns.items()}
    timed.update({f'{p}:vjp': vjp(fn) for p, fn in fns.items()})
    t = paired(timed, rounds, iters)
    return {p: dict(fwd_s=t[f'{p}:fwd'] * 1e-6, fwd_vjp_s=t[f'{p}:vjp'] * 1e-6) for p in fns}


def paired(fns, rounds, iters):
    result = study.paired(fns, rounds, iters)
    return {k: v['median_us'] for k, v in result.items()}


def markdown(report):
    pk = report['peaks']
    lines = [f"# Memory and FLOP efficiency: {report['device']} (sm_{''.join(map(str, report['capability']))})", '',
             f"- torch {report['torch']} / CUDA {report['cuda']}",
             f"- FP32 peak {pk['fp32_flops'] / 1e12:.1f} TFLOP/s ({pk['sms']} SMs x {pk['fp32_lanes_per_sm']} lanes x 2 x "
             f"{pk['sm_clock_hz'] / 1e9:.2f} GHz); DRAM peak {pk['dram_bytes_per_s'] / 1e9:.0f} GB/s; measured copy "
             f"{pk['copy_bytes_per_s'] / 1e9:.0f} GB/s; L2 {pk['l2_bytes'] / 2 ** 20:.0f} MiB",
             f"- ncu: {report['ncu'] or 'not used'}", '',
             '| case | path | fwd ms | fwd+VJP ms | saved MiB | peak MiB | model FFT GFLOP/s (% FP32) | DRAM MB/call | '
             'DRAM GB/s (% peak) | measured FP32 GFLOP/s (% peak) |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, row in report['cases'].items():
        for p in row['paths']:
            r = row[p]
            n = r.get('ncu', {})
            hw = (f"{n['dram_bytes'] / 1e6:.0f} | {n['dram_bytes_per_s'] / 1e9:.0f} ({n['dram_pct_peak']:.0f}%) | "
                  f"{n['fp32_flops_per_s'] / 1e9:.0f} ({n['fp32_pct_peak']:.1f}%)" if 'dram_pct_peak' in n else
                  (n.get('error', '-') + ' | - | -' if n else '- | - | -'))
            lines.append(f"| {name} | {p} | {r['fwd_s'] * 1e3:.2f} | {r['fwd_vjp_s'] * 1e3:.2f} | "
                         f"{r['saved_bytes'] / 2 ** 20:.1f} | {r['peak_bytes'] / 2 ** 20:.1f} | "
                         f"{r['model_flops_per_s'] / 1e9:.0f} ({r['model_pct_fp32_peak']:.1f}%) | {hw} |")
    lines += ['', 'Model FLOPs count the standard algorithm\'s FFTs (5 N log2 N) once for every path; fused '
              'recomputation is not credited. DRAM bytes and executed FP32 FLOPs are ncu counters over one '
              'forward+VJP call (all kernels, including ATen kernel-FFT preparation), divided by the '
              'uninstrumented CUDA-event forward+VJP time. Operator level only.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
