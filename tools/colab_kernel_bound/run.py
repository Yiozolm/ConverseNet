"""Re-check the v3.0.0 scale1_forward bound classification on another GPU (Linux/Colab).

The RTX 5060 Ti capture (tools/v4_campaign/README.md on codex/v4.0.0) measured
scale1_forward at B4/C128/100x100, shared prior, kernel [1,128,3,3]:
DRAM 88.33652% / 395.358185 GB/s, SM 31.234921%. This script repeats that
workload on the current GPU:

1. checked build of the unmodified v3.0.0 production sources,
2. counter-free probes: device copy bandwidth and FP32 GEMM rate (TF32 off),
3. torch.profiler kernel timing; scale1_forward achieved bandwidth from a
   lower-bound byte count, compared with the copy probe,
4. Nsight Compute: (A) the original capture (scale1_forward, --set full,
   launch-count 1, default replay/cache/clock control) and (B) SpeedOfLight
   for every kernel in the same training call.

Counter permission failures (ERR_NVGPUCTRPERM) are recorded, never hidden;
step 3 then remains the only evidence. Profiler durations are not benchmarks.
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3_COMMIT = '0d636215e23d472825e09cc340a99f1d2ae2b78c'
REFERENCE_5060TI = dict(gpu='NVIDIA GeForce RTX 5060 Ti', kernel='scale1_forward', dram_pct=88.33652,
                        dram_gbps=395.358185, sm_pct=31.234921, source='codex/v4.0.0 tools/v4_campaign/README.md')
# Raw-page names first, older/alternative names after. Missing metrics stay None.
METRICS = {
    'duration_ns': ('gpu__time_duration.sum',),
    'sm_pct': ('sm__throughput.avg.pct_of_peak_sustained_elapsed',),
    'memory_pct': ('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',),
    'dram_pct': ('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'dram__throughput.avg.pct_of_peak_sustained_elapsed'),
    'l2_pct': ('lts__throughput.avg.pct_of_peak_sustained_elapsed',),
    'l1_pct': ('l1tex__throughput.avg.pct_of_peak_sustained_active',),
    'dram_bytes_per_s': ('dram__bytes.sum.per_second',),
    'dram_read_bytes': ('dram__bytes_read.sum', 'dram__bytes_op_read.sum'),
    'dram_write_bytes': ('dram__bytes_write.sum', 'dram__bytes_op_write.sum'),
    'achieved_occupancy_pct': ('sm__warps_active.avg.pct_of_peak_sustained_active',),
}
EXTRA_B = ('dram__bytes_read.sum', 'dram__bytes_write.sum', 'dram__bytes.sum.per_second')
HIGH = 60.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return result.returncode, result.stdout.strip()


def environment():
    import torch
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit',
                          '--format=csv'], capture_output=True, text=True).stdout.strip()
    nvcc = shutil.which('nvcc')
    head = git('rev-parse', 'HEAD')[1]
    # Production sources, models and loader must be byte-identical to v3.0.0.
    diff_code, _ = git('diff', '--quiet', V3_COMMIT, '--', 'Converse2D', 'models', 'test')
    return dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
                gpu=props.name, capability=f'{props.major}.{props.minor}', sm_count=props.multi_processor_count,
                total_memory=props.total_memory, l2_cache_bytes=getattr(props, 'L2_cache_size', None),
                nvidia_smi=smi, nvcc=nvcc, nvcc_version=subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout.strip() if nvcc else None,
                git_head=head, v3_commit=V3_COMMIT, production_matches_v3=diff_code == 0)


def build(env, output):
    with (output / 'build.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', str(ROOT / 'test/extension_loader.py')], env=env, cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'Build failed; see {output / "build.log"}')
    manifest = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text(encoding='utf-8'))
    return dict(binary_sha256=manifest['binary_sha256'], library=manifest['library'],
                torch_cuda_arch_list=env['TORCH_CUDA_ARCH_LIST'])


def cuda_time(fn, iters):
    import torch
    for _ in range(3):
        fn()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) / 1e3)
    return statistics.median(times)


def probes():
    """Practical ceilings for the timing-only comparison; not vendor peaks."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    nbytes = 1 << 30
    src = torch.empty(nbytes // 4, device='cuda')
    dst = torch.empty_like(src)
    copy_s = cuda_time(lambda: dst.copy_(src), 20)
    del src, dst
    n = 8192
    a, b = torch.randn(n, n, device='cuda'), torch.randn(n, n, device='cuda')
    gemm_s = cuda_time(lambda: a @ b, 10)
    del a, b
    torch.cuda.empty_cache()
    return dict(copy_bytes_per_s=2 * nbytes / copy_s, copy_buffer_bytes=nbytes,
                fp32_gemm_flop_per_s=2 * n ** 3 / gemm_s, gemm_n=n, tf32=False,
                note='Median CUDA-event timing; read+write bytes for copy, 2n^3 for GEMM.')


def timing(env, output, batch, copy_bytes_per_s):
    trace = output / 'timing_trace.json'
    with (output / 'timing.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', str(HERE / 'worker.py'), '--root', str(ROOT), '--mode', 'timing',
                                 '--batch', str(batch), '--trace', str(trace)], env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'Timing worker failed; see {output / "timing.log"}')
    events = json.loads(trace.read_text(encoding='utf-8'))['traceEvents']
    kernels = {}
    for event in events:
        if event.get('cat') == 'kernel':
            entry = kernels.setdefault(event['name'], dict(durations_us=[], grid=event['args'].get('grid'), block=event['args'].get('block')))
            entry['durations_us'].append(event['dur'])
    table = []
    for name, entry in kernels.items():
        d = entry.pop('durations_us')
        table.append(dict(name=name, calls=len(d), median_us=statistics.median(d), total_us=sum(d), **entry))
    table.sort(key=lambda row: -row['total_us'])
    scale1 = [row for row in table if 'scale1_forward' in row['name']]
    estimate = None
    if scale1:
        row = scale1[0]
        n = row['grid'][0] * row['block'][0]
        # Shared y/prior: read y, write out (complex64) per element; broadcast
        # k read and d written once per (C,H,W) plane. L2 hits only lower this.
        lower_bytes = n * 16 + n // batch * 12
        achieved = lower_bytes / (row['median_us'] * 1e-6)
        estimate = dict(kernel=row['name'], elements_from_grid=n, lower_bound_bytes=lower_bytes, median_us=row['median_us'],
                        achieved_bytes_per_s_lower_bound=achieved, fraction_of_copy_probe=achieved / copy_bytes_per_s,
                        note='Lower-bound bytes / kineto duration. Elements are grid*block (<= n+255).')
    return dict(kernels=table, scale1_forward=estimate, trace_sha256=sha(trace))


def find_ncu():
    found = shutil.which('ncu')
    if found:
        return found
    candidates = sorted(glob.glob('/usr/local/cuda*/bin/ncu') + glob.glob('/usr/local/cuda*/nsight-compute*/ncu') +
                        glob.glob('/opt/nvidia/nsight-compute/*/ncu'))
    return candidates[-1] if candidates else None


def parse_raw(path):
    rows = list(csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    start = next(i for i, row in enumerate(rows) if row and row[0] == 'ID' and 'Kernel Name' in row)
    header, units = rows[start], rows[start + 1]
    kernels = []
    for row in rows[start + 2:]:
        if not row or not re.fullmatch(r'\d+', row[0]):
            continue
        values = dict(zip(header, row))
        entry = dict(id=int(row[0]), name=values.get('Kernel Name'), grid=values.get('Grid Size'), block=values.get('Block Size'))
        for key, names in METRICS.items():
            entry[key] = None
            for name in names:
                if name in values and values[name] not in ('', 'n/a'):
                    try:
                        entry[key] = float(values[name].replace(',', ''))
                        entry[key + '_unit'] = units[header.index(name)]
                    except ValueError:
                        pass
                    break
        entry['classification'] = classify(entry)
        kernels.append(entry)
    return kernels


def classify(m):
    units = {name: m.get(key) for name, key in (('DRAM', 'dram_pct'), ('L2', 'l2_pct'), ('L1', 'l1_pct'), ('SM', 'sm_pct'))}
    units = {k: v for k, v in units.items() if v is not None}
    if not units:
        return None
    top = max(units, key=units.get)
    if units[top] < HIGH:
        return f'latency/occupancy-limited (no unit >= {HIGH:g}% of peak; top {top} {units[top]:.1f}%)'
    return f'{"compute" if top == "SM" else "memory"}-bound ({top} {units[top]:.1f}%)'


def ncu_capture(ncu, env, output, name, extra, batch):
    folder = output / name
    folder.mkdir()
    command = [ncu, '--target-processes', 'all', '--profile-from-start', 'off', *extra,
               '--export', str(folder / 'capture'), sys.executable, '-u', str(HERE / 'worker.py'),
               '--root', str(ROOT), '--mode', 'ncu', '--batch', str(batch)]
    with (folder / 'capture.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (folder / 'capture.log').read_text(encoding='utf-8', errors='replace')
    record = dict(command=command, returncode=result.returncode)
    report = folder / 'capture.ncu-rep'
    if 'ERR_NVGPUCTRPERM' in text:
        record['status'] = 'counters_unavailable'
        record['detail'] = 'ERR_NVGPUCTRPERM: this host does not permit GPU performance counters'
        return record
    if result.returncode or not report.exists():
        record['status'] = 'error'
        record['detail'] = text[-2000:]
        return record
    csv_path = folder / 'raw.csv'
    with csv_path.open('w', encoding='utf-8') as stream:
        imported = subprocess.run([ncu, '--import', str(report), '--page', 'raw', '--csv', '--print-units', 'base'], stdout=stream,
                                  stderr=subprocess.PIPE, text=True)
    if imported.returncode:
        record.update(status='error', detail=imported.stderr[-2000:])
        return record
    record.update(status='complete', report_sha256=sha(report), csv_sha256=sha(csv_path), kernels=parse_raw(csv_path))
    return record


def markdown(summary):
    env, lines = summary['environment'], []
    lines += [f'# scale1_forward bound check: {env["gpu"]} (sm_{env["capability"].replace(".", "")})', '',
              f'- torch {env["torch"]} / CUDA {env["torch_cuda"]}; production matches v3.0.0: {env["production_matches_v3"]}',
              f'- copy probe: {summary["probes"]["copy_bytes_per_s"] / 1e9:.1f} GB/s; FP32 GEMM probe: '
              f'{summary["probes"]["fp32_gemm_flop_per_s"] / 1e12:.2f} TFLOP/s (TF32 off)', '']
    ref = REFERENCE_5060TI
    lines += ['| source | DRAM % | DRAM GB/s | SM % | L2 % | duration us | classification |', '|---|---|---|---|---|---|---|',
              f'| RTX 5060 Ti (recorded) | {ref["dram_pct"]:.2f} | {ref["dram_gbps"]:.1f} | {ref["sm_pct"]:.2f} | - | - | memory-bound (DRAM) |']
    capture = summary['ncu'].get('A_scale1_full', {})
    for k in capture.get('kernels', []):
        fmt = lambda v, s=1.0, p=2: '-' if v is None else f'{v / s:.{p}f}'
        lines.append(f'| {env["gpu"]} NCU | {fmt(k["dram_pct"])} | {fmt(k["dram_bytes_per_s"], 1e9, 1)} | {fmt(k["sm_pct"])} | '
                     f'{fmt(k["l2_pct"])} | {fmt(k["duration_ns"], 1e3, 1)} | {k["classification"]} |')
    if capture.get('status') != 'complete':
        lines.append(f'| {env["gpu"]} NCU | {capture.get("status", "not run")} | | | | | |')
    est = summary['timing']['scale1_forward']
    if est:
        lines += ['', f'Timing-only (no counters): scale1_forward median {est["median_us"]:.1f} us, >= '
                  f'{est["achieved_bytes_per_s_lower_bound"] / 1e9:.1f} GB/s = {100 * est["fraction_of_copy_probe"]:.1f}% of the copy probe.']
    all_kernels = summary['ncu'].get('B_all_sol', {})
    if all_kernels.get('status') == 'complete':
        lines += ['', '## All kernels in one training call (SpeedOfLight)', '', '| kernel | duration us | DRAM % | L2 % | SM % | classification |',
                  '|---|---|---|---|---|---|']
        for k in all_kernels['kernels']:
            fmt = lambda v, s=1.0: '-' if v is None else f'{v / s:.1f}'
            lines.append(f'| `{k["name"][:70]}` | {fmt(k["duration_ns"], 1e3)} | {fmt(k["dram_pct"])} | {fmt(k["l2_pct"])} | '
                         f'{fmt(k["sm_pct"])} | {k["classification"]} |')
    lines += ['', f'Classification rule: highest SOL unit; < {HIGH:g}% of peak for every unit is reported as latency/occupancy-limited. '
              'NCU defaults lock clocks to base and flush caches between replays, as in the recorded 5060 Ti capture.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--skip-ncu', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    output = args.output.resolve()
    output.mkdir(parents=True)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA device; select a GPU runtime')
    major, minor = torch.cuda.get_device_capability()
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST', f'{major}.{minor}'),
               MAX_JOBS=os.environ.get('MAX_JOBS', str(os.cpu_count() or 2)), PYTHONDONTWRITEBYTECODE='1')
    summary = dict(kind='colab_kernel_bound', started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), batch=args.batch,
                   reference=REFERENCE_5060TI, environment=environment(), script_sha256={p.name: sha(p) for p in (HERE / 'run.py', HERE / 'worker.py')})
    if not summary['environment']['production_matches_v3']:
        print('WARNING: Converse2D/models/test differ from v3.0.0; results are not a v3.0.0 measurement', file=sys.stderr)
    print('Building extension (first build takes several minutes)...', flush=True)
    summary['build'] = build(env, output)
    print('Probing copy bandwidth and FP32 GEMM...', flush=True)
    summary['probes'] = probes()
    print('Kernel timing via torch.profiler...', flush=True)
    summary['timing'] = timing(env, output, args.batch, summary['probes']['copy_bytes_per_s'])
    summary['ncu'] = {}
    ncu = None if args.skip_ncu else find_ncu()
    summary['ncu_path'] = ncu
    if ncu:
        summary['ncu_version'] = subprocess.run([ncu, '--version'], capture_output=True, text=True).stdout.strip()
        print('NCU capture A: scale1_forward, --set full...', flush=True)
        summary['ncu']['A_scale1_full'] = ncu_capture(ncu, env, output, 'A_scale1_full',
            ['--set', 'full', '--kernel-name', 'regex:.*scale1_forward.*', '--launch-count', '1'], args.batch)
        if summary['ncu']['A_scale1_full']['status'] != 'counters_unavailable':
            print('NCU capture B: SpeedOfLight for every kernel...', flush=True)
            summary['ncu']['B_all_sol'] = ncu_capture(ncu, env, output, 'B_all_sol',
                ['--section', 'SpeedOfLight', '--metrics', ','.join(EXTRA_B)], args.batch)
    else:
        summary['ncu']['A_scale1_full'] = dict(status='skipped' if args.skip_ncu else 'ncu_not_found')
    summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    text = markdown(summary)
    (output / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
