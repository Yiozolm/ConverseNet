"""Windows application-replay NCU DRAM counters, never an unprofiled benchmark.

Run serially for --dtype fp32/fp16/bf16 in fresh directories. For subsequent
dtypes, --identity-from FIRST/launcher.json enforces the same measured sources
and checked build. --summarize-only parses an existing raw CSV without GPU use.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NCU = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe')
# These names are verified in the existing Nsight 2025.3/SM120 raw capture.
# Newer names for the task's dram__bytes_read/write and dram__throughput counters.
METRICS = {'dram_read_bytes': 'dram__bytes_op_read.sum', 'dram_write_bytes': 'dram__bytes_op_write.sum',
           'duration': 'gpu__time_duration.sum', 'dram_throughput_percent': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'}
ALIASES = {'dram_read_bytes': 'dram__bytes_read.sum', 'dram_write_bytes': 'dram__bytes_write.sum',
           'duration': 'gpu__time_duration.sum', 'dram_throughput_percent': 'dram__throughput.avg.pct_of_peak_sustained_elapsed'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def summarize(path):
    rows = list(csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    start = next(i for i, row in enumerate(rows) if row and row[0] == 'ID' and 'Kernel Name' in row)
    header, units = rows[start], rows[start+1]
    kernels = []
    for row in rows[start+2:]:
        if not row or not re.fullmatch(r'\d+', row[0]):
            continue
        values = dict(zip(header, row))
        entry = {name: values.get(name) for name in ('ID', 'Process ID', 'Kernel Name', 'Context', 'Stream', 'Block Size', 'Grid Size')}
        entry['metrics'] = {}
        for name, metric in METRICS.items():
            key = metric if metric in values else ALIASES[name]
            try:
                value = float(values[key].replace(',', ''))
                if not math.isfinite(value):
                    raise ValueError('nonfinite metric')
                unit = units[header.index(key)]
            except (KeyError, ValueError, IndexError):
                raise RuntimeError(f'Missing/non-numeric {metric} for kernel {row[0]}')
            entry['metrics'][name] = dict(value=value, unit=unit, metric=key)
        kernels.append(entry)
    if not kernels:
        raise RuntimeError('No kernels in selected-metric CSV')
    byte_units = {'byte': 1, 'Kbyte': 1000, 'Mbyte': 1000000, 'Gbyte': 1000000000}
    time_units = {'nsecond': 1, 'ns': 1, 'usecond': 1000, 'us': 1000, 'µs': 1000,
                  'msecond': 1000000, 'ms': 1000000, 'second': 1000000000, 's': 1000000000}
    totals = dict(dram_read_bytes=0., dram_write_bytes=0., profiled_kernel_duration_sum_ns=0.)
    for kernel in kernels:
        for name in ('dram_read_bytes', 'dram_write_bytes'):
            metric = kernel['metrics'][name]
            totals[name] += metric['value'] * byte_units[metric['unit']]
        metric = kernel['metrics']['duration']
        totals['profiled_kernel_duration_sum_ns'] += metric['value'] * time_units[metric['unit']]
    totals['dram_total_bytes'] = totals['dram_read_bytes'] + totals['dram_write_bytes']
    return dict(kernel_count=len(kernels), kernels=kernels, aggregate=totals,
                profiler_only=True, unprofiled_timing=False,
                note='Aggregate actual DRAM counters across captured kernels. Duration sum is profiler kernel time, not whole-call wall time. Throughput percentages are not summed.')


def identity(root):
    paths = ('tools/v4_mixed_precision/profile_ncu.py', 'tools/v4_mixed_precision/profile_worker.py',
             'tools/v4_mixed_precision/adapter.py', 'models/converse_core.py', 'test/extension_loader.py',
             'Converse2D/build_config.py')
    manifest = json.loads((root / '.build/cuda/source_manifest.json').read_text(encoding='utf-8'))
    binary = root / '.build/cuda' / manifest['library']
    if sha(binary) != manifest['binary_sha256']:
        raise RuntimeError('Checked production binary hash mismatch')
    return dict(source_sha256={name: sha(root / name) for name in paths}, checked_manifest=manifest,
                binary_sha256=sha(binary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ncu', type=Path, default=DEFAULT_NCU)
    parser.add_argument('--identity-from', type=Path)
    parser.add_argument('--summarize-only', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    if args.summarize_only:
        dump(args.output / 'summary.json', summarize(args.summarize_only))
        return
    if args.dtype is None:
        parser.error('--dtype is required for a capture')
    root, output = args.root.resolve(), args.output.resolve()
    measured = identity(root)
    if args.identity_from and json.loads(args.identity_from.read_text())['identity'] != measured:
        raise RuntimeError('Source/build differs from the first dtype run')
    worker = root / 'tools/v4_mixed_precision/profile_worker.py'
    worker_args = [str(worker), '--root', str(root), '--dtype', args.dtype, '--metadata-dir', str(output)]
    bootstrap = output / 'worker_bootstrap.py'
    bootstrap.write_text('import site,sys,runpy\nsite.addsitedir(' + repr(str(Path(sys.prefix) / 'Lib/site-packages')) +
                         ')\nsys.argv=' + repr(worker_args) + '\nrunpy.run_path(sys.argv[0],run_name="__main__")\n', encoding='utf-8')
    command = [str(args.ncu), '--target-processes', 'all', '--profile-from-start', 'off',
               '--replay-mode', 'application', '--app-replay-mode', 'strict', '--app-replay-match', 'grid',
               '--cache-control', 'none', '--clock-control', 'none', '--metrics', ','.join(METRICS.values()),
               '--export', str(output / 'capture'), getattr(sys, '_base_executable', sys.executable), '-u', str(bootstrap)]
    report = dict(kind='mixed_level1_ncu_capture', status='running', dtype=args.dtype, identity=measured,
                  metric_names=METRICS, requested_semantic_aliases=ALIASES, command=command, worker_args=worker_args,
                  ncu_path=str(args.ncu), ncu_sha256=sha(args.ncu), bootstrap_sha256=sha(bootstrap),
                  replay_mode='application', cache_control='none', clock_control='none', profiler_only=True,
                  purpose='Test whether reduced external storage changes actual complete-call DRAM traffic; no inferred 2x spectral-bandwidth claim')
    try:
        with (output / 'capture.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report['capture_returncode'] = result.returncode
        capture = output / 'capture.ncu-rep'
        if result.returncode or not capture.exists():
            raise RuntimeError('NCU capture failed; retain capture.log and this directory')
        csv_path = output / 'metrics.csv'
        export = [str(args.ncu), '--import', str(capture), '--page', 'raw', '--csv', '--print-units', 'base']
        with csv_path.open('w', encoding='utf-8') as stream, (output / 'import.log').open('w', encoding='utf-8') as log:
            imported = subprocess.run(export, stdout=stream, stderr=log)
        report.update(import_command=export, import_returncode=imported.returncode)
        if imported.returncode:
            raise RuntimeError('NCU raw CSV import failed')
        summary = summarize(csv_path)
        workers = [json.loads(path.read_text()) for path in sorted(output.glob('worker-*.json'))]
        if not workers or any(w['status'] != 'complete' or w['checked_manifest'] != measured['checked_manifest'] for w in workers):
            raise RuntimeError('Missing or inconsistent application-replay worker metadata')
        if any(w['fixtures'] != workers[0]['fixtures'] for w in workers):
            raise RuntimeError('Application replay did not reproduce the same fixture bytes')
        if args.identity_from:
            first = json.loads(args.identity_from.read_text())
            prior_worker = json.loads(Path(first['worker_reports'][0]['path']).read_text())
            if any(workers[0]['fixtures'][k] != prior_worker['fixtures'][k] for k in ('original_x', 'weight', 'bias')):
                raise RuntimeError('Original FP32 fixture values differ across dtype captures')
        if identity(root) != measured:
            raise RuntimeError('Source/build identity changed during capture')
        summary['raw_capture'] = dict(path=str(capture), sha256=sha(capture))
        summary['raw_csv'] = dict(path=str(csv_path), sha256=sha(csv_path))
        dump(output / 'summary.json', summary)
        report.update(status='complete', application_executions=len(workers), kernel_count=summary['kernel_count'],
                      aggregate=summary['aggregate'], worker_reports=[dict(path=str(p), sha256=sha(p)) for p in sorted(output.glob('worker-*.json'))],
                      recorded_kernel_pass_counts=[dict(kernel=m.group(1), passes=int(m.group(2))) for m in
                          re.finditer(r'Profiling "([^"]+)"[^\r\n]*? - (\d+) passes?', (output / 'capture.log').read_text(encoding='utf-8'))],
                      counter_pass_count_note='Application executions are observed from worker records; exact hardware pass count remains in the raw NCU capture/log.')
    except Exception as error:
        report.update(status='error', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report['capture_log_sha256'] = sha(output / 'capture.log') if (output / 'capture.log').exists() else None
        dump(output / 'launcher.json', report)
    print(json.dumps(dict(status=report['status'], dtype=args.dtype, aggregate=report['aggregate'], output=str(output)), indent=2))


if __name__ == '__main__':
    main()
