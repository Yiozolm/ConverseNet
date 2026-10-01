"""Serial native Windows NCU capture for an explicitly selected study worker.

Example: --worker baseline.py --output NEW_DIR -- --profile baseline
         --case b4_c128_96 --padding-mode circular
Workers accept --output DIR and bracket exactly one call with CUDA profiler API.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NCU = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ncu', type=Path, default=NCU)
    # PowerShell script argument binding can consume a standalone '--'.
    # Forward unknown worker options directly instead of relying on REMAINDER.
    args, worker_options = parser.parse_known_args()
    worker = (HERE / args.worker).resolve()
    if not worker.is_relative_to(HERE) or not worker.is_file():
        raise ValueError('Select an existing worker within this experiment directory')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve prior profiles; choose a fresh directory')
    output.mkdir(parents=True)
    common_path = ROOT / 'tools/v4_mixed_precision/profile_ncu.py'
    spec = importlib.util.spec_from_file_location('mixed_fusion_counter_parser', common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    extra = worker_options[1:] if worker_options[:1] == ['--'] else worker_options
    worker_args = [str(worker), *extra, '--output', str(output)]
    bootstrap = output / 'bootstrap.py'
    bootstrap.write_text('import site,sys,runpy\nsite.addsitedir(' + repr(str(Path(sys.prefix) / 'Lib/site-packages')) +
                         ')\nsys.argv=' + repr(worker_args) + '\nrunpy.run_path(sys.argv[0],run_name="__main__")\n', encoding='utf-8')
    identities = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), worker, common_path)}
    capture = output / 'capture.ncu-rep'
    command = [str(args.ncu), '--target-processes', 'all', '--profile-from-start', 'off',
               '--replay-mode', 'application', '--app-replay-mode', 'strict', '--app-replay-match', 'grid',
               '--cache-control', 'none', '--clock-control', 'none', '--metrics', ','.join(common.METRICS.values()),
               '--export', str(output / 'capture'), getattr(sys, '_base_executable', sys.executable), '-u', str(bootstrap)]
    record = dict(kind='mixed_fusion_ncu', status='running', command=command, worker_args=worker_args,
                  source_sha256=identities, bootstrap_sha256=sha(bootstrap), ncu_sha256=sha(args.ncu),
                  metrics=common.METRICS, profiler_only=True, timing_admission=False)
    try:
        with (output / 'capture.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record['capture_returncode'] = result.returncode
        if result.returncode or not capture.exists():
            raise RuntimeError('NCU capture failed; inspect preserved capture.log')
        csv_path = output / 'metrics.csv'
        imported_command = [str(args.ncu), '--import', str(capture), '--page', 'raw', '--csv', '--print-units', 'base']
        with csv_path.open('w', encoding='utf-8') as stream, (output / 'import.log').open('w', encoding='utf-8') as log:
            imported = subprocess.run(imported_command, stdout=stream, stderr=log)
        record.update(import_command=imported_command, import_returncode=imported.returncode)
        if imported.returncode:
            raise RuntimeError('NCU CSV export failed')
        summary = common.summarize(csv_path)
        workers = sorted(output.glob('worker-*.json'))
        if not workers:
            raise RuntimeError('No completed worker identity reports')
        records = [json.loads(p.read_text(encoding='utf-8')) for p in workers]
        if any(r['status'] != 'complete' or not r.get('passed', False) for r in records):
            raise RuntimeError('Unverified worker; no profiling admission')
        if any(sha(ROOT / path) != digest for path, digest in identities.items()):
            raise RuntimeError('Measured source changed during capture')
        record.update(status='complete', application_executions=len(workers),
                      worker_reports=[dict(path=str(p), sha256=sha(p)) for p in workers],
                      capture_sha256=sha(capture), csv_sha256=sha(csv_path),
                      aggregate=summary['aggregate'], kernel_count=summary['kernel_count'])
        dump(output / 'summary.json', summary)
    except Exception as error:
        record.update(status='error', error=repr(error))
        raise
    finally:
        dump(output / 'launcher.json', record)
    print(json.dumps(dict(status=record['status'], aggregate=record['aggregate'], output=str(output))))


if __name__ == '__main__':
    main()
