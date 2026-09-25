"""Run a fixed ABBA/BAAB short full-workflow diagnostic with the frozen worker.

This is not a convergence test or a replacement for the historical long run.
Only one child uses the GPU at a time. Read each run only after it exits.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ORDER = ('before', 'current', 'current', 'before', 'current', 'before', 'before', 'current')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT/'artifacts/fp32_roadmap/baseline')
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a fresh output directory')
    if args.steps < 6 or args.steps % 2:
        parser.error('Steps must be an even number >=6 for warmed diagnostics')
    output.mkdir(parents=True)
    shell = shutil.which('pwsh') or shutil.which('powershell')
    env = os.environ.copy()
    env.update(CONVERSE_MSVC_VERSION='14.44', TORCH_CUDA_ARCH_LIST='12.0')
    worker = ROOT/'tools/roadmap_quality/train_usrnet_dataset.py'
    expected = {}
    for variant in ('before', 'current'):
        path = ROOT/f'artifacts/fp32_roadmap/quality_long_{variant}17/run.json'
        historical = json.loads(path.read_text(encoding='utf-8'))
        expected[variant] = dict(source_sha256=historical['source_sha256'],
                                 build_manifest=historical['backend']['build_manifest'],
                                 historical_run_sha256=sha(path))
    record = dict(kind='interleaved_training_workflow_diagnostic', status='running', started_utc=now(),
                  scope=f'Fresh processes, identical pretrained seed17 data and worker, {args.steps} updates and three full 100-image evaluations per process. No profiler or model/worker changes.',
                  order=ORDER, steps=args.steps, eval_every=args.steps//2, seed=17,
                  deterministic_algorithms=True, tf32=False, amp=False, affinity='0xC03C03',
                  worker_sha256=sha(worker), tool_sha256=sha(__file__),
                  baseline=str(args.baseline.resolve()), current=str(ROOT), runs=[], expected_identity_by_variant=expected,
                  telemetry_scope='Read-only GPU telemetry at 1 Hz across all children. It is coarse context, not attribution of individual steps or a control of clocks/power.')
    def save():
        (output/'campaign.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    save()
    telemetry = None
    telemetry_file = (output/'gpu_telemetry.csv').open('w', encoding='utf-8')
    smi = shutil.which('nvidia-smi')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        if smi:
            telemetry = subprocess.Popen([smi, '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate', '--format=csv', '-lms', '1000'], stdout=telemetry_file, stderr=subprocess.STDOUT, creationflags=flags)
        for index, variant in enumerate(ORDER):
            name = f'{index+1:02d}_{variant}'
            command = [shell, '-NoProfile', '-File', str(ROOT/'tools/run_affinity.ps1'),
                       '-Mask', '0xC03C03', '-MetadataPath', str(output/f'{name}_affinity.json'),
                       str(worker), '--root', str(args.baseline.resolve() if variant=='before' else ROOT),
                       '--variant', variant, '--purpose', 'pilot', '--seed', '17',
                       '--steps', str(args.steps), '--eval-every', str(args.steps//2),
                       '--batch-size', '4', '--microbatch-size', '4', '--patch-size', '96', '--scale', '3',
                       '--deterministic-algorithms', '--run-dir', str(output/name)]
            began = time.perf_counter()
            row = dict(name=name, variant=variant, command=command, started_utc=now())
            print(json.dumps(dict(event='start', name=name, utc=row['started_utc'])), flush=True)
            with (output/f'{name}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            row.update(exit_code=result.returncode, wrapper_process_wall_s=time.perf_counter()-began, completed_utc=now())
            for filename in ('run.json', 'training.jsonl', 'evaluations.jsonl', 'final.pth'):
                path = output/name/filename
                if path.exists():
                    row.setdefault('input_sha256', {})[filename] = sha(path)
            record['runs'].append(row)
            save()
            print(json.dumps(dict(event='end', name=name, code=result.returncode, wall_s=row['wrapper_process_wall_s'])), flush=True)
            if result.returncode:
                raise RuntimeError(f'Child failed: {name}; original files retained')
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
        telemetry_file.close()
        record.update(completed_utc=now(), telemetry_sha256=sha(output/'gpu_telemetry.csv'))
        save()


if __name__ == '__main__':
    main()
