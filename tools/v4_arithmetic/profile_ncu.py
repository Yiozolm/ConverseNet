"""Native Windows NCU launcher; profiler results are not benchmark timings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError('Use a new profile directory')
args.output.mkdir(parents=True)
worker = Path(__file__).with_name('profile_worker.py').resolve()
bootstrap = args.output.resolve() / 'worker.py'
bootstrap.write_text('import site,sys,runpy\nsite.addsitedir(' +
    repr(str(Path(sys.prefix) / 'Lib/site-packages')) + ')\nsys.argv=' +
    repr([str(worker), '--root', str(args.root.resolve())]) +
    '\nrunpy.run_path(sys.argv[0], run_name="__main__")\n', encoding='utf-8')
ncu = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe')
command = [str(ncu), '--target-processes', 'all', '--profile-from-start', 'off',
           '--set', 'full', '--kernel-name', 'regex:.*scale1_forward.*', '--launch-count', '1',
           '--export', str(args.output.resolve() / 'capture'),
           getattr(sys, '_base_executable', sys.executable), '-u', str(bootstrap)]
with (args.output / 'capture.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
record = dict(command=command, returncode=result.returncode,
              worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
              checked_build=json.loads((args.root / '.build/cuda/source_manifest.json').read_text()),
              profiler_only=True)
(args.output / 'launcher.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
if result.returncode or not (args.output / 'capture.ncu-rep').exists():
    raise SystemExit(result.returncode or 2)
print('NCU report:', args.output / 'capture.ncu-rep')
