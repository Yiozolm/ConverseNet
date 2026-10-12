"""Paired USRNet short fine-tuning (tools/roadmap_quality) on a Colab GPU.

Runs inside the clone made by run_command.py (cwd = the checked HEAD checkout, already built):

1. unpack the uploaded dataset tarball (Open Images subset + split manifest) into this checkout;
2. the CPU data-protocol checks (image hashes, split, holdout isolation);
3. a detached git worktree of the baseline commit, built by its own checked build (pilot run);
4. capacity pilots on both checkouts;
5. the six formal runs, seeds 17/29/43, both variants, order alternating between seeds;
6. summarize_usrnet_training.py: per-seed PSNR/SSIM gates (drop <= 0.05 dB, <= 0.001 SSIM).

    paired_quality.py --output <dir> [--baseline 0d636215e23d] [--dataset /content/usrnet_dataset.tar]
                      [--steps 250] [--eval-every 125]

Every worker runs in its own process with its log kept; failures are recorded, not hidden.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--baseline', default='0d636215e23d472825e09cc340a99f1d2ae2b78c')
parser.add_argument('--baseline-dir', type=Path, default=Path('/content/baseline'))
parser.add_argument('--dataset', type=Path, default=Path('/content/usrnet_dataset.tar'),
                    help='optional tarball with dataset/ and artifacts/dataset_training/; otherwise fetch from S3')
parser.add_argument('--manifest', type=Path, default=Path('/content/split_900_100.json'),
                    help='split manifest to copy into the checkout when fetching from S3')
parser.add_argument('--steps', type=int, default=250)
parser.add_argument('--eval-every', type=int, default=125)
parser.add_argument('--seeds', type=int, nargs='*', default=[17, 29, 43])
parser.add_argument('--skip-pilots', action='store_true')
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
ENV.pop('CONVERSE2D_BACKEND', None)
ENV.pop('CONVERSE2D_SKIP_BUILD', None)
summary = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), root=str(ROOT), baseline=args.baseline,
               steps=args.steps, eval_every=args.eval_every, steps_log={})


def save():
    (OUT / 'orchestrator.json').write_text(json.dumps(summary, indent=2))


def run(label, command, cwd=ROOT):
    log = OUT / f'{label}.log'
    print(f'[{label}] $ {" ".join(map(str, command))}', flush=True)
    t0 = time.time()
    with log.open('w', encoding='utf-8') as stream:
        code = subprocess.run([str(c) for c in command], cwd=str(cwd), env=ENV,
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    tail = ''.join(log.read_text(encoding='utf-8', errors='replace').splitlines(True)[-4:])
    summary['steps_log'][label] = dict(returncode=code, seconds=round(time.time() - t0, 1), tail=tail[-600:])
    save()
    print(f'[{label}] rc={code} in {time.time() - t0:.0f}s\n{tail}', flush=True)
    return code


# 1. dataset: the uploaded tarball if present, otherwise the manifest's images fetched from the
# public Open Images V6 bucket and verified against the manifest's file SHA-256.
manifest = ROOT / 'artifacts/dataset_training/split_900_100.json'
images = ROOT / 'dataset/.fiftyone/zoo/open-images-v6/validation/data'
if args.dataset.is_file():
    print('[dataset] unpacking', args.dataset, '->', ROOT, flush=True)
    with tarfile.open(args.dataset) as tar:
        tar.extractall(ROOT)
else:
    import hashlib
    import shutil
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    source = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if source != manifest:
        shutil.copy2(source, manifest)
    rows = json.loads(manifest.read_text(encoding='utf-8-sig'))['images']
    images.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        target = images / row['relative_path']
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == row['file_sha256']:
            return True
        url = f"https://open-images-dataset.s3.amazonaws.com/validation/{row['image_id']}.jpg"
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                if hashlib.sha256(data).hexdigest() == row['file_sha256']:
                    target.write_bytes(data)
                    return True
                return False
            except Exception:
                time.sleep(2 * (attempt + 1))
        return False

    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        verified = sum(pool.map(fetch, rows))
    summary['dataset_fetch'] = dict(rows=len(rows), verified=verified, seconds=round(time.time() - t0, 1))
    print(f'[dataset] fetched and verified {verified}/{len(rows)} in {time.time() - t0:.0f}s', flush=True)
    if verified != len(rows):
        save()
        raise SystemExit('dataset incomplete; see orchestrator.json')
assert manifest.is_file(), manifest
summary['images'] = len(list(images.glob('*.jpg')))
print('[dataset] images', summary['images'], flush=True)

# 2. CPU protocol checks
run('data_protocol_checks', [PY, '-u', 'tools/roadmap_quality/test_usrnet_training_data.py',
                             '--output', OUT / 'data_protocol_checks.json'])

# 3. baseline worktree
if not args.baseline_dir.exists():
    subprocess.run(['git', 'worktree', 'add', '--detach', str(args.baseline_dir), args.baseline], cwd=ROOT, check=True)
summary['baseline_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=args.baseline_dir,
                                          capture_output=True, text=True).stdout.strip()
summary['current_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
save()

worker = [PY, '-u', 'tools/roadmap_quality/train_usrnet_dataset.py', '--manifest', manifest, '--data-root', ROOT,
          '--checkpoint', ROOT / 'model_zoo/converse_usrnet.pth']
roots = {'before': args.baseline_dir.resolve(), 'current': ROOT}

# 4. pilots (the baseline's first run carries its checked build)
if not args.skip_pilots:
    for variant in ('before', 'current'):
        extra = ['--build'] if variant == 'before' else []
        run(f'pilot_{variant}', worker + ['--root', roots[variant], '--variant', variant, '--purpose', 'pilot',
                                          '--steps', 5, '--eval-every', 5, '--run-dir', OUT / f'pilot_{variant}', *extra])

# 5. formal runs, order alternating between seeds
for index, seed in enumerate(args.seeds):
    order = ('before', 'current') if index % 2 == 0 else ('current', 'before')
    for variant in order:
        run(f'{variant}_seed{seed}', worker + ['--root', roots[variant], '--variant', variant, '--seed', seed,
                                               '--steps', args.steps, '--eval-every', args.eval_every,
                                               '--run-dir', OUT / f'{variant}_seed{seed}'])

# 6. audit
code = run('paired_summary', [PY, '-u', 'tools/roadmap_quality/summarize_usrnet_training.py', '--root', OUT,
                              '--output', OUT / 'paired_summary.json', '--markdown', OUT / 'paired_summary.md',
                              '--steps', args.steps, '--eval-every', args.eval_every])
summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
save()
md = OUT / 'paired_summary.md'
if md.exists():
    print(md.read_text(encoding='utf-8'), flush=True)
# Keep the archive small: drop checkpoints after the audit has read them.
for path in OUT.rglob('*.pth'):
    summary.setdefault('removed_checkpoints', []).append(str(path.relative_to(OUT)))
    path.unlink()
save()
print('paired quality rc', code, flush=True)
sys.exit(code)
