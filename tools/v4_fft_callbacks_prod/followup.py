"""A100 follow-up: attribute release failures and per-site callback timing.

The first A100 run (run_colab.py) had 54 FP64-budget failures, all on planes
below the 16-side callback threshold, and a circular-s1 slowdown. This runs:
1. checked build of this checkout;
2. release suite here with CONVERSE2D_FFT_CALLBACKS=0 and with the default;
3. release suite at control refs in fresh git worktrees, each with its own
   build (default: 5a5f0ba, just before callbacks; 347b040, codex/v4.0.0
   before any claude/v4-dev commit; v3.0.0);
4. paired timing with ATen, all callback sites, and each site alone
   (CONVERSE2D_FFT_CALLBACKS=<site>), with per-kernel-name times.
Writes summary.json and summary.md into --output. Failures are recorded, not hidden.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITES = ('real', 'circular', 'inverse', 'crop_embed')


def run(command, log, env, cwd=ROOT):
    with log.open('w', encoding='utf-8') as stream:
        return subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def release(py, log, env, cwd):
    code = run([py, '-m', 'unittest', 'discover', '-s', 'test', '-p', 'test_*.py', '-v'], log, env, cwd)
    text = log.read_text(encoding='utf-8', errors='replace')
    ran = re.search(r'Ran (\d+) tests', text)
    return dict(returncode=code, ran=int(ran.group(1)) if ran else None,
                failures=re.findall(r'^(?:FAIL|ERROR): (.+)$', text, re.M))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refs', nargs='*', default=['5a5f0ba', '347b040', 'v3.0.0'])
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    out = args.output.resolve()
    out.mkdir(parents=True)
    import torch
    major, minor = torch.cuda.get_device_capability()
    base = dict(os.environ, TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST', f'{major}.{minor}'),
                MAX_JOBS=os.environ.get('MAX_JOBS', str(os.cpu_count() or 2)), PYTHONDONTWRITEBYTECODE='1')
    base.pop('CONVERSE2D_FFT_CALLBACKS', None)
    py = sys.executable
    git = lambda *a, cwd=ROOT: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
    summary = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gpu=torch.cuda.get_device_name(0),
                   capability=f'{major}.{minor}', torch=torch.__version__, torch_cuda=torch.version.cuda,
                   git_head=git('rev-parse', 'HEAD'), release={}, steps={})

    def save():
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('1/4 build...', flush=True)
    code = run([py, '-u', 'test/extension_loader.py'], out / 'build.log', base)
    summary['steps']['build'] = dict(returncode=code)
    if code:
        save()
        raise SystemExit(f'Build failed; see {out / "build.log"}')
    summary['binary_sha256'] = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())['binary_sha256']

    print('2/4 release suite here, callbacks off and on...', flush=True)
    summary['release']['head_aten'] = release(py, out / 'release_head_aten.log',
                                              dict(base, CONVERSE2D_FFT_CALLBACKS='0'), ROOT)
    summary['release']['head'] = release(py, out / 'release_head.log', base, ROOT)
    save()

    print('3/4 release suite at control refs...', flush=True)
    controls = Path('/tmp/converse2d_controls') if os.name != 'nt' else out / 'controls'
    for ref in args.refs:
        if not git('rev-parse', '--verify', '--quiet', ref + '^{commit}'):
            subprocess.run(['git', 'fetch', '--quiet', 'origin', f'+refs/tags/{ref}:refs/tags/{ref}'], cwd=ROOT)
        tree = controls / ref
        if tree.exists():
            subprocess.run(['git', 'worktree', 'remove', '--force', str(tree)], cwd=ROOT)
        added = subprocess.run(['git', 'worktree', 'add', '--detach', str(tree), ref], cwd=ROOT,
                               capture_output=True, text=True)
        name = re.sub(r'[^\w.-]', '_', ref)
        if added.returncode:
            summary['release'][ref] = dict(error=added.stderr.strip())
            continue
        code = run([py, '-u', 'test/extension_loader.py'], out / f'build_{name}.log', base, tree)
        summary['release'][ref] = (release(py, out / f'release_{name}.log', base, tree) if not code
                                   else dict(error=f'build failed ({code})'))
        summary['release'][ref]['commit'] = git('rev-parse', 'HEAD', cwd=tree)
        save()

    print('4/4 per-site paired timing...', flush=True)
    perf = out / 'perf.jsonl'
    labels = [('aten', '0'), ('all', None)] + [(site, site) for site in SITES]
    codes = []
    for r in range(args.rounds):
        order = labels[r % len(labels):] + labels[:r % len(labels)]
        for label, value in (order if r % 2 == 0 else order[::-1]):
            env = dict(base) if value is None else dict(base, CONVERSE2D_FFT_CALLBACKS=value)
            codes.append(run([py, '-u', str(HERE / 'perf.py'), '--label', label, '--output', str(perf)],
                             out / f'perf_{r}_{label}.log', env))
    timing = subprocess.run([py, str(HERE / 'summarize.py'), str(perf), '--kernels'],
                            capture_output=True, text=True).stdout if perf.exists() else ''
    (out / 'perf_summary.txt').write_text(timing, encoding='utf-8')
    summary['steps']['perf'] = dict(returncodes=codes)
    summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()

    head = set(summary['release']['head']['failures'])
    lines = [f"# A100 follow-up: {summary['gpu']} (sm_{major}{minor})", '',
             f"- torch {summary['torch']} / CUDA {summary['torch_cuda']}",
             f"- git {summary['git_head'][:12]}, binary {summary['binary_sha256'][:12]}", '',
             '## Release suite', '', '| run | commit | ran | failures | also failing at HEAD | only here |',
             '|---|---|---|---|---|---|']
    for name, result in summary['release'].items():
        if 'error' in result:
            lines.append(f"| {name} | {result.get('commit', '')[:12]} | - | {result['error']} | | |")
            continue
        fails = set(result['failures'])
        lines.append(f"| {name} | {result.get('commit', summary['git_head'])[:12]} | {result['ran']} | "
                     f"{len(fails)} | {len(fails & head)} | {len(fails - head)} |")
    for name, result in summary['release'].items():
        if name != 'head' and 'failures' in result:
            new = sorted(head - set(result['failures']))
            lines += ['', f"Failing at HEAD but not in {name} ({len(new)}):"] + [f'  - {f}' for f in new]
    lines += ['', '## Per-site timing (see perf_summary.txt for per-kernel times)', '', '```',
              '\n'.join(l for l in timing.splitlines() if not l.startswith('    ')), '```', '',
              'Operator-level forward+VJP timings; not a whole-model or convergence result.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
