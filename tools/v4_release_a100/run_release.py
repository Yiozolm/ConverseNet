"""Release suite and new-kernel checks for claude/v4-dev on a Colab GPU, driven by the Colab CLI.

Runs inside the remote Jupyter kernel (``colab exec -f tools/v4_release_a100/run_release.py``);
configuration comes from environment variables passed with ``--env``:

  CONVERSE_REF      commit to check out (required; the clone is verified to be at this SHA)
  CONVERSE_BRANCH   branch to clone (default claude/v4-dev)
  CONVERSE_REPO     repository URL (default https://github.com/Yiozolm/ConverseNet.git)
  CONVERSE_OUT      output directory (default /content/results/<gpu>-<utc time>)
  CONVERSE_GATE     1 = run the split-K pointwise weight-gradient gate if its fixtures are present
  CONVERSE_PERF     1 = paired timing of the pointwise and LayerNorm kernels including the model step
  CONVERSE_FIXTURES directory holding wgrad_capture_*.pt uploaded from the workstation (optional)

Steps, each in a fresh process with its log kept; failures are recorded, never hidden:
1. clone and pin the commit, install ninja;
2. checked build (test/extension_loader.py);
3. the complete, unmodified release suite at the default (full) numerical level;
4. optional gate and timing of the two kernels added on 2026-10-12;
5. summary.json / summary.md and a tarball of the output directory for ``colab download``.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time


def log_run(command, log, env, cwd):
    print(f'  $ {" ".join(map(str, command))}  -> {log.name}', flush=True)
    with Path(log).open('w', encoding='utf-8') as stream:
        return subprocess.run([str(c) for c in command], cwd=str(cwd), env=env,
                              stdout=stream, stderr=subprocess.STDOUT).returncode


def tail(path, n=6):
    try:
        return ''.join(Path(path).read_text(encoding='utf-8', errors='replace').splitlines(True)[-n:])
    except OSError:
        return ''


def main():
    ref = os.environ.get('CONVERSE_REF')
    if not ref:
        raise SystemExit('CONVERSE_REF is required')
    branch = os.environ.get('CONVERSE_BRANCH', 'claude/v4-dev')
    repo = os.environ.get('CONVERSE_REPO', 'https://github.com/Yiozolm/ConverseNet.git')
    import torch
    gpu = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability()
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    out = Path(os.environ.get('CONVERSE_OUT') or f'/content/results/v4_release-{gpu.replace(" ", "_")}-{stamp}')
    if out.exists():
        raise SystemExit(f'refusing to reuse {out}')
    out.mkdir(parents=True)
    root = Path('/content/ConverseNet')
    py = sys.executable
    summary = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gpu=gpu,
                   capability=f'{major}.{minor}', torch=torch.__version__, torch_cuda=torch.version.cuda,
                   python=sys.version.split()[0], ref=ref, steps={})
    smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
    (out / 'nvidia-smi.txt').write_text(smi)
    summary['driver'] = (re.search(r'Driver Version: (\S+)', smi) or [None, None])[1]

    def save():
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    print(f'[0] {gpu} sm_{major}{minor}, torch {torch.__version__}+cu{torch.version.cuda}, out {out}', flush=True)

    print('[1] clone and pin', flush=True)
    if root.exists():
        shutil.rmtree(root)
    base_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_CUDA_ARCH_LIST=f'{major}.{minor}',
                    MAX_JOBS=str(os.cpu_count() or 2), PIP_DISABLE_PIP_VERSION_CHECK='1')
    base_env.pop('CONVERSE2D_FFT_CALLBACKS', None)
    base_env.pop('CONVERSE2D_BACKEND', None)
    code = log_run(['git', 'clone', '--quiet', '--branch', branch, repo, root], out / 'clone.log', base_env, '/content')
    if code:
        raise SystemExit(f'clone failed: {tail(out / "clone.log")}')
    code = subprocess.run(['git', 'checkout', '--quiet', ref], cwd=root).returncode
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
    if code or not head.startswith(ref):
        raise SystemExit(f'could not pin {ref}: HEAD is {head}')
    summary['git_head'] = head
    summary['steps']['ninja'] = log_run([py, '-m', 'pip', 'install', '-q', 'ninja'], out / 'pip_ninja.log', base_env, root)
    fixtures = Path(os.environ.get('CONVERSE_FIXTURES', '/content/fixtures'))
    copied = []
    if fixtures.is_dir():
        (root / 'artifacts/v4_campaign').mkdir(parents=True, exist_ok=True)
        for item in sorted(fixtures.glob('wgrad_capture_*.pt')):
            shutil.copy2(item, root / 'artifacts/v4_campaign' / item.name)
            copied.append(item.name)
    summary['fixtures'] = copied
    save()

    print('[2] checked build', flush=True)
    t0 = time.time()
    code = log_run([py, '-u', 'test/extension_loader.py'], out / 'build.log', base_env, root)
    summary['steps']['build'] = dict(returncode=code, seconds=round(time.time() - t0, 1))
    manifest = root / '.build/cuda/source_manifest.json'
    if manifest.exists():
        summary['binary_sha256'] = json.loads(manifest.read_text()).get('binary_sha256')
    save()
    if code:
        print(tail(out / 'build.log', 40))
        raise SystemExit('build failed; see build.log')

    print('[3] release suite (full numerical level)', flush=True)
    t0 = time.time()
    code = log_run([py, '-m', 'unittest', 'discover', '-s', 'test', '-p', 'test_*.py', '-v'],
                   out / 'release_head.log', base_env, root)
    text = (out / 'release_head.log').read_text(encoding='utf-8', errors='replace')
    ran = re.search(r'Ran (\d+) tests', text)
    failures = sorted(set(re.findall(r'^(?:FAIL|ERROR): (.+)$', text, re.M)))
    summary['release'] = dict(returncode=code, ran=int(ran.group(1)) if ran else None, failures=failures,
                              skipped=len(re.findall(r'\.\.\. skipped', text)), seconds=round(time.time() - t0, 1))
    (out / 'release_failures.txt').write_text('\n'.join(failures) + ('\n' if failures else ''))
    for report in sorted((root / 'artifacts/fp32_release').glob('*.json')):
        shutil.copy2(report, out / report.name)
    print(f'  ran {summary["release"]["ran"]}, failures {len(failures)}, rc {code}', flush=True)
    save()

    if os.environ.get('CONVERSE_GATE') == '1':
        print('[4a] pointwise weight-gradient gate', flush=True)
        command = [py, '-u', 'tools/v4_pointwise_wgrad/gate.py', '--output', out / 'pointwise_gate.json']
        if not copied:
            command.append('--fixtures')  # empty list: the 14 synthetic geometries only (captures are 4 GB each)
        code = log_run(command, out / 'pointwise_gate.log', base_env, root)
        summary['steps']['pointwise_gate'] = dict(returncode=code, captured_fixtures=copied,
                                                  tail=tail(out / 'pointwise_gate.log'))
        save()

    if os.environ.get('CONVERSE_PERF') == '1':
        for name, script in (('pointwise', 'tools/v4_pointwise_wgrad/perf.py'),
                             ('layernorm', 'tools/v4_layernorm_training/perf.py')):
            print(f'[4b] {name} paired timing', flush=True)
            t0 = time.time()
            command = [py, '-u', script, '--output', out / f'{name}_perf.json', '--model']
            if name == 'pointwise':
                command += ['--idle-threshold', '100']  # dedicated Colab GPU; no foreign-load wait
            code = log_run(command, out / f'{name}_perf.log', base_env, root)
            summary['steps'][f'{name}_perf'] = dict(returncode=code, seconds=round(time.time() - t0, 1),
                                                    tail=tail(out / f'{name}_perf.log', 12))
            save()

    summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
    lines = [f'# {gpu} (sm_{major}{minor}) at {head[:12]}', '',
             f'- torch {torch.__version__} / CUDA {torch.version.cuda}, driver {summary["driver"]}',
             f'- binary {summary.get("binary_sha256", "?")[:12]}, build {summary["steps"]["build"]["seconds"]} s', '',
             f'## Release suite: ran {summary["release"]["ran"]}, failures {len(failures)}, '
             f'returncode {summary["release"]["returncode"]}', '']
    lines += [f'- {f}' for f in failures]
    for key in ('pointwise_gate', 'pointwise_perf', 'layernorm_perf'):
        if key in summary['steps']:
            lines += ['', f'## {key}', '', '```', json.dumps(summary['steps'][key], indent=1), '```']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    archive = out.with_suffix('.tgz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.name)
    print('\n'.join(lines), flush=True)
    print(f'\nARCHIVE {archive}', flush=True)


main()
