"""Clone, pin, build, then run one repository command on a Colab GPU (Colab CLI).

Companion to run_release.py for follow-up studies. Runs inside the remote kernel:

    colab exec -s <session> --timeout 3600 --env CONVERSE_REF=<sha> \
        --env CONVERSE_CMD="tools/v4_release_a100/layernorm_seeds.py --output {out}/layernorm_seeds.json" \
        -f tools/v4_release_a100/run_command.py

`{out}` in CONVERSE_CMD expands to the output directory. The command runs with the
checkout's Python in a fresh process; its log, summary.json and a tarball of the
output directory are written for `colab download`. An existing clone at the same
commit with a finished build is reused (CONVERSE_FRESH=1 forces a new clone).
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


def main():
    ref = os.environ.get('CONVERSE_REF')
    command = os.environ.get('CONVERSE_CMD')
    if not ref or not command:
        raise SystemExit('CONVERSE_REF and CONVERSE_CMD are required')
    branch = os.environ.get('CONVERSE_BRANCH', 'claude/v4-dev')
    repo = os.environ.get('CONVERSE_REPO', 'https://github.com/Yiozolm/ConverseNet.git')
    import torch
    gpu = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability()
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    name = os.environ.get('CONVERSE_NAME', 'command')
    out = Path(f'/content/results/{name}-{gpu.replace(" ", "_")}-{stamp}')
    out.mkdir(parents=True)
    root = Path('/content/ConverseNet')
    py = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_CUDA_ARCH_LIST=f'{major}.{minor}',
               MAX_JOBS=str(os.cpu_count() or 2), PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key in ('CONVERSE2D_FFT_CALLBACKS', 'CONVERSE2D_BACKEND'):
        env.pop(key, None)
    summary = dict(gpu=gpu, capability=f'{major}.{minor}', torch=torch.__version__, torch_cuda=torch.version.cuda,
                   ref=ref, command=command, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), steps={})

    def log_run(cmd, log, cwd):
        print(f'  $ {" ".join(map(str, cmd))}  -> {Path(log).name}', flush=True)
        with Path(log).open('w', encoding='utf-8') as stream:
            return subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                                  stdout=stream, stderr=subprocess.STDOUT).returncode

    head = (subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
            if (root / '.git').exists() else '')
    reuse = head.startswith(ref) and (root / '.build/cuda/source_manifest.json').exists() \
        and os.environ.get('CONVERSE_FRESH') != '1'
    print(f'[0] {gpu} sm_{major}{minor}; {"reusing" if reuse else "fresh"} checkout; out {out}', flush=True)
    if not reuse:
        if root.exists():
            shutil.rmtree(root)
        if log_run(['git', 'clone', '--quiet', '--branch', branch, repo, root], out / 'clone.log', '/content'):
            raise SystemExit('clone failed')
        subprocess.run(['git', 'checkout', '--quiet', ref], cwd=root, check=True)
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
        if not head.startswith(ref):
            raise SystemExit(f'could not pin {ref}: HEAD is {head}')
        log_run([py, '-m', 'pip', 'install', '-q', 'ninja'], out / 'pip_ninja.log', root)
        t0 = time.time()
        code = log_run([py, '-u', 'test/extension_loader.py'], out / 'build.log', root)
        summary['steps']['build'] = dict(returncode=code, seconds=round(time.time() - t0, 1))
        if code:
            print(Path(out / 'build.log').read_text(errors='replace')[-4000:])
            raise SystemExit('build failed')
    summary['git_head'] = head
    summary['binary_sha256'] = json.loads((root / '.build/cuda/source_manifest.json').read_text()).get('binary_sha256')

    print('[1] command', flush=True)
    argv = shlex.split(command.replace('{out}', str(out)))
    if argv and argv[0].endswith('.py'):
        argv = [py, '-u', *argv]
    t0 = time.time()
    code = log_run(argv, out / 'command.log', root)
    summary['steps']['command'] = dict(returncode=code, seconds=round(time.time() - t0, 1))
    summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    text = (out / 'command.log').read_text(encoding='utf-8', errors='replace')
    print(text[-6000:], flush=True)
    archive = out.with_suffix('.tgz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.name)
    print(f'command rc={code}\nARCHIVE {archive}', flush=True)


main()
