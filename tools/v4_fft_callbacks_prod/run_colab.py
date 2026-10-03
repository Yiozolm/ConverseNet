"""Linux/Colab validation of the cuFFT callback training FFTs on one GPU.

Steps (each in fresh processes, logs kept, failures recorded rather than hidden):
1. checked build (also exercises the Linux -lcufft -lnvrtc link);
2. complete release suite;
3. byte captures with CONVERSE2D_FFT_CALLBACKS=0 (unchanged ATen sequence)
   versus default in the same binary: public module and spectral cases;
4. callback admission per shape on this GPU (admission.py);
5. paired timing, alternating ATen/callback fresh-process rounds (perf.py).
Writes summary.json and summary.md into --output.
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


def run(command, log, env, cwd=ROOT):
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
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
    aten = dict(base, CONVERSE2D_FFT_CALLBACKS='0')
    py = sys.executable
    nvcc = shutil.which('nvcc')
    summary = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gpu=torch.cuda.get_device_name(0),
                   capability=f'{major}.{minor}', torch=torch.__version__, torch_cuda=torch.version.cuda,
                   nvcc=subprocess.run([nvcc, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1] if nvcc else None,
                   git_head=subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                   steps={})

    print('1/5 build...', flush=True)
    code = run([py, '-u', 'test/extension_loader.py'], out / 'build.log', base)
    summary['steps']['build'] = dict(returncode=code)
    if code:
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        raise SystemExit(f'Build failed; see {out / "build.log"}')
    summary['binary_sha256'] = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())['binary_sha256']

    print('2/5 release suite...', flush=True)
    code = run([py, '-m', 'unittest', 'discover', '-s', 'test', '-p', 'test_*.py', '-v'], out / 'release.log', base)
    tail = (out / 'release.log').read_text(encoding='utf-8', errors='replace')
    ran = re.search(r'Ran (\d+) tests', tail)
    failures = re.findall(r'^(?:FAIL|ERROR): (.+)$', tail, re.M)
    summary['steps']['release'] = dict(returncode=code, ran=int(ran.group(1)) if ran else None, failures=failures)

    print('3/5 byte captures (ATen vs callbacks)...', flush=True)
    for name, script in (('module', 'tools/v4_spatial_real_crop/capture.py'), ('spectral', 'tools/v4_scale1_planes/capture.py')):
        baseline, candidate = out / f'{name}_aten.json', out / f'{name}_callbacks.json'
        run([py, '-u', script, '--output', str(baseline)], out / f'{name}_aten.log', aten)
        code = run([py, '-u', script, '--output', str(candidate), '--compare', str(baseline)], out / f'{name}_callbacks.log', base)
        comparison = json.loads(candidate.read_text())['comparison'] if candidate.exists() else None
        summary['steps'][f'capture_{name}'] = dict(returncode=code, comparison=comparison)

    print('4/5 admission...', flush=True)
    code = run([py, '-u', str(HERE / 'admission.py'), '--output', str(out / 'admission.json')], out / 'admission.log', base)
    summary['steps']['admission'] = dict(returncode=code, rows=json.loads((out / 'admission.json').read_text())['rows']
                                         if (out / 'admission.json').exists() else None)

    print('5/5 paired timing...', flush=True)
    perf = out / 'perf.jsonl'
    codes = []
    for r in range(args.rounds):
        for label in (('aten', 'callbacks') if r % 2 == 0 else ('callbacks', 'aten')):
            codes.append(run([py, '-u', str(HERE / 'perf.py'), '--label', label, '--output', str(perf)],
                             out / f'perf_{r}_{label}.log', aten if label == 'aten' else base))
    timing = subprocess.run([py, str(HERE / 'summarize.py'), str(perf)], capture_output=True, text=True).stdout if perf.exists() else ''
    summary['steps']['perf'] = dict(returncodes=codes, summary=timing)
    summary['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    s = summary['steps']
    lines = [f"# cuFFT callback validation: {summary['gpu']} (sm_{major}{minor})", '',
             f"- torch {summary['torch']} / CUDA {summary['torch_cuda']}; {summary['nvcc']}",
             f"- git {summary['git_head'][:12]}, binary {summary['binary_sha256'][:12]}", '',
             f"Release suite: ran {s['release']['ran']}, failures {len(s['release']['failures'])}"]
    lines += [f'  - {f}' for f in s['release']['failures']]
    for name in ('module', 'spectral'):
        c = s[f'capture_{name}']['comparison']
        lines.append(f"Byte capture {name} (ATen vs callbacks): " + (
            f"{c['cases'] - len(c['mismatches'])}/{c['cases']} identical; mismatches {c['mismatches']}" if c else 'not available'))
    lines += ['', '## Admission (per training call)', '', '| shape | lto_fft | ATen 1/N passes | ATen real_crop VJP | pad kernels |',
              '|---|---|---|---|---|']
    for name, row in (s['admission']['rows'] or {}).items():
        lines.append(f"| {name} | {row['lto_fft']} | {row['scale']} | {row['real_crop']} | {row['pad']} |")
    lines += ['', '## Paired timing', '', '```', timing.rstrip(), '```', '',
              'Operator-level forward+VJP timings; not a whole-model or convergence result.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
