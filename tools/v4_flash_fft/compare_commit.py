"""Bitwise check of the working-tree fused extension against the same sources at
another commit, over every locally eligible flash_study case (one kernel per
channel, circular or unpadded). Shows that a change confined to the new
operations (kernel batches, other padding modes) left those paths' arithmetic
untouched. The reference sources are built in .build/flash_fft_<commit>.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import torch
from torch.utils import cpp_extension
import study
import flash_study as fs

parser = argparse.ArgumentParser()
parser.add_argument('--commit', default='HEAD')
args = parser.parse_args()
tag = subprocess.check_output(['git', 'rev-parse', '--short', args.commit], cwd=ROOT).decode().strip()
src_dir = ROOT / '.build' / f'flash_fft_{tag}'
src_dir.mkdir(parents=True, exist_ok=True)
for name in ('fused.cu', 'bind.cpp'):
    (src_dir / name).write_bytes(subprocess.check_output(['git', 'show', f'{args.commit}:tools/v4_flash_fft/{name}'], cwd=ROOT))
study.load_extension()
new = study.build()
if os.name == 'nt':
    cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
old = cpp_extension.load(name='flash_fft_reference', sources=[str(src_dir / 'fused.cu'), str(src_dir / 'bind.cpp')],
                         extra_cflags=['/O2', '/std:c++17'] if os.name == 'nt' else ['-O3', '-std=c++17'],
                         extra_cuda_cflags=['-O3', '-lineinfo', '-std=c++17'], build_directory=str(src_dir))
device = torch.cuda.current_device()
total = mismatched = 0
for name, (b, c, h, w, scale, pad) in fs.CASES.items():
    if not new.supported(h, w, scale, pad, device):
        continue
    torch.manual_seed(77001)
    x = torch.randn(b, c, h, w, device='cuda', requires_grad=True)
    x0 = torch.randn(b, c, h * scale, w * scale, device='cuda', requires_grad=scale > 1)
    weight = torch.rand(1, c, 3, 3, device='cuda', requires_grad=True)
    bias = torch.randn(1, c, 1, 1, device='cuda', requires_grad=True)
    upstream = torch.randn(b, c, h * scale, w * scale, device='cuda')
    inputs = (x, weight, bias) if scale == 1 else (x, x0, weight, bias)

    def run(ext):
        out = fs.flash(ext, x, x0, weight, bias, scale, pad)
        return [out.detach(), *torch.autograd.grad(out, inputs, upstream)]

    a, z = run(new), run(old)
    same = all(torch.equal(p.view(torch.int32), q.view(torch.int32)) for p, q in zip(a, z))
    total += 1
    mismatched += not same
    print(f'{name:28s} {"identical" if same else "DIFFERENT"}', flush=True)
print(f'{total - mismatched}/{total} cases bitwise identical to {args.commit} ({tag})')
sys.exit(1 if mismatched else 0)
