"""Byte capture for the s1 full-spectrum forward and its VJPs.

Run once on the baseline build and once on the candidate; --compare requires
identical SHA-256 for every output and gradient. The indexing rewrite keeps
per-element FP32 arithmetic, so any byte difference is a failure, not a budget
question. Outputs go to a fresh JSON path; existing reports are never replaced.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
import torch
from extension_loader import load_extension


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def spectral_case(batch, channels, h, w, kb, kc, shared, reg, scale, seed):
    torch.manual_seed(seed)
    y = (torch.randn(batch, channels, h, w, device='cuda', dtype=torch.complex64) * scale).requires_grad_()
    p = y if shared else (torch.randn_like(y) * scale).requires_grad_()
    k = torch.randn(kb, kc, h, w, device='cuda', dtype=torch.complex64)
    k[..., 0, 0] = 0  # zero kernel bins make the denominator equal the regularizer
    k.requires_grad_()
    regularizer = torch.full((1, channels, 1, 1), reg, device='cuda').mul(torch.rand(1, channels, 1, 1, device='cuda') + .5).requires_grad_()
    out = torch.ops.converse2d._training_full_spectral(y, p, k, regularizer, 1)
    upstream = torch.randn_like(out)
    inputs = (y, k, regularizer) if shared else (y, p, k, regularizer)
    grads = torch.autograd.grad(out, inputs, upstream)
    return [out, *grads]


def module_case(batch, seed):
    torch.manual_seed(seed)
    x = torch.randn(batch, 128, 100, 100, device='cuda', requires_grad=True)
    weight = torch.rand(1, 128, 3, 3, device='cuda', requires_grad=True)
    bias = torch.zeros(1, 128, 1, 1, device='cuda', requires_grad=True)
    out = torch.ops.converse2d.forward(x, x, weight, bias, 1, 1e-5)
    return [out, *torch.autograd.grad(out, (x, weight, bias), torch.randn_like(out))]


def cases():
    seed = 7100
    for batch, channels, (h, w) in itertools.product((1, 2, 3, 4, 5), (1, 3, 8), ((5, 7), (20, 30), (33, 17))):
        for kb, kc, shared in itertools.product(sorted({1, batch}), sorted({1, channels}), (False, True)):
            for reg, scale in ((.1, 1.), (1e-8, 1.), (.1, 1e18)):
                seed += 1
                name = f'spectral_b{batch}_c{channels}_{h}x{w}_kb{kb}_kc{kc}_{"shared" if shared else "indep"}_r{reg:g}_s{scale:g}'
                yield name, lambda a=(batch, channels, h, w, kb, kc, shared, reg, scale, seed): spectral_case(*a)
    yield 'spectral_b4_c128_100x100_kb1_kc128_shared', lambda: spectral_case(4, 128, 100, 100, 1, 128, True, 1e-5, 1., 7000)
    yield 'spectral_b4_c64_96x96_kb4_kc64_indep', lambda: spectral_case(4, 64, 96, 96, 4, 64, False, 1e-3, 1., 7001)
    for batch in (1, 4):
        yield f'module_b{batch}_c128_100x100', lambda b=batch: module_case(b, 7002 + b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    load_extension()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text(encoding='utf-8'))
    report = dict(binary_sha256=manifest['binary_sha256'], torch=torch.__version__, gpu=torch.cuda.get_device_name(0), cases={})
    for name, run in cases():
        report['cases'][name] = [dict(shape=list(t.shape), dtype=str(t.dtype), sha256=digest(t),
                                      finite=bool(torch.isfinite(t).all())) for t in run()]
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding='utf-8'))
        mismatches = [name for name in baseline['cases'] if baseline['cases'][name] != report['cases'].get(name)]
        missing = sorted(set(baseline['cases']) ^ set(report['cases']))
        report['comparison'] = dict(baseline=str(args.compare), baseline_binary_sha256=baseline['binary_sha256'],
                                    cases=len(baseline['cases']), mismatches=mismatches, missing=missing,
                                    identical=not mismatches and not missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(cases=len(report['cases']), comparison=report.get('comparison')), indent=2))
    if args.compare and not report['comparison']['identical']:
        sys.exit(1)


if __name__ == '__main__':
    main()
