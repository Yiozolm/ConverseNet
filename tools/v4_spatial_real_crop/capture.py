"""Byte and layout capture for the public differentiable forward at every scale.

The unpadded spatial() path now ends in real_crop(z, 0) instead of at::real.
Its VJP writes (g, +0) in one kernel instead of a zero fill plus strided copy,
so every output and gradient must keep identical bytes, shape, stride and
storage offset. Run on the baseline build, then on the candidate with --compare.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
import torch
from extension_loader import load_extension


def record(t):
    data = t.detach()
    return dict(shape=list(data.shape), stride=list(data.stride()), offset=data.storage_offset(),
                dtype=str(data.dtype), sha256=hashlib.sha256(data.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())


def upstream(shape, kind, generator):
    g = torch.randn(shape, generator=generator).cuda()
    if kind == 'transposed':
        return torch.randn(shape[0], shape[1], shape[3], shape[2], generator=generator).cuda().transpose(2, 3)
    if kind == 'expanded':
        return torch.randn(1, shape[1], shape[2], shape[3], generator=generator).cuda().expand(shape)
    if kind == 'zeros':
        return torch.zeros(shape, device='cuda').neg()  # -0 must still embed as (-0, +0)
    return g


def module_case(batch, channels, h, w, scale, kb, kc, shared, kind, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, channels, h, w, generator=generator).cuda().requires_grad_()
    x0 = x if shared else torch.randn(batch, channels, h * scale, w * scale, generator=generator).cuda().requires_grad_()
    weight = torch.rand(kb, kc, 3, 3, generator=generator).cuda().requires_grad_()
    bias = torch.randn(1, channels, 1, 1, generator=generator).cuda().requires_grad_()
    out = torch.ops.converse2d.forward(x, x0, weight, bias, scale, 1e-5)
    inputs = (x, weight, bias) if shared else (x, x0, weight, bias)
    grads = torch.autograd.grad(out, inputs, upstream(out.shape, kind, generator))
    return [out, *grads]


def double_backward_case(seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 6, 7, generator=generator).cuda().requires_grad_()
    weight = torch.rand(1, 3, 3, 3, generator=generator).cuda().requires_grad_()
    bias = torch.randn(1, 3, 1, 1, generator=generator).cuda().requires_grad_()
    out = torch.ops.converse2d.forward(x, x, weight, bias, 1, 1e-5)
    gx, gw = torch.autograd.grad(out.square().sum(), (x, weight), create_graph=True)
    return [out, gx, gw, *torch.autograd.grad(gx.sum() + gw.sum(), (x, weight, bias))]


def cases():
    seed = 9100
    for scale, (batch, channels, h, w), kind in itertools.product(
            (1, 2, 3), ((1, 1, 5, 7), (2, 3, 8, 6), (4, 8, 12, 9), (3, 5, 16, 16)), ('plain', 'transposed', 'expanded', 'zeros')):
        for kb, kc, shared in itertools.product(sorted({1, batch}), sorted({1, channels}), (True, False) if scale == 1 else (False,)):
            seed += 1
            yield (f'module_s{scale}_b{batch}_c{channels}_{h}x{w}_kb{kb}_kc{kc}_{"shared" if shared else "indep"}_{kind}',
                   lambda a=(batch, channels, h, w, scale, kb, kc, shared, kind, seed): module_case(*a))
    for batch in (1, 4):
        yield f'module_s1_b{batch}_c128_100x100', lambda b=batch: module_case(b, 128, 100, 100, 1, 1, 128, True, 'plain', 9000 + b)
    yield 'module_s2_b4_c64_48x48', lambda: module_case(4, 64, 48, 48, 2, 1, 64, False, 'plain', 9005)
    yield 'module_s3_b2_c32_32x32', lambda: module_case(2, 32, 32, 32, 3, 1, 32, False, 'plain', 9006)
    yield 'double_backward_s1', lambda: double_backward_case(9007)


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
    report = dict(binary_sha256=manifest['binary_sha256'], torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  cases={name: [record(t) for t in run()] for name, run in cases()})
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
