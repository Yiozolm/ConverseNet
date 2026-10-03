"""P0 feasibility study for folding FFT-adjacent work into cuFFT LTO callbacks.

1. Plan equivalence: our callback-free plans vs ATen fft2 / unscaled ifft2.
2. Load callbacks: real->complex (and circular pad) inside the forward FFT vs
   ATen promote(+pad)+fft2. A sentinel tail detects cuFFT using the real
   input buffer as complex-sized scratch.
3. Store callback: 1/N scaling inside the inverse FFT vs ATen ifft2.
4. Kernel names and paired CUDA-event timing at production shapes.

Byte identity is reported first; if bytes differ, errors against an FP64
FFT of the same FP32 input are recorded for both paths. Research only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT), str(HERE)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import torch
import torch.nn.functional as F
from extension_loader import load_extension
import loader


def same(a, b):
    a, b = a.contiguous(), b.contiguous()
    return a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype != torch.complex64 else torch.view_as_real(a).view(torch.int32),
                                               b.view(torch.uint8) if b.dtype != torch.complex64 else torch.view_as_real(b).view(torch.int32))


def errors(candidate, aten, exact):
    def stats(v):
        d = (v.to(torch.complex128) - exact)
        return dict(rel_l2=float(d.abs().norm() / exact.abs().norm().clamp_min(1e-300)), max_abs=float(d.abs().max()))
    diff = torch.view_as_real(candidate) != torch.view_as_real(aten)
    return dict(candidate=stats(candidate), aten=stats(aten), differing_components=int(diff.sum()))


def inputs(shape, kind, generator):
    x = torch.randn(shape, generator=generator)
    if kind == 'specials':
        atoms = torch.tensor([0., -0., 2. ** -149, -(2. ** -149), 2. ** -126, 1e30, -1e30, 3.])
        x = atoms[torch.randint(0, len(atoms), shape, generator=generator)]
    if kind == 'scaled':
        x = x * 1e18
    return x.cuda()


def correctness(ext):
    generator = torch.Generator().manual_seed(51001)
    shapes = [(4, 128, 100, 100), (4, 128, 96, 96), (4, 64, 48, 48), (2, 3, 33, 17), (1, 1, 7, 9), (3, 5, 64, 64), (2, 2, 1, 5)]
    pads = [((4, 128, 96, 96), 2), ((2, 3, 31, 15), 1), ((2, 3, 8, 6), 3), ((1, 4, 24, 28), 2)]
    results = []
    for kind in ('randn', 'specials', 'scaled'):
        for shape in shapes:
            x = inputs(shape, kind, generator)
            z = torch.complex(x, inputs(shape, kind, generator))
            row = dict(kind=kind, shape=list(shape))
            for name, ours, aten, exact_in, inverse in (
                    ('plan_forward', lambda: ext.fft2(z, False), lambda: torch.fft.fft2(z), z, False),
                    ('plan_inverse_unscaled', lambda: ext.fft2(z, True), lambda: torch.fft.ifft2(z, norm='forward'), z, True),
                    ('load_real', None, lambda: torch.fft.fft2(x), x, False),
                    ('store_scaled', lambda: ext.ifft2_scaled(z), lambda: torch.fft.ifft2(z), z, True)):
                try:
                    if name == 'load_real':
                        out, guard = guarded_real(ext, x, 0)
                        row['load_real_buffer_intact'] = guard
                    else:
                        out = ours()
                    ref = aten()
                    torch.cuda.synchronize()
                    row[name] = dict(identical=same(out, ref))
                    if not row[name]['identical']:
                        exact = exact_in.to(torch.complex128)
                        exact = torch.fft.ifft2(exact, norm='forward' if name == 'plan_inverse_unscaled' else 'backward') if inverse else torch.fft.fft2(exact)
                        row[name].update(errors(out, ref, exact))
                except RuntimeError as error:
                    row[name] = dict(error=str(error).splitlines()[0])
            results.append(row)
        for shape, pad in pads:
            x = inputs(shape, kind, generator)
            row = dict(kind=kind, shape=list(shape), pad=pad)
            try:
                out, guard = guarded_real(ext, x, pad)
                ref = torch.fft.fft2(F.pad(x, (pad,) * 4, mode='circular').to(torch.complex64))
                torch.cuda.synchronize()
                row['load_circular'] = dict(identical=same(out, ref), buffer_intact=guard)
                if not row['load_circular']['identical']:
                    exact = torch.fft.fft2(F.pad(x, (pad,) * 4, mode='circular').to(torch.complex128))
                    row['load_circular'].update(errors(out, ref, exact))
            except RuntimeError as error:
                row['load_circular'] = dict(error=str(error).splitlines()[0])
            results.append(row)
    return results


def guarded_real(ext, x, pad):
    """Run the load-callback FFT on x placed in a buffer with a sentinel tail."""
    n = x.numel()
    out_numel = x.size(0) * x.size(1) * (x.size(2) + 2 * pad) * (x.size(3) + 2 * pad)
    buffer = torch.full((n + 2 * out_numel,), float('nan'), device='cuda')
    buffer.view(torch.int32)[n:] = 0x7fc0dead
    view = buffer[:n].view(x.shape)
    view.copy_(x)
    before = hashlib.sha256(buffer.cpu().numpy().tobytes()).hexdigest()
    out = ext.fft2_real(view, pad)
    torch.cuda.synchronize()
    return out, hashlib.sha256(buffer.cpu().numpy().tobytes()).hexdigest() == before


def kernels(fn):
    fn(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return [dict(name=e.name[:120], us=e.device_time) for e in prof.events() if e.device_time > 0]


def timing(ext, rounds, iters):
    torch.manual_seed(51002)
    x96 = torch.randn(4, 128, 96, 96, device='cuda')
    x100 = torch.randn(4, 128, 100, 100, device='cuda')
    z100 = torch.randn(4, 128, 100, 100, device='cuda', dtype=torch.complex64)
    pad = torch.ops.converse2d._training_pad_complex
    pairs = {
        'promote_fft2_B4C128_100': (lambda: torch.fft.fft2(x100), lambda: ext.fft2_real(x100, 0)),
        'pad2_fft2_B4C128_96to100': (lambda: torch.fft.fft2(pad(x96, 2)), lambda: ext.fft2_real(x96, 2)),
        'ifft2_scale_B4C128_100': (lambda: torch.fft.ifft2(z100), lambda: ext.ifft2_scaled(z100)),
    }
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def measure(fn):
        values = []
        for _ in range(iters):
            start.record(); fn(); stop.record(); stop.synchronize()
            values.append(start.elapsed_time(stop) * 1e3)
        return statistics.median(values)

    out = {}
    for name, (aten, ours) in pairs.items():
        for fn in (aten, ours):
            for _ in range(5):
                fn()
        torch.cuda.synchronize()
        a, o = [], []
        for r in range(rounds):
            order = (aten, ours) if r % 2 == 0 else (ours, aten)
            for fn in order:
                (a if fn is aten else o).append(measure(fn))
        out[name] = dict(aten_us=a, callback_us=o, speedup=statistics.median(a) / statistics.median(o),
                         aten_kernels=kernels(aten), callback_kernels=kernels(ours))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    load_extension()
    ext, identity = loader.build()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(identity=identity, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  correctness=correctness(ext), timing=timing(ext, args.rounds, args.iters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    summary = {}
    for row in report['correctness']:
        for key in ('plan_forward', 'plan_inverse_unscaled', 'load_real', 'store_scaled', 'load_circular'):
            if key in row:
                v = row[key]
                state = 'error' if 'error' in v else ('identical' if v['identical'] else 'differs')
                summary.setdefault(key, {}).setdefault(state, 0)
                summary[key][state] += 1
    print(json.dumps(dict(summary=summary, intact=all(r.get('load_real_buffer_intact', True) and r.get('load_circular', {}).get('buffer_intact', True)
                                                     for r in report['correctness']),
                          speedups={k: round(v['speedup'], 3) for k, v in report['timing'].items()}), indent=2))


if __name__ == '__main__':
    main()
