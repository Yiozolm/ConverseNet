"""Half-spectrum fused inference forward vs production's half-spectrum path.

Under no_grad, production runs rfft2 -> correction kernel -> irfft2 with the
kernel spectrum cached per weight tensor. Three candidates get the same
precomputed k and l (as the cache provides them):
- `full`: the training-plane fused forward (full spectrum, 80 KB at 100x100);
- `half`: the half-spectrum fused forward (R2C rows, C2C columns, C2R rows,
  41 KB at 100x100, s2 128x128 and s3 144x144 planes under 99 KB).
Output accuracy is gated by numerical_policy.comparison against production
with the FP64 reference; timing is paired CUDA events. `debug_rfft2` is also
checked against torch.fft.rfft2 as a per-transform proxy. Research only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import torch
import torch.nn.functional as F
import study
from study import comparison, paired
import ops_study as ops
from ops_study import Case, MODES

CASES = {
    'pblock_s1_b4_c128_96_pad2': Case(4, 128, 96, 96, 1, 2, 'circular', 1, 3, 1e-5, 'shared'),
    'pblock_s1_b4_c64_96_pad2': Case(4, 64, 96, 96, 1, 2, 'circular', 1, 3, 1e-5, 'shared'),
    'data_s1_b4_c64_96_k7': Case(4, 64, 96, 96, 1, 0, 'circular', 4, 7, 1e-3, 'shared'),
    'replicate_s1_b4_c64_92_pad4_k5': Case(4, 64, 92, 92, 1, 4, 'replicate', 1, 5, 1e-5, 'shared'),
    'forward_s1_b4_c64_64': Case(4, 64, 64, 64, 1, 0, 'circular', 1, 3, 1e-5, 'shared'),
    'data_s3_b4_c64_32_k7': Case(4, 64, 32, 32, 3, 0, 'circular', 4, 7, 1e-3, 'nearest'),
    'data_s2_b4_c64_48_k7': Case(4, 64, 48, 48, 2, 0, 'circular', 4, 7, 1e-3, 'nearest'),
    'sr_s2_b4_c64_64_k7': Case(4, 64, 64, 64, 2, 0, 'circular', 4, 7, 1e-3, 'nearest'),
    'sr_s3_b2_c32_48_k7': Case(2, 32, 48, 48, 3, 0, 'circular', 2, 7, 1e-3, 'nearest'),
}


def half_spectrum(weight, bias, H, W, eps):
    kh, kw = weight.shape[-2:]
    psf = torch.roll(F.pad(weight, (0, W - kw, 0, H - kh)), (-(kh // 2), -(kw // 2)), (-2, -1))
    return torch.fft.rfft2(psf).contiguous(), (torch.sigmoid(bias - 9.0) + eps).reshape(-1).contiguous()


def rfft_check(ext, sizes):
    rows = {}
    for n in sizes:
        torch.manual_seed(70000 + n)
        x = torch.randn(4, 8, n, n, device='cuda')
        ref = torch.fft.rfft2(x.double())
        rows[n] = comparison(ext.debug_rfft2(x), torch.fft.rfft2(x), ref)
        r = rows[n]
        print(f"rfft2 {n:3d}x{n:<3d} {'ok  ' if r['passed'] else 'FAIL'} rel_l2 {r['candidate']['rel_l2']:.3e} "
              f"(cufft {r['baseline']['rel_l2']:.3e}, {r['ratio']['rel_l2']:.2f}x) max_abs {r['ratio']['max_abs']:.2f}x",
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--cases', nargs='*', default=list(CASES))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; keep earlier evidence')
    args.output.mkdir(parents=True)
    study.load_extension()
    ext = study.build()
    device = torch.cuda.current_device()
    report = dict(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  torch=torch.__version__, cuda=torch.version.cuda, smem_optin_bytes=ext.smem_capacity(device),
                  started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cases={},
                  occupancy={n: ext.half_occupancy(n, device) for n in (64, 96, 100)})
    print(f"{report['device']}: opt-in shared memory per block {report['smem_optin_bytes']} bytes; "
          f"half s1 blocks per SM {report['occupancy']}", flush=True)
    report['rfft2'] = rfft_check(ext, (24, 32, 48, 64, 72, 96, 100, 128, 144))
    for name in args.cases:
        case = CASES[name]
        full_ok, half_ok = ops.supported(ext, case, device), ext.half_supported(case.h, case.w, case.scale, case.pad, device)
        side = case.scale * (case.h + 2 * case.pad)
        row = dict(case=case._asdict(), full='fused' if full_ok else 'production', half='fused' if half_ok else 'production',
                   full_plane_bytes=side * side * 8, half_plane_bytes=side * (side // 2 + 1) * 8)
        report['cases'][name] = row
        print(f"{name:32s} full plane {row['full_plane_bytes']:7d} B -> {row['full']}; half plane "
              f"{row['half_plane_bytes']:7d} B -> {row['half']}", flush=True)
        if not half_ok:
            continue
        x, weight, bias, _ = ops.make_inputs(case, 74001)
        x, weight, bias = x.detach(), weight.detach(), bias.detach()
        with torch.no_grad():
            if case.scale == 1:
                H, W = case.h + 2 * case.pad, case.w + 2 * case.pad
                kf, l = ops.kernel_spectrum(weight, bias, H, W, case.eps)
                l = l.reshape(-1).contiguous()
                kh, _ = half_spectrum(weight, bias, H, W, case.eps)
                fns = {'production': lambda: ops.production(case, x, weight, bias),
                       'half': lambda: ext.half_forward(x, kh, l, case.pad, MODES[case.mode])}
                if full_ok:
                    fns['full'] = lambda: ext.forward(x, kf, l, case.pad, 1, MODES[case.mode])
            else:
                x0 = ops.prior(case, x)
                kf, l = ops.kernel_spectrum(weight, bias, x0.shape[-2], x0.shape[-1], case.eps)
                l = l.reshape(-1).contiguous()
                kh, _ = half_spectrum(weight, bias, x0.shape[-2], x0.shape[-1], case.eps)
                fns = {'production': lambda: ops.production(case, x, weight, bias),
                       'half': lambda: ext.half_scaled_forward(x, x0, kh, l, case.scale)}
                if full_ok:
                    fns['full'] = lambda: ext.scaled_forward(x, x0, kf, l, case.scale)
            try:
                baseline = fns['production']()
                ref = ops.reference64(case, x, weight, bias)
                row['accuracy'] = {}
                for label in ('half', 'full'):
                    if label not in fns:
                        continue
                    acc = row['accuracy'][label] = comparison(fns[label](), baseline, ref)
                    print(f"{'':32s} {label:5s} {'ok  ' if acc['passed'] else 'FAIL'} rel_l2 {acc['candidate']['rel_l2']:.3e} "
                          f"(prod {acc['baseline']['rel_l2']:.3e}, {acc['ratio']['rel_l2'] or 0:.2f}x) "
                          f"max_abs {acc['ratio']['max_abs'] or 0:.2f}x", flush=True)
                row['bitwise_repeatable'] = torch.equal(fns['half']().view(torch.int32), fns['half']().view(torch.int32))
            except Exception as error:  # keep failures in the report
                row['error'] = repr(error)
                print(f"{'':32s} ERROR {error!r}", flush=True)
                continue
            row['timing'] = paired(fns, args.rounds, args.iters)
        t = {k: v['median_us'] for k, v in row['timing'].items()}
        row['speedup'] = {k: t['production'] / v for k, v in t.items() if k != 'production'}
        print(f"{'':32s} fwd production {t['production']:7.1f}us | half {t['half']:7.1f}us ({row['speedup']['half']:.2f}x)"
              + (f" | full {t['full']:7.1f}us ({row['speedup']['full']:.2f}x)" if 'full' in t else '')
              + f" repeatable {row['bitwise_repeatable']}", flush=True)
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / 'half.json').write_text(json.dumps(report, indent=1))
    lines = [f"# Half-spectrum fused inference: {report['device']} (sm_{''.join(map(str, report['capability']))})", '',
             f"- torch {report['torch']} / CUDA {report['cuda']}; opt-in shared memory per block "
             f"{report['smem_optin_bytes']} B; half s1 blocks per SM {report['occupancy']}", '',
             '| case | half plane | production fwd us | half us | full us | half rel-L2 ratio | full rel-L2 ratio | budgets | repeatable |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, row in report['cases'].items():
        t = {k: v['median_us'] for k, v in row.get('timing', {}).items()}
        acc = row.get('accuracy', {})
        budgets = ('pass' if all(a['passed'] for a in acc.values()) else 'FAIL') if acc else row.get('error', '-')
        if 'half' in t:
            full = f"{t['full']:.0f} ({row['speedup']['full']:.2f}x)" if 'full' in t else f"- ({row['full']})"
            lines.append(f"| {name} | {row['half_plane_bytes'] // 1024} KB | {t['production']:.0f} | "
                         f"{t['half']:.0f} ({row['speedup']['half']:.2f}x) | {full} | "
                         f"{acc['half']['ratio']['rel_l2']:.2f} | {acc['full']['ratio']['rel_l2'] if 'full' in acc else 0:.2f} | "
                         f"{budgets} | {row.get('bitwise_repeatable')} |")
        else:
            lines.append(f"| {name} | {row['half_plane_bytes'] // 1024} KB | - | - ({row['half']}) | - | - | - | {budgets} | - |")
    lines += ['', 'Forward only under no_grad with k and l given; not a whole-model result.']
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
