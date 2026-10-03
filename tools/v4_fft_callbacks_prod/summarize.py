"""Summarize perf.py rounds: medians of per-round medians, every label vs ATen.

With --kernels, also list per kernel name the median time per call for each
label, largest first (records without kernel_us_by_name are skipped).
"""
import argparse
import json
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('perf')
parser.add_argument('--kernels', action='store_true')
args = parser.parse_args()
rows = [json.loads(line) for line in open(args.perf, encoding='utf-8')]
labels = list(dict.fromkeys(r['label'] for r in rows))
others = [label for label in labels if label != 'aten']
m = lambda values, key: statistics.median(v[key] for v in values)
for cfg in rows[0]['results']:
    a = [r['results'][cfg] for r in rows if r['label'] == 'aten']
    ka, ca = m(a, 'kernel_time_per_call_us'), m(a, 'call_median_ms')
    print(f"{cfg:28s} aten rounds {len(a)} kernels {a[0]['kernels_per_call']:.0f} | kernel us {ka:6.0f} | call ms {ca:.3f}")
    print(f"    call ms aten {[round(v['call_median_ms'], 2) for v in a]}")
    for label in others:
        c = [r['results'][cfg] for r in rows if r['label'] == label]
        kc, cc = m(c, 'kernel_time_per_call_us'), m(c, 'call_median_ms')
        print(f"  {label:26s} rounds {len(c)} kernels {c[0]['kernels_per_call']:.0f} lto {c[0]['lto_fft_per_call']:.0f} | "
              f"kernel us {kc:6.0f} ({ka / kc:.3f}x) | call ms {cc:.3f} ({ca / cc:.3f}x)")
        print(f"    call ms {label} {[round(v['call_median_ms'], 2) for v in c]}")
    if args.kernels:
        per = {label: [r['results'][cfg]['kernel_us_by_name'] for r in rows
                       if r['label'] == label and 'kernel_us_by_name' in r['results'][cfg]] for label in labels}
        names = sorted({n for runs in per.values() for run in runs for n in run},
                       key=lambda n: -max(statistics.median(run.get(n, 0) for run in runs) if runs else 0
                                          for runs in per.values()))
        print('    kernel us/call: ' + ' | '.join(labels))
        for name in names:
            values = [f"{statistics.median(run.get(name, 0) for run in per[label]):7.1f}" if per[label] else '      -'
                      for label in labels]
            print(f"    {' '.join(values)}  {name[:110]}")
