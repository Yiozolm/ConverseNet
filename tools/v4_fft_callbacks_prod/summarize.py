"""Summarize perf.py rounds: medians of per-round medians, callbacks vs ATen."""
import json
import statistics
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding='utf-8')]
for cfg in rows[0]['results']:
    a = [r['results'][cfg] for r in rows if r['label'] == 'aten']
    c = [r['results'][cfg] for r in rows if r['label'] == 'callbacks']
    m = lambda values, key: statistics.median(v[key] for v in values)
    ka, kc = m(a, 'kernel_time_per_call_us'), m(c, 'kernel_time_per_call_us')
    ca, cc = m(a, 'call_median_ms'), m(c, 'call_median_ms')
    print(f"{cfg:28s} rounds {len(a)}/{len(c)} kernels {a[0]['kernels_per_call']:.0f}->{c[0]['kernels_per_call']:.0f} "
          f"lto {c[0]['lto_fft_per_call']:.0f} | kernel us {ka:6.0f}->{kc:6.0f} ({ka / kc:.3f}x) | "
          f"call ms {ca:.3f}->{cc:.3f} ({ca / cc:.3f}x)")
    print('    call ms aten', [round(v['call_median_ms'], 2) for v in a], 'callbacks', [round(v['call_median_ms'], 2) for v in c])
