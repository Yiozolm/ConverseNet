"""Offline Nsight Compute import and compact training-kernel summaries.

This script only imports existing reports; it never launches a GPU workload.
Recorded replay durations and rule speedup estimates are diagnostic metrics,
not end-to-end training timings or predicted benchmark improvements.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess


DEFAULT_NCU = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe')
METRICS = {
    'duration_ns_diagnostic': 'gpu__time_duration.avg',
    'replay_passes': 'profiler__replayer_passes',
    'block_threads': 'launch__block_size',
    'registers_per_thread': 'launch__registers_per_thread',
    'registers_allocated_per_thread': 'launch__registers_per_thread_allocated',
    'register_block_limit': 'launch__occupancy_limit_registers',
    'warp_block_limit': 'launch__occupancy_limit_warps',
    'theoretical_occupancy_pct': 'sm__maximum_warps_per_active_cycle_pct',
    'achieved_occupancy_pct': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'sm_throughput_pct': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'fp64_pipe_active_pct': 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram_sol_throughput_pct': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'dram_bytes_per_second': 'dram__bytes.sum.per_second',
    'l1_hit_pct': 'l1tex__t_sector_hit_rate.pct',
    'l2_hit_pct': 'lts__t_sector_hit_rate.pct',
    'l2_throughput_pct': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'active_warps_per_scheduler': 'smsp__warps_active.avg.per_cycle_active',
    'eligible_warps_per_scheduler': 'smsp__warps_eligible.avg.per_cycle_active',
    'issue_active_pct': 'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'no_eligible_pct': 'smsp__issue_inst0.avg.pct_of_peak_sustained_active',
    'warp_cycles_per_issued_instruction': 'smsp__average_warp_latency_per_inst_issued.ratio',
    'local_spilling_requests': 'derived__local_spilling_requests',
    'local_spilling_request_pct': 'derived__local_spilling_requests_pct',
}
STALL_PREFIX = 'smsp__average_warps_issue_stalled_'
STALL_SUFFIX = '_per_issue_active.ratio'


def number(value):
    try:
        return float(value.replace(',', ''))
    except (AttributeError, ValueError):
        return None


def export(ncu, report, raw, details):
    for page, options, path in (
            ('raw', ['--csv', '--print-units', 'base'], raw),
            ('details', ['--print-details', 'header', '--print-rule-details',
                         '--print-metric-name', 'label-name'], details)):
        subprocess.run([str(ncu), '--import', str(report), '--page', page, *options,
                        '--log-file', str(path)], check=True)


def read_raw(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.reader(source))
    start = next(i for i, row in enumerate(rows) if row and row[0] == 'ID')
    header = rows[start]
    units = dict(zip(header, rows[start+1]))
    launches = [dict(zip(header, row)) for row in rows[start+2:]
                if row and row[0].strip().isdigit()]
    return units, launches


def read_rules(path):
    rules, kernel, section, launch_index = [], '', '', -1
    current = None
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if raw.startswith('  ') and not raw.startswith('    ') and 'Context ' in raw:
            kernel = line.split(' (')[0]
            launch_index += 1
        if line.startswith('Section:'):
            section = line.removeprefix('Section:').strip()
        match = re.match(r'^(OPT|INF|WRN)\s+(.*)', line)
        if match:
            current = {'kernel': kernel, 'launch_index': launch_index,
                       'section': section, 'severity': match[1], 'text': match[2]}
            rules.append(current)
        elif not line or line.startswith(('>', '---')):
            current = None
        elif current is not None:
            current['text'] += ' ' + line
    return rules


def summarize(report):
    raw, details = report.with_suffix('.raw.csv'), report.with_suffix('.details.txt')
    units, launches = read_raw(raw)
    records = []
    for launch in launches:
        metrics = {name: number(launch.get(key)) for name, key in METRICS.items()}
        stalls = {key[len(STALL_PREFIX):-len(STALL_SUFFIX)]: number(value)
                  for key, value in launch.items()
                  if key.startswith(STALL_PREFIX) and key.endswith(STALL_SUFFIX)}
        interval = metrics['warp_cycles_per_issued_instruction']
        shares = {name: 100*value/interval if value is not None and interval else None
                  for name, value in stalls.items()}
        pipes = {key: number(value) for key, value in launch.items()
                 if ('__pipe_' in key or 'inst_executed_pipe_' in key)
                 and '.avg.' in key}
        records.append({'id': int(launch['ID']), 'kernel': launch['Kernel Name'],
                        'grid': launch.get('Grid Size'), 'block': launch.get('Block Size'),
                        'metrics': metrics, 'stall_cycles_per_issued_instruction': stalls,
                        'stall_share_of_warp_issue_interval_pct': shares,
                        'collected_pipe_metrics': pipes})
    command_path = report.with_suffix('.command.json')
    metadata_path = report.with_suffix('.metadata.json')
    return {'report': str(report), 'raw_csv': str(raw), 'details': str(details),
            'command': json.loads(command_path.read_text(encoding='utf-8-sig')) if command_path.exists() else None,
            'metadata': json.loads(metadata_path.read_text(encoding='utf-8-sig')) if metadata_path.exists() else None,
            'metric_names': METRICS, 'metric_units': {name: units.get(key) for name, key in METRICS.items()},
            'launches': records, 'rules': read_rules(details)}


def table(records):
    lines = ['| Report / launch | Registers | Occ. theoretical / achieved % | DRAM SOL % / GB/s | SM / FP64 pipe % | Eligible warps | Long scoreboard % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'n/a' if v is None else f'{v:.2f}'
    for report in records:
        for launch in report['launches']:
            m = launch['metrics']
            name = launch['kernel'].split('(')[0].removeprefix('void ').replace('<unnamed>::', '')
            if len(name) > 90:
                name = name[:87] + '...'
            bandwidth = m['dram_bytes_per_second']
            lines.append(f'| {Path(report["report"]).parent.name}/{Path(report["report"]).name}: {launch["id"]} {name} | '
                         f'{fmt(m["registers_per_thread"])} | {fmt(m["theoretical_occupancy_pct"])} / {fmt(m["achieved_occupancy_pct"])} | '
                         f'{fmt(m["dram_sol_throughput_pct"])} / {fmt(bandwidth/1e9 if bandwidth is not None else None)} | '
                         f'{fmt(m["sm_throughput_pct"])} / {fmt(m["fp64_pipe_active_pct"])} | {fmt(m["eligible_warps_per_scheduler"])} | '
                         f'{fmt(launch["stall_share_of_warp_issue_interval_pct"].get("long_scoreboard"))} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--ncu', type=Path, default=DEFAULT_NCU)
    parser.add_argument('--export', action='store_true', help='Offline import/export each report before parsing')
    parser.add_argument('--output', type=Path, default=Path('artifacts/nsight_training/ncu_summary.json'))
    args = parser.parse_args()
    records = []
    for report in args.reports:
        report = report.resolve()
        if args.export:
            export(args.ncu, report, report.with_suffix('.raw.csv'), report.with_suffix('.details.txt'))
        records.append(summarize(report))
    output = {'scope': 'Offline NCU diagnostic counters; replay duration and rule speedups are not training benchmarks.',
              'unavailable_metrics': 'null means unavailable, never zero', 'reports': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    markdown = ('NCU diagnostic metrics only. DRAM SOL percentage and measured byte rate are distinct counters.\n\n'
                + table(records) + '\n\nMetric definitions:\n\n'
                '- DRAM SOL %: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, '
                'not the multi-level `gpu__compute_memory_throughput` metric.\n'
                '- GB/s: `dram__bytes.sum.per_second / 1e9`.\n'
                '- SM %: `sm__throughput.avg.pct_of_peak_sustained_elapsed`.\n'
                '- FP64 pipe %: `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed`.\n'
                '- Occupancy theoretical/achieved: `sm__maximum_warps_per_active_cycle_pct` / '
                '`sm__warps_active.avg.pct_of_peak_sustained_active`.\n'
                '- Eligible warps: `smsp__warps_eligible.avg.per_cycle_active`, per scheduler.\n'
                '- Long scoreboard %: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` '
                'divided by `smsp__average_warp_latency_per_inst_issued.ratio`, times 100; '
                'this is a warp issue-interval share, not wall-time share.\n'
                '- Null or n/a means unavailable. In particular, unavailable spill metrics do not prove zero spilling.\n')
    args.output.with_suffix('.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
