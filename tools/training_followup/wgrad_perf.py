"""Numerically admitted, complete single-layer FWD+VJP paired measurements.

All NHWC layout copies and native input/bias VJPs remain inside each call.
One untimed diagnostic profile records the actual kernel/copy routes. No model
replacement or training claim is authorized by this single-layer benchmark.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

import wgrad_common as common
from wgrad_gate import device_case, values


def profile_summary(path):
    events = json.load(gzip.open(path, 'rt', encoding='utf-8'))['traceEvents']
    cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'user_annotation')]
    threads = defaultdict(list)
    for event in cpu:
        threads[(event['pid'], event['tid'])].append(event)
    parents, external = {}, {}
    for group in threads.values():
        stack = []
        for event in sorted(group, key=lambda e: (e['ts'], -e['dur'])):
            while stack and (event['ts'] >= stack[-1]['ts']+stack[-1]['dur']
                             or event['ts']+event['dur'] > stack[-1]['ts']+stack[-1]['dur']+.01):
                stack.pop()
            parents[id(event)] = stack[-1] if stack else None
            external[event.get('args', {}).get('External id')] = event
            stack.append(event)
    sequence_routes = {}
    for event in cpu:
        sequence = event.get('args', {}).get('Sequence number')
        if sequence is None or 'Backward' in event['name'] or event['name'].startswith('autograd::'):
            continue
        parent = event
        while parent is not None:
            if parent['name'].startswith('wgrad_variant/'):
                sequence_routes.setdefault(sequence, parent['name'].split('/', 1)[1])
                break
            parent = parents.get(id(parent))
    routes = {name: dict(kernel_count=0, kernel_gpu_us=0., names={}, kernels=[], copy_kernel_ids=[])
              for name in ('native', 'candidate')}
    for index, event in enumerate(e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'):
        parent = external.get(event.get('args', {}).get('External id'))
        route, ancestors = None, []
        while parent is not None:
            ancestors.append(parent['name'])
            if parent['name'].startswith('wgrad_variant/'):
                route = parent['name'].split('/', 1)[1]
            if 'Backward' in parent['name'] or parent['name'].startswith('autograd::engine::evaluate_function:'):
                route = route or sequence_routes.get(parent.get('args', {}).get('Sequence number'))
            parent = parents.get(id(parent))
        if route not in routes:
            raise RuntimeError('Unmapped kernel in isolated profile')
        row = routes[route]
        row['kernel_count'] += 1
        row['kernel_gpu_us'] += event['dur']
        totals = row['names'].setdefault(event['name'], dict(count=0, gpu_us=0.))
        totals['count'] += 1
        totals['gpu_us'] += event['dur']
        row['kernels'].append(dict(id=index, name=event['name'], duration_us=event['dur'], cpu_ancestry=ancestors))
        if 'aten::contiguous' in ancestors or 'aten::copy_' in ancestors or 'aten::clone' in ancestors:
            row['copy_kernel_ids'].append(index)
    return routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=common.ROOT)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    args.root, args.output, args.gate = args.root.resolve(), args.output.resolve(), args.gate.resolve()
    if args.output.exists():
        parser.error('Use new evidence filenames')
    gate = json.loads(args.gate.read_text())
    if not gate.get('all_gpu_cases_passed') or gate.get('status') != 'complete':
        raise RuntimeError('Complete all-case numerical gate required')
    for filename in ('wgrad_channels_last.py', 'wgrad_common.py', 'wgrad_gate.py'):
        if common.sha(Path(__file__).parent / filename) != gate['metadata']['tool_sha256'][filename]:
            raise RuntimeError('Candidate/gate dependency changed: ' + filename)
    torch, loader = common.setup(args.root)
    import torch.nn.functional as F
    import wgrad_channels_last as candidate
    native = lambda x, w, b: F.conv2d(x, w, b)
    fixture_path = Path(gate['fixture_path'])
    if common.sha(fixture_path) != gate['fixture_sha256']:
        raise RuntimeError('Gate fixtures changed')
    captured = torch.load(fixture_path, map_location='cpu', weights_only=True)
    report = dict(kind='isolated_wgrad_complete_layer_performance', status='running', metadata=common.metadata(args.root, loader),
        gate_sha256=common.sha(args.gate), gate_path=str(args.gate), fixture_sha256=common.sha(fixture_path),
        settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters), cases={},
        scope='Only captured p.m_body.0.conv1.5 calls; complete forward+dx+dweight+dbias, including all layout copies, Python/custom-autograd and synchronization. No model replacement. Profile times are diagnostic, not benchmark samples.')
    single = argparse.Namespace(warmup=0, rounds=1, iters=args.iters)
    try:
        for index, (name, row) in enumerate(captured.items()):
            raw = tuple(row[key] for key in ('x', 'weight', 'bias', 'grad_output'))
            data, upstream = device_case(torch, raw, torch.float32, (True, True, True))
            functions = dict(native=lambda: values(torch, native, data, upstream),
                             candidate=lambda: values(torch, candidate.conv1x1, data, upstream))
            snapshots = {route: common.tensors.records(function()) for route, function in functions.items()}
            hashes_equal = all(snapshots['native'][key]['sha256'] == snapshots['candidate'][key]['sha256'] for key in snapshots['native'])
            if not hashes_equal:
                report['cases'][name] = dict(snapshots=snapshots, byte_equal=False)
                raise RuntimeError('Snapshot changed from admitted numerical lane')
            for function in functions.values():
                for _ in range(args.warmup):
                    function()
            rounds = {route: [] for route in functions}
            for round_index in range(args.rounds):
                order = ('native', 'candidate') if round_index % 2 == 0 else ('candidate', 'native')
                for route in order:
                    value = common.tensors.measure(torch, functions[route], single)['rounds'][0]
                    value.update(paired_round=round_index, order=list(order))
                    rounds[route].append(value)
            metric_names = ('wall_ms', 'cuda_event_ms', 'peak_extra_allocated_bytes')
            timings = {route: dict(rounds=rows, median={key: statistics.median(row[key] for row in rows) for key in metric_names})
                       for route, rows in rounds.items()}
            ratios = {key: [rounds['native'][i][key]/rounds['candidate'][i][key] for i in range(args.rounds)]
                      for key in ('wall_ms', 'cuda_event_ms')}
            report['cases'][name] = dict(byte_equal=True, snapshots=snapshots, timings=timings, paired_speedups=ratios,
                median_speedup={key: statistics.median(value) for key, value in ratios.items()})
            print(name, report['cases'][name]['median_speedup'], flush=True)
            if index == 0:
                trace = args.output.with_suffix('.trace.json.gz')
                if trace.exists():
                    raise RuntimeError('Trace path already exists')
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                    for route, function in functions.items():
                        with torch.profiler.record_function('wgrad_variant/' + route):
                            function()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(trace))
                report['diagnostic_profile'] = dict(path=str(trace), sha256=common.sha(trace), routes=profile_summary(trace))
            del data, upstream
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['affinity_after'] = common.affinity()
        common.write_report(args.output, report)


if __name__ == '__main__':
    main()
