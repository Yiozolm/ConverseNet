"""Fresh validation of installed models.pointwise, never a research patch.

Default --stage all runs the production operator gate, model regression, then
complete B4 Adam timings. Prototype evidence is provenance, not admission.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import traceback

import torch
import torch.nn.functional as F
import wgrad_study_c128_to64 as common
import wgrad_model_c128_to64 as model_helpers
from numerical_policy import comparison, error_metrics


def read_prototypes(args):
    reports = {}
    for phase in ('gate', 'perf', 'model'):
        path = getattr(args, 'prototype_' + phase).resolve()
        report = json.loads(path.read_text())
        if report.get('status') != 'complete' or not report.get('passed') or report.get('phase') != phase:
            raise RuntimeError('Successful complete prototype ' + phase + ' evidence required')
        for name, digest in report['metadata']['source_sha256'].items():
            # Production models deliberately changed during integration. Only
            # immutable prototype code dependencies are compared here.
            if Path(name).parts[0] == 'tools' and common.sha(common.ROOT / name) != digest:
                raise RuntimeError('Prototype source changed: ' + name)
        reports[phase] = dict(path=str(path), sha256=common.sha(path), report=report)
    if reports['perf']['report']['gate_sha256'] != reports['gate']['sha256']:
        raise RuntimeError('Prototype performance/gate evidence mismatch')
    if reports['model']['report']['gate_sha256'] != reports['gate']['sha256'] or \
            reports['model']['report']['perf_sha256'] != reports['perf']['sha256']:
        raise RuntimeError('Prototype model evidence mismatch')
    return {phase: {key: value for key, value in row.items() if key != 'report'} for phase, row in reports.items()}


def arithmetic_identity(production_path):
    """Compare complete forward/backward ASTs, allowing one policy-helper rename."""
    class Canonical(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == '_fp32_policy':
                node.id = 'fp32_policy'
            return node
    def methods(path):
        module = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
                   and node.name == 'PointwiseWeightGradient')
        return {node.name: ast.dump(Canonical().visit(copy.deepcopy(node)), include_attributes=False)
                for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ('forward', 'backward')}
    prototype_path = Path(common.candidate.base.__file__)
    prototype, production = methods(prototype_path), methods(production_path)
    if prototype.keys() != {'forward', 'backward'} or prototype != production:
        raise RuntimeError('Production PointwiseWeightGradient arithmetic differs from validated prototype')
    return dict(passed=True, allowed_normalization='_fp32_policy -> fp32_policy identifier only',
                prototype_path=str(prototype_path), prototype_sha256=common.sha(prototype_path),
                production_path=str(production_path), production_sha256=common.sha(production_path),
                methods={name: hashlib.sha256(value.encode()).hexdigest() for name, value in production.items()})


def metadata(args):
    production = common.ROOT / 'models/pointwise.py'
    result = common.identity()
    dependencies = (Path(__file__), Path(model_helpers.__file__), production,
                    common.ROOT / 'models/converse_core.py', common.ROOT / 'tools/roadmap_quality/usrnet_training_data.py')
    result['source_sha256'].update({str(path.relative_to(common.ROOT)): common.sha(path) for path in dependencies})
    result['prototype_candidate_source_sha256'] = result['candidate_source_sha256']
    result['candidate_source_sha256'] = common.sha(production)
    result['implementation'] = 'Actual models.pointwise.PointwiseConv2d._conv_forward; no research patch'
    result['baseline'] = 'Same checked CUDA model; only PointwiseConv2d.backend=pytorch'
    result['prototype_evidence'] = read_prototypes(args)
    result['arithmetic_ast'] = arithmetic_identity(production)
    return result


def check_parent(path, report, required):
    parent = json.loads(path.read_text())
    if parent.get('status') != 'complete' or not parent.get('passed'):
        raise RuntimeError('Complete passed production parent required')
    for stage in required:
        if not parent.get(stage, {}).get('passed'):
            raise RuntimeError('Missing passed production stage: ' + stage)
    for key in ('source_sha256', 'checked_build', 'prototype_evidence'):
        if parent['metadata'][key] != report['metadata'][key]:
            raise RuntimeError('Production parent identity changed: ' + key)
    report['parent'] = dict(path=str(path), sha256=common.sha(path))
    for stage in required:
        report[stage] = parent[stage]


def production_function(raw, backend='cuda'):
    from models.pointwise import PointwiseConv2d
    layer = PointwiseConv2d(raw[1].shape[1], raw[1].shape[0], 1,
                           bias=raw[2] is not None, backend=backend)
    return layer._conv_forward


def production_cases(payload):
    active = None
    for name, row in payload['captures'].items():
        raw = common.raw_capture(row)
        expected = tuple(raw[1].shape) == (64, 128, 1, 1)
        if expected and active is None:
            active = raw
        yield dict(name='actual/' + name, raw=raw, needs=(True, True, True), expected=expected)
    if active is None:
        raise RuntimeError('No C128->64 actual fixture')
    for mask in range(8):
        needs = tuple(bool(mask & (1 << i)) for i in range(3))
        yield dict(name=f'mask/{mask:03b}', raw=active, needs=needs, expected=needs[1])
    yield dict(name='bias_none', raw=(*active[:2], None, active[3]), needs=(True, True, False), expected=True)
    yield dict(name='shared_leaves', raw=active, needs=(True, True, True), expected=True, shared=True)
    yield dict(name='higher_order/native_backward', raw=active, needs=(True, True, True), expected=True, higher=True)
    for key, index, expected in (('input', 0, False), ('upstream', 3, True)):
        changed = list(active)
        changed[index] = changed[index].transpose(-1, -2).contiguous().transpose(-1, -2)
        yield dict(name='noncontiguous/' + key, raw=tuple(changed), needs=(True, True, True), expected=expected)
    changed = list(active)
    changed[0] = changed[0].contiguous(memory_format=torch.channels_last)
    yield dict(name='channels_last_fallback', raw=tuple(changed), needs=(True, True, True), expected=False)
    for batch_size, h, w in ((1, 96, 96), (4, 95, 96), (4, 7, 9)):
        raw = (active[0][:batch_size, :, :h, :w].contiguous(), active[1], active[2],
               active[3][:batch_size, :, :h, :w].contiguous())
        yield dict(name=f'shape_fallback/B{batch_size}_{h}_{w}', raw=raw, needs=(True, True, True), expected=False)


def gate(args, report):
    payload = common.load_fixtures(args.fixtures)
    result = dict(passed=False, fixture_path=str(args.fixtures), fixture_sha256=common.sha(args.fixtures),
                  source=payload['source'], rows={}, failures=[], active_actual=0, fallback_actual=0)
    report['gate'] = result
    for case in production_cases(payload):
        name, raw, expected = case.pop('name'), case.pop('raw'), case.pop('expected')
        route = {}
        reference = common.values(F.conv2d, raw, dtype=torch.float64, **case)
        baseline = common.values(production_function(raw, 'pytorch'), raw, **case)
        actual = common.values(production_function(raw, 'cuda'), raw, route=route, **case)
        if route['candidate_active'] != expected:
            raise RuntimeError('Production dispatch differs: ' + name + ' ' + str(route))
        if name.startswith('actual/'):
            result['active_actual' if expected else 'fallback_actual'] += 1
        tests = {key: comparison(actual[key], baseline[key], reference[key]) for key in reference}
        row = dict(**route, expected_active=expected, tensors=tests, passed=all(v['passed'] for v in tests.values()),
                   bitwise_diagnostic={key: bool(torch.equal(actual[key], baseline[key])) for key in reference})
        result['rows'][name] = row
        result['failures'].extend(name + '/' + key for key, value in tests.items() if not value['passed'])
        print('production gate', name, 'active', route['candidate_active'], 'passed', row['passed'], flush=True)
        del actual, baseline, reference
    if result['active_actual'] != 70 or result['fallback_actual'] != 70:
        raise RuntimeError('Production actual fixture scope must be 70 active and 70 fallback')
    result['passed'] = not result['failures']
    return result['passed']


def model_with_backend(backend):
    from models.pointwise import PointwiseConv2d
    net = common.model()
    pointwise_names = []
    for name, module in net.named_modules():
        if isinstance(module, PointwiseConv2d):
            module.backend = backend
            pointwise_names.append(name)
    if not pointwise_names:
        raise RuntimeError('Production model does not contain PointwiseConv2d')
    return net, pointwise_names


def compare_snapshots(actual, baseline, *, exact=False):
    if actual.keys() != baseline.keys():
        raise RuntimeError('Production model gradient/state masks differ')
    result = {}
    for name, expected in baseline.items():
        value = actual[name]
        metrics = error_metrics(value, expected.double())
        passed = metrics['finite'] and bool(torch.equal(value, expected) if exact else
                                            torch.allclose(value, expected, atol=3e-5, rtol=3e-5))
        result[name] = dict(passed=passed, **metrics)
        if name == 'loss':
            result[name].update(native=expected.item(), candidate=value.item())
    return result


def model(args, report):
    result = dict(passed=False, convergence_evidence=False, rows=[], failures=[], active_counts=[],
                  seeds=[17, 29, 43], steps=3, batch=4, atol=3e-5, rtol=3e-5,
                  B1_fallback='One full-model Adam step with exact native agreement and zero active calls')
    report['model'] = result
    for batch_size, seeds, steps in ((4, (17, 29, 43), 3), (1, (17,), 1)):
        for seed in seeds:
            torch.manual_seed(seed)
            baseline, names = model_with_backend('pytorch')
            candidate, candidate_names = model_with_backend('cuda')
            if names != candidate_names:
                raise RuntimeError('Production model module lists differ')
            candidate.load_state_dict(baseline.state_dict())
            optimizers = [torch.optim.Adam(net.parameters(), lr=1e-4) for net in (baseline, candidate)]
            batches, source = model_helpers.make_batches(args, seed, batch_size, steps)
            result.setdefault('data_sources', {})[f'B{batch_size}/seed{seed}'] = source
            counters, handles = [], []
            for net in (baseline, candidate):
                counter, registrations = model_helpers.active_counter(net, names)
                counters.append(counter)
                handles.extend(registrations)
            try:
                for index, data in enumerate(batches):
                    snapshots = []
                    for net, optimizer, counter in zip((baseline, candidate), optimizers, counters):
                        model_helpers.reset_count(counter)
                        snapshots.append(model_helpers.step(net, optimizer, data, snapshot=True))
                    expected = 70 if batch_size == 4 else 0
                    count = dict(batch=batch_size, seed=seed, step=index, baseline=dict(counters[0]),
                                 candidate=dict(counters[1]), expected_candidate=expected)
                    result['active_counts'].append(count)
                    if counters[0]['active'] != 0 or counters[1]['active'] != expected:
                        raise RuntimeError('Production model actual dispatch mismatch: ' + str(count))
                    comparisons = compare_snapshots(snapshots[1], snapshots[0], exact=batch_size == 1)
                    for name, metrics in comparisons.items():
                        result['rows'].append(dict(batch=batch_size, seed=seed, step=index, tensor=name, **metrics))
                        if not metrics['passed']:
                            result['failures'].append(f'B{batch_size}/seed{seed}/step{index}/' + name)
                    print('production model', batch_size, seed, index, 'failures', len(result['failures']),
                          'active', counters[1]['active'], flush=True)
            finally:
                for handle in handles:
                    handle.remove()
            del baseline, candidate, optimizers, batches
            torch.cuda.empty_cache()
    result['passed'] = not result['failures']
    return result['passed']


def perf(args, report):
    result = dict(passed=False, batch=4, independent_runs=[], settings=dict(warmup=args.warmup,
                  rounds=args.rounds, iters=args.iters), active_calls_per_forward=70,
                  scope='Actual production dispatch, complete B4 full pretrained Adam steps, identical state/data per round')
    report['perf'] = result
    net, names = model_with_backend('pytorch')
    data, source = model_helpers.make_batches(args, args.seed, 4, max(args.iters, args.warmup))
    result['source'] = source
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    original = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    model_helpers.step(net, optimizer, data[0])
    frozen_optimizer = copy.deepcopy(optimizer.state_dict())
    def restore(backend):
        net.load_state_dict(original)
        optimizer.load_state_dict(copy.deepcopy(frozen_optimizer))
        optimizer.zero_grad(set_to_none=True)
        modules = dict(net.named_modules())
        for name in names:
            modules[name].backend = backend
    # Actual production route counts are checked outside timing.
    counter, handles = model_helpers.active_counter(net, names)
    try:
        result['route_probe'] = {}
        for backend, expected in (('pytorch', 0), ('cuda', 70)):
            restore(backend)
            model_helpers.reset_count(counter)
            model_helpers.step(net, optimizer, data[0])
            result['route_probe'][backend] = dict(counter)
            if counter['active'] != expected:
                raise RuntimeError('Wrong production performance dispatch for ' + backend)
    finally:
        for handle in handles:
            handle.remove()
    for repeat in range(2):
        rounds = dict(native=[], candidate=[])
        run = dict(rounds=rounds)
        result['independent_runs'].append(run)
        for backend in ('pytorch', 'cuda'):
            restore(backend)
            for index in range(args.warmup):
                model_helpers.step(net, optimizer, data[index])
        for index in range(args.rounds):
            order = ('native', 'candidate') if index % 2 == 0 else ('candidate', 'native')
            for route in order:
                restore('pytorch' if route == 'native' else 'cuda')
                cursor, loss = [0], [None]
                def invoke():
                    loss[0] = model_helpers.step(net, optimizer, data[cursor[0]])
                    cursor[0] += 1
                timing = common.measure(invoke, args.iters)
                rounds[route].append(dict(round=index, order=list(order), **timing))
                model_helpers.assert_finite_state(net, optimizer, loss[0])
        speedups = {metric: [rounds['native'][i][metric] / rounds['candidate'][i][metric]
                            for i in range(args.rounds)] for metric in ('wall_ms', 'cuda_event_ms')}
        run.update(paired_speedups=speedups, median_speedup={key: statistics.median(value) for key, value in speedups.items()})
        run['passed'] = run['median_speedup']['wall_ms'] > 1 and run['median_speedup']['cuda_event_ms'] >= 1 / 1.02
        print('production B4 Adam performance', repeat, run['median_speedup'], 'passed', run['passed'], flush=True)
    result['passed'] = all(run['passed'] for run in result['independent_runs'])
    return result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'gate', 'model', 'perf'), default='all')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--prototype-gate', type=Path, required=True)
    parser.add_argument('--prototype-perf', type=Path, required=True)
    parser.add_argument('--prototype-model', type=Path, required=True)
    parser.add_argument('--parent', type=Path, help='Passed production gate for model stage, or production model for perf stage')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve evidence: choose a new output path')
    if args.stage in ('model', 'perf') and not args.parent:
        parser.error('--parent production evidence required for separate model/perf stages')
    if args.warmup < 5 or args.rounds < 9 or args.iters < 10:
        parser.error('Require >=5 warmups, >=9 rounds, >=10 iterations')
    for field in ('fixtures', 'manifest', 'output', 'parent'):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    report = dict(phase='production_' + args.stage, status='running', passed=False,
                  fresh_production_validation=True, prototype_pass_is_not_production_admission=True)
    try:
        common.setup()
        report['metadata'] = metadata(args)
        if args.stage in ('model', 'perf'):
            check_parent(args.parent, report, ('gate',) if args.stage == 'model' else ('gate', 'model'))
            if report['gate']['fixture_sha256'] != common.sha(args.fixtures):
                raise RuntimeError('Production parent fixture mismatch')
        passed = True
        for stage in ('gate', 'model', 'perf') if args.stage == 'all' else (args.stage,):
            if not globals()[stage](args, report):
                passed = False
                break
        report.update(status='complete', passed=passed)
    except Exception as error:
        report.update(status='failed', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        common.exclusive_json(args.output, report)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
