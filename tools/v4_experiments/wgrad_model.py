"""Gated full-pretrained-USRNet regression and complete Adam-step timing.

The control is the unmodified checked CUDA model with native Conv2d. This
isolates the pointwise VJP change; it does not replace independent FP64 gates.
Three short trajectories provide regression coverage, not convergence proof.
"""
import argparse
from contextlib import ExitStack
import copy
from pathlib import Path
import statistics
import traceback

import torch
import torch.nn.functional as F
import wgrad_study as study
from numerical_policy import error_metrics


def make_batches(args, seed, batch_size, count):
    if args.manifest:
        from tools.roadmap_quality.usrnet_training_data import DatasetProtocol
        protocol = DatasetProtocol(args.manifest, patch_size=96, scale=3, seed=seed)
        return [tuple(v.cuda() for v in protocol.train_batch(index, batch_size)) for index in range(count)], protocol.metadata
    rows, source = [], None
    for index in range(count):
        row, source = study.batch(args, batch_size, index, seed)
        rows.append(row)
    return rows, source


def step(net, optimizer, data, *, snapshot=False):
    x, kernel, target = data
    optimizer.zero_grad(set_to_none=True)
    output = net(x, kernel, 3)
    loss = F.mse_loss(output, target)
    loss.backward()
    record = {}
    if snapshot:
        record = dict(output=output.detach().cpu().clone(), loss=loss.detach().cpu().clone())
        record.update({'gradient/' + name: p.grad.detach().cpu().clone() for name, p in net.named_parameters()
                       if p.grad is not None})
    optimizer.step()
    if snapshot:
        record.update({'parameter/' + name: p.detach().cpu().clone() for name, p in net.named_parameters()})
        for name, parameter in net.named_parameters():
            for field, value in optimizer.state[parameter].items():
                if torch.is_tensor(value):
                    record[f'optimizer/{name}/{field}'] = value.detach().cpu().clone()
    return record if snapshot else loss.detach()


def assert_finite_state(net, optimizer, loss):
    """Untimed integrity check for the entire ten-step measurement sequence."""
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite loss after timed model trajectory')
    for name, parameter in net.named_parameters():
        tensors = {'parameter': parameter, 'gradient': parameter.grad}
        tensors.update({'optimizer/' + key: value for key, value in optimizer.state[parameter].items()
                        if torch.is_tensor(value)})
        for key, value in tensors.items():
            if value is not None and not bool(torch.isfinite(value).all()):
                raise RuntimeError('Nonfinite timed model state: ' + name + '/' + key)


def active_counter(net, allowed):
    counter, handles = dict(total=0, active=0), []
    def hook(module, inputs, output):
        counter['total'] += 1
        counter['active'] += type(output.grad_fn).__name__ == 'PointwiseWeightGradientBackward'
    for name, module in net.named_modules():
        if name in allowed:
            handles.append(module.register_forward_hook(hook))
    return counter, handles


def reset_count(counter):
    counter.update(total=0, active=0)


def smoke(args, allowed, report):
    report['smoke'] = dict(seeds=[17, 29, 43], steps=3, batch=4, lr=32, scale=3,
                           model_iterations=5, model_blocks=7, atol=3e-5, rtol=3e-5,
                           convergence_evidence=False, rows=[], failures=[], active_counts=[])
    result = report['smoke']
    expected_count = len(allowed) * 5
    if expected_count not in (70, 140):
        raise RuntimeError('Expected exactly one or both prior pointwise channel directions')
    for seed in (17, 29, 43):
        torch.manual_seed(seed)
        baseline, candidate = study.model(), study.model()
        candidate.load_state_dict(baseline.state_dict())
        optimizers = [torch.optim.Adam(net.parameters(), lr=1e-4) for net in (baseline, candidate)]
        batches, source = make_batches(args, seed, 4, 3)
        result.setdefault('data_sources', {})[str(seed)] = source
        counters, handles = [], []
        for net in (baseline, candidate):
            counter, registrations = active_counter(net, allowed)
            counters.append(counter)
            handles.extend(registrations)
        try:
            with study.candidate.patch_modules(candidate, allowed):
                for index, data in enumerate(batches):
                    snapshots = []
                    for net, optimizer, counter in zip((baseline, candidate), optimizers, counters):
                        reset_count(counter)
                        snapshots.append(step(net, optimizer, data, snapshot=True))
                    count_row = dict(seed=seed, step=index, expected=expected_count,
                                     baseline=dict(counters[0]), candidate=dict(counters[1]))
                    result['active_counts'].append(count_row)
                    if counters[0]['active'] != 0 or counters[1]['active'] != expected_count:
                        raise RuntimeError('Full-model candidate active count differs from admission: ' + str(count_row))
                    if snapshots[0].keys() != snapshots[1].keys():
                        raise RuntimeError('Model gradient/state masks differ')
                    for name, reference in snapshots[0].items():
                        value = snapshots[1][name]
                        metrics = error_metrics(value, reference.double())
                        passed = metrics['finite'] and bool(torch.allclose(value, reference, atol=3e-5, rtol=3e-5))
                        row = dict(seed=seed, step=index, tensor=name, passed=passed, **metrics)
                        if name == 'loss':
                            row.update(native=reference.item(), candidate=value.item())
                        result['rows'].append(row)
                        if not passed:
                            result['failures'].append(f'seed{seed}/step{index}/' + name)
                    print('model smoke', seed, index, 'failures', len(result['failures']),
                          'active', counters[1]['active'], flush=True)
        finally:
            for handle in handles:
                handle.remove()
        del baseline, candidate, optimizers, batches
        torch.cuda.empty_cache()
    result['passed'] = not result['failures']
    return result['passed']


def model_perf(args, allowed, report):
    report['performance'] = dict(settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters,
                                               independent_runs=2), cases={},
        reset='Before every timed route/round restore identical pretrained parameters and populated Adam state; reset is outside timing',
        scope='Complete zero_grad + full pretrained forward + MSE + backward + Adam step; same data sequence; no hooks or profiler')
    expected_count = len(allowed) * 5
    for batch_size in (1, 4):
        net = study.model()
        data, source = make_batches(args, args.seed, batch_size, max(args.iters, args.warmup))
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        # Populate Adam's state once, then restore original parameters. Both
        # routes receive this identical optimizer state at every paired round.
        original = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        step(net, optimizer, data[0])
        frozen_optimizer = copy.deepcopy(optimizer.state_dict())
        def restore():
            net.load_state_dict(original)
            optimizer.load_state_dict(copy.deepcopy(frozen_optimizer))
            optimizer.zero_grad(set_to_none=True)
        with study.candidate.patch_modules(net, allowed):
            counter, handles = active_counter(net, allowed)
            try:
                restore()
                step(net, optimizer, data[0])
                if counter['active'] != expected_count:
                    raise RuntimeError(f'B{batch_size} model performance route inactive: {counter}')
            finally:
                for handle in handles:
                    handle.remove()
        runs = []
        case = dict(source=source, independent_runs=runs, active_calls_per_forward=expected_count, passed=False)
        report['performance']['cases'][f'B{batch_size}'] = case
        for run_index in range(2):
            rounds = dict(native=[], candidate=[])
            summary = dict(rounds=rounds)
            runs.append(summary)
            for route in ('native', 'candidate'):
                restore()
                with ExitStack() as stack:
                    if route == 'candidate':
                        stack.enter_context(study.candidate.patch_modules(net, allowed))
                    for index in range(args.warmup):
                        step(net, optimizer, data[index])
            for round_index in range(args.rounds):
                order = ('native', 'candidate') if round_index % 2 == 0 else ('candidate', 'native')
                for route in order:
                    restore()
                    counter = [0]
                    last_loss = [None]
                    def invoke():
                        row = data[counter[0]]
                        counter[0] += 1
                        last_loss[0] = step(net, optimizer, row)
                    with ExitStack() as stack:
                        if route == 'candidate':
                            stack.enter_context(study.candidate.patch_modules(net, allowed))
                        timing = study.measure(invoke, args.iters)
                    rounds[route].append(dict(round=round_index, order=list(order), **timing))
                    assert_finite_state(net, optimizer, last_loss[0])
            speedups = {metric: [rounds['native'][i][metric] / rounds['candidate'][i][metric]
                                for i in range(args.rounds)] for metric in ('wall_ms', 'cuda_event_ms')}
            summary.update(paired_speedups=speedups,
                           median_speedup={key: statistics.median(value) for key, value in speedups.items()})
            print('model perf', batch_size, run_index, summary['median_speedup'], flush=True)
        # Overall model direction must improve for B4 in both repeats, and
        # neither B1 nor B4 may materially regress (>2%). Per-op 3%/7-of-9
        # admission has already been verified by the independent perf report.
        passed = all(run['median_speedup']['wall_ms'] >= 1 / 1.02 and
                     run['median_speedup']['cuda_event_ms'] >= 1 / 1.02 and
                     (batch_size != 4 or run['median_speedup']['wall_ms'] > 1)
                     for run in runs)
        case['passed'] = passed
        del net, optimizer, data, frozen_optimizer
        torch.cuda.empty_cache()
    report['performance']['passed'] = all(row['passed'] for row in report['performance']['cases'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--perf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve evidence: choose a new result path')
    if args.warmup < 5 or args.rounds < 9 or args.iters < 10:
        parser.error('Require >=5 warmups, >=9 rounds, >=10 iterations')
    args.output, args.gate, args.perf = args.output.resolve(), args.gate.resolve(), args.perf.resolve()
    if args.manifest:
        args.manifest = args.manifest.resolve()
    report = dict(phase='model', status='running', passed=False)
    try:
        study.setup()
        report['metadata'] = study.identity()
        report['metadata']['model_harness_sha256'] = study.sha(__file__)
        gate = study.verify_admission(args.gate, 'gate')
        perf = study.verify_admission(args.perf, 'perf')
        if perf['gate_sha256'] != study.sha(args.gate):
            raise RuntimeError('Performance report belongs to a different gate')
        if gate['allowed_modules'] != perf['allowed_modules']:
            raise RuntimeError('Gate/performance scope mismatch')
        report.update(gate_path=str(args.gate), gate_sha256=study.sha(args.gate),
                      perf_path=str(args.perf), perf_sha256=study.sha(args.perf),
                      fixture_path=gate['fixture_path'], fixture_sha256=gate['fixture_sha256'],
                      allowed_modules=perf['allowed_modules'])
        if smoke(args, perf['allowed_modules'], report):
            model_perf(args, perf['allowed_modules'], report)
            report['passed'] = report['performance']['passed']
        else:
            report['performance'] = dict(status='blocked_by_model_smoke')
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        study.exclusive_json(args.output, report)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
