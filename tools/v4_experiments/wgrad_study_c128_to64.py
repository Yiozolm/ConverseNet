"""Capture, gate and time the restricted C128->64 direct-FP32-GEMM candidate.

Every output path must be new. No production module is patched by import.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from numerical_policy import comparison, BUDGETS, BASELINE_COMMIT, BASELINE_SHA256
import pointwise_wgrad_c128_to64 as candidate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def setup():
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend/CPU-only overrides')
    os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    from extension_loader import load_extension
    load_extension()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')


def identity():
    files = [Path(__file__), Path(candidate.__file__), Path(candidate.base.__file__), ROOT / 'test/numerical_policy.py',
        ROOT / 'test/fp32_baseline.py', ROOT / 'test/extension_loader.py',
        ROOT / 'models/converse_usrnet.py', ROOT / 'models/util_converse.py',
        ROOT / 'model_zoo/converse_usrnet.pth']
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        candidate_source_sha256=candidate.source_sha256(),
        deployment_scope='C128->64 only; C64->128 native fallback. Original v1 failure evidence unchanged.',
        checked_build=json.loads((ROOT / '.build/cuda/source_manifest.json').read_text()),
        torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        tf32=False, amp=False, deterministic_algorithms=True,
        frozen_converse_baseline=dict(commit=BASELINE_COMMIT, sha256=BASELINE_SHA256),
        conv_baseline='Native FP32 torch.nn.functional.conv2d, unchanged by baseline commit',
        oracle='Independent native FP64 convolution on identical FP32-origin inputs', budgets=BUDGETS)


def model():
    from models.converse_usrnet import ConverseUSRNet
    result = ConverseUSRNet(backend='cuda').cuda().train()
    result.load_state_dict(torch.load(ROOT / 'model_zoo/converse_usrnet.pth',
                                     map_location='cpu', weights_only=True), strict=True)
    return result


def batch(args, batch_size=4, step=0, seed=None):
    seed = args.seed if seed is None else seed
    if args.manifest:
        from tools.roadmap_quality.usrnet_training_data import DatasetProtocol
        protocol = DatasetProtocol(args.manifest, patch_size=96, scale=3, seed=seed)
        values = protocol.train_batch(step, batch_size)
        source = dict(kind='real_photographs_declared_synthetic_degradation', **protocol.metadata)
    else:
        gen = torch.Generator().manual_seed(seed + step)
        values = (torch.rand(batch_size, 3, 32, 32, generator=gen),
                  torch.softmax(torch.randn(batch_size, 1, 49, generator=gen), -1).reshape(batch_size, 1, 7, 7),
                  torch.rand(batch_size, 3, 96, 96, generator=gen))
        source = dict(kind='synthetic_fixed_seed', seed=seed, step=step,
                      limitation='Real model activations from synthetic LR/target, not a real image dataset')
    return tuple(v.cuda() for v in values), source


def capture(args, report):
    if args.fixtures.exists():
        raise RuntimeError('Choose a new fixtures path')
    torch.manual_seed(args.seed)
    net = model()
    data, source = batch(args)
    captures, counts, handles = {}, Counter(), []
    selected = {name: mod for name, mod in net.named_modules()
                if name.startswith('p.m_body.') and isinstance(mod, torch.nn.Conv2d)
                and mod.kernel_size == (1, 1)
                and (mod.in_channels, mod.out_channels) in ((64, 128), (128, 64))}
    def hook(name):
        def forward(mod, inputs, output):
            call = counts[name]
            counts[name] += 1
            key = f'{name}/call{call}'
            captures[key] = dict(module=name, call=call, x=inputs[0].detach().cpu().clone(),
                                 weight=mod.weight.detach().cpu().clone(),
                                 bias=mod.bias.detach().cpu().clone() if mod.bias is not None else None)
            def backward(upstream):
                captures[key]['grad_output'] = upstream.detach().cpu().clone()
            output.register_hook(backward)
        return forward
    for name, mod in selected.items():
        handles.append(mod.register_forward_hook(hook(name)))
    try:
        x, kernel, target = data
        output = net(x, kernel, 3)
        loss = F.mse_loss(output, target)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    if set(counts) != set(selected) or len(selected) != 28 or any(value != 5 for value in counts.values()):
        raise RuntimeError('Expected five invocations of every selected full-model module')
    if any('grad_output' not in value for value in captures.values()):
        raise RuntimeError('Missing captured upstream')
    payload = dict(schema_version=1, source=source, checkpoint_sha256=sha(ROOT / 'model_zoo/converse_usrnet.pth'),
                   model=dict(iterations=5, blocks=7, batch=4, lr=32, scale=3),
                   calls=dict(counts), captures=captures, metadata=report['metadata'])
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    with args.fixtures.open('xb') as stream:
        torch.save(payload, stream)
    report.update(passed=True, fixture_path=str(args.fixtures), fixture_sha256=sha(args.fixtures),
                  source=source, calls=dict(counts), selected_calls=list(captures), loss=loss.item())


def load_fixtures(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not payload.get('captures') or payload['model'] != dict(iterations=5, blocks=7, batch=4, lr=32, scale=3):
        raise RuntimeError('Full pretrained B4 LR32/s3 model fixtures required')
    counts = Counter(row['module'] for row in payload['captures'].values())
    if len(counts) != 28 or counts != Counter(payload['calls']) or any(value != 5 for value in counts.values()):
        raise RuntimeError('All 28 modules and all five invocations must be captured')
    if any({row['call'] for row in payload['captures'].values() if row['module'] == name} != set(range(5))
           for name in counts):
        raise RuntimeError('Missing/duplicate fixture invocation')
    return payload


def values(fn, raw, *, dtype=torch.float32, needs=(True, True, True), shared=False,
           higher=False, convolution=None, route=None):
    data = tuple(None if v is None else v.to(device='cuda', dtype=dtype).detach().requires_grad_(need)
                 for v, need in zip(raw[:3], needs))
    upstream = raw[3].to(device='cuda', dtype=dtype)
    x, w, b = data
    convolution = convolution or {}
    output = fn(x, w, b, **convolution)
    if route is not None:
        route['forward_grad_fn'] = type(output.grad_fn).__name__
        route['candidate_active'] = type(output.grad_fn).__name__ == 'PointwiseWeightGradientBackward'
    if shared:
        output = output + fn(x * 0.375, w, b, **convolution)
    targets = [(name, value) for name, value in zip(('dx', 'dweight', 'dbias'), data)
               if value is not None and value.requires_grad]
    gradients = torch.autograd.grad(output, [v for _, v in targets], upstream,
                                    create_graph=higher) if targets else ()
    result = dict(output=output, **{n: v for (n, _), v in zip(targets, gradients)})
    if higher:
        scalar = sum(g.square().sum() for g in gradients if g.requires_grad)
        second = torch.autograd.grad(scalar, [v for _, v in targets], allow_unused=True)
        result.update({'d' + n: v for (n, _), v in zip(targets, second) if v is not None})
    return result


def raw_capture(row):
    return tuple(row[key] for key in ('x', 'weight', 'bias', 'grad_output'))


def cases(payload, seed):
    for name, row in payload['captures'].items():
        yield dict(name='actual/' + name, raw=raw_capture(row), needs=(True, True, True))
    gen = torch.Generator().manual_seed(seed)
    synthetic = []
    for batch_size in (1, 4):
        for ci, co in ((64, 128), (128, 64)):
            raw = (torch.randn(batch_size, ci, 96, 96, generator=gen),
                   torch.randn(co, ci, 1, 1, generator=gen) / ci ** .5,
                   torch.randn(co, generator=gen),
                   torch.randn(batch_size, co, 96, 96, generator=gen) / (batch_size * 96 * 96) ** .5)
            synthetic.append(raw)
            yield dict(name=f'synthetic/B{batch_size}_C{ci}_{co}', raw=raw, needs=(True, True, True))
    # All semantic probes below must exercise the restricted active direction.
    # The synthetic B1/B4 C64->128 cases above independently verify fallback.
    raw = synthetic[1]
    for bits in range(8):
        yield dict(name=f'mask/{bits:03b}', raw=raw, needs=tuple(bool(bits & (1 << i)) for i in range(3)))
    yield dict(name='bias_none', raw=(raw[0], raw[1], None, raw[3]), needs=(True, True, False))
    yield dict(name='shared_leaves', raw=raw, needs=(True, True, True), shared=True)
    for name, field in (('input', 0), ('upstream', 3)):
        changed = list(raw)
        changed[field] = changed[field].transpose(-1, -2).contiguous().transpose(-1, -2)
        yield dict(name='noncontiguous/' + name, raw=tuple(changed), needs=(True, True, True))
    changed = list(raw)
    changed[0] = changed[0].contiguous(memory_format=torch.channels_last)
    yield dict(name='channels_last', raw=tuple(changed), needs=(True, True, True))
    for h, w in ((63, 67), (7, 9)):
        changed = (torch.randn(1, 128, h, w, generator=gen), raw[1], raw[2],
                   torch.randn(1, 64, h, w, generator=gen) / (h * w) ** .5)
        yield dict(name=f'spatial/{h}_{w}', raw=changed, needs=(True, True, True))
    for name, kwargs, weight in (
        ('stride', dict(stride=2), raw[1]),
        ('padding', dict(padding=1), raw[1]),
        ('dilation', dict(dilation=2), raw[1]),
        ('groups', dict(groups=2), raw[1][:, :64].contiguous()),
        ('channel_tail', {}, torch.randn(63, 128, 1, 1, generator=gen)),
    ):
        h = (96 + 2 * kwargs.get('padding', 0) - 1) // kwargs.get('stride', 1) + 1
        changed = (raw[0], weight, torch.randn(weight.shape[0], generator=gen),
                   torch.randn(1, weight.shape[0], h, h, generator=gen))
        yield dict(name='fallback/' + name, raw=changed, needs=(True, True, True), convolution=kwargs)
    yield dict(name='higher_order/native_fallback', raw=raw, needs=(True, True, True), higher=True)


def gate(args, report):
    payload = load_fixtures(args.fixtures)
    report.update(fixture_path=str(args.fixtures), fixture_sha256=sha(args.fixtures),
                  fixture_source=payload['source'], cases={}, failures=[])
    for case in cases(payload, args.seed):
        name, raw = case.pop('name'), case.pop('raw')
        ref = values(F.conv2d, raw, dtype=torch.float64, **case)
        baseline = values(F.conv2d, raw, **case)
        route = {}
        actual = values(candidate.conv1x1, raw, route=route, **case)
        expected_active = (case['needs'][1] and raw[0].shape[0] * raw[0].shape[2] * raw[0].shape[3] >= 4096
                           and (raw[1].shape[1], raw[1].shape[0]) == (128, 64)
                           and not name.startswith('fallback/'))
        route['expected_active'] = expected_active
        if route['candidate_active'] != expected_active:
            raise RuntimeError('Wrong candidate route: ' + name)
        if actual.keys() != ref.keys() or baseline.keys() != ref.keys():
            raise RuntimeError('Mismatch in differentiation mask')
        tests = {key: comparison(actual[key], baseline[key], ref[key], distribution=True) for key in ref}
        row = dict(**case, **route, passed=all(v['passed'] for v in tests.values()), tensors=tests,
                   shape=list(raw[0].shape), weight_shape=list(raw[1].shape),
                   bitwise_diagnostic={key: bool(torch.equal(actual[key], baseline[key])) for key in ref})
        report['cases'][name] = row
        report['failures'].extend(name + '/' + key for key, value in tests.items() if not value['passed'])
        print(name, 'passed', row['passed'], flush=True)
        del ref, baseline, actual
    active_captures = [row for row in payload['captures'].values() if tuple(row['weight'].shape[:2]) == (64, 128)]
    if len(active_captures) != 70:
        raise RuntimeError('Expected 14 C128->64 modules and five invocations each')
    graph_stream_gate(raw_capture(active_captures[0]), report)
    active_count = sum(row['candidate_active'] for row in report['cases'].values())
    if active_count == 0:
        raise RuntimeError('Gate never exercised candidate')
    report.update(active_cases=active_count, passed=not report['failures'], all_gpu_cases_passed=not report['failures'],
                  allowed_modules=sorted({row['module'] for row in active_captures}))


def graph_stream_gate(raw, report):
    """Exercise first-order VJPs in a nondefault stream and replayable graph."""
    data = tuple(v.cuda().detach().requires_grad_(True) for v in raw[:3])
    upstream = raw[3].cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def invoke():
        output = candidate.conv1x1(*data)
        gradients = torch.autograd.grad(output, data, upstream)
        return dict(output=output, **dict(zip(('dx', 'dweight', 'dbias'), gradients)))
    with torch.cuda.stream(stream):
        stream_result = invoke()
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    reference = values(F.conv2d, raw, dtype=torch.float64)
    baseline = values(F.conv2d, raw)
    tests = {key: comparison(stream_result[key], baseline[key], reference[key]) for key in reference}
    report['cases']['stream/nondefault'] = dict(candidate_active=True, passed=all(v['passed'] for v in tests.values()), tensors=tests)
    report['failures'].extend('stream/nondefault/' + key for key, value in tests.items() if not value['passed'])
    del stream_result, reference, baseline
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = invoke()
    for replay in range(2):
        changed = tuple(value * (0.875 if replay else 1.0) for value in raw)
        with torch.no_grad():
            for target, source in zip((*data, upstream), changed):
                target.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        reference = values(F.conv2d, changed, dtype=torch.float64)
        baseline = values(F.conv2d, changed)
        tests = {key: comparison(captured[key], baseline[key], reference[key]) for key in reference}
        name = f'graph/replay{replay}'
        report['cases'][name] = dict(candidate_active=True, passed=all(v['passed'] for v in tests.values()), tensors=tests)
        report['failures'].extend(name + '/' + key for key, value in tests.items() if not value['passed'])
        print(name, report['cases'][name]['passed'], flush=True)


def verify_admission(path, phase):
    previous = json.loads(path.read_text())
    if previous.get('phase') != phase or previous.get('status') != 'complete' or not previous.get('passed'):
        raise RuntimeError('A complete successful ' + phase + ' report is required')
    for name, digest in previous['metadata']['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Evidence source changed: ' + name)
    if previous['metadata']['checked_build'] != json.loads((ROOT / '.build/cuda/source_manifest.json').read_text()):
        raise RuntimeError('Checked build identity changed')
    path = Path(previous['fixture_path'])
    if sha(path) != previous['fixture_sha256']:
        raise RuntimeError('Fixtures changed')
    return previous


def measure(fn, iters):
    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    start = time.perf_counter()
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iters
    return dict(wall_ms=elapsed, cuda_event_ms=start_event.elapsed_time(end_event) / iters,
                peak_extra_allocated_bytes=torch.cuda.max_memory_allocated() - allocated)


def paired(functions, args):
    for fn in functions.values():
        for _ in range(args.warmup):
            fn()
    rounds = {name: [] for name in functions}
    for index in range(args.rounds):
        order = ['native', 'candidate'] if index % 2 == 0 else ['candidate', 'native']
        for name in order:
            rounds[name].append(dict(round=index, order=order, **measure(functions[name], args.iters)))
    speedups = {metric: [rounds['native'][i][metric] / rounds['candidate'][i][metric]
                        for i in range(args.rounds)] for metric in ('wall_ms', 'cuda_event_ms')}
    return dict(rounds=rounds, paired_speedups=speedups,
                median_speedup={metric: statistics.median(vals) for metric, vals in speedups.items()})


def perf(args, report):
    admission = verify_admission(args.gate, 'gate')
    payload = load_fixtures(Path(admission['fixture_path']))
    report.update(gate_path=str(args.gate), gate_sha256=sha(args.gate),
                  fixture_path=admission['fixture_path'], fixture_sha256=admission['fixture_sha256'],
                  settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters), cases={},
                  scope='Complete forward+dx+dw+db with layout copies and Python/autograd overhead; no input transfer in timing')
    # Every captured module and all repeated invocations.
    for name, captured in payload['captures'].items():
        if captured['module'] not in admission['allowed_modules']:
            continue
        raw = raw_capture(captured)
        data = tuple(None if v is None else v.cuda().detach().requires_grad_(True) for v in raw[:3])
        upstream = raw[3].cuda()
        def invoke(fn):
            output = fn(*data)
            grads = torch.autograd.grad(output, tuple(v for v in data if v is not None), upstream)
            return output, grads
        functions = dict(native=lambda: invoke(F.conv2d), candidate=lambda: invoke(candidate.conv1x1))
        runs = [paired(functions, args) for _ in range(2)]
        row = dict(independent_runs=runs,
                   passed=all(run['median_speedup']['wall_ms'] >= 1.03 and
                              run['median_speedup']['cuda_event_ms'] >= 1 / 1.02 and
                              sum(value > 1 for value in run['paired_speedups']['wall_ms']) >= math.ceil(7 * args.rounds / 9)
                              for run in runs))
        report['cases'][name] = row
        print(name, [run['median_speedup'] for run in runs], 'passed', row['passed'], flush=True)
        del data, upstream
    report['criterion'] = dict(min_median_wall_speedup=1.03, positive_rounds=math.ceil(7 * args.rounds / 9),
                               min_median_cuda_speedup=1 / 1.02, independent_runs=2, all_cases_required=True)
    report['passed'] = bool(report['cases']) and all(row['passed'] for row in report['cases'].values())
    report['allowed_modules'] = admission['allowed_modules'] if report['passed'] else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture', 'gate', 'perf'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Preserve evidence: select a new output path')
    if args.phase in ('capture', 'gate') and args.fixtures is None:
        parser.error('--fixtures required')
    if args.phase == 'perf' and args.gate is None:
        parser.error('--gate required')
    if args.rounds < 9 or args.iters < 10 or args.warmup < 5:
        parser.error('Use >= 9 rounds, >= 10 iterations and >= 5 warmups')
    for key in ('fixtures', 'manifest', 'gate'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    report = dict(phase=args.phase, status='running', passed=False)
    try:
        setup()
        report['metadata'] = identity()
        globals()[args.phase](args, report)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        exclusive_json(args.output, report)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
