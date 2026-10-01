"""Post-gate complete-call and pretrained-model FFT-free inference experiment.

Standalone helper: do not place this file inside the gated candidate directory.
No production source is modified. Candidate replacement is local and temporary.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / 'models/converse_srresnet.py').exists())
RESEARCH = ROOT / 'tools/v4_fftfree_compensated'
sys.path[:0] = [str(RESEARCH), str(ROOT / 'test'), str(ROOT)]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from numerical_policy import error_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exclusive_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def verify_gate(args, study, research_loader):
    gate = json.loads(args.gate.read_text())
    if (gate.get('status') != 'complete' or not gate.get('passed')
            or gate.get('kind') != 'v4_explicit_nearest_fftfree_compensated'
            or len(gate.get('cases', [])) != 3459
            or not all(row.get('passed') for row in gate['cases'])
            or not gate.get('contracts', {}).get('passed')):
        raise RuntimeError('All 3459 compensated numerical cases and execution contracts must pass')
    if gate['source_sha256'] != study.source_identity():
        raise RuntimeError('Gated source identity changed; this helper must stay outside the candidate directory')
    for name, digest in gate['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Gated dependency changed: ' + name)
    manifest = json.loads((args.artifacts / 'manifest.json').read_text())
    if manifest != gate['checked_research_manifest']:
        raise RuntimeError('Research build manifest does not match admitted gate')
    if manifest['identity'] != research_loader.build_identity():
        raise RuntimeError('Research build/toolchain identity changed')
    if sha(manifest['binary']) != manifest['binary_sha256']:
        raise RuntimeError('Research PyBind binary changed')
    for name, digest in manifest['identity']['sources'].items():
        if sha(args.artifacts / 'sources' / name) != digest:
            raise RuntimeError('Archived research build source changed: ' + name)
    return gate, manifest


def setup(args, report):
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend and CPU-only overrides')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    study = import_file('compensated_gate_study_for_perf', RESEARCH / 'study.py')
    research_loader = import_file('compensated_research_loader_for_perf', RESEARCH / 'loader.py')
    manifest = json.loads((args.artifacts / 'manifest.json').read_text())
    research_arch = manifest['identity']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
    if not research_arch:
        raise RuntimeError('Research manifest lacks explicit CUDA architecture')
    current_arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if current_arch not in (None, '', research_arch):
        raise RuntimeError('Caller CUDA architecture differs from research manifest')
    os.environ['TORCH_CUDA_ARCH_LIST'] = research_arch
    gate, manifest = verify_gate(args, study, research_loader)
    extension, loaded_manifest = research_loader.load_checked(args.artifacts, build=False)
    if loaded_manifest != manifest:
        raise RuntimeError('Loaded research manifest mismatch')
    from inference import NearestK2Inference
    solve = NearestK2Inference(extension)

    production_manifest_path = ROOT / '.build/cuda/source_manifest.json'
    production_manifest_hash = sha(production_manifest_path)
    production_manifest = json.loads(production_manifest_path.read_text())
    production_arch = production_manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
    # The existing production build was checked with the empty architecture
    # override. Loading it under 12.0 would falsely label its manifest stale.
    try:
        if production_arch:
            os.environ['TORCH_CUDA_ARCH_LIST'] = production_arch
        else:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        production_loader = import_file('checked_production_loader_for_fftfree_perf', ROOT / 'test/extension_loader.py')
        production_loader.load_extension()
    finally:
        os.environ['TORCH_CUDA_ARCH_LIST'] = research_arch
    if sha(production_manifest_path) != production_manifest_hash:
        raise RuntimeError('Production manifest was modified while loading')
    production_binary = ROOT / '.build/cuda' / production_manifest['library']
    if sha(production_binary) != production_manifest['binary_sha256']:
        raise RuntimeError('Production binary changed')
    report['identity'] = dict(helper_path=str(Path(__file__).resolve()), helper_sha256=sha(__file__),
        helper_is_new_post_gate_dependency=True, gate_path=str(args.gate), gate_sha256=sha(args.gate),
        gate_source_sha256=gate['source_sha256'], checked_research_manifest=manifest,
        checked_production_manifest=production_manifest, production_manifest_sha256=production_manifest_hash,
        production_sources=production_loader.production_source_hashes(),
        production_load_architecture=production_arch, research_architecture_restored=research_arch,
        model_sources={name: sha(ROOT / name) for name in ('models/converse_srresnet.py', 'models/util_converse.py',
                                                          'model_zoo/converse_srresnet.pth')},
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), tf32=False, amp=False,
                         affinity=study.affinity(), deterministic_algorithms=True))
    return study, solve, research_loader, production_loader


def layer_candidate(layer, value, solve):
    if layer.kernel_size != 2 or layer.scale != 2 or layer.variant != 'v7':
        raise ValueError('Only explicit-nearest k2/s2 v7 layers belong to this experiment')
    pad = layer.padding
    padded = F.pad(value, (pad,) * 4, mode=layer.padding_mode) if pad else value
    # Explicit nearest-prior API: the compensated formula creates its phase
    # outputs directly; it needs no materialized nearest tensor or FFT.
    output = solve(padded, layer.weight, layer.bias, float(layer.eps))
    return output[..., 2 * pad:-2 * pad, 2 * pad:-2 * pad] if pad else output


@contextmanager
def model_candidate(model, names, solve, counter=None):
    previous = []
    modules = dict(model.named_modules())
    try:
        for name in names:
            layer = modules[name]
            previous.append((layer, 'forward' in vars(layer), vars(layer).get('forward')))
            def forward(this, value):
                if counter is not None:
                    counter['calls'] += 1
                return layer_candidate(this, value, solve)
            layer.forward = types.MethodType(forward, layer)
        yield
    finally:
        for layer, had_override, old_forward in previous:
            if had_override:
                layer.forward = old_forward
            else:
                del layer.forward


def one_call(call, *, cold):
    if cold:
        torch.ops.converse2d.clear_cache()
    # All cache clearing and event construction are outside wall timing. The
    # actual pad/prior/regularizer/solve/crop and synchronization stay inside.
    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    start_event.record()
    output = call()
    end_event.record()
    end_event.synchronize()
    wall = (time.perf_counter() - tick) * 1000
    record = dict(wall_ms=wall, cuda_ms=start_event.elapsed_time(end_event),
                  peak_extra_allocated_bytes=max(0, torch.cuda.max_memory_allocated() - allocated))
    if output.requires_grad:
        raise RuntimeError('Inference measurement unexpectedly created a gradient graph')
    return record


def paired(routes, args, scope, contexts=None):
    contexts = contexts or (lambda name: nullcontext())
    result = dict(cache_scope=scope, independent_runs=[], passed=False)
    for repeat in range(2):
        # Warm scope clears once at the start of each independent group, then
        # fixed weights/biases and warmed production caches stay alive.
        torch.ops.converse2d.clear_cache()
        with torch.no_grad():
            for name, call in routes.items():
                with contexts(name):
                    for _ in range(args.warmup):
                        if scope == 'cold':
                            torch.ops.converse2d.clear_cache()
                        call()
        torch.cuda.synchronize()
        rounds = dict(native=[], candidate=[])
        run = dict(rounds=rounds)
        result['independent_runs'].append(run)
        for index in range(args.rounds):
            order = ('native', 'candidate') if index % 2 == 0 else ('candidate', 'native')
            for name in order:
                with torch.no_grad(), contexts(name):
                    samples = [one_call(routes[name], cold=scope == 'cold') for _ in range(args.iters)]
                row = dict(round=index, order=list(order), samples=samples,
                           mean={key: statistics.mean(sample[key] for sample in samples) for key in samples[0]})
                rounds[name].append(row)
        speedups = {metric: [rounds['native'][i]['mean'][metric] / rounds['candidate'][i]['mean'][metric]
                            for i in range(args.rounds)] for metric in ('wall_ms', 'cuda_ms')}
        run.update(paired_speedups=speedups, median_speedup={key: statistics.median(value) for key, value in speedups.items()})
        run['passed'] = (run['median_speedup']['wall_ms'] >= 1.03
                         and sum(v > 1 for v in speedups['wall_ms']) >= math.ceil(7 * args.rounds / 9)
                         and run['median_speedup']['cuda_ms'] >= 1 / 1.02)
        print(scope, repeat, run['median_speedup'], 'passed', run['passed'], flush=True)
    result['passed'] = all(run['passed'] for run in result['independent_runs'])
    return result


def make_model():
    from models.converse_srresnet import ConverseMSRResNet
    from models.util_converse import Converse2D
    model = ConverseMSRResNet().cuda().eval()
    model.load_state_dict(torch.load(ROOT / 'model_zoo/converse_srresnet.pth',
                                     map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    names = []
    for name, layer in model.named_modules():
        if hasattr(layer, 'backend'):
            layer.backend = 'cuda'
        if isinstance(layer, Converse2D) and layer.kernel_size == 2 and layer.scale == 2:
            names.append(name)
    if names != ['up1', 'up2']:
        raise RuntimeError('Actual checkpoint upsampling modules differ from inspected up1/up2: ' + str(names))
    return model, names


def model_quality(model, names, solve, study, report):
    result = dict(passed=False, replaced_modules=names,
                  unmodified_preceding_modules=['upconv1', 'upconv2'],
                  checkpoint_sha256=sha(ROOT / 'model_zoo/converse_srresnet.pth'),
                  convergence_evidence=False, rows=[], actual_layer_checks=[])
    report['model_quality'] = result
    captured = {}
    for seed in (17, 29, 43):
        generator = torch.Generator().manual_seed(seed)
        value = torch.rand(1, 3, 24, 28, generator=generator).cuda()
        for mode in ('no_grad', 'inference_mode', 'frozen_gradmode'):
            handles = []
            if seed == 17 and mode == 'no_grad':
                for name in names:
                    def hook(layer, inputs, name=name):
                        captured[name] = (inputs[0].detach().clone(), layer.weight.detach(), layer.bias.detach())
                    handles.append(dict(model.named_modules())[name].register_forward_pre_hook(hook))
            try:
                torch.ops.converse2d.clear_cache()
                with study.mode_context(mode):
                    baseline = model(value)
            finally:
                for handle in handles:
                    handle.remove()
            counter = dict(calls=0)
            torch.ops.converse2d.clear_cache()
            with study.mode_context(mode), model_candidate(model, names, solve, counter):
                actual = model(value)
            metrics = error_metrics(actual, baseline.double())
            smoke = metrics['finite'] and bool(torch.isclose(actual, baseline, atol=1e-5, rtol=1e-5).all())
            passed = smoke and counter['calls'] == 2 and not actual.requires_grad
            result['rows'].append(dict(seed=seed, mode=mode, **metrics, smoke_1e_minus5=smoke,
                                       candidate_calls=counter['calls'], passed=passed))
            print('SRResNet quality', seed, mode, 'passed', passed, metrics, flush=True)
    for name in names:
        layer = dict(model.named_modules())[name]
        data = captured[name]
        spec = dict(name='checkpoint/' + name, seed=17, shape=tuple(data[0].shape),
                    kb=data[1].shape[0], kc=data[1].shape[1], kind='checkpoint_actual',
                    layout='contiguous', mode='no_grad', padding=layer.padding, eps=layer.eps)
        check = study.numeric_case(solve, spec, data)
        result['actual_layer_checks'].append(check)
        print('SRResNet actual layer budget', name, check['passed'], flush=True)
    result['passed'] = all(row['passed'] for row in result['rows'] + result['actual_layer_checks'])
    return captured


def operator_perf(args, study, solve, captured, model, report):
    from models.util_converse import Converse2D
    result = dict(passed=False, cases=[], criterion=dict(independent_runs=2, median_wall_speedup=1.03,
                  positive_rounds=math.ceil(7 * args.rounds / 9), no_cuda_slowdown_above=0.02),
                  scope='Complete eager inference caller including pad, nearest preparation where needed, regularizer, solve, crop, copies and synchronization')
    report['operator_performance'] = result
    fixtures = []
    for spec in study.case_specs():
        if spec.get('timing_fixture'):
            data = study.materialize(spec, 'cuda')
            layer = Converse2D(data[0].shape[1], data[0].shape[1], 2, scale=2,
                               padding=spec['padding'], padding_mode='replicate', eps=spec['eps'], backend='cuda').cuda()
            layer.weight = torch.nn.Parameter(data[1], requires_grad=False)
            layer.bias = torch.nn.Parameter(data[2], requires_grad=False)
            fixtures.append((spec['name'], layer, data[0]))
    modules = dict(model.named_modules())
    fixtures.extend(('checkpoint/' + name, modules[name], raw[0]) for name, raw in captured.items())
    for name, layer, value in fixtures:
        # Both routes traverse nn.Module.__call__. Only the bound forward
        # implementation changes, outside each timed group of invocations.
        routes = dict(native=lambda: layer(value), candidate=lambda: layer(value))
        contexts = lambda route: model_candidate(layer, [''], solve) if route == 'candidate' else nullcontext()
        row = dict(name=name, input=study.record(value), weight=study.record(layer.weight), bias=study.record(layer.bias),
                   scopes={})
        result['cases'].append(row)
        for scope in ('cold', 'warm'):
            print('operator', name, scope, flush=True)
            row['scopes'][scope] = paired(routes, args, scope, contexts)
        row['passed'] = all(value['passed'] for value in row['scopes'].values())
    result['passed'] = bool(result['cases']) and all(row['passed'] for row in result['cases'])
    return result['passed']


def model_perf(args, model, names, solve, report):
    generator = torch.Generator().manual_seed(17)
    value = torch.rand(1, 3, 24, 28, generator=generator).cuda()
    routes = dict(native=lambda: model(value), candidate=lambda: model(value))
    contexts = lambda name: model_candidate(model, names, solve) if name == 'candidate' else nullcontext()
    report['model_performance'] = paired(routes, args, 'warm', contexts)
    report['model_performance'].update(scope='Same pretrained SRResNet; only up1/up2 local explicit-nearest solve replacement; warm eager',
                                       input_shape=list(value.shape), candidate_calls_per_forward=2,
                                       convergence_evidence=False)
    return report['model_performance']['passed']


def verify_unchanged(args, report, study, research_loader, production_loader):
    identity = report['identity']
    if study.source_identity() != identity['gate_source_sha256']:
        raise RuntimeError('Gated source changed during experiment')
    if research_loader.build_identity() != identity['checked_research_manifest']['identity']:
        raise RuntimeError('Research toolchain inputs changed during experiment')
    if sha(identity['checked_research_manifest']['binary']) != identity['checked_research_manifest']['binary_sha256']:
        raise RuntimeError('Research binary changed during experiment')
    if sha(ROOT / '.build/cuda/source_manifest.json') != identity['production_manifest_sha256']:
        raise RuntimeError('Production manifest changed during experiment')
    if production_loader.production_source_hashes() != identity['production_sources']:
        raise RuntimeError('Production sources changed during experiment')
    binary = ROOT / '.build/cuda' / identity['checked_production_manifest']['library']
    if sha(binary) != identity['checked_production_manifest']['binary_sha256']:
        raise RuntimeError('Production binary changed during experiment')
    if sha(__file__) != identity['helper_sha256']:
        raise RuntimeError('Post-gate helper changed during experiment')
    for name, digest in identity['model_sources'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Model/checkpoint source changed during experiment: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve evidence: choose a new output path')
    if args.warmup < 5 or args.rounds < 9 or args.iters < 10:
        parser.error('Require >=5 warmups, >=9 alternating rounds, >=10 calls')
    args.gate, args.artifacts, args.output = args.gate.resolve(), args.artifacts.resolve(), args.output.resolve()
    report = dict(kind='fftfree_compensated_post_gate_performance', status='running', passed=False,
                  settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters, independent_runs=2),
                  convergence_evidence=False, production_modified=False)
    try:
        study, solve, research_loader, production_loader = setup(args, report)
        model, names = make_model()
        captured = model_quality(model, names, solve, study, report)
        if not report['model_quality']['passed']:
            report['timing_status'] = 'blocked_by_checkpoint_model_or_actual_layer_numerical_gate'
        elif not operator_perf(args, study, solve, captured, model, report):
            report['timing_status'] = 'blocked_by_complete_call_performance_gate'
        else:
            report['passed'] = model_perf(args, model, names, solve, report)
            report['timing_status'] = 'complete'
        verify_unchanged(args, report, study, research_loader, production_loader)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='error', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        exclusive_report(args.output, report)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
