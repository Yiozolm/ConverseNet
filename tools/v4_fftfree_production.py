"""Fresh numerical/model/performance admission for the actual production op.

Research reports are provenance only. No research library is loaded. The old
model control is the exact Converse2D.forward from commit 09f0f12, evaluated in
the real util_converse globals and bound only to the two controlled layers.
"""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'tools/v4_fftfree_compensated_fused_lambda'
sys.path[:0] = [str(RESEARCH), str(ROOT / 'test'), str(ROOT)]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from numerical_policy import error_metrics
import v4_fftfree_fused_lambda_perf as timing

LEGACY_COMMIT = '09f0f12'
PRIVATE_OP = 'converse2d::_nearest_k2_s2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def provenance(args):
    if args.prototype_gate is None:
        return dict(purpose='Fresh production admission; no prototype evidence substituted',
                    research_binary_loaded=False, research_toolchain_revalidation=False,
                    prototype_reports=None)
    gate = json.loads(args.prototype_gate.read_text())
    perf = json.loads(args.prototype_perf.read_text())
    if (gate.get('status') != 'complete' or not gate.get('passed') or len(gate.get('cases', [])) != 3519
            or not all(row['passed'] for row in gate['cases']) or not gate.get('contracts', {}).get('passed')):
        raise RuntimeError('Successful 3519-case prototype gate required as provenance')
    if (perf.get('status') != 'complete' or not perf.get('passed')
            or not perf.get('model_quality', {}).get('passed')
            or not perf.get('operator_performance', {}).get('passed')
            or not perf.get('model_performance', {}).get('passed')):
        raise RuntimeError('Successful prototype complete-call/model evidence required')
    if perf['identity']['gate_sha256'] != sha(args.prototype_gate):
        raise RuntimeError('Prototype evidence chain mismatch')
    for name, digest in gate['source_sha256'].items():
        # Production source/build configuration deliberately changed. The
        # research source itself and the numerical policy remain immutable.
        path = Path(name)
        if (path.parts[0] == 'tools' or name.replace('\\', '/').startswith('test/numerical_policy')) and sha(ROOT / path) != digest:
            raise RuntimeError('Prototype dependency changed: ' + name)
    if sha(perf['identity']['helper_path']) != perf['identity']['helper_sha256']:
        raise RuntimeError('Prototype timing helper changed')
    return dict(gate=dict(path=str(args.prototype_gate), sha256=sha(args.prototype_gate)),
                performance=dict(path=str(args.prototype_perf), sha256=sha(args.prototype_perf)),
                purpose='Historical provenance only; never substituted for fresh production evidence',
                research_binary_loaded=False, research_toolchain_revalidation=False)


def load_legacy_forward():
    import models.util_converse as util
    commit = subprocess.check_output(['git', 'rev-parse', LEGACY_COMMIT], cwd=ROOT, text=True).strip()
    raw = subprocess.check_output(['git', 'show', commit + ':models/util_converse.py'], cwd=ROOT)
    tree = ast.parse(raw.decode('utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Converse2D')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    canonical = ast.dump(method, include_attributes=False)
    if '_nearest_k2_s2' in canonical:
        raise RuntimeError('Legacy control already contains optimized nearest dispatch')
    method.name = '_v4_legacy_converse_forward'
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    # Deliberately use the module's real dictionary, not a copy: the checked
    # extension's _HAS_CONVERSE2D_EXT flag must update in the same globals.
    exec(compile(module, f'git:{commit}:models/util_converse.py', 'exec'), util.__dict__)
    function = util.__dict__['_v4_legacy_converse_forward']
    if function.__globals__ is not util.__dict__:
        raise RuntimeError('Legacy function does not share actual production module globals')
    return function, dict(commit=commit, file_sha256=hashlib.sha256(raw).hexdigest(),
                          forward_ast_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                          uses_real_util_globals=True, binding_scope=['up1', 'up2'])


@contextmanager
def legacy_layers(model, names, forward):
    modules, previous = dict(model.named_modules()), []
    try:
        for name in names:
            layer = modules[name]
            previous.append((layer, 'forward' in vars(layer), vars(layer).get('forward')))
            layer.forward = types.MethodType(forward, layer)
        yield
    finally:
        for layer, had_override, old in previous:
            if had_override:
                layer.forward = old
            else:
                del layer.forward


def setup(args, report):
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend and CPU-only overrides')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    evidence = provenance(args)
    manifest_path = ROOT / '.build/cuda/source_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    arch = manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
    caller_arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    try:
        if arch:
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        else:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        loader = import_file('production_fftfree_checked_loader', ROOT / 'test/extension_loader.py')
        loader.load_extension()
    finally:
        if caller_arch is None:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        else:
            os.environ['TORCH_CUDA_ARCH_LIST'] = caller_arch
    if sha(manifest_path) != manifest_hash:
        raise RuntimeError('Production loader changed its manifest')
    if not hasattr(torch.ops.converse2d, '_nearest_k2_s2'):
        raise RuntimeError('Actual production private op is unavailable')
    study = import_file('production_fftfree_spec_provider', RESEARCH / 'study.py')
    legacy, legacy_identity = load_legacy_forward()
    dependencies = [Path(__file__), Path(timing.__file__), RESEARCH / 'study.py', RESEARCH / 'inference.py',
                    RESEARCH / 'historical_fixture.json', ROOT / 'test/numerical_policy.py',
                    ROOT / 'test/fp32_baseline.py', ROOT / 'test/extension_loader.py',
                    ROOT / 'models/converse_core.py', ROOT / 'models/util_converse.py',
                    ROOT / 'models/converse_srresnet.py', ROOT / 'models/pointwise.py',
                    ROOT / 'model_zoo/converse_srresnet.pth']
    report['identity'] = dict(stage_helper_sha256=sha(__file__), prototype_provenance=evidence,
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in dependencies},
        checked_production_manifest=manifest, production_manifest_sha256=manifest_hash,
        production_sources=loader.production_source_hashes(), legacy_forward=legacy_identity,
        production_load_architecture=arch, binary_scope='Production checked extension only',
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         tf32=False, amp=False, cudnn_benchmark=False, deterministic_algorithms=True,
                         torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                         affinity=study.affinity()))
    return study, legacy, loader


def solve(x, weight, bias, eps=1e-5):
    return torch.ops.converse2d._nearest_k2_s2(x, weight, bias, float(eps), 'v7')


def module_numeric_case(study, spec, data, layer=None):
    """Test actual pad-cancelling module against the old padded FFT problem.

Direct-op fixtures with padding zero receive a legal positive replicate pad,
so every appended module probe exercises removal of real padding/cropping.
The original direct-op specification and its comparison remain unchanged.
"""
    from models.util_converse import Converse2D
    x, weight, bias = data
    before = [study.record(value) for value in data]
    original_requires_grad = [value.requires_grad for value in data]
    if layer is None:
        pad = spec['padding'] or min(2, x.shape[-2], x.shape[-1])
        layer = Converse2D(x.shape[1], x.shape[1], 2, scale=2, padding=pad,
                           padding_mode='replicate', eps=spec['eps'], backend='cuda')
        # Preserve the exact supplied values and layout/broadcast metadata.
        # Frozen GradMode must select inference, just like the other two modes.
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    else:
        pad = layer.padding
    if layer.weight.requires_grad or layer.bias.requires_grad:
        raise RuntimeError('Module numerical probe requires frozen parameters')
    wrapped_inputs = [study.record(value) for value in (x, layer.weight, layer.bias)]
    wrapping_preserved = wrapped_inputs == before
    expected_shape = (*x.shape[:2], 2 * x.shape[-2], 2 * x.shape[-1])
    kernel_input_shapes = []
    original_op = torch.ops.converse2d._nearest_k2_s2
    def record_call(*args, **kwargs):
        kernel_input_shapes.append(list(args[0].shape))
        return original_op(*args, **kwargs)
    with study.mode_context(spec['mode']):
        padded = torch.nn.functional.pad(x, (pad,) * 4, mode=layer.padding_mode, value=0) if pad else x
        values = padded, weight, bias
        baseline = study.reference_solve(study.frozen_half, values, spec['eps'])
        high = study.reference_solve(study.reference64, tuple(value.double() for value in values), spec['eps'])
        if pad:
            crop = (..., slice(2 * pad, -2 * pad), slice(2 * pad, -2 * pad))
            baseline, high = baseline[crop], high[crop]
        prior = torch.nn.functional.interpolate(padded, scale_factor=2, mode='nearest')
        denominator = study.denominator_statistics((padded, prior, weight, bias), 2, spec['eps'], padded.device)
        # Instrument only the Python entry, then invoke the actual production
        # operator unchanged. No profiler or replacement arithmetic is used.
        with patch.object(torch.ops.converse2d, '_nearest_k2_s2', new=record_call):
            actual = layer(x)
            repeated = layer(x)
            # A lazy negative view with the same logical input values also
            # exercises the new pre-padding branch's metadata handling.
            negative_view = torch._neg_view(-x)
            negative_actual = layer(negative_view)
        shape_matches = tuple(actual.shape) == expected_shape and tuple(negative_actual.shape) == expected_shape
        if not shape_matches:
            return dict(passed=False, module_padding=pad, padding_mode=layer.padding_mode,
                        actual_shape=list(actual.shape), expected_shape=list(expected_shape),
                        failure='Production module did not return the selected 2x output')
        regime = study.regime_for(spec)
        checks = {name: study.output_check(value, baseline, high, regime,
                                            dynamic_range=spec['kind'] == 'dynamic')
                  for name, value in (('module_output', actual), ('lazy_negative_input', negative_actual))}
        repeated_equal = study.record(actual) == study.record(repeated)
        no_grad = not actual.requires_grad and not negative_actual.requires_grad
    unchanged = (before == [study.record(value) for value in data]
                 and original_requires_grad == [value.requires_grad for value in data])
    original_domain_selected = kernel_input_shapes == [list(x.shape)] * 3
    return dict(passed=all(check['passed'] for check in checks.values()) and repeated_equal and unchanged
                       and wrapping_preserved and no_grad and original_domain_selected,
                module_padding=pad, padding_mode=layer.padding_mode,
                mode=spec['mode'], regime=regime, checks=checks, denominator_statistics=denominator,
                reference='Pad FP32 input, frozen FP32 / independent FP64 solve, crop complete 2x blocks',
                input_shape=list(x.shape), padded_reference_shape=list(padded.shape),
                expected_shape=list(expected_shape), output=study.record(actual),
                input_strides=list(x.stride()), weight_strides=list(weight.stride()), bias_strides=list(bias.stride()),
                kernel_input_shapes=kernel_input_shapes, unpadded_kernel_input_selected=original_domain_selected,
                repeated_equal=repeated_equal, inputs_unchanged=unchanged, output_requires_grad=not no_grad,
                input_tensors_before=before, wrapped_input_tensors=wrapped_inputs,
                parameter_wrapping_preserved_values_and_strides=wrapping_preserved,
                original_requires_grad=original_requires_grad,
                frozen_parameter_requires_grad=[layer.weight.requires_grad, layer.bias.requires_grad],
                lazy_negative_view_present=negative_view.is_neg())


def gate(args, study, report):
    specs = list(study.case_specs())
    if len(specs) != 3519 or len({row['name'] for row in specs}) != 3519:
        raise RuntimeError('Incomplete production test matrix')
    result = dict(passed=False, cases=[], failed_cases=[], direct_cases=3519, module_cases=3519,
                  implementation='Actual production _nearest_k2_s2 plus actual Converse2D.forward pad cancellation',
                  contracts='Production release tests cover CPU and differentiable GradMode fallback; research rejection contracts intentionally not invoked')
    report['gate'] = result
    for index, spec in enumerate(specs):
        data = study.materialize(spec, 'cuda')
        row = study.numeric_case(solve, spec, data)
        row['direct_passed'] = row['passed']
        row['module_check'] = module_numeric_case(study, spec, data)
        row['passed'] = row['direct_passed'] and row['module_check']['passed']
        result['cases'].append(row)
        if not row['passed']:
            result['failed_cases'].append(row['name'])
            print('FAIL production', row['name'], flush=True)
        if (index + 1) % 100 == 0:
            print(f'Production numerical gate {index + 1}/{len(specs)}', flush=True)
    result['passed'] = not result['failed_cases']
    return result['passed']


def traced(call):
    # Untimed route evidence only. Performance below has no profiler/hook.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        value = call()
    counts = {event.key: event.count for event in trace.key_averages()}
    return value, counts.get(PRIVATE_OP, 0)


def model_quality(args, study, legacy, report):
    model, names = timing.make_model()
    result = dict(passed=False, candidate='Default production Converse2D.forward',
                  baseline='Legacy CUDA forward bound only to up1/up2', convergence_evidence=False,
                  modules=names, rows=[], actual_layer_checks=[])
    report['model'] = result
    captured = {}
    for seed in (17, 29, 43):
        value = torch.rand(1, 3, 24, 28, generator=torch.Generator().manual_seed(seed)).cuda()
        for mode in ('no_grad', 'inference_mode', 'frozen_gradmode'):
            handles = []
            if seed == 17 and mode == 'no_grad':
                for name in names:
                    def hook(layer, inputs, name=name):
                        captured[name] = (inputs[0].detach().clone(), layer.weight.detach(), layer.bias.detach())
                    handles.append(dict(model.named_modules())[name].register_forward_pre_hook(hook))
            try:
                torch.ops.converse2d.clear_cache()
                with study.mode_context(mode), legacy_layers(model, names, legacy):
                    baseline, old_calls = traced(lambda: model(value))
            finally:
                for handle in handles:
                    handle.remove()
            torch.ops.converse2d.clear_cache()
            with study.mode_context(mode):
                actual, new_calls = traced(lambda: model(value))
            metrics = error_metrics(actual, baseline.double())
            smoke = metrics['finite'] and bool(torch.isclose(actual, baseline, atol=1e-5, rtol=1e-5).all())
            passed = smoke and old_calls == 0 and new_calls == 2 and not actual.requires_grad
            result['rows'].append(dict(seed=seed, mode=mode, **metrics, smoke_1e_minus5=smoke,
                                       legacy_private_op_calls=old_calls, candidate_private_op_calls=new_calls, passed=passed))
            print('production SRResNet', seed, mode, 'calls', old_calls, new_calls, 'passed', passed, metrics, flush=True)
    for name in names:
        layer = dict(model.named_modules())[name]
        data = captured[name]
        spec = dict(name='production_checkpoint/' + name, seed=17, shape=tuple(data[0].shape),
                    kb=data[1].shape[0], kc=data[1].shape[1], kind='checkpoint_actual',
                    layout='contiguous', mode='no_grad', padding=layer.padding, eps=layer.eps)
        check = study.numeric_case(solve, spec, data)
        check['direct_passed'] = check['passed']
        check['module_check'] = module_numeric_case(study, spec, data, layer)
        check['passed'] = check['direct_passed'] and check['module_check']['passed']
        result['actual_layer_checks'].append(check)
    result['passed'] = all(row['passed'] for row in result['rows'] + result['actual_layer_checks'])
    return result['passed'], (model, names, captured)


def capture_for_perf(model, names, legacy):
    captured, handles = {}, []
    for name in names:
        def hook(layer, inputs, name=name):
            captured[name] = (inputs[0].detach().clone(), layer.weight.detach(), layer.bias.detach())
        handles.append(dict(model.named_modules())[name].register_forward_pre_hook(hook))
    try:
        value = torch.rand(1, 3, 24, 28, generator=torch.Generator().manual_seed(17)).cuda()
        with torch.no_grad(), legacy_layers(model, names, legacy):
            model(value)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def perf(args, study, legacy, report, state=None):
    from models.util_converse import Converse2D
    if state is None:
        model, names = timing.make_model()
        captured = capture_for_perf(model, names, legacy)
    else:
        model, names, captured = state
    result = dict(passed=False, operator_cases=[], model=None,
                  scope='Actual production versus legacy module; both use nn.Module dispatch; fixed weights and inputs')
    report['perf'] = result
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
    fixtures.extend(('production_checkpoint/' + name, modules[name], raw[0]) for name, raw in captured.items())
    for name, layer, value in fixtures:
        routes = dict(native=lambda: layer(value), candidate=lambda: layer(value))
        contexts = lambda route: legacy_layers(layer, [''], legacy) if route == 'native' else nullcontext()
        row = dict(name=name, input=study.record(value), weight=study.record(layer.weight), bias=study.record(layer.bias), scopes={})
        result['operator_cases'].append(row)
        for scope in ('cold', 'warm'):
            print('production operator performance', name, scope, flush=True)
            row['scopes'][scope] = timing.paired(routes, args, scope, contexts)
        row['passed'] = all(check['passed'] for check in row['scopes'].values())
    if not all(row['passed'] for row in result['operator_cases']):
        result['model'] = dict(status='blocked_by_operator_performance')
        return False
    value = torch.rand(1, 3, 24, 28, generator=torch.Generator().manual_seed(17)).cuda()
    routes = dict(native=lambda: model(value), candidate=lambda: model(value))
    contexts = lambda route: legacy_layers(model, names, legacy) if route == 'native' else nullcontext()
    result['model'] = timing.paired(routes, args, 'warm', contexts)
    result['model'].update(candidate_private_op_calls_per_forward=2, legacy_private_op_calls_per_forward=0,
                           convergence_evidence=False, input_shape=list(value.shape), modules=names)
    result['passed'] = result['model']['passed']
    return result['passed']


def verify_identity(report, loader):
    identity = report['identity']
    for name, digest in identity['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Production stage source changed: ' + name)
    if sha(ROOT / '.build/cuda/source_manifest.json') != identity['production_manifest_sha256']:
        raise RuntimeError('Production manifest changed')
    if loader.production_source_hashes() != identity['production_sources']:
        raise RuntimeError('Production C++ source changed')
    manifest = identity['checked_production_manifest']
    if sha(ROOT / '.build/cuda' / manifest['library']) != manifest['binary_sha256']:
        raise RuntimeError('Production binary changed')


def parent_evidence(args, report, required):
    parent = json.loads(args.parent.read_text())
    if parent.get('status') != 'complete' or not parent.get('passed'):
        raise RuntimeError('Complete passed fresh production parent required')
    for key in ('checked_production_manifest', 'production_sources', 'legacy_forward', 'prototype_provenance'):
        if parent['identity'][key] != report['identity'][key]:
            raise RuntimeError('Production parent identity mismatch: ' + key)
    # Each stage records its own helper SHA. Existing report hashes are never
    # rewritten if a later stage adds harness code; shared math/data sources and
    # the production implementation must still exactly match the parent.
    for name, digest in parent['identity']['source_sha256'].items():
        if (ROOT / name).resolve() == Path(__file__).resolve():
            continue
        if sha(ROOT / name) != digest:
            raise RuntimeError('Production parent dependency changed: ' + name)
    for stage in required:
        if not parent.get(stage, {}).get('passed'):
            raise RuntimeError('Missing successful fresh production ' + stage)
        report[stage] = parent[stage]
    report['parent'] = dict(path=str(args.parent), sha256=sha(args.parent),
                          parent_stage_helper_sha256=parent['identity']['stage_helper_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('gate', 'model', 'perf', 'all'), default='gate')
    parser.add_argument('--prototype-gate', type=Path)
    parser.add_argument('--prototype-perf', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    if (args.prototype_gate is None) != (args.prototype_perf is None):
        parser.error('Supply both prototype reports for optional historical provenance, or neither')
    if args.output.exists():
        parser.error('Preserve evidence: choose a new output path')
    if args.stage in ('model', 'perf') and args.parent is None:
        parser.error('Separate model/perf stages require --parent fresh production evidence')
    if args.warmup < 5 or args.rounds < 9 or args.iters < 10:
        parser.error('Require >=5 warmups, >=9 alternating rounds and >=10 calls')
    for key in ('prototype_gate', 'prototype_perf', 'parent', 'output'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    report = dict(kind='actual_production_nearest_k2_s2', stage=args.stage, status='running', passed=False,
                  fresh_production_evidence=True, research_binary_loaded=False, convergence_evidence=False,
                  settings=dict(warmup=args.warmup, rounds=args.rounds, iters=args.iters, independent_runs=2))
    try:
        study, legacy, loader = setup(args, report)
        if args.stage in ('model', 'perf'):
            parent_evidence(args, report, ('gate',) if args.stage == 'model' else ('gate', 'model'))
        passed, state = True, None
        if args.stage in ('gate', 'all'):
            passed = gate(args, study, report)
        if passed and args.stage in ('model', 'all'):
            passed, state = model_quality(args, study, legacy, report)
        if passed and args.stage in ('perf', 'all'):
            passed = perf(args, study, legacy, report, state)
        verify_identity(report, loader)
        report.update(status='complete', passed=passed)
    except Exception as error:
        report.update(status='error', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        timing.exclusive_report(args.output, report)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
