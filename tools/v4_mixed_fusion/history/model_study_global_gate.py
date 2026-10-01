"""Actual DnCNN / USRNet-s1 model study for an independently admitted fusion.

All three routes use the same pretrained FP32 model. mixed_baseline and fused
cast every Converse2D input to FP16 at the same location inside the real model
call. Only fused uses the research adapter. USRNet's five DataNet calls remain
unmodified FP32; this differs from the earlier forty-boundary sensitivity run.

No CUDA candidate is defined here, no output casts or master-weight changes
are made, and model quality relative to original FP32 remains diagnostic.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from tools.v4_mixed_precision import model_study as data_tools
from tools.v4_mixed_fusion import baseline

ROUTES = ('original_fp32', 'mixed_baseline', 'fused')
EXPECTED_CALLS = {'dncnn': 20, 'usrnet': 35}
ADMISSION = dict(minimum_median_wall_speedup=1.03, minimum_median_cuda_speedup=1.03,
    independent_repeats=2, minimum_rounds=9, calls_per_round=10,
    positive_wall_rounds='ceil(rounds*7/9) in each repeat',
    accuracy='All >=6 fixed real crops: fused exactly equals mixed_baseline and both PSNR/SSIM are identical',
    model_requirement='At least one measured model passes both repeat performance requirements; all measured models remain correct',
    comparison='mixed_baseline / fused; original_fp32 comparisons are separate diagnostics',
    production_quality_threshold=None, production_promotion=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(*, include_candidate=True):
    names = ['tools/v4_mixed_fusion/model_study.py', 'tools/v4_mixed_fusion/baseline.py',
             'tools/v4_mixed_precision/model_study.py', 'tools/roadmap_quality/usrnet_training_data.py',
             'utils/utils_image.py', 'test/extension_loader.py']
    if include_candidate:
        names.extend(('tools/v4_mixed_fusion/adapter.py', 'tools/v4_mixed_fusion/loader.py'))
    names.extend(path.relative_to(ROOT).as_posix() for path in sorted((ROOT / 'models').glob('*.py')))
    return {name: sha(ROOT / name) for name in names}


class CountedExtension:
    """Untimed call evidence only. Timing uses the original extension object."""
    def __init__(self, extension):
        self.extension = extension
        self.calls = 0
        self.arguments = []

    def pad_cast(self, value, padding, mode):
        self.calls += 1
        self.arguments.append(dict(shape=list(value.shape), dtype=str(value.dtype),
                                   contiguous=value.is_contiguous(), padding=padding, mode=mode))
        return self.extension.pad_cast(value, padding, mode)

    def __getattr__(self, name):
        return getattr(self.extension, name)


@contextmanager
def forward_scope(model, route, extension=None, *, adapter_function=None,
                  instrument=False, verify_contents=True):
    """Temporarily replace forward, preserving the real nn.Module.__call__.

    Saved bound originals prevent recursion. Existing pre/post hooks, parameter
    objects, versions, and training flags are retained. Nothing changes GradMode.
    """
    from models.util_converse import Converse2D
    if route not in ROUTES:
        raise ValueError('Unknown model route')
    if route == 'fused' and adapter_function is None:
        from tools.v4_mixed_fusion.adapter import mixed_module_forward
        adapter_function = mixed_module_forward
    selected = [(name, module) for name, module in model.named_modules() if isinstance(module, Converse2D)]
    if not selected:
        raise RuntimeError('No Converse2D module found')
    before_hash = data_tools.state_hash(model) if verify_contents else None
    parameters = [(name, value, value.data_ptr(), value._version) for name, value in model.named_parameters()]
    training = {name: module.training for name, module in model.named_modules()}
    originals, handles, calls = [], [], Counter()
    original_hooks = {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                      for name, module in selected}
    counted = CountedExtension(extension) if route == 'fused' and instrument else None
    chosen_extension = counted if counted is not None else extension
    probe = dict(calls=calls, extension=counted,
                 modules={name: dict(kernel=module.kernel_size, scale=module.scale,
                                    padding=module.padding, padding_mode=module.padding_mode)
                          for name, module in selected})
    try:
        for name, module in selected:
            if instrument:
                def count(this, arguments, name=name):
                    calls[name] += 1
                handles.append(module.register_forward_pre_hook(count))
            if route == 'original_fp32':
                continue
            original = module.forward
            originals.append((module, 'forward' in vars(module), vars(module).get('forward')))
            def forward(this, value, original_forward=original):
                if value.dtype != torch.float32:
                    raise ValueError('Model boundary must receive FP32 from unchanged surrounding layers')
                low = value.to(torch.float16)
                if route == 'mixed_baseline' or this.scale != 1:
                    return original_forward(low.float())
                return adapter_function(this, low, chosen_extension, original_forward=original_forward)
            module.forward = types.MethodType(forward, module)
        yield probe
    finally:
        for handle in handles:
            handle.remove()
        for module, had_override, original in reversed(originals):
            if had_override:
                module.forward = original
            else:
                del module.forward
        current = dict(model.named_parameters())
        if any(current.get(name) is not value or value.data_ptr() != pointer or value._version != version
               for name, value, pointer, version in parameters):
            raise RuntimeError('Model master parameter identity/version changed')
        if training != {name: module.training for name, module in model.named_modules()}:
            raise RuntimeError('Model training flags changed')
        if any(original_hooks[name] != (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
               for name, module in selected):
            raise RuntimeError('Existing module hook registrations changed')
        if before_hash is not None and data_tools.state_hash(model) != before_hash:
            raise RuntimeError('Model parameter/buffer contents changed')


def verify_operator_gate(path, artifacts, block_threads, research, production):
    gate = json.loads(path.read_text(encoding='utf-8'))
    if gate.get('status') != 'complete' or gate.get('passed') is not True:
        raise RuntimeError('Complete, successful operator gate required before model execution')
    if gate.get('block_threads') != block_threads:
        raise RuntimeError('Operator gate belongs to a different block configuration')
    if Path(gate['research_artifacts']).resolve() != artifacts.resolve():
        raise RuntimeError('Operator gate artifact directory mismatch')
    if gate['research_manifest'] != research:
        raise RuntimeError('Operator gate research source/build identity mismatch')
    if gate['checked_manifest'] != production['checked_manifest']:
        raise RuntimeError('Operator gate production build mismatch')
    if gate['production']['production_source_sha256'] != production['production_source_sha256']:
        raise RuntimeError('Operator gate production sources changed')
    for name, digest in gate['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Operator gate dependency changed: ' + name)
    return dict(path=str(path), sha256=sha(path), block_threads=block_threads,
                research_artifacts=str(artifacts), passed=True)


def route_probe(model, route, inputs, extension):
    with forward_scope(model, route, extension, instrument=True) as probe, torch.no_grad():
        output = model(*inputs).detach().clone()
    return output, dict(module_calls=dict(probe['calls']), modules=probe['modules'],
        pad_cast_calls=probe['extension'].calls if probe['extension'] is not None else 0,
        pad_cast_arguments=probe['extension'].arguments if probe['extension'] is not None else [],
        instrumentation='Forward pre-hooks and extension proxy used only outside timing')


def accuracy(model, name, examples, args, extension, result):
    result['accuracy'] = dict(rows=[], passed=False, scope='Only Converse2D input boundaries are quantized; USRNet DataNet remains unmodified FP32')
    rows = result['accuracy']['rows']
    expected_calls = EXPECTED_CALLS[name]
    border = 0 if name == 'dncnn' else 1
    timing_input = None
    for example in examples:
        cpu_inputs, target, provenance = data_tools.model_sample(name, example, args)
        inputs = tuple(value.cuda() if torch.is_tensor(value) else value for value in cpu_inputs)
        if timing_input is None:
            timing_input = inputs, target, provenance
        input_hashes = [data_tools.tensor_record(value) for value in inputs if torch.is_tensor(value)]
        outputs, probes = {}, {}
        for route in ROUTES:
            torch.ops.converse2d.clear_cache()
            outputs[route], probes[route] = route_probe(model, route, inputs, extension)
        finite = all(bool(torch.isfinite(value).all()) and value.dtype == torch.float32 for value in outputs.values())
        exact = bool(torch.equal(outputs['fused'], outputs['mixed_baseline']))
        counts_ok = all(sum(probe['module_calls'].values()) == expected_calls for probe in probes.values())
        counts_ok &= probes['fused']['pad_cast_calls'] == expected_calls
        if name == 'usrnet':
            counts_ok &= len(probes['fused']['module_calls']) == 7 and all(count == 5 for count in probes['fused']['module_calls'].values())
        quality = {route: data_tools.quality(output, target, border) for route, output in outputs.items()}
        identical_quality = quality['fused'] == quality['mixed_baseline']
        with torch.no_grad():
            restored = model(*inputs)
        restored_exact = bool(torch.equal(restored, outputs['original_fp32']))
        inputs_unchanged = input_hashes == [data_tools.tensor_record(value) for value in inputs if torch.is_tensor(value)]
        row = dict(sample=example['key'], provenance=provenance, finite=finite, exact_fused_vs_mixed=exact,
            quality_identical_fused_vs_mixed=identical_quality, route_counts_valid=bool(counts_ok),
            restored_original_output_exact=restored_exact, caller_inputs_unchanged=inputs_unchanged,
            outputs={route: data_tools.tensor_record(value) for route, value in outputs.items()},
            errors=dict(fused_vs_mixed=data_tools.error_metrics(outputs['fused'], outputs['mixed_baseline']),
                        mixed_vs_original=data_tools.error_metrics(outputs['mixed_baseline'], outputs['original_fp32']),
                        fused_vs_original=data_tools.error_metrics(outputs['fused'], outputs['original_fp32'])),
            quality=quality, probes=probes,
            passed=bool(finite and exact and identical_quality and counts_ok and restored_exact and inputs_unchanged))
        rows.append(row)
        print('model accuracy', name, example['key'], 'passed', row['passed'], 'pad_cast_calls', probes['fused']['pad_cast_calls'], flush=True)
    result['accuracy']['passed'] = bool(rows) and all(row['passed'] for row in rows)
    return timing_input


def timing(model, name, inputs, provenance, args, extension, result):
    report = dict(passed=False, sample=provenance, repeats=[], settings=dict(warmup=args.warmup,
        rounds=args.rounds, calls=args.iters, repeats=args.repeats),
        scope='Actual whole-model __call__, including every FP32->FP16 cast, adapter dispatch, pad/cast and unchanged solver. No counters, hooks or profiler inside timing.',
        primary_ratio='mixed_baseline / fused', original_fp32_ratio='Diagnostic only; not an accuracy-preserving optimization claim')
    result['performance'] = report
    initial_state = data_tools.state_hash(model)
    with torch.no_grad():
        for repeat_index in range(args.repeats):
            torch.ops.converse2d.clear_cache()
            for route in ROUTES:
                with forward_scope(model, route, extension, instrument=False, verify_contents=False):
                    for _ in range(args.warmup):
                        output = model(*inputs)
                        del output
            run = dict(pairs=[])
            report['repeats'].append(run)
            for index in range(args.rounds):
                order = list(ROUTES) if index % 2 == 0 else list(reversed(ROUTES))
                pair = dict(round=index, order=order)
                for route in order:
                    # Scope setup/restoration is outside timing. Each patched
                    # forward and all its casts run inside the model call.
                    with forward_scope(model, route, extension, instrument=False, verify_contents=False):
                        pair[route] = baseline.measure(lambda: model(*inputs), args.iters)
                        probe_output = model(*inputs)
                    if not bool(torch.isfinite(probe_output).all()):
                        raise RuntimeError('Nonfinite whole-model timing probe')
                    del probe_output
                run['pairs'].append(pair)
            run['medians'] = {route: {metric: statistics.median(pair[route][metric] for pair in run['pairs'])
                for metric in ('wall_ms', 'cuda_event_ms', 'peak_increment_bytes')} for route in ROUTES}
            speedups = {metric: [pair['mixed_baseline'][metric] / pair['fused'][metric] for pair in run['pairs']]
                        for metric in ('wall_ms', 'cuda_event_ms')}
            run['mixed_to_fused_paired_speedups'] = speedups
            run['mixed_to_fused_median_speedup'] = {key: statistics.median(values) for key, values in speedups.items()}
            run['original_to_fused_median_speedup_diagnostic'] = {metric: statistics.median(
                pair['original_fp32'][metric] / pair['fused'][metric] for pair in run['pairs'])
                for metric in ('wall_ms', 'cuda_event_ms')}
            run['positive_wall_rounds'] = sum(value > 1 for value in speedups['wall_ms'])
            run['required_positive_wall_rounds'] = math.ceil(args.rounds * 7 / 9)
            run['passed'] = (run['mixed_to_fused_median_speedup']['wall_ms'] >= 1.03
                and run['mixed_to_fused_median_speedup']['cuda_event_ms'] >= 1.03
                and run['positive_wall_rounds'] >= run['required_positive_wall_rounds'])
            print('model performance', name, repeat_index, run['mixed_to_fused_median_speedup'], 'passed', run['passed'], flush=True)
    if data_tools.state_hash(model) != initial_state:
        raise RuntimeError('Whole-model timing changed master parameters or buffers')
    report['passed'] = bool(report['repeats']) and all(run['passed'] for run in report['repeats'])


def self_check(report):
    """CPU mechanics only; the stub is explicitly not a fusion candidate."""
    from models.util_converse import Converse2D
    torch.set_num_threads(1)
    model = torch.nn.Sequential(Converse2D(2, 2, 3, scale=1, padding=2, backend='pytorch')).eval()
    value = torch.randn(1, 2, 7, 9)
    original_hook_calls = Counter()
    handle = model[0].register_forward_hook(lambda module, inputs, output: original_hook_calls.update(('post',)))
    original_state = data_tools.state_hash(model)
    def stub(module, low, extension, *, original_forward):
        return original_forward(low.float())
    with torch.no_grad():
        control = model(value)
        with forward_scope(model, 'mixed_baseline', instrument=True) as plain:
            mixed = model(value)
        with forward_scope(model, 'fused', object(), adapter_function=stub, instrument=True) as fused:
            actual = model(value)
        restored = model(value)
    assert torch.equal(mixed, actual) and torch.equal(control, restored)
    assert sum(plain['calls'].values()) == sum(fused['calls'].values()) == 1
    assert original_hook_calls['post'] == 4 and data_tools.state_hash(model) == original_state
    try:
        with forward_scope(model, 'mixed_baseline'):
            raise RuntimeError('intentional cleanup probe')
    except RuntimeError as error:
        assert str(error) == 'intentional cleanup probe'
    assert 'forward' not in vars(model[0])
    assert data_tools.state_hash(model) == original_state
    handle.remove()
    assert not torch.cuda.is_initialized()
    report.update(status='complete', passed=True, gpu_evidence=False,
        self_check=dict(exact_stub_vs_mixed=True, master_state_restored=True,
                        original_hooks_preserved=True, exception_restoration=True,
                        limitation='Tiny CPU stub tests scope mechanics only; no CUDA fusion or model admission'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--operator-gate', type=Path)
    parser.add_argument('--block-threads', type=int, choices=(128, 256, 512), default=256)
    parser.add_argument('--models', nargs='+', choices=('dncnn', 'usrnet'), default=['dncnn', 'usrnet'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Use a fresh report path; retain all earlier failures')
    if not args.self_check and (args.artifacts is None or args.operator_gate is None):
        parser.error('--artifacts and --operator-gate are REQUIRED for GPU model execution/timing')
    if args.samples < 6 or args.patch_size != 96 or args.warmup < 5 or args.rounds < 9 or args.iters < 10 or args.repeats < 2:
        parser.error('Require >=6 real 96x96 crops, >=5 warmups, >=9 rounds, >=10 calls and >=2 repeats')
    args.usr_scale, args.synthetic = 1, False
    report = dict(kind='mixed_fusion_full_model_study', status='running', passed=False,
        source_sha256=source_identity(include_candidate=not args.self_check), models={},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        admission=ADMISSION, precision='FP16 activation storage at Converse2D boundaries; masters/bias/output/solver remain FP32',
        usrnet_scope='s1; only 35 prior Converse2D calls quantized; five DataNet calls stay unmodified FP32',
        production_promotion=False, model_quality_thresholds=None, convergence_evidence=False)
    try:
        if args.self_check:
            self_check(report)
        else:
            if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
                raise RuntimeError('Unset production backend/CPU overrides')
            if not torch.cuda.is_available():
                raise RuntimeError('Root-owned CUDA execution required')
            from tools.v4_mixed_fusion import loader
            production = loader.load_production_checked()
            extension, research = loader.load_checked(args.artifacts, block_threads=args.block_threads)
            if not hasattr(extension, 'pad_cast'):
                raise RuntimeError('Expected actual extension.pad_cast binding')
            gate_identity = verify_operator_gate(args.operator_gate.resolve(), args.artifacts.resolve(),
                                                args.block_threads, research, production)
            report.update(production=production, research_manifest=research, operator_gate=gate_identity,
                          gpu_evidence=True, torch=str(torch.__version__), cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name())
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(False)
            report['environment'] = dict(tf32=False, autocast=False, deterministic_algorithms=False,
                                         cudnn_deterministic=True, no_implicit_deterministic_fill=True)
            examples = data_tools.image_samples(args)
            report['samples'] = [dict(key=row['key'], **row['provenance']) for row in examples]
            for name in args.models:
                result = dict(status='running', passed=False)
                report['models'][name] = result
                model = None
                try:
                    model, checkpoint = data_tools.make_model(name, torch.device('cuda'))
                    result['checkpoint'] = checkpoint
                    selected = accuracy(model, name, examples, args, extension, result)
                    if result['accuracy']['passed']:
                        inputs, target, provenance = selected
                        timing(model, name, inputs, provenance, args, extension, result)
                        result['passed'] = result['performance']['passed']
                    else:
                        result['performance'] = dict(passed=False, status='blocked_by_model_accuracy')
                    result['status'] = 'complete'
                except Exception as error:
                    result.update(status='error', passed=False, error=repr(error), traceback=traceback.format_exc())
                    print('model retained failure', name, repr(error), flush=True)
                finally:
                    del model
                    torch.ops.converse2d.clear_cache()
            if source_identity() != report['source_sha256']:
                raise RuntimeError('A model study dependency changed during execution')
            if loader.load_production_checked()['checked_manifest'] != production['checked_manifest']:
                raise RuntimeError('Production build changed')
            _, final_research = loader.load_checked(args.artifacts, block_threads=args.block_threads)
            if final_research != research:
                raise RuntimeError('Research artifact identity changed')
            qualified = [name for name, value in report['models'].items() if value.get('passed')]
            all_correct = len(report['models']) == len(args.models) and all(
                value.get('status') == 'complete' and value.get('accuracy', {}).get('passed')
                for value in report['models'].values())
            report.update(status='complete', all_measured_models_correct=all_correct,
                          performance_qualified_models=qualified, passed=bool(all_correct and qualified))
    except Exception as error:
        report.update(status='error', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
