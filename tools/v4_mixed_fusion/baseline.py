"""Measure the unchanged FP16-activation -> FP32 -> padded s1 module baseline.

No fusion candidate is implemented or registered here. FP32 weight/bias/output
and all existing solver routing remain unchanged. Stage measurements are
diagnostics, not additive estimates of complete-call latency.

Study: python tools/v4_mixed_fusion/baseline.py --output NEW_BASELINE.json
NCU worker: ... --profile baseline --case b4_c128_96 --padding-mode circular
            --output NEW_NCU_DIRECTORY
CPU harness check: ... --self-check --output NEW_CPU_CHECK.json
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
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
from fp32_baseline import converse2d_fp32 as frozen_quantized_fp32
from models.converse_core import converse2d_reference as independent_fp64
from numerical_policy import comparison, error_metrics, BUDGETS

WORKLOADS = {
    'b4_c128_96': dict(shape=(4, 128, 96, 96), padding=2, kernel=3),
    'b1_c64_96': dict(shape=(1, 64, 96, 96), padding=2, kernel=3),
    'b1_c64_31x37': dict(shape=(1, 64, 31, 37), padding=2, kernel=3),
    'b1_c64_96_k7p6': dict(shape=(1, 64, 96, 96), padding=6, kernel=7),
    # Explicitly optional: k3 cannot fit an unpadded singleton spatial axis.
    'edge_1x17_k1': dict(shape=(1, 64, 1, 17), padding=0, kernel=1),
}
DEFAULT_CASES = ('b4_c128_96', 'b1_c64_96', 'b1_c64_31x37', 'b1_c64_96_k7p6')
PADDING_MODES = ('circular', 'reflect', 'replicate', 'constant')
ADMISSION = {
    'scope': 'Predeclared future candidate requirements, not a claim about this baseline study',
    'complete_call_minimum_median_wall_speedup': 1.03,
    'complete_call_minimum_median_cuda_speedup': 1.03,
    'independent_repeats': 2,
    'minimum_rounds_per_repeat': 9,
    'positive_wall_round_fraction': 'at least ceil(rounds*7/9) in each repeat',
    'large_case_peak_requirement': 'candidate allocator peak increment lower than baseline for B4/C128/96x96/k3/p2',
    'model_requirement': 'At least one measured whole model has exact output agreement with its quantized-input FP32 control and >=1.03x median wall/CUDA speedups under the same repeated paired protocol',
    'fp32_model_speed_claim': 'Requires a separate unchanged-original-FP32 model measurement',
    'fallback_requirement': 'Unsupported layouts and differentiable GradMode retain module(xlow.float()); frozen GradMode keeps its existing ATen route without a forced no_grad context',
}
DEPENDENCIES = (
    'tools/v4_mixed_fusion/baseline.py', 'models/util_converse.py',
    'models/converse_core.py', 'models/pointwise.py', 'test/fp32_baseline.py',
    'test/numerical_policy.py', 'test/extension_loader.py', 'Converse2D/build_config.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT / name) for name in DEPENDENCIES}


def tensor_record(value):
    dense = value.detach().resolve_neg().resolve_conj().cpu().contiguous()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                sha256=hashlib.sha256(dense.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def load_checked():
    path = ROOT / '.build/cuda/source_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    original = {name: os.environ.get(name) for name in ('TORCH_CUDA_ARCH_LIST', 'CONVERSE2D_SKIP_BUILD')}
    try:
        arch = manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
        if arch:
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        else:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        spec = importlib.util.spec_from_file_location('mixed_fusion_baseline_checked_loader', ROOT / 'test/extension_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.load_extension()
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Read-only checked loading changed the manifest')
    return manifest, loader


def pad(value, specification):
    amount = specification['padding']
    return F.pad(value, (amount,) * 4, mode=specification['padding_mode'], value=0) if amount else value


def crop(value, specification):
    amount = specification['padding']
    return value[..., amount:-amount, amount:-amount] if amount else value


def make_fixture(specification, device):
    from models.util_converse import Converse2D
    generator = torch.Generator(device='cpu').manual_seed(specification['seed'])
    shape, kernel = specification['shape'], specification['kernel']
    original = torch.randn(shape, generator=generator)
    weight = torch.softmax(torch.randn(1, shape[1], kernel * kernel, generator=generator), -1)
    weight = weight.reshape(1, shape[1], kernel, kernel)
    bias = torch.randn(1, shape[1], 1, 1, generator=generator) * .2
    module = Converse2D(shape[1], shape[1], kernel, scale=1,
        padding=specification['padding'], padding_mode=specification['padding_mode'],
        eps=specification['eps'], backend='cuda' if str(device).startswith('cuda') else 'auto').to(device).eval()
    with torch.no_grad():
        module.weight.copy_(weight.to(device))
        module.bias.copy_(bias.to(device))
    stored = original.to(device=device, dtype=torch.float16)
    return dict(module=module, stored=stored, original_cpu=original,
                fixtures=dict(original=tensor_record(original), stored=tensor_record(stored),
                              weight=tensor_record(module.weight), bias=tensor_record(module.bias)))


def boundary(fixture):
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise RuntimeError('Baseline requires autocast disabled')
    if fixture['stored'].dtype != torch.float16:
        raise ValueError('The measured boundary uses preexisting FP16 activation storage')
    # Do not change GradMode: the unchanged module decides full training,
    # no_grad CUDA inference, or frozen-input GradMode's existing ATen path.
    return fixture['module'](fixture['stored'].float())


def solver(value, module):
    if value.is_cuda:
        return torch.ops.converse2d.forward(value, value, module.weight, module.bias, 1, float(module.eps), module.variant)
    from models.converse_core import converse2d_fp32
    return converse2d_fp32(value, value, module.weight, module.bias, 1, module.eps)


def staged_boundary(fixture, specification):
    with torch.profiler.record_function('mixed_fusion/upcast'):
        value = fixture['stored'].float()
    with torch.profiler.record_function('mixed_fusion/padding'):
        value = pad(value, specification)
    with torch.profiler.record_function('mixed_fusion/fp32_solver'):
        output = solver(value, fixture['module'])
    with torch.profiler.record_function('mixed_fusion/crop'):
        return crop(output, specification)


def validate_fixture(fixture, specification):
    module, stored = fixture['module'], fixture['stored']
    with torch.no_grad():
        q = stored.float()
        actual = boundary(fixture)
        literal = module(stored.float())
        staged = staged_boundary(fixture, specification)
        exact_literal, exact_staged = torch.equal(actual, literal), torch.equal(actual, staged)
        padded = pad(q, specification)
        reference32 = crop(frozen_quantized_fp32(padded, padded, module.weight, module.bias, 1, module.eps), specification)
        # Quantize first, then promote the identical physical FP32 inputs.
        high_input = pad(q.double(), specification)
        reference64 = crop(independent_fp64(high_input, high_input, module.weight.double(), module.bias.double(), 1, module.eps), specification)
        budget = comparison(actual, reference32, reference64)
        original_output = module(fixture['original_cpu'].to(stored.device))
        row = dict(exact_current_module=bool(exact_literal), exact_staged_pipeline=bool(exact_staged),
                   quantized_fp32_budget=budget,
                   input_quantization_vs_original_fp32_output=error_metrics(actual, original_output.double()),
                   output=tensor_record(actual), reference32=tensor_record(reference32),
                   output_dtype=str(actual.dtype), expected_output_shape=list(stored.shape),
                   shared_s1_prior=True, spectrum='half (inference)',
                   passed=bool(exact_literal and exact_staged and budget['passed'] and actual.dtype == torch.float32))
    if not row['passed']:
        raise ValueError('Baseline fixture failed its unchanged-reference gate: ' + json.dumps(row))
    return row


def cpu_event_graph(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as trace:
        output = call()
        if output.is_cuda:
            torch.cuda.synchronize(output.device)
    del output
    counts, phases = Counter(), defaultdict(Counter)
    for event in trace.events():
        if event.device_type != torch.autograd.DeviceType.CPU or not event.name.startswith(('aten::', 'converse2d::')):
            continue
        counts[event.name] += 1
        parent = event.cpu_parent
        while parent is not None:
            if parent.name.startswith('mixed_fusion/'):
                phases[parent.name][event.name] += 1
                break
            parent = parent.cpu_parent
    return dict(op_counts=dict(counts), staged_op_counts={key: dict(value) for key, value in phases.items()},
                scope='CPU dispatcher event graph only; profiler durations are not benchmark samples')


def measure(call, iterations):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        output = call()
        del output
    end.record()
    end.synchronize()
    return dict(wall_ms=(time.perf_counter() - wall) * 1000 / iterations,
                cuda_event_ms=begin.elapsed_time(end) / iterations,
                initial_allocated_bytes=before,
                peak_increment_bytes=torch.cuda.max_memory_allocated() - before,
                calls=iterations)


def benchmark(fixture, specification, args):
    module = fixture['module']
    with torch.no_grad():
        # Resident control has the SAME quantized values, isolating upcast cost.
        resident = fixture['stored'].float()
        functions = {'resident_quantized_fp32': lambda: module(resident),
                     'fp16_boundary': lambda: boundary(fixture)}
        torch.ops.converse2d.clear_cache()
        for call in functions.values():
            for _ in range(args.warmup):
                output = call()
                del output
        runs = []
        for _ in range(args.repeats):
            pairs = []
            for index in range(args.rounds):
                order = list(functions) if index % 2 == 0 else list(reversed(functions))
                pair = dict(round=index, order=order)
                for name in order:
                    pair[name] = measure(functions[name], args.iters)
                pairs.append(pair)
            runs.append(dict(pairs=pairs, median={name: {metric: statistics.median(row[name][metric] for row in pairs)
                for metric in ('wall_ms', 'cuda_event_ms', 'peak_increment_bytes')} for name in functions},
                resident_to_boundary_speedup={metric: statistics.median(row['resident_quantized_fp32'][metric] /
                    row['fp16_boundary'][metric] for row in pairs) for metric in ('wall_ms', 'cuda_event_ms')}))
        actual_graph = cpu_event_graph(functions['fp16_boundary'])
        staged_graph = cpu_event_graph(lambda: staged_boundary(fixture, specification))
        if actual_graph['op_counts'].get('converse2d::forward') != 1:
            raise RuntimeError('Unexpected public operator count in baseline module')
        if actual_graph['op_counts'].get('aten::fft_fft2', 0) or not actual_graph['op_counts'].get('aten::fft_rfft2', 0):
            raise RuntimeError('Expected unchanged half-spectrum inference')
        if any('_training_' in name for name in actual_graph['op_counts']):
            raise RuntimeError('Baseline accidentally selected training')
        padded = pad(resident, specification)
        stage_calls = {'upcast_only': lambda: fixture['stored'].float(),
                       'padding_only_resident_fp32': lambda: pad(resident, specification),
                       'upcast_plus_padding': lambda: pad(fixture['stored'].float(), specification),
                       'solver_only_resident_padded_fp32': lambda: solver(padded, module)}
        stages = {}
        for name, call in stage_calls.items():
            for _ in range(args.warmup):
                output = call()
                del output
            stages[name] = [measure(call, args.iters) for _ in range(3)]
    return dict(complete_calls=runs, stage_probes=stages, current_module_cpu_graph=actual_graph,
                staged_cpu_graph=staged_graph,
                stage_scope='Isolated pre-resident stages have different allocation lifetimes and are not additive. Only complete_calls ranks complete latency.',
                cache_scope='Warm inference with unchanged FP32 weight identity; no training spectrum reuse')


def profile_worker(fixture, specification, report):
    torch.ops.converse2d.clear_cache()
    with torch.no_grad():
        for _ in range(5):
            output = boundary(fixture)
            if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
                raise RuntimeError('Profile warmup produced invalid output')
            del output
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            output = boundary(fixture)
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
    report.update(profile='baseline', warmup_calls=5, captured_calls=1, output=tensor_record(output),
                  profiler_only=True, unprofiled_timing=False,
                  profile_scope='One unchanged module call including FP16->FP32 cast, existing padding/solver/crop; every warmup output released')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('baseline',))
    parser.add_argument('--case', choices=tuple(WORKLOADS), nargs='+')
    parser.add_argument('--padding-mode', choices=PADDING_MODES, nargs='+')
    parser.add_argument('--seed', type=int, default=41001)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.profile and args.self_check:
        parser.error('Profile and CPU self-check are separate runs')
    names = args.case or ([DEFAULT_CASES[0]] if args.profile else list(DEFAULT_CASES))
    modes = args.padding_mode or ([PADDING_MODES[0]] if args.profile else list(PADDING_MODES))
    if args.profile and (len(names) != 1 or len(modes) != 1):
        parser.error('An NCU application worker must select one case and one padding mode')
    if args.warmup < 5 or args.rounds < 9 or args.iters < 10 or args.repeats < 2:
        parser.error('Require >=5 warmups, >=9 rounds, >=10 calls and >=2 repeats')
    args.output = args.output.resolve()
    if not args.profile and args.output.exists():
        parser.error('Preserve previous reports; select a fresh output path')
    destination = (args.output / f'worker-{os.getpid()}-{time.time_ns()}.json') if args.profile else args.output
    report = dict(kind='mixed_fusion_unchanged_baseline', status='running', passed=False,
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources(), rows=[],
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        settings=dict(seed=args.seed, cases=names, padding_modes=modes, warmup=args.warmup,
                      rounds=args.rounds, calls=args.iters, repeats=args.repeats, profile=args.profile),
        candidate_generated=False, production_changes=False, optimization_admission=False,
        future_candidate_admission=ADMISSION,
        precision='FP16 activation-only storage -> FP32; FP32 master weight/bias/output, complex64 spectra',
        reference='Frozen FP32 and independent FP64 solve on the same FP16-quantized input, with original padding/crop',
        numerical_budgets=BUDGETS, self_check=args.self_check, gpu_evidence=not args.self_check)
    try:
        if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
            raise RuntimeError('Unset backend/CPU-only overrides')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Match the complete-call and NCU protocols without implicit FillFunctor.
        torch.use_deterministic_algorithms(False)
        report['environment'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            deterministic_algorithms=False, cudnn_deterministic=True, cudnn_benchmark=False,
            tf32=False, autocast=False, cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        device = 'cpu' if args.self_check else 'cuda'
        loader = None
        if not args.self_check:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; only the root runs GPU experiments')
            manifest, loader = load_checked()
            report.update(checked_manifest=manifest, production_sources=loader.production_source_hashes())
            report['environment']['gpu'] = torch.cuda.get_device_name()
        else:
            torch.set_num_threads(1)
        for name in names:
            for mode in modes:
                specification = dict(WORKLOADS[name], name=name, padding_mode=mode,
                                     seed=args.seed + list(WORKLOADS).index(name), eps=1e-5, scale=1)
                if args.self_check:
                    specification.update(shape=(1, 3, 5, 7), padding=2, kernel=3,
                                         self_check_scope='Small CPU harness exercise, not the requested CUDA workload')
                fixture = make_fixture(specification, device)
                row = dict(specification=specification, fixtures=fixture['fixtures'])
                report['rows'].append(row)
                row['accuracy'] = validate_fixture(fixture, specification)
                if args.profile:
                    profile_worker(fixture, specification, row)
                elif args.self_check:
                    with torch.no_grad():
                        row['staged_cpu_graph'] = cpu_event_graph(lambda: staged_boundary(fixture, specification))
                else:
                    row['timing'] = benchmark(fixture, specification, args)
                if fixture['fixtures']['stored'] != tensor_record(fixture['stored']) or \
                        fixture['fixtures']['weight'] != tensor_record(fixture['module'].weight) or \
                        fixture['fixtures']['bias'] != tensor_record(fixture['module'].bias):
                    raise RuntimeError('Fixture or master parameters changed')
                print(name, mode, 'accuracy', row['accuracy']['passed'], flush=True)
        if sources() != report['source_sha256']:
            raise RuntimeError('A measured source changed during baseline execution')
        if loader is not None:
            if loader.production_source_hashes() != report['production_sources'] or \
                    json.loads((ROOT / '.build/cuda/source_manifest.json').read_text()) != manifest or \
                    sha(ROOT / '.build/cuda' / manifest['library']) != manifest['binary_sha256']:
                raise RuntimeError('Checked source/build identity changed')
        elif torch.cuda.is_initialized():
            raise RuntimeError('CPU self-check initialized CUDA')
        report.update(status='complete', passed=all(row['accuracy']['passed'] for row in report['rows']))
    except Exception as error:
        report.update(status='error', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Report:', destination, flush=True)


if __name__ == '__main__':
    main()
