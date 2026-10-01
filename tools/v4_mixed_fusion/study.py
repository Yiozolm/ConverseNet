"""Post-gate complete-call timing and NCU worker for pad/cast fusion."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from tools.v4_mixed_fusion import baseline, loader
from tools.v4_mixed_fusion.adapter import mixed_module_forward


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_gate(args):
    gate = json.loads(args.gate.read_text(encoding='utf-8'))
    expected_scope = dict(input_dtype='float16', padding_mode='circular', scale=1, min_eps=1e-5,
                          weight_dtype='float32', bias_dtype='float32', output_dtype='float32',
                          layout='contiguous_NCHW', execution='inference_or_frozen_gradmode')
    if (gate.get('status') != 'complete' or not gate.get('complete') or not gate.get('active_domain_admitted')
            or gate.get('admission_scope') != expected_scope or gate.get('block_threads') != args.block_threads):
        raise ValueError('The explicitly restricted normal-regularization domain must pass for this exact build')
    if len(gate.get('rows', [])) != gate.get('expected_cases'):
        raise ValueError('Numerical matrix is incomplete')
    active = [row for row in gate['rows'] if row.get('expected_fused')]
    if not active or not all(row.get('passed') and row.get('route_verified') for row in active):
        raise ValueError('An active numerical case failed')
    if gate.get('verified_fused_module_cases') != len(active):
        raise ValueError('Actual fused route coverage differs from claimed admission')
    for path, digest in gate['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Numerical dependency changed: ' + path)
    production = loader.load_production_checked()
    extension, manifest = loader.load_checked(args.artifacts, block_threads=args.block_threads)
    if manifest != gate['research_manifest'] or production['checked_manifest'] != gate['checked_manifest']:
        raise ValueError('Measured extension differs from the numerically gated build')
    return gate, production, extension, manifest


def measure(call, cold):
    if not cold:
        return baseline.measure(call, 10)
    samples = []
    for _ in range(10):
        torch.ops.converse2d.clear_cache()
        samples.append(baseline.measure(call, 1))
    return dict(wall_ms=statistics.mean(r['wall_ms'] for r in samples),
                cuda_event_ms=statistics.mean(r['cuda_event_ms'] for r in samples),
                peak_increment_bytes=max(r['peak_increment_bytes'] for r in samples),
                calls=10, samples=samples)


@contextmanager
def module_route(module, extension, route):
    """Both operator timing routes retain identical nn.Module.__call__ work.

    Install/restore outside timing. The baseline keeps its ORIGINAL public call
    module(xlow.float()), including the upcast's lifetime in __call__ arguments.
    Candidate hooks see FP16; baseline hooks see the original FP32 boundary.
    Standalone fixtures have no hooks. Whole-model frontends are tested apart.
    """
    original = module.forward
    if route == 'baseline':
        yield
        return
    had = 'forward' in vars(module)
    previous = vars(module).get('forward')
    def forward(this, value):
        return mixed_module_forward(this, value, extension, original_forward=original)
    module.forward = types.MethodType(forward, module)
    try:
        yield
    finally:
        if had:
            module.forward = previous
        else:
            del module.forward


def exact_fixture(fixture, extension):
    original = baseline.boundary(fixture)
    with module_route(fixture['module'], extension, 'candidate'):
        candidate = fixture['module'](fixture['stored'])
    exact = torch.equal(original.contiguous().view(torch.uint8), candidate.contiguous().view(torch.uint8))
    metadata = dict(shape=list(candidate.shape), stride=list(candidate.stride()),
                    offset=candidate.storage_offset(), dtype=str(candidate.dtype))
    expected_meta = dict(shape=list(original.shape), stride=list(original.stride()),
                         offset=original.storage_offset(), dtype=str(original.dtype))
    if not exact or metadata != expected_meta or not bool(torch.isfinite(candidate).all()):
        raise RuntimeError('Actual complete-call fixture failed exact numerical/layout preflight')
    return dict(exact=True, output=baseline.tensor_record(candidate), metadata=metadata)


def benchmark(fixture, extension, specification):
    routes = ('baseline', 'candidate')
    functions = {'baseline': lambda: fixture['module'](fixture['stored'].float()),
                 'candidate': lambda: fixture['module'](fixture['stored'])}
    rows = []
    with torch.no_grad():
        accuracy = exact_fixture(fixture, extension)
        for cold in (False, True):
            repeats = []
            for repeat in range(2):
                torch.ops.converse2d.clear_cache()
                for route in routes:
                    with module_route(fixture['module'], extension, route):
                        for _ in range(5):
                            output = functions[route]()
                            del output
                pairs = []
                for index in range(9):
                    order = ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline']
                    pair = dict(round=index, order=order)
                    for name in order:
                        with module_route(fixture['module'], extension, name):
                            pair[name] = measure(functions[name], cold)
                    pairs.append(pair)
                ratio = {metric: statistics.median(p['baseline'][metric] / p['candidate'][metric] for p in pairs)
                         for metric in ('wall_ms', 'cuda_event_ms')}
                positive = sum(p['candidate']['wall_ms'] < p['baseline']['wall_ms'] for p in pairs)
                peak = {name: statistics.median(p[name]['peak_increment_bytes'] for p in pairs) for name in routes}
                timing_pass = all(value >= 1.03 for value in ratio.values()) and positive >= 7
                large_peak_pass = specification['name'] != 'b4_c128_96' or peak['candidate'] < peak['baseline']
                repeats.append(dict(repeat=repeat, pairs=pairs, speedup=ratio, positive_wall_pairs=positive,
                                    median_peak_increment_bytes=peak, timing_passed=timing_pass,
                                    large_peak_passed=large_peak_pass, passed=timing_pass and large_peak_pass))
            rows.append(dict(specification=specification, cache='cold' if cold else 'warm',
                             accuracy=accuracy, repeats=repeats, passed=all(r['passed'] for r in repeats)))
        # Component comparison is diagnostic only. Both sides include their
        # real conversion; a preconverted FP32 input is never substituted.
        amount = specification['padding']
        prep = {'baseline': lambda: torch.nn.functional.pad(fixture['stored'].float(), (amount,) * 4, mode='circular'),
                'candidate': lambda: extension.pad_cast(fixture['stored'], amount, 'circular')}
        component = {}
        for name, call in prep.items():
            for _ in range(5):
                output = call()
                del output
            component[name] = [baseline.measure(call, 10) for _ in range(3)]
    return rows, component


def profile(fixture, extension, route):
    with torch.no_grad():
        accuracy = exact_fixture(fixture, extension)
        torch.ops.converse2d.clear_cache()
        with module_route(fixture['module'], extension, route):
            call = ((lambda: fixture['module'](fixture['stored'].float())) if route == 'baseline'
                    else (lambda: fixture['module'](fixture['stored'])))
            for _ in range(5):
                output = call()
                del output
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            try:
                output = call()
                torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
    return dict(route=route, warmup_calls=5, captured_calls=1, accuracy=accuracy,
                output=baseline.tensor_record(output), profiler_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--block-threads', type=int, choices=(128, 256, 512), default=256)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('baseline', 'candidate'))
    parser.add_argument('--case', choices=baseline.DEFAULT_CASES, nargs='+')
    args = parser.parse_args()
    if not args.profile and args.output.exists():
        raise FileExistsError('Keep prior measurements; select a fresh output path')
    names = args.case or ([baseline.DEFAULT_CASES[0]] if args.profile else baseline.DEFAULT_CASES)
    if args.profile and len(names) != 1:
        raise ValueError('Capture one complete-call fixture')
    destination = args.output / f'worker-{os.getpid()}-{time.time_ns()}.json' if args.profile else args.output
    sources = dict(baseline.sources())
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in
                    (Path(__file__), HERE / 'adapter.py', HERE / 'loader.py')})
    report = dict(kind='mixed_padding_fusion_complete_call', status='running', passed=False, rows=[],
                  source_sha256=sources, block_threads=args.block_threads,
                  numerical_gate=dict(path=str(args.gate), sha256=sha(args.gate)),
                  admission=baseline.ADMISSION, component_probes={},
                  frontend_protocol='Unmodified public baseline module(xlow.float()) versus candidate module(xlow), both real nn.Module.__call__. Only candidate forward is wrapped, outside timing. Baseline retains its original FP32 argument lifetime; standalone fixtures have no hooks.',
                  scope='FP16 activation, FP32 weights/bias/output, s1 positive circular padding; public FP32 path unchanged')
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(False)
        gate, production, extension, manifest = verify_gate(args)
        report.update(research_manifest=manifest, checked_manifest=production['checked_manifest'],
                      admission_scope=gate['admission_scope'], full_numerical_matrix_passed=gate['passed'],
                      active_numerical_domain_admitted=gate['active_domain_admitted'],
                      environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                       gpu=torch.cuda.get_device_name(), deterministic_algorithms=False,
                                       tf32=False, autocast=False))
        for name in names:
            specification = dict(baseline.WORKLOADS[name], name=name, padding_mode='circular',
                                 seed=41001 + list(baseline.WORKLOADS).index(name), eps=1e-5, scale=1)
            fixture = baseline.make_fixture(specification, 'cuda')
            if args.profile:
                report['profile'] = profile(fixture, extension, args.profile)
            else:
                rows, component = benchmark(fixture, extension, specification)
                for row in rows:
                    row['fixtures'] = fixture['fixtures']
                report['rows'].extend(rows)
                report['component_probes'][name] = component
                print(name, [(r['cache'], r['passed'], [v['speedup'] for v in r['repeats']]) for r in rows], flush=True)
            torch.ops.converse2d.clear_cache()
        if any(sha(ROOT / name) != digest for name, digest in sources.items()):
            raise RuntimeError('Measured Python source changed')
        if loader.build_identity(block_threads=args.block_threads) != manifest['identity'] or sha(manifest['binary']) != manifest['binary_sha256']:
            raise RuntimeError('Measured research build changed')
        if loader.load_production_checked()['checked_manifest'] != report['checked_manifest']:
            raise RuntimeError('Measured production build changed')
        report.update(status='complete', passed=True if args.profile else all(r['passed'] for r in report['rows']),
                      production_promotion=False, model_admission_required=True)
    except Exception as exc:
        report.update(status='error', passed=False, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
