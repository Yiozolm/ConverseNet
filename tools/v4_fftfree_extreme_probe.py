"""Fresh actual-production overflow probe; nonfinite old baselines are excluded.

Fixtures exactly reproduce the 144-case independent CPU diagnostic. This tool
loads only the checked production extension and never edits build manifests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from fp32_baseline import converse2d_fp32 as frozen_half
from models.converse_core import converse2d_reference as reference64
from numerical_policy import BUDGETS, comparison


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures():
    for magnitude in (1e19, 1e20, 1e30, float(np.finfo(np.float32).max)):
        for pattern, weights in (
            ('all_positive', [magnitude] * 4),
            ('checkerboard', [magnitude, -magnitude, -magnitude, magnitude]),
            ('mixed3negative', [magnitude, -magnitude, -magnitude, -magnitude]),
            ('unequal', [magnitude, magnitude * .5, -magnitude * .25, magnitude * .125]),
        ):
            for x in (1., -1., 1e-20, 1e20):
                yield f'{magnitude:g}/{pattern}/x{x:g}', np.float32(x), np.asarray(weights, dtype=np.float32)
    rng = np.random.default_rng(171029)
    for index in range(80):
        magnitude = np.float32(10. ** rng.uniform(18.7, 37.5))
        weights = rng.uniform(-1, 1, 4).astype(np.float32) * magnitude
        x = np.float32((-1 if index % 2 else 1) * 10. ** rng.uniform(-20, 20))
        yield f'random/{index}', x, weights


def tensor_record(value):
    cpu = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())


def numeric_values(value):
    # JSON retains every nonfinite result explicitly, never as invalid NaN JSON.
    return [float(v) if np.isfinite(float(v)) else str(float(v)) for v in value.detach().cpu().reshape(-1)]


def setup(report):
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend and CPU-only overrides')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    manifest_path = ROOT / '.build/cuda/source_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    production_arch = manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
    old_arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    try:
        if production_arch:
            os.environ['TORCH_CUDA_ARCH_LIST'] = production_arch
        else:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        import extension_loader
        extension_loader.load_extension()
    finally:
        if old_arch is None:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        else:
            os.environ['TORCH_CUDA_ARCH_LIST'] = old_arch
    if sha(manifest_path) != manifest_hash:
        raise RuntimeError('Production loader changed the manifest')
    report['identity'] = dict(helper_sha256=sha(__file__), checked_production_manifest=manifest,
        production_manifest_sha256=manifest_hash, production_sources=extension_loader.production_source_hashes(),
        reference_sources={name: sha(ROOT / name) for name in ('test/numerical_policy.py', 'test/fp32_baseline.py',
                                                               'models/converse_core.py', 'test/extension_loader.py')},
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         tf32=False, amp=False, deterministic_algorithms=True))
    return extension_loader


def probe(name, xvalue, weights, mode):
    x = torch.full((1, 1, 1, 1), float(np.float32(xvalue)), dtype=torch.float32, device='cuda')
    weight = torch.from_numpy(np.asarray(weights, dtype=np.float32).reshape(1, 1, 2, 2)).cuda()
    bias = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device='cuda')
    data = x, weight, bias
    before = [tensor_record(value) for value in data]
    context = {'no_grad': torch.no_grad, 'inference_mode': torch.inference_mode,
               'frozen_gradmode': torch.enable_grad}[mode]
    with context():
        prior = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        baseline = frozen_half(x, prior, weight, bias, 2, 1e-5)
        high = reference64(x.double(), prior.double(), weight.double(), bias.double(), 2, 1e-5)
        actual = torch.ops.converse2d._nearest_k2_s2(x, weight, bias, 1e-5, 'v7')
    baseline_finite = bool(torch.isfinite(baseline).all())
    candidate_finite = bool(torch.isfinite(actual).all())
    high_finite = bool(torch.isfinite(high).all())
    unchanged = before == [tensor_record(value) for value in data]
    accuracy = comparison(actual, baseline, high) if baseline_finite else None
    # A baseline-nonfinite row has no accuracy pass/fail and is never silently
    # included among admitted cases. Its raw candidate/reference remain visible.
    return dict(name=name, mode=mode, inputs=before, x=float(np.float32(xvalue)),
                weights=[float(value) for value in weights], baseline_fp32=numeric_values(baseline),
                reference_fp64=numeric_values(high), candidate_fp32=numeric_values(actual),
                baseline_finite=baseline_finite, candidate_finite=candidate_finite,
                reference_finite=high_finite, inputs_unchanged=unchanged,
                output_requires_grad=actual.requires_grad, accuracy_comparison=accuracy,
                excluded_from_accuracy_gate=not baseline_finite,
                accuracy_claim='Independent FP64 budget against finite old FP32 baseline' if baseline_finite
                               else 'No accuracy claim: old FP32 baseline is nonfinite',
                passed=(bool(accuracy['passed']) and candidate_finite and unchanged and not actual.requires_grad)
                       if baseline_finite else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('all', 'no_grad', 'inference_mode', 'frozen_gradmode'), default='all')
    parser.add_argument('--cpu-evidence', type=Path,
                        default=ROOT / 'artifacts/v4_campaign/fftfree_overflow_cpu_diagnostic.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve evidence: choose a new output path')
    report = dict(kind='actual_production_extreme_fp32_probe', status='running', passed=False,
                  budgets=BUDGETS, bias=0., eps=1e-5, random_seed=171029, cases=[], excluded=[], failures=[])
    try:
        loader = setup(report)
        specs = list(fixtures())
        if len(specs) != 144 or len({name for name, _, _ in specs}) != 144:
            raise RuntimeError('Wrong fixed fixture count')
        if args.cpu_evidence.exists():
            old = json.loads(args.cpu_evidence.read_text())
            expected = {row['name']: (row['x'], row['weights']) for row in old['cases']}
            actual = {name: (float(x), [float(value) for value in weights]) for name, x, weights in specs}
            if actual != expected:
                raise RuntimeError('Fixture values differ from independent CPU evidence')
            report['cpu_provenance'] = dict(path=str(args.cpu_evidence.resolve()), sha256=sha(args.cpu_evidence),
                                             fixture_values_identical=True, gpu_admission_reused=False)
        modes = ('no_grad', 'inference_mode', 'frozen_gradmode') if args.mode == 'all' else (args.mode,)
        report['modes'] = list(modes)
        for mode in modes:
            for name, x, weights in specs:
                row = probe(name, x, weights, mode)
                report['cases'].append(row)
                if row['excluded_from_accuracy_gate']:
                    report['excluded'].append(mode + '/' + name)
                elif not row['passed']:
                    report['failures'].append(mode + '/' + name)
                    print('FAIL', mode, name, flush=True)
            print('Checked mode', mode, 'cases', len(report['cases']), 'failures', len(report['failures']), flush=True)
        identity = report['identity']
        manifest = identity['checked_production_manifest']
        if sha(ROOT / '.build/cuda/source_manifest.json') != identity['production_manifest_sha256']:
            raise RuntimeError('Production manifest changed during probe')
        if sha(ROOT / '.build/cuda' / manifest['library']) != manifest['binary_sha256']:
            raise RuntimeError('Production binary changed during probe')
        if loader.production_source_hashes() != identity['production_sources']:
            raise RuntimeError('Production sources changed during probe')
        if sha(__file__) != identity['helper_sha256']:
            raise RuntimeError('Probe source changed during execution')
        for name, digest in identity['reference_sources'].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError('Probe reference source changed during execution: ' + name)
        report['counts'] = dict(total=len(report['cases']), finite_baseline=sum(row['baseline_finite'] for row in report['cases']),
                                explicitly_excluded=len(report['excluded']), failures=len(report['failures']),
                                candidate_nonfinite_diagnostic=sum(not row['candidate_finite'] for row in report['cases']))
        report['passed'] = report['counts']['finite_baseline'] > 0 and not report['failures']
        report['status'] = 'complete'
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
