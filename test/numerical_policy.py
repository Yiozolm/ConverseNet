"""Test-only FP32 error budgets. Never imported by the production operator."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch

BASELINE_COMMIT = '0d636215e23d472825e09cc340a99f1d2ae2b78c'
# Normalize newlines so Git's Windows checkout policy cannot change identity.
BASELINE_FILE = Path(__file__).with_name('fp32_baseline.py')
BASELINE_SHA256 = '8b31f77ae03fafad69f6e8d3f696fe02166aa0d8fe258a2937de3ab619041ccd'
if hashlib.sha256(BASELINE_FILE.read_text(encoding='utf-8').encode()).hexdigest() != BASELINE_SHA256:
    raise RuntimeError('Frozen FP32 baseline changed; do not recalibrate against a candidate')
BUDGETS = {
    'normal': dict(rel_l2_factor=1.25, max_abs_factor=1.50,
                   rel_l2_floor=1e-7, max_abs_floor=1e-6),
    # The document leaves this floor unspecified. Start with the normal floor;
    # do not silently promote weak cases to the optional 2x extreme allowance.
    'weak': dict(rel_l2_factor=1.50, max_abs_factor=1.50,
                 rel_l2_floor=1e-7, max_abs_floor=1e-6),
}


def error_metrics(value, reference, *, distribution=False):
    if value.shape != reference.shape:
        raise ValueError('numerical comparison requires identical shapes')
    if value.is_complex() != reference.is_complex():
        raise ValueError('numerical comparison requires matching real/complex types')
    dtype = torch.complex128 if reference.is_complex() else torch.float64
    value, reference = value.detach().to(dtype), reference.detach().to(dtype)
    if not bool(torch.isfinite(value).all() and torch.isfinite(reference).all()):
        return dict(finite=False, max_abs=None, rel_l2=None)
    delta = (value - reference).abs()
    max_abs = delta.max().item()
    rel_l2 = (delta.norm() / reference.norm().clamp_min(1e-300)).item()
    if not math.isfinite(max_abs) or not math.isfinite(rel_l2):
        return dict(finite=False, max_abs=None, rel_l2=None)
    result = dict(finite=True, max_abs=max_abs, rel_l2=rel_l2)
    if distribution:
        # Scale-aware floor is diagnostic only; never used by the L2 gate.
        tau = max(1e-12, reference.abs().max().item() * 1e-6)
        relative = (delta / reference.abs().clamp_min(tau)).reshape(-1)
        quantiles = torch.quantile(relative, relative.new_tensor([.5, .9, .99, .999, 1.]))
        result['relative_error'] = dict(zip(('p50', 'p90', 'p99', 'p999', 'max'), quantiles.tolist()))
        result['near_zero_floor'] = tau
    return result


def comparison(candidate, baseline, ref64, *, regime='normal', distribution=False):
    """Both hard gates apply to each output/VJP; finite is unconditional."""
    if candidate.dtype not in (torch.float32, torch.complex64):
        raise ValueError('candidate must remain FP32/complex64')
    if baseline.dtype != candidate.dtype:
        raise ValueError('baseline and candidate dtypes must match')
    if ref64.dtype not in (torch.float64, torch.complex128):
        raise ValueError('independent reference must be FP64/complex128')
    policy = BUDGETS[regime]
    c = error_metrics(candidate, ref64, distribution=distribution)
    b = error_metrics(baseline, ref64, distribution=distribution)
    finite = c['finite'] and b['finite']
    limits, ratios = {}, {}
    for metric in ('max_abs', 'rel_l2'):
        limits[metric] = max(b[metric] * policy[metric + '_factor'],
                             policy[metric + '_floor']) if b['finite'] else None
        # A zero baseline and nonzero candidate has no finite ratio. The floor
        # still decides admission; serialize null, never nonstandard JSON Infinity.
        ratios[metric] = ((c[metric] / b[metric]) if b[metric] else
                          (1.0 if c[metric] == 0 else None)) if finite else None
        if ratios[metric] is not None and not math.isfinite(ratios[metric]):
            ratios[metric] = None
    passed = finite and all(c[m] <= limits[m] for m in limits)
    return dict(regime=regime, baseline=b, candidate=c, ratio=ratios,
                limits=limits, passed=bool(passed))


def assert_budget(test, actual, baseline, high, *, regime='normal'):
    test.assertEqual(len(actual), len(baseline))
    test.assertEqual(len(actual), len(high))
    for i, (a, b, r) in enumerate(zip(actual, baseline, high)):
        row = comparison(a, b, r, regime=regime)
        test.assertTrue(row['passed'], f'output/VJP {i}: {row}')
    if os.environ.get('CONVERSE2D_EXACT_REGRESSION') == '1':
        test.assert_results_equal(actual, baseline)


def denominator_statistics(raw, scale, eps, device):
    """Independent full-spectrum FP64 diagnostic, including kernel broadcasts."""
    from models.converse_core import alias_mean
    h, w = raw[0].shape[-2:]
    weight, bias = (v.detach().to(device=device, dtype=torch.float64) for v in raw[-2:])
    kh, kw = weight.shape[-2:]
    psf = torch.nn.functional.pad(weight, (0, w * scale - kw, 0, h * scale - kh))
    fb = torch.fft.fft2(torch.roll(psf, (-(kh // 2), -(kw // 2)), (-2, -1)))
    d = alias_mean(fb.real.square() + fb.imag.square(), scale) + torch.sigmoid(bias - 9) + eps
    d = d.expand(raw[0].shape).flatten()
    if not bool(torch.isfinite(d).all()) or d.min().item() <= 0:
        raise AssertionError('nonfinite/nonpositive FP64 denominator')
    q = torch.quantile(d, d.new_tensor([0., .01, .5, 1.])).tolist()
    return dict(zip(('min_denominator', 'p01_denominator', 'median_denominator', 'max_denominator'), q))


def write_report(name, rows, *, device, extra=None):
    """Retain failed rows and uniquely named runs instead of overwriting evidence."""
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    source_sha = hashlib.sha256(BASELINE_FILE.read_text(encoding='utf-8').encode()).hexdigest()
    manifest = root / '.build' / ('cuda' if str(device).startswith('cuda') else 'cpu') / 'source_manifest.json'
    report = dict(schema_version=1, created_utc=stamp, git_head=git('rev-parse', 'HEAD'),
                  git_dirty=bool(git('status', '--porcelain')), torch=str(torch.__version__),
                  cuda=torch.version.cuda, device=str(device),
                  gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
                  baseline=dict(commit=BASELINE_COMMIT, source_sha256=source_sha,
                                operator_training='frozen Python full-spectrum FP32',
                                operator_inference='frozen Python half-spectrum FP32'),
                  budgets=BUDGETS, rows=rows, passed=bool(rows) and all(r['passed'] for r in rows))
    report['test_source_sha256'] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / 'test').glob('*.py'))
    }
    report['fp64_reference_sha256'] = hashlib.sha256((root / 'models/converse_core.py').read_bytes()).hexdigest()
    report['tf32'] = dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                          cudnn=torch.backends.cudnn.allow_tf32)
    if manifest.exists():
        report['checked_build'] = json.loads(manifest.read_text(encoding='utf-8'))
    if extra:
        report.update(extra)
    if not report.get('complete', True):
        report['passed'] = False
    directory = Path(os.environ.get('CONVERSE2D_REPORT_DIR', root / 'artifacts/fp32_release'))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}-{stamp}-{os.getpid()}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return path
