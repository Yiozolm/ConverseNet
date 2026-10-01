"""Reexecute pinned nearest constructions under v4 budgets, without promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(ROOT / 'tools/nearest_phase_repair'), str(ROOT)]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import gate as legacy_gate
from fp32_baseline import converse2d_reference as frozen_fp32
from numerical_policy import comparison, denominator_statistics

parser = argparse.ArgumentParser()
parser.add_argument('--historical', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError('Preserve earlier results; select a new output')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
historical = json.loads(args.historical.read_text())
legacy, specs, identity, algorithm = legacy_gate.load_legacy(torch, historical)
candidate, candidate_identity = legacy_gate.load_candidate(
    ROOT / 'tools/nearest_phase_repair/candidate.py', algorithm, 'v4_recheck')
report = dict(kind='fresh_cuda_nearest_v4_recheck', old_verdicts_unchanged=True,
              legacy=identity, candidate=candidate_identity,
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
              routes={}, passed=False, performance='not run')
try:
    for name, fn in (('old_nearest_spectral', legacy['nearest_spectral']),
                     ('exact_phase_nearest_spectral', candidate.nearest_spectral)):
        rows = []
        report['routes'][name] = dict(rows=rows, passed=False)
        for index, spec in enumerate(specs):
            raw, upstream = legacy['build_fixture'](index, spec)
            capture = legacy['capture_cuda']
            high = capture(legacy['converse2d_reference'], raw, upstream, spec, torch.float64)
            baseline = capture(frozen_fp32, raw, upstream, spec, torch.float32)
            actual = capture(fn, raw, upstream, spec, torch.float32)
            denominator = denominator_statistics(raw, spec['scale'], spec['eps'], 'cuda')
            for label in high:
                rows.append(dict(index=index, spec=spec, tensor=label, **denominator,
                    **comparison(actual[label], baseline[label], high[label],
                                 regime='weak' if spec['weak'] else 'normal', distribution=True)))
        result = report['routes'][name]
        result.update(passed=all(row['passed'] for row in rows),
                      tensor_count=len(rows), failed_tensors=sum(not row['passed'] for row in rows))
        print(name, result['tensor_count'], result['failed_tensors'], flush=True)
    report.update(status='complete', passed=all(row['passed'] for row in report['routes'].values()))
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
if not report['passed']:
    raise SystemExit(2)
