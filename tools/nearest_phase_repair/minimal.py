"""First admission step: the saved k3/s2 identity/impulse counterexample."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import candidate
import route_b_nearest as frozen

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single_insertion_check():
    old = ast.parse(Path(frozen.__file__).read_text(encoding='utf-8'))
    new = ast.parse(Path(candidate.__file__).read_text(encoding='utf-8'))
    old_fn = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'nearest_spectral')
    new_fn = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'nearest_spectral')
    class StripRepair(ast.NodeTransformer):
        removed = 0
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'z'
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'apply_exact_coefficients'):
                self.removed += 1
                return None
            return node
    strip = StripRepair()
    stripped = strip.visit(new_fn)
    if strip.removed != 1 or ast.dump(old_fn, include_attributes=False) != ast.dump(stripped, include_attributes=False):
        raise RuntimeError('Candidate differs from frozen function beyond the single phase-repair insertion')
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu', 'cuda'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output name')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    packet = ROOT/'docs/training_followup_route_b_inputs.json'
    raw = frozen.decode_inputs(json.loads(packet.read_text(encoding='utf-8')))['minimal_identity_impulse']
    legacy, _ = frozen.verify_legacy(ROOT/'.build/roadmap-research')
    paths = [Path(__file__), Path(candidate.__file__), Path(frozen.__file__), packet]
    report = dict(kind='nearest_phase_repair_minimal_gate', status='running',
                  one_phase_insertion_only=single_insertion_check(), legacy_function_identity=legacy,
                  source_sha256={str(p):sha(p) for p in paths}, torch=str(torch.__version__),
                  policy='FP32/complex64 candidate; unchanged full-spectrum FFT/solver/cache. FP64 only independent reference.',
                  cases={})
    try:
        for device in args.devices:
            candidate._geometry.clear()
            frozen._geometry.clear()
            with torch.no_grad():
                x, k, b = [v.to(device) for v in raw]
                p = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
                high = frozen.converse2d_reference(x.double(), p.double(), k.double(), b.double(), 2, 1e-5)
                control = frozen.converse2d_reference(x, p, k, b, 2, 1e-5)
                old = frozen.nearest_spectral(x, p, k, b, 2, 1e-5)
                repaired = candidate.nearest_spectral(x, p, k, b, 2, 1e-5)
                cached = candidate.nearest_spectral(x, p, k, b, 2, 1e-5)
                phases = candidate._geometry[(4, 4, 2, x.device)]
                exact = torch.tensor([2+0j, 1-1j, 0j, 1+1j], device=device, dtype=torch.complex64)
                row = dict(old_gate=frozen.gate(old, control, high), repaired_gate=frozen.gate(repaired, control, high),
                           output_byte_equal=frozen.raw_bytes(repaired) == frozen.raw_bytes(control),
                           cached_call_byte_equal=frozen.raw_bytes(cached) == frozen.raw_bytes(repaired),
                           exact_axis_values=[bool(torch.equal(axis, exact)) for axis in phases],
                           old=frozen.snapshot(old, True), repaired=frozen.snapshot(repaired, True),
                           python_fp32=frozen.snapshot(control, True), reference_fp64=frozen.snapshot(high, True),
                           axes=[frozen.snapshot(axis, True) for axis in phases], input={name:frozen.snapshot(v,True) for name,v in zip(('x','k','b'),raw)})
                row['passed'] = (not row['old_gate']['passed'] and row['repaired_gate']['passed']
                                 and row['output_byte_equal'] and row['cached_call_byte_equal'] and all(row['exact_axis_values']))
                report['cases'][device] = row
                print(device, 'old_failed', not row['old_gate']['passed'], 'repaired_passed', row['passed'], flush=True)
        report['status'] = 'minimal_gate_passed' if all(r['passed'] for r in report['cases'].values()) else 'minimal_gate_failed'
        report['full_matrix_authorized'] = report['status'] == 'minimal_gate_passed' and 'cuda' in report['cases']
        report['production_admitted'] = False
    finally:
        report['sources_unchanged'] = report['source_sha256'] == {str(p):sha(p) for p in paths}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if report['status'] != 'minimal_gate_passed' or not report['sources_unchanged']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
