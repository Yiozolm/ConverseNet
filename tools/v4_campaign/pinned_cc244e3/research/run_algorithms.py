"""Zero-margin per-tensor admission for isolated FP32 algorithm prototypes."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from models.converse_core import converse2d_reference
import algorithms
import training_candidates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(value, high):
    value, high = value.detach().double().cpu(), high.detach().double().cpu()
    delta = value-high
    return {'max_abs': delta.abs().max().item(),
            'relative_l2': (delta.norm()/high.norm().clamp_min(1e-300)).item(),
            'finite': bool(torch.isfinite(value).all())}


def capture(fn, raw, upstream, spec, dtype, *, scope=None):
    x, raw_p, k, b = [v.to(device='cuda', dtype=dtype).detach().requires_grad_() for v in raw]
    if spec['prior'] == 'shared':
        p = x
    elif spec['prior'] == 'nearest':
        p = torch.nn.functional.interpolate(x, scale_factor=spec['scale'], mode='nearest')
    else:
        p = raw_p
    requested = [x] + ([p] if spec['prior'] == 'independent' else []) + [k, b]
    labels = ['dx'] + (['dprior'] if spec['prior'] == 'independent' else []) + ['dweight', 'dbias']
    with scope() if scope else contextlib.nullcontext():
        outputs = [fn(x*(1+i*.03), p if i else p, k, b, spec['scale'], spec['eps'])
                   for i in range(spec.get('calls', 1))] if spec.get('calls', 1) > 1 else [fn(x, p, k, b, spec['scale'], spec['eps'])]
    gradients = torch.autograd.grad(outputs, requested, [upstream.to(device='cuda', dtype=dtype)]*len(outputs))
    return {**{f'output{i}': out for i, out in enumerate(outputs)}, **dict(zip(labels, gradients))}


def specifications(name):
    scales = (2,) if name == 'disjoint_k2_s2' else (1,) if name == 'transfer_shared' else (2, 3, 4) if name == 'nearest_spectral' else (1, 2, 3)
    priors = ('shared',) if name == 'transfer_shared' else ('nearest',) if name == 'nearest_spectral' else ('independent', 'nearest')
    cases = []
    for scale in scales:
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            for weak in (False, True):
                for prior in priors:
                    cases.append(dict(scale=scale, kb=kb, kc=kc, weak=weak, prior=prior,
                                      kernel=2 if name == 'disjoint_k2_s2' else 3,
                                      eps=1e-8 if weak else 1e-5,
                                      calls=2 if name == 'within_forward_spectrum_reuse' else 1))
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', nargs='*')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    registry = dict(disjoint_k2_s2=algorithms.disjoint_k2_s2, transfer_shared=algorithms.transfer_shared,
                    nearest_spectral=algorithms.nearest_spectral, direct_dft=algorithms.direct_dft,
                    **training_candidates.REGISTRY)
    result = {'scope': 'Pure FP32 research admission, not production or quality approval',
              'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
              'source_sha256': {p.name:sha(p) for p in (Path(__file__), Path(algorithms.__file__), Path(training_candidates.__file__))},
              'candidates': {}}
    for name in args.candidates or registry:
        cases = []
        for index, spec in enumerate(specifications(name)):
            generator = torch.Generator().manual_seed(41191+index)
            s, support = spec['scale'], spec['kernel']
            raw = [torch.randn(2, 3, 5, 7, generator=generator), torch.randn(2, 3, 5*s, 7*s, generator=generator),
                   torch.rand(spec['kb'], spec['kc'], support, support, generator=generator)/(support*support),
                   torch.randn(1, 3, 1, 1, generator=generator)]
            up = torch.randn(raw[1].shape, generator=generator)/raw[1].numel()**.5
            if spec['weak']:
                raw[0].mul_(1e-5); raw[1].mul_(1e-5); raw[2].mul_(1e-6); raw[3].fill_(-40); up.mul_(1e-5)
            high = capture(converse2d_reference, raw, up, spec, torch.float64)
            control = capture(converse2d_reference, raw, up, spec, torch.float32)
            scope = training_candidates.CANDIDATE_SCOPES.get(name)
            actual = capture(registry[name], raw, up, spec, torch.float32, scope=scope)
            rows = {}
            for label in actual:
                a, p = record(actual[label], high[label]), record(control[label], high[label])
                rows[label] = {'candidate': a, 'python_fp32': p,
                               'passed': a['finite'] and a['max_abs'] <= p['max_abs'] and a['relative_l2'] <= p['relative_l2']}
            cases.append({'spec': spec, 'tensors': rows})
            del high, control, actual
        all_rows = [v for case in cases for v in case['tensors'].values()]
        failures = sum(not row['passed'] for row in all_rows)
        result['candidates'][name] = {'cases': cases, 'tensor_count': len(all_rows), 'failures': failures,
                                      'status': 'candidate_failed_numeric_gate' if failures else 'numeric_gate_passed_quality_pending'}
        print(name, len(all_rows), 'tensors,', failures, 'failures', flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
