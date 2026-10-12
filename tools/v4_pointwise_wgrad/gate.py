"""FP64-budgeted gate of the split-K pointwise weight VJP.

Candidate: `torch.ops.converse2d._pointwise_wgrad` on the captured activations
and upstream gradients of every pointwise call in the B4 USRNet training
captures (70 C64->128 and 70 C128->64 calls per fixture file). Baseline: the
deterministic cuDNN `F.conv2d` weight gradient, which is the production path
the kernel replaces. Reference: FP64 `F.conv2d`. Synthetic geometries exercise
K tails, batch boundaries inside a K tile, partial output tiles and multiple
tiles through the raw operator. One production-module probe per direction
checks routing and that the module gradient is the operator result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--fixtures', type=Path, nargs='*', default=None)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seeds', type=int, default=2)
args = parser.parse_args()
ROOT = args.root.resolve()
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
os.environ.pop('CONVERSE2D_BACKEND', None)
if args.output.exists():
    raise SystemExit(f'refusing to overwrite {args.output}')
import torch
import torch.nn.functional as F
from extension_loader import load_extension
from numerical_policy import comparison, BUDGETS
from models.pointwise import PointwiseConv2d

load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if args.fixtures is None:
    args.fixtures = [ROOT / 'artifacts/v4_campaign/wgrad_capture_v1.pt',
                     ROOT / 'artifacts/v4_campaign/wgrad_capture_seed29.pt']

SYNTHETIC = (  # (batch, cin, h, w, cout)
    (1, 64, 96, 96, 128), (1, 128, 96, 96, 64), (4, 64, 64, 64, 128), (2, 128, 48, 48, 64),
    (4, 64, 96, 96, 64), (4, 128, 96, 96, 128), (3, 64, 7, 5, 128), (3, 128, 5, 7, 64),
    (2, 3, 10, 10, 5), (2, 5, 10, 10, 3), (1, 200, 17, 17, 130), (5, 130, 9, 9, 200),
    (1, 64, 1, 1, 128), (2, 128, 4, 4, 64),
)


def native_wgrad(x, grad_output, dtype=torch.float32):
    xd = x.to(dtype).requires_grad_(True)
    weight = torch.zeros(grad_output.shape[1], x.shape[1], 1, 1, device=x.device, dtype=dtype, requires_grad=True)
    output = F.conv2d(xd, weight)
    return torch.autograd.grad(output, weight, grad_output.to(dtype))[0]


def gate_case(x, grad_output):
    x = x.cuda().contiguous()
    grad_output = grad_output.cuda().contiguous()
    candidate = torch.ops.converse2d._pointwise_wgrad(grad_output, x)
    baseline = native_wgrad(x, grad_output)
    reference = native_wgrad(x, grad_output, torch.float64)
    repeat = torch.ops.converse2d._pointwise_wgrad(grad_output, x)
    row = comparison(candidate, baseline, reference)
    row['deterministic'] = bool(torch.equal(candidate, repeat))
    row['passed'] = bool(row['passed'] and row['deterministic'])
    return row, candidate


def compact(row):
    return dict(passed=row['passed'], deterministic=row['deterministic'],
                baseline={k: row['baseline'][k] for k in ('rel_l2', 'max_abs')},
                candidate={k: row['candidate'][k] for k in ('rel_l2', 'max_abs')},
                ratio=row['ratio'], limits=row['limits'])


def summarize(rows):
    def stat(key, metric):
        values = [r[key][metric] for r in rows if r[key][metric] is not None]
        return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None
    return dict(cases=len(rows), passed=sum(r['passed'] for r in rows), deterministic=all(r['deterministic'] for r in rows),
                ratio_rel_l2=stat('ratio', 'rel_l2'), ratio_max_abs=stat('ratio', 'max_abs'),
                baseline_rel_l2=stat('baseline', 'rel_l2'), candidate_rel_l2=stat('candidate', 'rel_l2'))


report = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
              budgets=BUDGETS, binary_sha256=json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())['binary_sha256'],
              source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                  'Converse2D/torch_converse2d/peripheral/pointwise_wgrad.cu',
                  'Converse2D/torch_converse2d/peripheral/pointwise_wgrad.cpp', 'models/pointwise.py')},
              captured={}, synthetic={}, production_probe={})

for fixture in args.fixtures:
    payload = torch.load(fixture, map_location='cpu', weights_only=False)
    rows = {}
    for name, capture in payload['captures'].items():
        direction = f"C{capture['x'].shape[1]}->{capture['grad_output'].shape[1]}"
        row, _ = gate_case(capture['x'], capture['grad_output'])
        rows[name] = dict(direction=direction, shape=list(capture['x'].shape), **compact(row))
    directions = sorted({r['direction'] for r in rows.values()})
    report['captured'][fixture.name] = dict(
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(), cases=rows,
        summary={d: summarize([r for r in rows.values() if r['direction'] == d]) for d in directions})
    for d in directions:
        s = report['captured'][fixture.name]['summary'][d]
        print(fixture.name, d, f"{s['passed']}/{s['cases']}", 'ratio rel_l2 max', round(s['ratio_rel_l2']['max'], 4),
              'max_abs max', round(s['ratio_max_abs']['max'], 4), flush=True)

for geometry in SYNTHETIC:
    batch, cin, h, w, cout = geometry
    rows = []
    for seed in range(args.seeds):
        generator = torch.Generator().manual_seed(5101 + seed * 7 + sum(geometry))
        x = torch.randn(batch, cin, h, w, generator=generator)
        grad_output = torch.randn(batch, cout, h, w, generator=generator) / (batch * h * w) ** .5
        row, _ = gate_case(x, grad_output)
        rows.append(compact(row))
    key = 'x'.join(map(str, geometry[:4])) + f'->{cout}'
    report['synthetic'][key] = dict(seeds=rows, summary=summarize(rows))
    s = report['synthetic'][key]['summary']
    print('synthetic', key, f"{s['passed']}/{s['cases']}", 'ratio rel_l2 max', round(s['ratio_rel_l2']['max'], 4), flush=True)

# Production module: routing through PointwiseConv2d gives exactly the operator result.
first = torch.load(args.fixtures[0], map_location='cpu', weights_only=False)['captures']
for name in ('p.m_body.0.conv1.1/call0', 'p.m_body.0.conv1.5/call0'):
    capture = first[name]
    x = capture['x'].cuda().requires_grad_(True)
    layer = PointwiseConv2d(capture['weight'].shape[1], capture['weight'].shape[0], 1, backend='cuda').cuda()
    with torch.no_grad():
        layer.weight.copy_(capture['weight'])
        layer.bias.copy_(capture['bias'])
    output = layer(x)
    grads = torch.autograd.grad(output, (x, layer.weight, layer.bias), capture['grad_output'].cuda())
    expected = torch.ops.converse2d._pointwise_wgrad(capture['grad_output'].cuda().contiguous(), x.detach())
    report['production_probe'][name] = dict(grad_fn=type(output.grad_fn).__name__,
                                            active=type(output.grad_fn).__name__ == 'PointwiseWeightGradientBackward',
                                            dweight_equals_operator=bool(torch.equal(grads[1], expected)))
    print('probe', name, report['production_probe'][name], flush=True)

report['passed'] = (all(s['passed'] == s['cases'] for f in report['captured'].values() for s in f['summary'].values())
                    and all(v['summary']['passed'] == v['summary']['cases'] for v in report['synthetic'].values())
                    and all(p['active'] and p['dweight_equals_operator'] for p in report['production_probe'].values()))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('passed', report['passed'], 'wrote', args.output)
