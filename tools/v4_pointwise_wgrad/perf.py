"""Paired timing of the split-K pointwise weight VJP.

Operator rows time, per geometry: the deterministic cuDNN weight-only
`convolution_backward` (the production kernel being replaced), the new
`_pointwise_wgrad`, and for the C128->64 direction the previous
permute/reshape + cuBLAS GEMM formulation. Layer rows time the complete
PointwiseConv2d forward + (dx, dweight, dbias) VJP with backend `pytorch`
(native convolution) against `auto`. With --model, the complete pretrained
USRNet B4 LR32->HR96 s3 training step (zero_grad, forward, MSE, backward,
Adam) switches only the PointwiseConv2d modules between routes; every other
Converse operator stays on the same CUDA binary. Rounds alternate routes.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=6)
parser.add_argument('--iters', type=int, default=30)
parser.add_argument('--model', action='store_true')
parser.add_argument('--model-rounds', type=int, default=5)
parser.add_argument('--model-iters', type=int, default=5)
parser.add_argument('--idle-threshold', type=int, default=8, help='max foreign GPU utilization percent before timing')
parser.add_argument('--idle-samples', type=int, default=3)
args = parser.parse_args()
ROOT = args.root.resolve()
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
os.environ.pop('CONVERSE2D_BACKEND', None)
if args.output.exists():
    raise SystemExit(f'refusing to overwrite {args.output}')
import torch
from torch.profiler import ProfilerActivity, profile
from extension_loader import load_extension
from models.pointwise import PointwiseConv2d

load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = False
ROUTES = ('pytorch', 'auto')
GEOMETRIES = ((4, 64, 96, 96, 128), (4, 128, 96, 96, 64), (1, 64, 96, 96, 128), (1, 128, 96, 96, 64))


def wait_idle():
    idle = 0
    samples = []
    while idle < args.idle_samples:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True).stdout.strip()
        value = int(out.splitlines()[0])
        samples.append(value)
        idle = idle + 1 if value <= args.idle_threshold else 0
        if idle < args.idle_samples:
            time.sleep(5)
    return samples


def timed(call, iters):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(iters):
        start.record()
        call()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) * 1000)
    return values


def kernel_sum_us(call, iters):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
    return sum(e.device_time for e in prof.events() if e.device_type.name == 'CUDA') / iters


def paired(calls):
    result = {name: dict(call_us=[], kernel_us=[]) for name in calls}
    for name, call in calls.items():
        for _ in range(5):
            call()
    for _ in range(args.rounds):
        for name, call in calls.items():
            result[name]['call_us'].append(statistics.median(timed(call, args.iters)))
            result[name]['kernel_us'].append(kernel_sum_us(call, args.iters))
    rows = {name: {k: statistics.median(v) for k, v in values.items()} for name, values in result.items()}
    rows['rounds'] = result
    return rows


def operator_case(geometry):
    batch, cin, h, w, cout = geometry
    torch.manual_seed(41201 + cin + cout)
    x = torch.randn(batch, cin, h, w, device='cuda')
    grad_output = torch.randn(batch, cout, h, w, device='cuda')
    weight = torch.randn(cout, cin, 1, 1, device='cuda') / cin ** .5
    calls = {
        'cudnn_wgrad': lambda: torch.ops.aten.convolution_backward.default(
            grad_output, x, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])[1],
        'split_k_wgrad': lambda: torch.ops.converse2d._pointwise_wgrad(grad_output, x),
    }
    if cout < cin:
        calls['gemm_wgrad'] = lambda: (grad_output.permute(1, 0, 2, 3).reshape(cout, -1)
                                       @ x.permute(1, 0, 2, 3).reshape(cin, -1).t()).reshape(cout, cin, 1, 1)
    rows = paired(calls)
    rows['speedup_vs_cudnn'] = {k: rows['cudnn_wgrad'][k] / rows['split_k_wgrad'][k] for k in ('call_us', 'kernel_us')}
    return rows


def layer_case(geometry):
    batch, cin, h, w, cout = geometry
    torch.manual_seed(41301 + cin + cout)
    x = torch.randn(batch, cin, h, w, device='cuda', requires_grad=True)
    layer = PointwiseConv2d(cin, cout, 1).cuda()
    upstream = torch.randn(batch, cout, h, w, device='cuda')

    def make(route):
        def run():
            layer.backend = route
            return torch.autograd.grad(layer(x), (x, layer.weight, layer.bias), upstream)
        return run

    rows = paired({route: make(route) for route in ROUTES})
    rows['speedup'] = {k: rows['pytorch'][k] / rows['auto'][k] for k in ('call_us', 'kernel_us')}
    return rows


def model_case():
    from models.converse_usrnet import ConverseUSRNet
    torch.manual_seed(72001)
    checkpoint = ROOT / 'model_zoo/converse_usrnet.pth'
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ConverseUSRNet(backend='cuda').cuda()
    model.load_state_dict(initial, strict=True)
    model.train()
    layers = [m for m in model.modules() if isinstance(m, PointwiseConv2d)]
    x = torch.rand(4, 3, 32, 32, device='cuda')
    k = torch.rand(4, 1, 7, 7, device='cuda')
    k /= k.sum((-2, -1), keepdim=True)
    target = torch.rand(4, 3, 96, 96, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, k, 3) - target).square().mean()
        loss.backward()
        optimizer.step()
        return loss

    def restore(route):
        for m in layers:
            m.backend = 'cuda' if route == 'auto' else route
        model.load_state_dict(initial, strict=True)
        optimizer.state.clear()
        loss = step()  # allocate Adam state through one identical warm step
        torch.cuda.synchronize()
        return loss.item()

    result = {route: dict(step_ms=[], first_loss=None) for route in ROUTES}
    for route in ROUTES:
        result[route]['first_loss'] = restore(route)
        for _ in range(2):
            step()
    for _ in range(args.model_rounds):
        for route in ROUTES:
            restore(route)
            result[route]['step_ms'].append(statistics.median(timed(step, args.model_iters)) / 1000)
    rows = {route: dict(step_ms=statistics.median(v['step_ms']), rounds=v['step_ms'], first_loss=v['first_loss'])
            for route, v in result.items()}
    rows['speedup'] = rows['pytorch']['step_ms'] / rows['auto']['step_ms']
    rows['pointwise_modules'] = len(layers)
    return rows


record = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rounds=args.rounds, iters=args.iters,
              binary_sha256=json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())['binary_sha256'],
              idle_samples=wait_idle(), operator={}, layer={})
for geometry in GEOMETRIES:
    key = 'x'.join(map(str, geometry[:4])) + f'->{geometry[4]}'
    record['operator'][key] = operator_case(geometry)
    row = record['operator'][key]
    print(key, {k: round(v['call_us']) for k, v in row.items() if k not in ('rounds', 'speedup_vs_cudnn')},
          'speedup vs cudnn call', round(row['speedup_vs_cudnn']['call_us'], 2),
          'kernel', round(row['speedup_vs_cudnn']['kernel_us'], 2), flush=True)
    record['layer'][key] = layer_case(geometry)
    row = record['layer'][key]
    print('  layer', {r: round(row[r]['call_us']) for r in ROUTES}, 'speedup call', round(row['speedup']['call_us'], 2),
          'kernel', round(row['speedup']['kernel_us'], 2), flush=True)
if args.model:
    record['model'] = model_case()
    print('usrnet step ms', {r: round(record['model'][r]['step_ms'], 2) for r in ROUTES},
          'speedup', round(record['model']['speedup'], 3), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
print('wrote', args.output)
