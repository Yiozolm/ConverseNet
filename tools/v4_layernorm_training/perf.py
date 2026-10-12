"""Paired timing of the fused channels-first LayerNorm training path.

Both routes run in one process on the same checked build: `pytorch` keeps the
original ATen expression (ten forward and about twenty-five backward kernels),
`auto` selects the fused statistics forward and the fused first-order VJP.
Rounds alternate routes. Records per-call forward+VJP medians (CUDA events),
the per-call GPU kernel sum (torch.profiler) and, with --model, the complete
pretrained USRNet B4 LR32->HR96 s3 training step (zero_grad, forward, MSE,
backward, Adam) with only the LayerNorm modules switched between routes.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=6)
parser.add_argument('--iters', type=int, default=30)
parser.add_argument('--model', action='store_true')
parser.add_argument('--model-rounds', type=int, default=5)
parser.add_argument('--model-iters', type=int, default=5)
args = parser.parse_args()
sys.path[:0] = [str(args.root.resolve() / 'test'), str(args.root.resolve())]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
os.environ.pop('CONVERSE2D_BACKEND', None)
import torch
from torch.profiler import ProfilerActivity, profile
from extension_loader import load_extension
from models.util_converse import LayerNorm

load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
ROUTES = ('pytorch', 'auto')
SHAPES = ((4, 64, 96, 96), (2, 128, 100, 100), (1, 64, 96, 96))


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


def operator_case(shape):
    torch.manual_seed(41101 + shape[1])
    x = torch.randn(shape, device='cuda', requires_grad=True)
    layer = LayerNorm(shape[1], eps=1e-5, data_format='channels_first').cuda()
    with torch.no_grad():
        layer.weight.add_(torch.randn_like(layer.weight) * .1)
        layer.bias.add_(torch.randn_like(layer.bias) * .1)
    upstream = torch.randn_like(x)

    def run():
        return torch.autograd.grad(layer(x), (x, layer.weight, layer.bias), upstream)

    result = {route: dict(call_us=[], kernel_us=[]) for route in ROUTES}
    checks = {}
    for route in ROUTES:
        layer.backend = route
        checks[route] = [g.detach().clone() for g in run()]
        for _ in range(5):
            run()
    for _ in range(args.rounds):
        for route in ROUTES:
            layer.backend = route
            result[route]['call_us'].append(statistics.median(timed(run, args.iters)))
            result[route]['kernel_us'].append(kernel_sum_us(run, args.iters))
    rows = {route: {k: statistics.median(v) for k, v in values.items()} for route, values in result.items()}
    rows['speedup'] = {k: rows['pytorch'][k] / rows['auto'][k] for k in ('call_us', 'kernel_us')}
    rows['rounds'] = result
    rows['gradient_rel_l2_pytorch_vs_auto'] = [
        ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()
        for a, b in zip(checks['pytorch'], checks['auto'])]
    del x, layer, upstream
    return rows


def model_case():
    from models.converse_usrnet import ConverseUSRNet
    torch.manual_seed(72001)
    checkpoint = args.root / 'model_zoo/converse_usrnet.pth'
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ConverseUSRNet(backend='cuda').cuda()
    model.load_state_dict(initial, strict=True)
    model.train()
    norms = [m for m in model.modules() if isinstance(m, LayerNorm)]
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
        for m in norms:
            m.backend = route
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
    rows['layernorm_modules'] = len(norms)
    return rows


record = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rounds=args.rounds, iters=args.iters,
              binary_sha256=json.loads((args.root / '.build/cuda/source_manifest.json').read_text())['binary_sha256'],
              operator={})
for shape in SHAPES:
    record['operator']['x'.join(map(str, shape))] = operator_case(shape)
    row = record['operator']['x'.join(map(str, shape))]
    print(shape, {r: round(row[r]['call_us']) for r in ROUTES}, 'call', round(row['speedup']['call_us'], 2),
          'kernel', round(row['speedup']['kernel_us'], 2), flush=True)
if args.model:
    record['model'] = model_case()
    print('usrnet step ms', {r: round(record['model'][r]['step_ms'], 2) for r in ROUTES},
          'speedup', round(record['model']['speedup'], 3), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
print('wrote', args.output)
