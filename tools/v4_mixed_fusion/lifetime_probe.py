"""Diagnose public-module versus direct-bound-forward upcast lifetimes."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import types
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from tools.v4_mixed_fusion import baseline, loader


@contextmanager
def wrapper_control(module):
    original = module.forward
    def forward(this, value):
        return original(value.float())
    module.forward = types.MethodType(forward, module)
    try:
        yield
    finally:
        del module.forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior evidence')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    production = loader.load_production_checked()
    spec = dict(baseline.WORKLOADS['b4_c128_96'], name='b4_c128_96', padding_mode='circular', seed=41001, eps=1e-5, scale=1)
    fixture = baseline.make_fixture(spec, 'cuda')
    original_pad, original_op = torch.nn.functional.pad, torch.ops.converse2d.forward
    rows = []
    for route in ('original_module_call', 'common_wrapper_control'):
        observed = {}
        def pad(tensor, *args, **kwargs):
            observed['upcast_weakref'] = weakref.ref(tensor)
            return original_pad(tensor, *args, **kwargs)
        def op(*args, **kwargs):
            observed['upcast_alive_at_solver'] = observed['upcast_weakref']() is not None
            observed['allocated_at_solver'] = torch.cuda.memory_allocated()
            return original_op(*args, **kwargs)
        torch.ops.converse2d.clear_cache()
        with torch.no_grad(), patch('torch.nn.functional.pad', new=pad), patch.object(torch.ops.converse2d, 'forward', new=op):
            before = torch.cuda.memory_allocated()
            if route == 'original_module_call':
                output = fixture['module'](fixture['stored'].float())
            else:
                with wrapper_control(fixture['module']):
                    output = fixture['module'](fixture['stored'])
            del output
        rows.append(dict(route=route, upcast_alive_at_solver=observed['upcast_alive_at_solver'],
                         incremental_allocation_at_solver=observed['allocated_at_solver'] - before))
    if not rows[0]['upcast_alive_at_solver'] or rows[1]['upcast_alive_at_solver']:
        raise RuntimeError('Lifetime distinction not reproduced on this runtime')
    report = dict(kind='mixed_fusion_frontend_lifetime', passed=True, rows=rows, production=production,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=str(torch.__version__), scope='Diagnostic only; no latency or promotion claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
