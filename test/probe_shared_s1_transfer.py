"""Fixed pretrained precision gate for the exact shared-input s1 transfer.

For x0 IS x and scale1, H=(conj(K)+lambda)/(|K|^2+lambda).
Build H before broadcasting over batch, retain ATen automatic differentiation.
No parameters or spectra are cached, no fixture or tolerance is changed.
The precision grid diagnoses FFT, transfer and product rounding independently;
only passing candidates may be timed. Production remains unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = {
    'transfer32': (False, False, False, False),
    'transfer64': (True, False, False, False),
    'transfer64_input64_round': (True, True, False, False),
    'transfer64_product64': (True, False, True, False),
    'transfer64_input64_product64': (True, True, True, False),
    'full64_internal': (True, True, True, True),
}


def transfer_core(x, weight, bias, eps, mode):
    import torch
    from diagnose_boundary_precision import kernel_fft64
    high_h, high_input, high_product, high_output = MODES[mode]
    height, width = x.shape[-2:]
    k = kernel_fft64(weight, height, width)
    if not high_h:
        k = k.cfloat().contiguous()
    lam = torch.sigmoid((bias.double() if high_h else bias) - 9.) + eps
    h = (k.conj() + lam) / (k.real.square() + k.imag.square() + lam)
    y = torch.fft.rfft2(x.double() if high_input else x)
    if high_product:
        spectrum = h.cdouble() * y.cdouble()
    else:
        spectrum = h.cfloat() * y.cfloat()
    if not high_output:
        spectrum = spectrum.cfloat()
    return torch.fft.irfft2(spectrum, s=(height, width)).to(x.dtype)


def method(mode, padding=2, eps=1e-5):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference

    def forward(x, weight, bias):
        padded = F.pad(x, (padding,) * 4, mode='circular') if padding else x
        if mode == 'reference':
            output = converse2d_reference(padded, padded, weight, bias, 1, eps)
        elif mode == 'production':
            output = torch.ops.converse2d.forward(padded, padded, weight, bias, 1, eps, 'v7')
        else:
            output = transfer_core(padded, weight, bias, eps, mode)
        return output[..., padding:-padding, padding:-padding] if padding else output
    return forward


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-passing', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite prior evidence')
    import os
    import torch
    import train_usrnet_dataset as worker
    from extension_loader import load_extension
    from diagnose_boundary_precision import fixture, comparison
    from probe_pointwise_training import capture, timed_fixture
    os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
    load_extension()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tensors = fixture()
    fixture_hash = worker.tensor_hash(dict(zip(('x','weight','bias','upstream'),tensors)))
    original = json.loads((ROOT/'artifacts/native_deconv_target/boundary_probe.json').read_text())
    if fixture_hash != original['input_sha256']:
        raise RuntimeError('Original failed pretrained fixture was modified')
    sources = worker.source_hashes()
    for name in ('probe_shared_s1_transfer.py','diagnose_boundary_precision.py','probe_pointwise_training.py'):
        sources['test/'+name] = worker.file_hash(ROOT/'test'/name)
    report = dict(status='running', scope=__doc__, source_sha256=sources,
        fixture_sha256=fixture_hash, precision_modes=MODES, validation={}, timing=[],
        checkpoint_sha256=worker.file_hash(ROOT/'model_zoo/converse_usrnet.pth'),
        config=worker.json_safe(vars(args)), production_eligible=False,
        environment=dict(torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), tf32=False,
            cudnn_deterministic=True, deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        expected = capture(tensors, torch.float64, method('reference'))
        passing = []
        for name in ('production', *MODES):
            actual = capture(tensors, torch.float32, method(name))
            checks = {key: comparison(actual[key], value, *((3e-5,3e-5) if key=='output' else (5e-5,5e-5)))
                      for key,value in expected.items()}
            passed = all(row['passed'] for row in checks.values())
            report['validation'][name] = dict(passed=passed, tensors=checks)
            if passed:
                passing.append(name)
            print(name, json.dumps(dict(passed=passed, dw=checks['dw'])), flush=True)
            worker.write_json(args.output, report)
        if args.time_passing:
            for index in range(args.rounds):
                order = passing[index % len(passing):] + passing[:index % len(passing)] if passing else []
                report['timing'].append(dict(order=order, results={name: timed_fixture(tensors,method(name),args) for name in order}))
                worker.write_json(args.output, report)
        report['status'] = 'complete_fixed_fixture_screen'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        worker.write_json(args.output, report)


if __name__ == '__main__':
    main()
