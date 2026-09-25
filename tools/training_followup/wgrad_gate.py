"""Zero-margin per-tensor FP64 noninferiority gate for NHWC-only weight VJP.

No timing/model replacement runs here. Every native FP32/candidate tensor is
compared with the same independent native FP64 convolution oracle. Historical
GEMM/split-K failures are separate evidence and are never changed by this gate.
"""
import argparse
import json
from pathlib import Path

import wgrad_common as common


def values(torch, function, data, upstream, create_graph=False):
    output = function(*data)
    targets = [(name, value) for name, value in zip(('dx', 'dweight', 'dbias'), data) if value.requires_grad]
    gradients = torch.autograd.grad(output, [value for _, value in targets], upstream, create_graph=create_graph)
    return dict(output=output, **{name: value for (name, _), value in zip(targets, gradients)})


def device_case(torch, raw, dtype, needs):
    data = tuple(value.to(device='cuda', dtype=dtype).detach().requires_grad_(need)
                 for value, need in zip(raw[:3], needs))
    return data, raw[3].to(device='cuda', dtype=dtype)


def higher_order(torch, candidate, native, raw):
    results = []
    # Use the qualifying production shape: create_graph=True must route to the
    # complete original ATen convolution backward, not NHWC conversion/GEMM.
    for function in (native, candidate.conv1x1):
        data, upstream = device_case(torch, raw, torch.float32, (True, True, True))
        first = values(torch, function, data, upstream, create_graph=True)
        scalar = sum(value.square().sum() for name, value in first.items() if name != 'output')
        second = torch.autograd.grad(scalar, data, allow_unused=True)
        results.append({name: value.detach() for name, value in zip(('ddx', 'ddweight', 'ddbias'), second) if value is not None})
    rows = {}
    for name, baseline in results[0].items():
        actual = results[1][name]
        rows[name] = dict(passed=bool(torch.allclose(actual, baseline, atol=3e-5, rtol=3e-5)),
                          max_abs=float((actual-baseline).abs().max()),
                          python_fp32=common.tensors.tensor_record(baseline), candidate=common.tensors.tensor_record(actual))
    return dict(passed=all(row['passed'] for row in rows.values()), atol=3e-5, rtol=3e-5, tensors=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=common.ROOT)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2801)
    args = parser.parse_args()
    args.root, args.output, args.fixtures = args.root.resolve(), args.output.resolve(), args.fixtures.resolve()
    if args.output.exists():
        parser.error('Choose a new result path')
    torch, loader = common.setup(args.root)
    import torch.nn.functional as F
    import wgrad_channels_last as candidate
    native = lambda x, w, b: F.conv2d(x, w, b)
    captured = torch.load(args.fixtures, map_location='cpu', weights_only=True)
    if not captured:
        raise RuntimeError('Real model fixtures are required')
    cases = [dict(name='actual/'+name, raw=tuple(row[key] for key in ('x', 'weight', 'bias', 'grad_output')),
                  needs=(True, True, True), source='Actual current B4 model invocation') for name, row in captured.items()]
    weight, bias = cases[0]['raw'][1:3]
    for index in range(3):
        generator = torch.Generator().manual_seed(args.seed + index)
        x = torch.randn(4, 128, 96, 96, generator=generator)
        g = torch.randn(4, 64, 96, 96, generator=generator) / (4*64*96*96)**.5
        cases.append(dict(name=f'synthetic/seed{args.seed+index}', raw=(x, weight, bias, g),
                          needs=(True, True, True), source='Independent synthetic activations/upstream and actual model weights'))
    first = cases[0]['raw']
    for name, needs in (('weight_only', (False, True, False)), ('input_weight', (True, True, False)),
                        ('weight_bias', (False, True, True)), ('input_only_fallback', (True, False, False))):
        cases.append(dict(name='subset/'+name, raw=first, needs=needs, source=cases[0]['source']))
    for name in ('transposed_input_fallback', 'channels_last_input_fallback', 'noncontiguous_upstream'):
        x, w, b, g = first
        if name == 'transposed_input_fallback':
            x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        elif name == 'channels_last_input_fallback':
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            g = g.transpose(-1, -2).contiguous().transpose(-1, -2)
        cases.append(dict(name='layout/'+name, raw=(x, w, b, g), needs=(True, True, True), source=cases[0]['source']))
    report = dict(kind='wgrad_channels_last_zero_margin_gate', status='running', metadata=common.metadata(args.root, loader),
        fixture_path=str(args.fixtures), fixture_sha256=common.sha(args.fixtures), seed=args.seed, cases={},
        candidate_scope='FP32 NCHW B4 C128->64 HR96 1x1 only. Forward unchanged; native original-layout input/bias VJP; weight-only cuDNN uses channels_last copies inside backward. Algorithm/reduction order may differ.',
        timing_status='not requested; full-model experiment requires all-case gate and separate authorization')
    try:
        for case in cases:
            reference_data, reference_upstream = device_case(torch, case['raw'], torch.float64, case['needs'])
            reference = values(torch, native, reference_data, reference_upstream)
            data, upstream = device_case(torch, case['raw'], torch.float32, case['needs'])
            baseline = values(torch, native, data, upstream)
            actual = values(torch, candidate.conv1x1, data, upstream)
            row = dict(source=case['source'], needs_grad=case['needs'], active=candidate.eligible(*data),
                fixture=common.tensors.records(dict(zip(('x', 'weight', 'bias', 'grad_output'), case['raw']))),
                strides={name: list(value.stride()) for name, value in zip(('x', 'weight', 'bias', 'grad_output'), data+(upstream,))},
                python_fp32=common.tensors.records(baseline, reference), snapshot=common.tensors.records(actual, reference))
            row['noninferiority'] = common.tensors.noninferiority(row)
            row['passed'] = all(item['passed'] for item in row['noninferiority'].values())
            report['cases'][case['name']] = row
            print(case['name'], 'active', row['active'], 'passed', row['passed'], flush=True)
            del reference, reference_data, reference_upstream, baseline, actual, data, upstream
        report['higher_order'] = higher_order(torch, candidate, native, first)
        report['active_cases'] = sum(row['active'] for row in report['cases'].values())
        report['failures'] = [name+'/'+tensor for name,row in report['cases'].items()
                              for tensor, value in row['noninferiority'].items() if not value['passed']]
        report['all_gpu_cases_passed'] = not report['failures'] and report['active_cases'] > 0 and report['higher_order']['passed']
        report['status'] = 'complete'
        if not report['all_gpu_cases_passed']:
            report['timing_status'] = 'blocked_by_numerical_gate; no timing or model replacement'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['affinity_after'] = common.affinity()
        common.write_report(args.output, report)
    if not report['all_gpu_cases_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
