"""Actual B4 LR32/s3 Adam-step module/wgrad attribution; diagnostic only.

Raw Chrome kernels are counted once (cat=kernel); annotations never contribute
to the GPU denominator. Forward output grad_fn sequence IDs map convolution
backward kernels to exact module paths and repeated invocations. LayerNorm
uses its actual graph-node sequences, with the input grad_fn as a boundary.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import wgrad_common as common


def graph_sequences(output, inputs):
    boundaries = {value.grad_fn for value in inputs if value.grad_fn is not None}
    pending, seen, result = [output.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in boundaries or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == 'AccumulateGrad':
            continue
        result.append(dict(sequence=int(node._sequence_nr()), name=type(node).__name__))
        pending.extend(child for child, _ in node.next_functions)
    return result


def classify(name):
    if 'wgrad' in name:
        return 'cuDNN_wgrad'
    if 'dgrad' in name:
        return 'cuDNN_dgrad'
    if 'cudnn' in name:
        return 'cuDNN_other'
    if 'fft' in name.lower():
        return 'FFT_named_kernels'
    if 'full_training' in name:
        return 'Converse_training'
    if 'psf' in name.lower():
        return 'PSF'
    return 'other'


def analyze(trace_path, calls):
    trace = json.load(gzip.open(trace_path, 'rt', encoding='utf-8'))
    events = trace['traceEvents']
    cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'user_annotation')]
    by_thread = defaultdict(list)
    for event in cpu:
        by_thread[(event['pid'], event['tid'])].append(event)
    parents, external = {}, {}
    for group in by_thread.values():
        stack = []
        for event in sorted(group, key=lambda e: (e['ts'], -e['dur'])):
            end = event['ts'] + event['dur']
            while stack and (event['ts'] >= stack[-1]['ts'] + stack[-1]['dur'] or end > stack[-1]['ts'] + stack[-1]['dur'] + .01):
                stack.pop()
            parents[id(event)] = stack[-1] if stack else None
            external[event.get('args', {}).get('External id')] = event
            stack.append(event)
    conv_seq = {row['sequence']: row['call_id'] for row in calls if row['type'] == 'Conv2d'}
    ln_seq = {node['sequence']: row['call_id'] for row in calls if row['type'] == 'LayerNorm' for node in row['nodes']}
    by_id = {row['call_id']: row for row in calls}
    for row in calls:
        row.update(wgrad_kernel_ids=[], backward_kernel_ids=[], forward_kernel_ids=[])
    kernels, unmatched_wgrad = [], []
    for index, event in enumerate(e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'):
        parent = external.get(event.get('args', {}).get('External id'))
        conv = ln = forward = None
        ancestry = []
        while parent is not None:
            args = parent.get('args', {})
            sequence = args.get('Sequence number')
            ancestry.append(dict(name=parent['name'], sequence=sequence, external_id=args.get('External id')))
            if 'Backward' in parent['name'] or parent['name'].startswith('autograd::engine::evaluate_function:'):
                conv = conv or conv_seq.get(sequence)
                ln = ln or ln_seq.get(sequence)
            if parent['name'].startswith('wgrad_module/'):
                forward = parent['name'].removeprefix('wgrad_module/')
            parent = parents.get(id(parent))
        kind = classify(event['name'])
        row = dict(id=index, name=event['name'], start_us=event['ts'], duration_us=event['dur'],
                   stream=event.get('args', {}).get('stream'), external_id=event.get('args', {}).get('External id'),
                   correlation=event.get('args', {}).get('correlation'), category=kind,
                   convolution_call=conv, layernorm_backward_call=ln, forward_call=forward)
        if kind == 'cuDNN_wgrad':
            row['cpu_ancestry'] = ancestry
            if conv:
                by_id[conv]['wgrad_kernel_ids'].append(index)
            else:
                unmatched_wgrad.append(index)
        if conv:
            by_id[conv]['backward_kernel_ids'].append(index)
        if ln:
            by_id[ln]['backward_kernel_ids'].append(index)
        if forward in by_id:
            by_id[forward]['forward_kernel_ids'].append(index)
        kernels.append(row)
    duration = lambda ids: sum(kernels[i]['duration_us'] for i in set(ids))
    for row in calls:
        for phase in ('wgrad', 'backward', 'forward'):
            row[phase + '_gpu_us'] = duration(row[phase + '_kernel_ids'])
    groups = defaultdict(lambda: dict(calls=0, wgrad_gpu_us=0., backward_gpu_us=0., forward_gpu_us=0., call_ids=[]))
    shapes = defaultdict(lambda: dict(calls=0, wgrad_gpu_us=0., backward_gpu_us=0., call_ids=[]))
    for row in calls:
        summary = groups[row['module']]
        summary['calls'] += 1
        summary['call_ids'].append(row['call_id'])
        summary['input_shape'] = row['input_shape']
        summary['weight_shape'] = row['weight_shape']
        summary['type'] = row['type']
        for phase in ('wgrad', 'backward', 'forward'):
            summary[phase + '_gpu_us'] += row[phase + '_gpu_us']
        if row['type'] == 'Conv2d':
            shape = shapes[str(row['input_shape']) + ' / ' + str(row['weight_shape'])]
            shape['calls'] += 1
            shape['call_ids'].append(row['call_id'])
            shape['wgrad_gpu_us'] += row['wgrad_gpu_us']
            shape['backward_gpu_us'] += row['backward_gpu_us']
    classes = defaultdict(lambda: dict(count=0, gpu_us=0.))
    for row in kernels:
        classes[row['category']]['count'] += 1
        classes[row['category']]['gpu_us'] += row['duration_us']
    origin = min(row['start_us'] for row in kernels)
    intervals = sorted((row['start_us'] - origin, row['start_us'] - origin + row['duration_us']) for row in kernels)
    union, left, right = 0., *intervals[0]
    for start, end in intervals[1:]:
        if start > right:
            union += right - left
            left, right = start, end
        else:
            right = max(right, end)
    union += right - left
    ln_forward = {i for row in calls if row['type'] == 'LayerNorm' for i in row['forward_kernel_ids']}
    ln_backward = {i for row in calls if row['type'] == 'LayerNorm' for i in row['backward_kernel_ids']}
    return dict(kernel_count=len(kernels), kernel_sum_us=sum(row['duration_us'] for row in kernels),
        kernel_union_us=union, kernel_span_us=max(end for _, end in intervals) - intervals[0][0],
        streams=sorted({row['stream'] for row in kernels}), mutually_exclusive_name_categories=dict(classes),
        exact_modules=dict(groups), convolution_shapes=dict(shapes), calls=calls, kernels=kernels, unmatched_wgrad_kernel_ids=unmatched_wgrad,
        layernorm=dict(forward_gpu_us=duration(ln_forward), backward_gpu_us=duration(ln_backward),
                       union_gpu_us=duration(ln_forward | ln_backward), forward_kernel_ids=sorted(ln_forward),
                       backward_kernel_ids=sorted(ln_backward), overlap_kernel_ids=sorted(ln_forward & ln_backward),
                       scope='Orthogonal module/sequence attribution, already included in name categories; not additive. No standalone backward speed claim.'),
        mapping='CUDA External id -> CPU event -> nested backward Sequence number -> exact forward output grad_fn sequence/module call. LayerNorm uses bounded internal graph-node sequences.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=common.ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--capture-module', default='p.m_body.0.conv1.5')
    parser.add_argument('--data', choices=('synthetic', 'dataset'), default='synthetic')
    parser.add_argument('--manifest', type=Path, default=common.ROOT / 'artifacts/dataset_training/split_900_100.json')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error('Preserve previous evidence; choose a new output')
    torch, loader = common.setup(args.root)
    from models.converse_usrnet import ConverseUSRNet
    torch.manual_seed(args.seed)
    model = ConverseUSRNet(backend='cuda').cuda().train()
    model.load_state_dict(torch.load(args.root / 'model_zoo/converse_usrnet.pth', map_location='cpu', weights_only=True))
    x = torch.rand(4, 3, 32, 32, device='cuda')
    kernel = torch.rand(4, 1, 7, 7, device='cuda')
    kernel /= kernel.sum((-2, -1), keepdim=True)
    target = torch.rand(4, 3, 96, 96, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)
    dataset_metadata, dataset_batches, dataset_hashes = None, None, None
    if args.data == 'dataset':
        sys.path.insert(0, str(args.root / 'tools/roadmap_quality'))
        import usrnet_training_data
        import train_usrnet_dataset as worker
        usrnet_training_data.ROOT = args.root
        protocol = usrnet_training_data.DatasetProtocol(args.manifest, patch_size=96, scale=3, seed=args.seed, noise_std=.01)
        dataset_metadata = dict(protocol.metadata,
            worker_sha256=common.sha(args.root / 'tools/roadmap_quality/train_usrnet_dataset.py'))
        dataset_batches = [protocol.train_batch(index, 4) for index in range(args.warmup + 1)]
        dataset_hashes = [common.tensors.records(dict(zip(('lr', 'kernel', 'hr'), batch))) for batch in dataset_batches]
        recipe = argparse.Namespace(batch_size=4, microbatch_size=4, patch_size=96, scale=3, loss='mse')
        for batch in dataset_batches:
            worker.check_cpu_batch(batch, recipe, 4)
    def step(index):
        if dataset_batches is not None:
            row = worker.train_step(model, optimizer, dataset_batches[index], recipe)
            return None, torch.tensor(row['loss'], dtype=torch.float32), row
        optimizer.zero_grad(set_to_none=True)
        output = model(x, kernel, 3)
        loss = (output - target).square().mean()
        loss.backward()
        optimizer.step()
        return output.detach(), loss.detach(), None
    for index in range(args.warmup):
        step(index)
    torch.cuda.synchronize()
    result = dict(kind='actual_B4_wgrad_module_profile', status='running', metadata=common.metadata(args.root, loader),
                  settings=dict(batch=4, lr_shape=[32, 32], scale=3, hr_shape=[96, 96], prior_fft_shape=[100, 100],
                                warmup=args.warmup, seed=args.seed, capture_module=args.capture_module, data=args.data),
                  dataset_protocol=dataset_metadata, dataset_batches=dataset_hashes,
                  profiler_scope=('One frozen-worker train_step on DatasetProtocol batch index '+str(args.warmup)+
                    ', after preceding distinct batches; F.mse_loss, microbatch4 weighting, H2D, finite checks/norm and Adam included. CPU batch generation/hash/validation outside profile; no evaluation/checkpoint. Diagnostic times are not speed benchmarks.'
                    if args.data == 'dataset' else
                    'One complete warm Adam step, fixed synthetic data/pretrained initialization; profiler/hook diagnostic times are not speed benchmarks.'))
    handles, stacks, counts, calls, captured = [], {}, defaultdict(int), [], {}
    last_model_output = []
    selected = dict(model.named_modules())[args.capture_module]
    selected_parameters = dict(weight=selected.weight.detach().cpu().clone(), bias=selected.bias.detach().cpu().clone())
    torch.cuda.synchronize()
    for path, module in model.named_modules():
        if type(module).__name__ not in ('Conv2d', 'LayerNorm'):
            continue
        def pre(m, inputs, path=path):
            index = counts[path]
            counts[path] += 1
            call_id = f'{path}#{index}'
            context = torch.profiler.record_function('wgrad_module/' + call_id)
            context.__enter__()
            stacks.setdefault(id(m), []).append((context, call_id))
        def post(m, inputs, output, path=path):
            context, call_id = stacks[id(m)].pop()
            try:
                row = dict(call_id=call_id, module=path, type=type(m).__name__,
                           input_shape=list(inputs[0].shape), input_stride=list(inputs[0].stride()),
                           output_shape=list(output.shape), weight_shape=list(m.weight.shape),
                           sequence=int(output.grad_fn._sequence_nr()))
                if row['type'] == 'LayerNorm':
                    row['nodes'] = graph_sequences(output, inputs)
                calls.append(row)
                if path == 'conv2':
                    last_model_output[:] = [output.detach()]
                if path == args.capture_module:
                    captured[call_id] = dict(x=inputs[0].detach(), **selected_parameters)
                    def capture_gradient(g, call_id=call_id):
                        captured[call_id]['grad_output'] = g.detach()
                    output.register_hook(capture_gradient)
            finally:
                context.__exit__(None, None, None)
        handles.extend((module.register_forward_pre_hook(pre), module.register_forward_hook(post)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix('.trace.json.gz')
    fixture_path = args.output.with_suffix('.pt')
    if trace_path.exists() or fixture_path.exists():
        raise RuntimeError('Refusing to overwrite trace/capture evidence')
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            output, loss, worker_row = step(args.warmup)
            torch.cuda.synchronize()
        if output is None:
            output = last_model_output[0]
        profile.export_chrome_trace(str(trace_path))
        captured = {name: {key: value.detach().cpu().clone() for key, value in row.items()} for name, row in captured.items()}
        torch.save(captured, fixture_path)
        result.update(status='complete', trace_path=str(trace_path), trace_sha256=common.sha(trace_path),
                      fixtures_path=str(fixture_path), fixtures_sha256=common.sha(fixture_path),
                      fixtures={name: common.tensors.records(values) for name, values in captured.items()},
                      frozen_worker_step_record=worker_row,
                      step_output=common.tensors.tensor_record(output), step_loss=common.tensors.tensor_record(loss),
                      attribution=analyze(trace_path, calls))
        if result['attribution']['unmatched_wgrad_kernel_ids']:
            raise RuntimeError('Unmapped wgrad kernels; attribution is incomplete')
        print(json.dumps({key: result['attribution'][key] for key in ('kernel_count', 'kernel_sum_us', 'mutually_exclusive_name_categories')}, indent=2))
        print('LayerNorm GPU us:', {key: result['attribution']['layernorm'][key] for key in
                                    ('forward_gpu_us', 'backward_gpu_us', 'union_gpu_us')})
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        for handle in handles:
            handle.remove()
        result['affinity_after'] = common.affinity()
        common.write_report(args.output, result)


if __name__ == '__main__':
    main()
