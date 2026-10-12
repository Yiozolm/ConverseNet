"""Three-seed, three-step operator-only Level 1A training diagnostic.

FP32 master inputs/weight/bias and Adam state are retained. Only activation
casts are low precision. FP16 uses GradScaler without autocast; BF16 does not.
Trajectory drift is reported, not relabeled as a same-input kernel error gate.
This is not whole-model training, model approval, or convergence evidence.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from fp32_baseline import converse2d_fp32 as frozen_fp32
from numerical_policy import error_metrics
from tools.v4_mixed_precision.adapter import mixed_converse2d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(value):
    cpu = value.detach().contiguous().cpu()
    return dict(dtype=str(value.dtype), shape=list(value.shape), finite=bool(torch.isfinite(value).all()),
                sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())


def fixture(seed, device):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 2, 5, 7, generator=generator) * .03
    prior = torch.randn(1, 2, 15, 21, generator=generator) * .03
    weight = torch.randn(1, 2, 9, generator=generator).softmax(-1).reshape(1, 2, 3, 3)
    bias = torch.zeros(1, 2, 1, 1)
    target = torch.randn(1, 2, 15, 21, generator=generator) * .03
    return tuple(value.to(device) for value in (x, prior, weight, bias, target))


def lane(raw, name, dtype, device, initial_scale):
    x, prior, weight, bias, target = raw
    state = dict(name=name, x=x.detach().clone().requires_grad_(),
                 prior=prior.detach().clone().requires_grad_(),
                 weight=torch.nn.Parameter(weight.detach().clone()), bias=torch.nn.Parameter(bias.detach().clone()),
                 target=target, dtype=dtype)
    state['optimizer'] = torch.optim.Adam((state['weight'], state['bias']), lr=1e-4)
    state['scaler'] = (torch.amp.GradScaler(device, init_scale=initial_scale, growth_interval=2000)
                       if name != 'original_fp32' and dtype == torch.float16 else None)
    return state


def optimizer_step_count(state):
    values = [state['optimizer'].state.get(state[name], {}).get('step', 0) for name in ('weight', 'bias')]
    return [int(value.item()) if torch.is_tensor(value) else int(value) for value in values]


def one_step(state, backend):
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    state['x'].grad = state['prior'].grad = None
    if torch.is_autocast_enabled(state['x'].device.type):
        raise RuntimeError('This manual mixed diagnostic must not enable autocast')
    if state['name'] == 'original_fp32':
        output = frozen_fp32(state['x'], state['prior'], state['weight'], state['bias'], 3, 1e-3)
    else:
        xlow = state['x'].to(state['dtype'])
        plow = state['prior'].to(state['dtype'])
        if state['name'] == 'quantized_frozen_reference':
            # Match the exact differentiable low activation boundary, including
            # its backward cast, before entering the frozen FP32 training solve.
            output = frozen_fp32(xlow.float(), plow.float(), state['weight'], state['bias'], 3, 1e-3)
        else:
            output = mixed_converse2d(xlow, plow, state['weight'], state['bias'], 3, 1e-3,
                                      output_dtype=torch.float32, backend=backend)
    loss = (output - state['target']).square().mean()
    before_steps = optimizer_step_count(state)
    scale = scaler.get_scale() if scaler is not None else 1.0
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    # GradScaler unscales optimizer parameters only. Activation master leaves
    # are not optimized, so their diagnostic VJPs are explicitly divided by the
    # same pre-update scale. Cast-induced Inf cannot be repaired by this divide.
    snapshot = dict(output=output.detach().clone(), loss=loss.detach().clone(),
                    dx=(state['x'].grad / scale).detach().clone(),
                    dprior=(state['prior'].grad / scale).detach().clone(),
                    dweight=state['weight'].grad.detach().clone(), dbias=state['bias'].grad.detach().clone())
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    for name in ('weight', 'bias'):
        snapshot['parameter/' + name] = state[name].detach().clone()
        for field, value in optimizer.state[state[name]].items():
            if torch.is_tensor(value):
                snapshot[f'optimizer/{name}/{field}'] = value.detach().clone()
    after_steps = optimizer_step_count(state)
    metadata = dict(loss_scale_before=scale, loss_scale_after=scaler.get_scale() if scaler is not None else None,
                    grad_scaler_enabled=scaler is not None, optimizer_steps_before=before_steps,
                    optimizer_steps_after=after_steps,
                    optimizer_step_executed=all(after == before + 1 for before, after in zip(before_steps, after_steps)),
                    master_dtypes={name: str(state[name].dtype) for name in ('x', 'prior', 'weight', 'bias')},
                    autocast_enabled=torch.is_autocast_enabled(state['x'].device.type),
                    tensors={name: record(value) for name, value in snapshot.items()})
    metadata['all_finite'] = all(row['finite'] for row in metadata['tensors'].values())
    metadata['master_and_optimizer_fp32'] = (all(state[name].dtype == torch.float32 for name in ('x', 'prior', 'weight', 'bias'))
        and all(value.dtype == torch.float32 for name, value in snapshot.items() if name.startswith(('parameter/', 'optimizer/'))))
    metadata['activation_boundary_overflow'] = not all(bool(torch.isfinite(snapshot[name]).all()) for name in ('dx', 'dprior'))
    return snapshot, metadata


def setup(args, report):
    if os.environ.get('CONVERSE2D_BACKEND') or (args.device == 'cuda' and os.environ.get('CONVERSE2D_CPU_ONLY') == '1'):
        raise RuntimeError('Unset conflicting backend/CPU-only overrides')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    paths = [Path(__file__), Path(__file__).with_name('adapter.py'), ROOT / 'test/fp32_baseline.py',
             ROOT / 'test/numerical_policy.py', ROOT / 'models/converse_core.py']
    report['source_sha256'] = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    report['environment'] = dict(torch=str(torch.__version__), device=args.device, autocast=False, tf32=False)
    loader = None
    if args.device == 'cuda':
        if args.backend != 'cuda':
            raise ValueError('CUDA campaign must exercise the production adapter backend')
        manifest_path = ROOT / '.build/cuda/source_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest_hash = sha(manifest_path)
        old_arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
        arch = manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
        try:
            if arch:
                os.environ['TORCH_CUDA_ARCH_LIST'] = arch
            else:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
            import extension_loader as loader
            loader.load_extension()
        finally:
            if old_arch is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = old_arch
        if sha(manifest_path) != manifest_hash:
            raise RuntimeError('Production loader changed its manifest')
        report.update(checked_production_manifest=manifest, production_manifest_sha256=manifest_hash,
                      production_sources=loader.production_source_hashes())
        report['environment'].update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    elif args.backend != 'pytorch':
        raise ValueError('CPU diagnostic requires --backend pytorch')
    return loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--backend', choices=('cuda', 'pytorch'), default='cuda')
    parser.add_argument('--initial-scale', type=float, default=128.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve evidence: select a new output file')
    if not 0 < args.initial_scale < float('inf'):
        parser.error('initial-scale must be finite and positive')
    report = dict(kind='mixed_level1a_operator_training_diagnostic', status='running', passed=False,
        output_dtype='torch.float32', seeds=[17, 29, 43], optimizer_steps=3, scale=3, eps=1e-3,
        master_parameters='FP32 weight/bias; activation leaves also FP32, not optimizer parameters',
        lanes=['original_fp32', 'quantized_frozen_reference', 'candidate'],
        precision_mode='Manual FP16/BF16 activation casts, no autocast, FP32 output/core/master/Adam moments',
        initial_fp16_loss_scale=args.initial_scale, rows=[], trajectory_drift=[],
        pass_scope='Finite outputs/VJPs/state, three executed updates, FP32 master/Adam state only',
        numerical_admission=False, whole_model_training=False, convergence_evidence=False)
    try:
        loader = setup(args, report)
        for dtype in (torch.float16, torch.bfloat16):
            for seed in report['seeds']:
                raw = fixture(seed, args.device)
                report.setdefault('fixtures', {})[f'{dtype}/seed{seed}'] = {name: record(value)
                    for name, value in zip(('x_master', 'prior_master', 'weight_master', 'bias_master', 'target'), raw)}
                states = [lane(raw, name, dtype, args.device, args.initial_scale) for name in report['lanes']]
                for step in range(3):
                    snapshots = []
                    for state in states:
                        snapshot, metadata = one_step(state, args.backend)
                        snapshots.append(snapshot)
                        report['rows'].append(dict(dtype=str(dtype), seed=seed, step=step, lane=state['name'], **metadata))
                    if not (snapshots[0].keys() == snapshots[1].keys() == snapshots[2].keys()):
                        report['trajectory_drift'].append(dict(dtype=str(dtype), seed=seed, step=step,
                            comparable=False, reason='Optimizer states differ because a lane skipped an update'))
                    else:
                        for name in snapshots[0]:
                            r32, rq, candidate = (snapshot[name] for snapshot in snapshots)
                            report['trajectory_drift'].append(dict(dtype=str(dtype), seed=seed, step=step,
                                tensor=name, comparable=True,
                                quantized_vs_original=error_metrics(rq, r32.double()),
                                candidate_vs_quantized=error_metrics(candidate, rq.double()),
                                candidate_vs_original=error_metrics(candidate, r32.double()),
                                same_input_kernel_error_claim=False))
                    print(dtype, seed, step, [(row['lane'], row['all_finite'], row['optimizer_step_executed'])
                          for row in report['rows'][-3:]], flush=True)
        for name, digest in report['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError('Training diagnostic source changed: ' + name)
        if loader is not None:
            manifest = report['checked_production_manifest']
            if sha(ROOT / '.build/cuda/source_manifest.json') != report['production_manifest_sha256']:
                raise RuntimeError('Production manifest changed')
            if sha(ROOT / '.build/cuda' / manifest['library']) != manifest['binary_sha256']:
                raise RuntimeError('Production binary changed')
            if loader.production_source_hashes() != report['production_sources']:
                raise RuntimeError('Production sources changed')
        report['passed'] = len(report['rows']) == 54 and all(row['all_finite'] and row['optimizer_step_executed']
            and row['master_and_optimizer_fp32'] and not row['autocast_enabled'] for row in report['rows'])
        report['status'] = 'complete'
        if args.device == 'cpu':
            report['cuda_initialized'] = torch.cuda.is_initialized()
            if report['cuda_initialized']:
                raise RuntimeError('CPU-only diagnostic unexpectedly initialized CUDA')
    except Exception as error:
        report.update(status='error', passed=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Report:', args.output, flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
