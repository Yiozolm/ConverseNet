"""Explicit current/PyTorch adapter around the unchanged real-data worker.

Run one process per backend/seed, using seeds 17, 29 and 43 for the formal
250-step comparison. All original worker arguments and defaults remain valid:
full pretrained 5-iteration/7-block USRNet, FP32 RGB MSE, Adam 1e-5 and batch 4.

Example:
  python test/train_usrnet_python_comparison.py --backend pytorch --seed 17 \
    --steps 250 --batch-size 4 --microbatch-size 4 \
    --run-dir artifacts/dataset_training/python_seed17

This adapter labels and verifies the execution backend. Quality must be paired
using initial tensor, split, recipe and per-step batch hashes; historical current
run times are not an independent paired speed benchmark for this adapter.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = Path(__file__).resolve()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def unchanged_namespace(_ops):
    # The reference adapter's small operations object is only for the worker's
    # optional clear_cache call. It must never replace torch.ops.converse2d.
    yield


def contains_spectral_solve(output):
    pending, seen = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = node.name() if callable(getattr(node, 'name', None)) else type(node).__name__
        if 'SpectralSolve' in str(name):
            return True
        pending.extend(edge for edge, _ in node.next_functions)
    return False


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('current', 'pytorch'), default='current')
    parser.add_argument('--variant', choices=('current',), default='current')
    adapter_args, worker_args = parser.parse_known_args()
    sys.path.insert(0, str(ROOT))
    import train_usrnet_dataset as worker
    original_load = worker.load_backend
    original_execute = worker.execute
    original_hashes = worker.source_hashes
    target = 'cuda' if adapter_args.backend == 'current' else 'pytorch'
    stats = dict(comparison_backend=adapter_args.backend, model_forwards=0,
                 grad_enabled_forwards=0, inference_forwards=0,
                 python_reference_calls=0, reference_calls_by_module={},
                 first_training_graph_checked=False, first_training_graph_has_spectral_solve=None)
    shared_files = {name: file_hash(ROOT/'test'/name) for name in
                    ('train_usrnet_dataset.py', 'usrnet_training_data.py', 'evaluate_usrnet_quality.py')}

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {'CONVERSE2D_BACKEND': target}))
        stack.enter_context(patch.object(sys, 'argv', [str(ADAPTER), *worker_args, '--variant', 'current']))
        installed = False

        def install_model_adapter():
            nonlocal installed
            if installed:
                return
            import torch
            # For current, import only after original_load has registered the
            # verified checkout extension, preserving the original import order.
            from models import converse_usrnet, util_converse
            model_class = converse_usrnet.ConverseUSRNet
            original_init = model_class.__init__
            for module, label in ((util_converse, 'Converse2D'), (converse_usrnet, 'DataNet')):
                reference = module.converse2d_reference
                def counted_reference(*args, _reference=reference, _label=label, **kwargs):
                    stats['python_reference_calls'] += 1
                    counts = stats['reference_calls_by_module']
                    counts[_label] = counts.get(_label, 0) + 1
                    if adapter_args.backend != 'pytorch':
                        raise RuntimeError('Current backend unexpectedly entered the Python reference')
                    if not args[0].is_cuda or args[0].dtype != torch.float32:
                        raise RuntimeError('Reference comparison must execute CUDA FP32 tensors')
                    return _reference(*args, **kwargs)
                stack.enter_context(patch.object(module, 'converse2d_reference', counted_reference))

            def configured_init(model, *args, **kwargs):
                kwargs['backend'] = target
                # Keep the module's class symbol intact: the implementation
                # calls super(ConverseUSRNet, self), which a factory would break.
                original_init(model, *args, **kwargs)
                if hasattr(model, 'reuse_training_spectra'):
                    model.reuse_training_spectra = False
                layers = [layer for layer in model.modules() if hasattr(layer, 'backend')]
                for layer in layers:
                    layer.backend = target
                # Each AlphaVariant contains ONE Converse2D in conv1; conv2
                # contains ordinary Conv2d layers. Count actual modules rather
                # than assuming two solvers per residual block.
                prior_solvers = sum(isinstance(layer, util_converse.Converse2D)
                                    for layer in model.p.modules())
                expected_calls = model.num_iterations * (prior_solvers + 1)
                stats['prior_solvers_per_iteration'] = prior_solvers
                stats['expected_reference_calls_per_forward'] = expected_calls
                stats['configured_backend_layers'] = len(layers)
                active = []

                def before_forward(module, inputs):
                    if os.environ.get('CONVERSE2D_BACKEND') != target or any(layer.backend != target for layer in layers):
                        raise RuntimeError('Comparison backend configuration changed during the run')
                    if getattr(module, 'reuse_training_spectra', False):
                        raise RuntimeError('This comparison excludes the optional spectrum-reuse candidate')
                    active.append(stats['python_reference_calls'])

                def after_forward(module, inputs, output):
                    if not active:
                        return
                    started = active.pop()
                    if output is None:
                        return
                    reference_calls = stats['python_reference_calls'] - started
                    wanted = expected_calls if adapter_args.backend == 'pytorch' else 0
                    if reference_calls != wanted:
                        raise RuntimeError(f'{adapter_args.backend}: expected {wanted} reference calls, got {reference_calls}')
                    stats['model_forwards'] += 1
                    mode = 'grad_enabled_forwards' if torch.is_grad_enabled() else 'inference_forwards'
                    stats[mode] += 1
                    if torch.is_grad_enabled() and not stats['first_training_graph_checked']:
                        if not output.requires_grad:
                            raise RuntimeError('Training output lost its autograd graph')
                        found = contains_spectral_solve(output)
                        if found != (adapter_args.backend == 'current'):
                            raise RuntimeError('Autograd graph does not match the explicitly selected backend')
                        stats['first_training_graph_checked'] = True
                        stats['first_training_graph_has_spectral_solve'] = found

                model.register_forward_pre_hook(before_forward)
                model.register_forward_hook(after_forward, always_call=True)
            stack.enter_context(patch.object(model_class, '__init__', configured_init))
            installed = True

        def load_backend(args):
            if args.variant != 'current':
                raise ValueError('This adapter compares only current versus pytorch, not frozen/reuse variants')
            if adapter_args.backend == 'current':
                result = original_load(args)
            else:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('The PyTorch comparison also requires CUDA')
                result = (SimpleNamespace(clear_cache=lambda: None),
                          dict(kind='PyTorch FP32 full-FFT autograd reference',
                               comparison_backend='pytorch', native_extension_build=False))
            install_model_adapter()
            return result

        def source_hashes():
            return {**original_hashes(), ADAPTER.relative_to(ROOT).as_posix(): file_hash(ADAPTER)}

        def execute(args, protocol, report):
            report['comparison_backend'] = adapter_args.backend
            report['comparison_adapter'] = dict(path=str(ADAPTER), sha256=file_hash(ADAPTER),
                shared_worker_files_sha256=shared_files, formal_seeds=[17, 29, 43], formal_steps=250,
                timing_scope='Use a separately paired benchmark for speed; these runs compare quality on matched training batches.')
            report['backend_verification'] = stats
            original_execute(args, protocol, report)
            if not stats['first_training_graph_checked'] or stats['grad_enabled_forwards'] < 1:
                raise RuntimeError('No verified training forward was observed')
            for name, expected_hash in shared_files.items():
                if file_hash(ROOT/'test'/name) != expected_hash:
                    raise RuntimeError(f'Shared worker/data/metric source changed during the run: {name}')

        stack.enter_context(patch.object(worker, 'load_backend', load_backend))
        stack.enter_context(patch.object(worker, 'source_hashes', source_hashes))
        stack.enter_context(patch.object(worker, 'execute', execute))
        if adapter_args.backend == 'pytorch':
            stack.enter_context(patch.object(worker, 'production_namespace', unchanged_namespace))
        worker.__doc__ = __doc__ + '\n\nOriginal worker options and protocol:\n' + worker.__doc__
        worker.main()


if __name__ == '__main__':
    main()
