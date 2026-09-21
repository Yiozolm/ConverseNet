"""Differentiable kernel-spectrum reuse is confined to one forward scope."""
import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch
from extension_loader import load_extension
import test_fp32_training as common
from models.converse_core import converse2d_reference


@contextmanager
def counted_scope():
    counts = []
    torch.ops.converse2d._begin_training_cache()
    try:
        yield counts
    finally:
        counts.extend(torch.ops.converse2d._end_training_cache())


class TrainingScope(unittest.TestCase):
    data = common.FP32Training.data
    compare = common.FP32Training.compare
    assert_numerical_close = common.FP32Training.assert_numerical_close

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(9214)

    @staticmethod
    def forward(args, scale=2, variant='v7'):
        return torch.ops.converse2d.forward(*args, scale, 1e-3, variant)

    def test_shared_spectrum_accumulates_gradients_and_higher_derivatives(self):
        for higher in (False, True):
            with self.subTest(higher=higher):
                x, prior, weight, bias = self.data(5, 7, 2)
                args = (x, prior, torch.randn_like(x).requires_grad_(),
                        torch.randn_like(prior).requires_grad_(), weight, bias)
                upstream = tuple(torch.randn_like(prior)/prior.numel()**.5 for _ in range(2))
                results = []
                for name in ('scoped', 'unscoped', 'fp64'):
                    values = tuple(t.detach().to(torch.float64 if name == 'fp64' else torch.float32)
                                   .requires_grad_() for t in args)
                    op = converse2d_reference if name == 'fp64' else torch.ops.converse2d.forward
                    with counted_scope() if name == 'scoped' else nullcontext([]) as counts:
                        outputs = (op(values[0], values[1], values[4], values[5], 2, 1e-3),
                                   op(values[2], values[3], values[4], values[5], 2, 1e-3))
                    if name == 'scoped':
                        self.assertEqual(counts, [1, 1])
                    # Ending a scope releases cache ownership without detaching
                    # either output from the one shared preparation graph.
                    grads = torch.autograd.grad(outputs, values,
                        tuple(u.to(outputs[0].dtype) for u in upstream), create_graph=higher)
                    results.append((outputs, grads, values))
                for group, tolerance in ((0, 3e-5), (1, 5e-5)):
                    for name, candidate in zip(('scoped', 'unscoped'), results[:2]):
                        for index, (actual, expected) in enumerate(zip(candidate[group], results[2][group])):
                            self.assert_numerical_close(actual, expected, atol=tolerance, rtol=tolerance,
                                                        label=f'{name}/group{group}/{index}')
                if higher:
                    vectors = [torch.randn_like(g) for g in results[0][1]]
                    seconds = []
                    for _, grads, values in results:
                        direction = sum((g*v.to(g.dtype)).sum() for g, v in zip(grads, vectors))
                        seconds.append(torch.autograd.grad(direction, values))
                    for actual, expected in zip(seconds[0], seconds[2]):
                        self.assert_numerical_close(actual, expected, atol=2e-3, rtol=2e-4,
                                                    label='scoped second derivative')

    def test_fresh_scope_after_optimizer_step(self):
        args = self.data(5, 7, 2)
        optimizer = torch.optim.SGD(args[2:], lr=1e-4)
        for step in range(3):
            with self.subTest(step=step):
                optimizer.zero_grad(set_to_none=True)
                before = args[2].detach().clone()
                with counted_scope() as counts:
                    first = self.forward(args)
                    second = self.forward((args[0]*.9, args[1]+.01, *args[2:]))
                self.assertEqual(counts, [1, 1])
                (first.square().mean()+second.square().mean()).backward()
                self.assertTrue(torch.isfinite(args[2].grad).all().item())
                optimizer.step()
                self.assertFalse(torch.equal(args[2], before))
                self.compare(args, 2)

    def test_source_version_identity_storage_and_requires_grad_keys(self):
        for change in ('version', 'identity', 'storage', 'enable_gradient', 'disable_gradient'):
            with self.subTest(change=change):
                args = list(self.data(5, 7, 2))
                if change == 'enable_gradient':
                    args[2].requires_grad_(False)
                with counted_scope() as counts:
                    self.forward(args)
                    with torch.no_grad():
                        if change == 'version':
                            args[2].mul_(.95)
                        elif change == 'identity':
                            args[2] = args[2].detach().clone().requires_grad_()
                        elif change == 'storage':
                            # Deliberately replace storage without a version
                            # bump to exercise the independent data_ptr key.
                            args[2].data = args[2].detach().mul(.95)
                        elif change == 'enable_gradient':
                            args[2].requires_grad_(True)
                        else:
                            args[2].requires_grad_(False)
                    self.compare(tuple(args), 2)
                self.assertEqual(counts, [0, 2])

    def test_spatial_shape_is_part_of_key(self):
        args = self.data(5, 7, 2)
        other = self.data(7, 5, 2)
        with counted_scope() as counts:
            self.forward(args)
            self.compare((*other[:2], *args[2:]), 2)
        self.assertEqual(counts, [0, 2])

    def test_detached_nonleaf_does_not_reuse_the_old_producer_graph(self):
        x, prior, base, bias = self.data(5, 7, 2)
        kernel = base.square()
        version, pointer = kernel._version, kernel.data_ptr()
        with counted_scope() as counts:
            self.forward((x, prior, kernel, bias))
            kernel.detach_().requires_grad_(True)
            self.assertEqual((kernel._version, kernel.data_ptr()), (version, pointer))
            output = self.forward((x, prior, kernel, bias))
        self.assertEqual(counts, [0, 2])
        producer_grad, kernel_grad = torch.autograd.grad(output.square().mean(),
                                                        (base, kernel), allow_unused=True)
        self.assertIsNone(producer_grad)
        self.assertIsNotNone(kernel_grad)
        self.assertTrue(torch.isfinite(kernel_grad).all())

    def test_nested_scopes_restore_the_outer_scope(self):
        args = self.data(5, 7, 2)
        with counted_scope() as outer:
            self.forward(args)
            with counted_scope() as inner:
                self.forward(args)
                self.forward(args)
            self.forward(args)
        self.assertEqual(inner, [1, 1])
        self.assertEqual(outer, [1, 1])
        with counted_scope() as fresh:
            self.forward(args)
        self.assertEqual(fresh, [0, 1])

    def test_explicit_weights_do_not_retain_dynamic_kernel_spectra(self):
        args = self.data(5, 7, 2)
        other = self.data(5, 7, 2)
        torch.ops.converse2d._begin_training_cache_for([args[2]])
        try:
            self.forward(args)
            self.forward(other)
            self.forward(other)
            self.forward(args)
        finally:
            counts = list(torch.ops.converse2d._end_training_cache())
        self.assertEqual(counts, [1, 1])

    def test_scope_is_thread_local(self):
        args = self.data(5, 7, 2)
        device = args[0].device
        def other_thread():
            with torch.cuda.device(device), counted_scope() as counts:
                self.forward(args)
                self.forward(args)
            return counts
        with counted_scope() as main:
            self.forward(args)
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(executor.submit(other_thread).result(), [1, 1])
            self.forward(args)
        self.assertEqual(main, [1, 1])

    def test_no_cache_entries_for_nontraining_paths(self):
        for mode in ('no_grad', 'frozen', 'fp64', 'full_fft'):
            with self.subTest(mode=mode):
                args = self.data(5, 7, 2)
                if mode == 'frozen':
                    args = tuple(t.detach() for t in args)
                elif mode == 'fp64':
                    args = tuple(t.detach().double().requires_grad_() for t in args)
                with counted_scope() as counts:
                    with torch.no_grad() if mode == 'no_grad' else torch.enable_grad():
                        for _ in range(2):
                            self.forward(args, variant='v2' if mode == 'full_fft' else 'v7')
                self.assertEqual(counts, [0, 0])

    def test_nondefault_stream_never_reuses_another_stream_entry(self):
        args = self.data(5, 7, 2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with counted_scope() as counts:
            self.forward(args)
            with torch.cuda.stream(stream):
                self.forward(args)
                self.forward(args)
            torch.cuda.current_stream().wait_stream(stream)
        self.assertEqual(counts, [1, 2])

    def test_python_context_cleans_up_an_exception(self):
        from models.util_converse import _training_kernel_scope
        args = self.data(5, 7, 2)
        real_end = torch.ops.converse2d._end_training_cache
        ended = []
        def record_end():
            result = real_end()
            ended.append(list(result))
            return result
        with patch.object(torch.ops.converse2d, '_end_training_cache', record_end):
            with self.assertRaisesRegex(RuntimeError, 'deliberate forward failure'):
                with _training_kernel_scope(args[0], 'cuda'):
                    self.forward(args)
                    raise RuntimeError('deliberate forward failure')
            with _training_kernel_scope(args[0], 'cuda'):
                self.forward(args)
                self.forward(args)
        self.assertEqual(ended, [[0, 1], [1, 1]])

    def test_full_usrnet_eval_with_gradients_matches_unscoped(self):
        from models.converse_usrnet import ConverseUSRNet
        # Default five iterations/seven blocks; nonzero residual gates make
        # this exercise the internal solvers' gradients on a small image.
        actual_model = ConverseUSRNet(backend='cuda', reuse_training_spectra=True).cuda().eval()
        with torch.no_grad():
            for name, p in actual_model.named_parameters():
                if name.endswith(('alpha1', 'alpha2')):
                    p.fill_(.1)
        # Isolate scope semantics with the same current FP32 model/operator.
        # The unchanged absolute FP64 gate lives in test_full_usrnet_precision.
        expected_model = copy.deepcopy(actual_model)
        x = (torch.rand(1, 3, 4, 5, device='cuda')*.2).requires_grad_()
        kernel = torch.rand(1, 1, 7, 7, device='cuda')
        kernel = (kernel/kernel.sum((-2, -1), keepdim=True)).requires_grad_()
        rx, rk = (t.detach().clone().requires_grad_() for t in (x, kernel))
        real_end = torch.ops.converse2d._end_training_cache
        ended = []
        def record_end():
            result = real_end()
            ended.append(list(result))
            return result
        with torch.backends.cudnn.flags(allow_tf32=False):
            with patch.object(torch.ops.converse2d, '_end_training_cache', record_end):
                actual = actual_model(x, kernel, 2)
                expected = expected_model._forward_impl(rx, rk, 2)
            self.assertTrue(actual.requires_grad)
            self.assertEqual(len(ended), 1)
            self.assertGreater(ended[0][0], 0)
            self.assertGreater(ended[0][1], 0)
            self.assert_numerical_close(actual, expected, atol=3e-5, rtol=3e-4, label='full USRNet output')
            upstream = torch.randn_like(actual)/actual.numel()**.5
            actual_grads = torch.autograd.grad(actual, (x, kernel, *actual_model.parameters()), upstream)
            expected_grads = torch.autograd.grad(expected, (rx, rk, *expected_model.parameters()), upstream)
            gradient_names = ['input', 'input_kernel'] + [name for name, _ in actual_model.named_parameters()]
            for name, a, e in zip(gradient_names, actual_grads, expected_grads):
                self.assert_numerical_close(a, e, atol=3e-5, rtol=3e-4,
                                            label=f'full USRNet gradient/{name}')
            self.assertGreater(torch.count_nonzero(actual_grads[1]).item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
