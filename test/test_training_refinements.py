"""Large scale-one specialization and dynamic-kernel preparation regressions.

Spatial comparisons reuse the established independent full-FFT FP64 oracle
and fixed tolerances. Complex tests use independent complex128 ATen graphs.
"""
import itertools
import unittest

import torch
from extension_loader import load_extension
import test_fp32_training as spatial
import test_training_fusion as spectral
from models.converse_core import converse2d_reference


class TrainingRefinements(unittest.TestCase):
    data = spatial.FP32Training.data
    compare = spatial.FP32Training.compare
    assert_numerical_close = spatial.FP32Training.assert_numerical_close

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(9214)

    def test_scale_one_threshold_and_broadcasts(self):
        # 255*257 == 65535 stays generic; 256*256 reaches the specialization.
        # The odd-width case also exercises the last non-Nyquist half bin.
        for height, width in ((255, 257), (256, 256), (256, 257)):
            broadcasts = ((1, 1),) if height*width < 65536 else ((1, 1), (1, 3), (2, 1), (2, 3))
            for kb, kc in broadcasts:
                for shared in (False, True):
                    with self.subTest(shape=(height, width), kb=kb, kc=kc, shared=shared):
                        x, prior, weight, bias = self.data(height, width, 1, kb, kc)
                        self.compare((x, weight, bias) if shared else (x, prior, weight, bias),
                                     1, nearest=shared)

    def test_scale_one_selective_gradients(self):
        # Large tensors are necessary: the existing small mask matrix exercises
        # the generic backend, not the H*W >= 65536 scale-one CUDA branch.
        for needs in itertools.product((False, True), repeat=4):
            if any(needs):
                with self.subTest(needs=needs):
                    args = tuple(t.detach().requires_grad_(need)
                                 for t, need in zip(self.data(256, 256, 1, 1, 1), needs))
                    self.compare(args, 1)

    def test_scale_one_higher_order_and_shared_input(self):
        for needs in ((True, True, True, True), (False, True, True, True), (True, True, False, False)):
            with self.subTest(needs=needs):
                args = tuple(t.detach().requires_grad_(need)
                             for t, need in zip(self.data(256, 256, 1), needs))
                self.compare(args, 1, higher=True)
        x, _, weight, bias = self.data(256, 257, 1, 1, 1)
        self.compare((x, weight, bias), 1, nearest=True, higher=True)

    def check_spectral(self, values, height, width, *, shared=False, higher=False):
        arguments = (values[0], values[2], values[3]) if shared else values
        refs = tuple(t.detach().to(torch.complex128 if t.is_complex() else torch.float64)
                     .requires_grad_() for t in arguments)
        actual_args = (arguments[0], arguments[0], *arguments[1:]) if shared else arguments
        expected_args = (refs[0], refs[0], *refs[1:]) if shared else refs
        actual = torch.ops.converse2d._training_spectral(*actual_args, height, width, 1)
        expected = spectral.spectral_reference(*expected_args, height, width, 1)
        upstream = (torch.randn_like(actual)/actual.numel()**.5).conj()
        actual_grads = torch.autograd.grad(actual, arguments, upstream, create_graph=higher)
        expected_grads = torch.autograd.grad(expected, refs, upstream.cdouble(), create_graph=higher)
        double = values[0].dtype == torch.complex128
        torch.testing.assert_close(actual.cdouble(), expected,
                                   atol=1e-12 if double else 1e-5, rtol=1e-12 if double else 1e-5)
        for a, e in zip(actual_grads, expected_grads):
            self.assertTrue(torch.isfinite(a).all().item())
            torch.testing.assert_close(a.to(e.dtype), e,
                                       atol=1e-11 if double else 2e-5, rtol=1e-11 if double else 2e-5)
        if higher:
            vectors = [torch.randn_like(g) for g in actual_grads]
            actual_direction = sum((g.conj()*v).real.sum() for g, v in zip(actual_grads, vectors))
            expected_direction = sum((g.conj()*v.to(g.dtype)).real.sum()
                                     for g, v in zip(expected_grads, vectors))
            aa = torch.autograd.grad(actual_direction, arguments)
            ee = torch.autograd.grad(expected_direction, refs)
            for a, e in zip(aa, ee):
                torch.testing.assert_close(a.to(e.dtype), e,
                                           atol=1e-10 if double else 2e-3, rtol=1e-10 if double else 2e-4)

    def test_scale_one_stream_strides_and_conjugate_views(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for dtype in (torch.complex64, torch.complex128):
                for shared in (False, True):
                    with self.subTest(dtype=dtype, shared=shared):
                        # Independent arbitrary complex spectra catch errors
                        # hidden by the Hermitian constraints of real inputs.
                        values = spectral.TrainingFusion.data(self, 256, 257, 1,
                                                              kb=1, kc=1, dtype=dtype)
                        values = tuple(torch.stack((t, t), -1)[..., 0].conj()
                                       if t.is_complex() else torch.stack((t, t), -1)[..., 0]
                                       for t in values)
                        self.check_spectral(values, 256, 257, shared=shared, higher=True)
        torch.cuda.current_stream().wait_stream(stream)

    def test_dynamic_kernel_preparation_total_size_boundary(self):
        # Both spatial areas are below 16k, while KB*KC*H*W straddles 1M.
        # The full per-sample, per-channel kernel tensor is trainable.
        for height, width in ((80, 96), (96, 96)):
            with self.subTest(batch=16, channels=8, height=height, width=width):
                x = torch.randn(16, 8, width, height, device='cuda').transpose(-1, -2)
                prior = torch.randn_like(x)
                weight = torch.rand(16, 8, 3, 3, device='cuda')/9
                bias = torch.randn(1, 8, 1, 1, device='cuda')
                args = tuple(t.requires_grad_() for t in (x, prior, weight, bias))
                self.compare(args, 1)

    def test_fp64_public_precision_is_preserved(self):
        args = tuple(t.detach().double().requires_grad_() for t in self.data(256, 256, 1, 2, 1))
        refs = tuple(t.detach().clone().requires_grad_() for t in args)
        actual = torch.ops.converse2d.forward(*args, 1, 1e-3)
        expected = converse2d_reference(*refs, 1, 1e-3)
        upstream = torch.randn_like(actual)/actual.numel()**.5
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        actual_grads = torch.autograd.grad(actual, args, upstream)
        expected_grads = torch.autograd.grad(expected, refs, upstream)
        for a, e in zip(actual_grads, expected_grads):
            self.assertEqual(a.dtype, torch.float64)
            torch.testing.assert_close(a, e, atol=1e-9, rtol=1e-9)

    def test_large_scale_one_weak_regularization(self):
        # Exercise the specialized branch with the same predeclared weak-
        # regularizer budgets as the original public training suite.
        for amplitude in (0., 1e-6, 1e-3):
            with self.subTest(amplitude=amplitude):
                args = self.data(256, 256, 1, 1, 1)
                with torch.no_grad():
                    args[0].mul_(1e-5)
                    args[1].mul_(1e-5)
                    args[2].mul_(amplitude)
                    args[3].fill_(-40.)
                refs = tuple(t.detach().double().requires_grad_() for t in args)
                actual = torch.ops.converse2d.forward(*args, 1, 1e-8)
                expected = converse2d_reference(*refs, 1, 1e-8)
                upstream = torch.randn_like(actual)*1e-5
                actual_grads = torch.autograd.grad(actual, args, upstream)
                expected_grads = torch.autograd.grad(expected, refs, upstream.double())
                self.assert_numerical_close(actual, expected, atol=1e-6, rtol=1e-5,
                                            label='large s1 weak output')
                for index, (a, e) in enumerate(zip(actual_grads, expected_grads)):
                    self.assert_numerical_close(a, e, atol=5e-5, rtol=5e-5,
                                                label=f'large s1 weak gradient {index}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
