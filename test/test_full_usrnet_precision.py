"""Strict full-USRNet FP64 quality gate, separate from cache semantics.

This gate retains its original fixture and pointwise tolerances. A failure is
reported normally with a nonzero exit status; the known full-model precision
gap is neither skipped nor marked as an expected failure.
"""
import copy
import unittest

import torch
from extension_loader import load_extension
import test_fp32_training as common


class FullUSRNetPrecision(unittest.TestCase):
    assert_numerical_close = common.FP32Training.assert_numerical_close

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(9214)

    def test_full_usrnet_eval_with_gradients_matches_fp64(self):
        from models.converse_usrnet import ConverseUSRNet
        # Default five iterations/seven blocks; nonzero residual gates make
        # this exercise the internal solvers' gradients on a small image.
        actual_model = ConverseUSRNet(backend='cuda').cuda().eval()
        with torch.no_grad():
            for name, p in actual_model.named_parameters():
                if name.endswith(('alpha1', 'alpha2')):
                    p.fill_(.1)
        expected_model = copy.deepcopy(actual_model).double()
        for module in expected_model.modules():
            if hasattr(module, 'backend'):
                module.backend = 'pytorch'
        x = (torch.rand(1, 3, 4, 5, device='cuda')*.2).requires_grad_()
        kernel = torch.rand(1, 1, 7, 7, device='cuda')
        kernel = (kernel/kernel.sum((-2, -1), keepdim=True)).requires_grad_()
        rx, rk = (t.detach().double().requires_grad_() for t in (x, kernel))
        with torch.backends.cudnn.flags(allow_tf32=False):
            actual = actual_model(x, kernel, 2)
            expected = expected_model(rx, rk, 2)
            self.assertTrue(actual.requires_grad)
            self.assert_numerical_close(actual, expected, atol=3e-5, rtol=3e-4, label='full USRNet output')
            upstream = torch.randn_like(actual)/actual.numel()**.5
            actual_grads = torch.autograd.grad(actual, (x, kernel, *actual_model.parameters()), upstream)
            expected_grads = torch.autograd.grad(expected, (rx, rk, *expected_model.parameters()), upstream.double())
            gradient_names = ['input', 'input_kernel'] + [name for name, _ in actual_model.named_parameters()]
            for name, a, e in zip(gradient_names, actual_grads, expected_grads):
                self.assert_numerical_close(a, e, atol=3e-5, rtol=3e-4,
                                            label=f'full USRNet gradient/{name}')
            self.assertGreater(torch.count_nonzero(actual_grads[1]).item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
