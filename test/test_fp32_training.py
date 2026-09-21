"""Public FP32 training against an independent full-FFT FP64 reference."""
import copy
import itertools
import sys
import unittest

import torch
from torch.nn import functional as F
from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference


def with_nearest_prior(op, x, weight, bias, scale, eps):
    """Exercise the same differentiable prior construction as production models."""
    prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode='nearest')
    return op(x, prior, weight, bias, scale, eps)


class FP32Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(9214)

    def assert_numerical_close(self, actual, expected, *, atol, rtol, label):
        # Keep absolute and relative diagnostics even for near-zero gradients;
        # acceptance uses the fixed pointwise absolute + relative error budget.
        actual = actual.detach().double()
        expected = expected.detach().double()
        self.assertTrue(torch.isfinite(actual).all().item(), label + ': nonfinite actual')
        self.assertTrue(torch.isfinite(expected).all().item(), label + ': nonfinite reference')
        difference = actual - expected
        maximum = difference.abs().max().item()
        relative = (difference.norm() / expected.norm().clamp_min(1e-30)).item()
        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol,
            msg=lambda message: f'{label}: max_abs={maximum:.6g}, relative_l2={relative:.6g}\n{message}')

    def data(self, h, w, s, kb=1, kc=3):
        # Strided public inputs exercise the FFT preparation as well as solve.
        x = torch.randn(2, 3, w, h, device='cuda').transpose(-1, -2)
        p = torch.randn(2, 3, w*s, h*s, device='cuda').transpose(-1, -2)
        kh, kw = min(3, h*s), min(3, w*s)
        k = torch.rand(kb, kc, kh, kw, device='cuda') / (kh*kw)
        b = torch.randn(1, 3, 1, 1, device='cuda')
        return tuple(t.requires_grad_() for t in (x, p, k, b))

    def compare(self, args, s, nearest=False, higher=False, eps=1e-3):
        op = torch.ops.converse2d.forward
        ref = converse2d_reference
        expected_args = tuple(t.detach().double().requires_grad_(t.requires_grad) for t in args)
        actual = with_nearest_prior(op, *args, s, eps) if nearest else op(*args, s, eps)
        expected = with_nearest_prior(ref, *expected_args, s, eps) if nearest else ref(*expected_args, s, eps)
        self.assertEqual(actual.dtype, args[0].dtype)
        upstream = torch.randn_like(actual) / actual.numel()**0.5
        inputs = [t for t in args if t.requires_grad]
        refs = [t for t in expected_args if t.requires_grad]
        grads = torch.autograd.grad(actual, inputs, upstream, create_graph=higher)
        expected_grads = torch.autograd.grad(expected, refs, upstream.double(), create_graph=higher)
        self.assert_numerical_close(actual, expected, atol=3e-5, rtol=3e-5, label='output')
        for index, (a, e) in enumerate(zip(grads, expected_grads)):
            self.assertEqual(a.dtype, inputs[index].dtype)
            self.assert_numerical_close(a, e, atol=5e-5, rtol=5e-5, label=f'gradient {index}')
        if higher:
            # A directional second derivative checks the public dispatch and
            # connected recomputation, including frozen formal arguments.
            vectors = [torch.randn_like(g) for g in grads]
            aa = sum((g*v).sum() for g, v in zip(grads, vectors))
            ee = sum((g*v.double()).sum() for g, v in zip(expected_grads, vectors))
            # A solve is linear in x/prior when both parameters are frozen.
            # Its constant VJP correctly has no second-derivative graph.
            self.assertEqual(aa.requires_grad, ee.requires_grad)
            if not ee.requires_grad:
                return
            ag = torch.autograd.grad(aa, inputs, allow_unused=True)
            eg = torch.autograd.grad(ee, refs, allow_unused=True)
            for a, e in zip(ag, eg):
                if e is None:
                    self.assertIsNone(a)
                else:
                    torch.testing.assert_close(a.double(), e, atol=2e-3, rtol=2e-4)

    def test_shapes_broadcasts_and_nearest(self):
        for h, w, s in ((1,1,1), (1,5,2), (4,1,3), (5,6,1), (5,7,2), (6,8,3), (3,5,4), (3,4,5)):
            for kb, kc in ((1,1), (1,3), (2,1), (2,3)):
                for nearest in (False, True):
                    with self.subTest(h=h, w=w, s=s, kb=kb, kc=kc, nearest=nearest):
                        x, p, k, b = self.data(h, w, s, kb, kc)
                        self.compare((x,k,b) if nearest else (x,p,k,b), s, nearest)

    def test_partial_gradients_and_higher_order(self):
        for scale in (1, 2):
            for needs in itertools.product((False,True), repeat=4):
                if not any(needs):
                    continue
                args = tuple(t.detach().requires_grad_(need) for t, need in zip(self.data(3,4,scale), needs))
                with self.subTest(scale=scale, needs=needs):
                    self.compare(args, scale, higher=True)

    def test_large_and_rectangular_kernel_preparation(self):
        # Cover large transforms, odd/even centering, both broadcast axes and
        # a larger PSF. Use the same predeclared FP64 error budgets as above.
        for h, w, scale, kh, kw, kb, kc, higher in (
                (128, 128, 1, 3, 3, 1, 3, False),
                (127, 131, 1, 4, 2, 2, 1, False),
                (65, 67, 2, 7, 5, 2, 3, True),
                (129, 129, 1, 9, 9, 1, 1, False)):
            with self.subTest(h=h, w=w, scale=scale, kernel=(kh, kw), kb=kb, kc=kc):
                x, prior, _, bias = self.data(h, w, scale, kb, kc)
                weight = (torch.rand(kb, kc, kw, kh, device='cuda').transpose(-1, -2)
                          / (kh * kw)).requires_grad_()
                self.compare((x, prior, weight, bias), scale, higher=higher)

    def test_shared_prior_higher_order(self):
        x, _, k, b = self.data(3,4,1)
        # Production's nearest prior at scale=1 uses the same observation twice.
        self.compare((x,k,b), 1, nearest=True, higher=True)
        expected_x = x.detach().double().requires_grad_()
        expected_k = k.detach().double().requires_grad_()
        expected_b = b.detach().double().requires_grad_()
        a = torch.ops.converse2d.forward(x,x,k,b,1,1e-3)
        e = converse2d_reference(expected_x,expected_x,expected_k,expected_b,1,1e-3)
        g = torch.randn_like(a)
        for aa, ee in zip(torch.autograd.grad(a,(x,k,b),g),
                          torch.autograd.grad(e,(expected_x,expected_k,expected_b),g.double())):
            torch.testing.assert_close(aa.double(),ee,atol=5e-5,rtol=5e-5)

    def test_weak_regularization_and_zero_filter_against_fp64(self):
        # Thresholds match the established spatial gradient / weak-regularizer
        # tests. Neither epsilon nor tolerance depends on the measured error.
        for scale in (1, 3):
            for amplitude in (0., 1e-6, 1e-3):
                with self.subTest(scale=scale, amplitude=amplitude):
                    args = self.data(3, 4, scale, kb=1, kc=1)
                    with torch.no_grad():
                        args[0].mul_(1e-5)
                        args[1].mul_(1e-5)
                        args[2].mul_(amplitude)
                        args[3].fill_(-40.)
                    refs = tuple(t.detach().double().requires_grad_() for t in args)
                    actual = torch.ops.converse2d.forward(*args, scale, 1e-8)
                    expected = converse2d_reference(*refs, scale, 1e-8)
                    upstream = torch.randn_like(actual) * 1e-5
                    actual_grads = torch.autograd.grad(actual, args, upstream)
                    expected_grads = torch.autograd.grad(expected, refs, upstream.double())
                    self.assert_numerical_close(actual, expected, atol=1e-6, rtol=1e-5, label='weak output')
                    for index, (a, e) in enumerate(zip(actual_grads, expected_grads)):
                        self.assert_numerical_close(a, e, atol=5e-5, rtol=5e-5,
                                                    label=f'weak gradient {index}')

    def test_module_modes_grad_requirements_and_cache_transitions(self):
        from models.util_converse import Converse2D

        for scale, training, input_grad, parameter_grad in itertools.product(
                (1, 2), (False, True), (False, True), (False, True)):
            with self.subTest(scale=scale, training=training,
                              input_grad=input_grad, parameter_grad=parameter_grad):
                actual_layer = Converse2D(3, 3, 3, scale=scale, padding=1, backend='cuda').cuda()
                actual_layer.train(training).requires_grad_(parameter_grad)
                reference_layer = copy.deepcopy(actual_layer).double()
                reference_layer.backend = 'pytorch'
                x = torch.randn(2, 3, 5, 7, device='cuda', requires_grad=input_grad)
                rx = x.detach().double().requires_grad_(input_grad)
                # Populate the fixed-weight inference cache before grad-enabled
                # execution, including eval + input-only gradients.
                for _ in range(2):
                    with torch.no_grad():
                        actual = actual_layer(x)
                        expected = reference_layer(rx)
                    self.assertFalse(actual.requires_grad)
                    self.assert_numerical_close(actual, expected, atol=3e-5, rtol=3e-5,
                                                label='cached module output')
                actual = actual_layer(x)
                expected = reference_layer(rx)
                self.assertEqual(actual.requires_grad, input_grad or parameter_grad)
                self.assert_numerical_close(actual, expected, atol=3e-5, rtol=3e-5,
                                            label='grad-enabled module output')
                if actual.requires_grad:
                    inputs = tuple(t for t in (x, *actual_layer.parameters()) if t.requires_grad)
                    refs = tuple(t for t in (rx, *reference_layer.parameters()) if t.requires_grad)
                    upstream = torch.randn_like(actual) / actual.numel()**0.5
                    actual_grads = torch.autograd.grad(actual, inputs, upstream)
                    expected_grads = torch.autograd.grad(expected, refs, upstream.double())
                    for index, (a, e) in enumerate(zip(actual_grads, expected_grads)):
                        self.assert_numerical_close(a, e, atol=5e-5, rtol=5e-5,
                                                    label=f'module gradient {index}')
                # Mutate the exact tensors whose spectra were cached, then
                # return to no_grad and compare using their new FP32 values.
                with torch.no_grad():
                    actual_layer.weight.mul_(.98)
                    actual_layer.bias.add_(.02)
                    reference_layer.load_state_dict(actual_layer.state_dict())
                    actual = actual_layer(x)
                    expected = reference_layer(rx)
                self.assertFalse(actual.requires_grad)
                self.assert_numerical_close(actual, expected, atol=3e-5, rtol=3e-5,
                                            label='updated module output')

    def test_usrnet_sgd_and_gradient_accumulation_against_fp64(self):
        from models.converse_usrnet import ConverseUSRNet

        # This is a short synthetic integration regression, not convergence.
        # Retain the prior experiment's model shape and fixed error budget.
        with torch.backends.cudnn.flags(allow_tf32=False):
            for accumulation in (1, 2):
                with self.subTest(accumulation=accumulation):
                    actual = ConverseUSRNet(num_iterations=2, num_blocks=1, backend='cuda').cuda()
                    with torch.no_grad():
                        for name, parameter in actual.named_parameters():
                            if name.endswith(('alpha1', 'alpha2')):
                                parameter.fill_(.1)
                    expected = copy.deepcopy(actual).double()
                    for layer in expected.modules():
                        if hasattr(layer, 'backend'):
                            layer.backend = 'pytorch'
                    models = (actual, expected)
                    optimizers = tuple(torch.optim.SGD(model.parameters(), lr=1e-4, momentum=.9,
                                                        foreach=False, fused=False) for model in models)
                    initial = {name: p.detach().clone() for name, p in actual.named_parameters()}
                    batches = []
                    for _ in range(accumulation):
                        x = torch.rand(1, 3, 8, 10, device='cuda') * .2
                        kernel = torch.rand(1, 1, 7, 7, device='cuda')
                        kernel = kernel / kernel.sum((-2, -1), keepdim=True)
                        target = torch.rand(1, 3, 16, 20, device='cuda') * .2
                        batches.append((x, kernel, target))
                    for step in range(4):
                        losses = []
                        for model, optimizer in zip(models, optimizers):
                            optimizer.zero_grad(set_to_none=True)
                            loss_sum = 0.
                            dtype = next(model.parameters()).dtype
                            for x, kernel, target in batches:
                                # Round inputs once in FP32 before making the
                                # FP64 copy so both paths see identical values.
                                changed_x = x * (1 + .02 * step) + .003 * step
                                changed_target = target * (1 - .01 * step) + .001 * step
                                output = model(changed_x.to(dtype), kernel.to(dtype), 2)
                                loss = (output - changed_target.to(dtype)).square().mean() / accumulation
                                loss.backward()
                                loss_sum = loss_sum + loss.detach()
                            optimizer.step()
                            losses.append(loss_sum)
                        self.assert_numerical_close(*losses, atol=3e-5, rtol=3e-4,
                                                    label=f'USRNet loss step {step}')
                        for (name, a), (reference_name, e) in zip(actual.named_parameters(), expected.named_parameters()):
                            self.assertEqual(name, reference_name)
                            self.assertEqual(a.grad is None, e.grad is None, name)
                            if a.grad is None:
                                continue
                            for label, av, ev in (
                                    ('parameter', a, e), ('gradient', a.grad, e.grad),
                                    ('momentum', optimizers[0].state[a]['momentum_buffer'],
                                     optimizers[1].state[e]['momentum_buffer'])):
                                self.assert_numerical_close(av, ev, atol=3e-5, rtol=3e-4,
                                                            label=f'USRNet {label}/{name} step {step}')
                    parameters = dict(actual.named_parameters())
                    changed = {name for name, p in parameters.items() if not torch.equal(p, initial[name])}
                    nonzero = {name for name, p in parameters.items()
                               if p.grad is not None and torch.count_nonzero(p.grad).item() > 0}
                    solver_weights = {name + '.weight' for name, layer in actual.named_modules()
                                      if layer.__class__.__name__ == 'Converse2D'}
                    for group in (solver_weights, {'d.alpha'},
                                  {name for name in parameters if name.startswith('kernelnet.')}):
                        self.assertTrue(group & changed, f'parameters never updated: {group}')
                        self.assertTrue(group & nonzero, f'no effective gradient: {group}')

    def test_stream_and_repeated_parameter_updates(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            args = self.data(7,8,3)
            for _ in range(3):
                self.compare(args,3)
                with torch.no_grad():
                    args[2].mul_(0.98)
                    args[3].add_(0.02)
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == '__main__':
    unittest.main(verbosity=2)
