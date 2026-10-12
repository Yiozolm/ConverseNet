"""Production integration gates for the split-K pointwise weight VJP."""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from support import CUDATestCase
from numerical_policy import comparison
from models.pointwise import PointwiseConv2d


ACTIVE_NODE = 'PointwiseWeightGradientBackward'


def fixture(seed=4101, cin=128, cout=64, batch=4, side=96):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, cin, side, side, generator=generator),
            torch.randn(cout, cin, 1, 1, generator=generator) / cin ** .5,
            torch.randn(cout, generator=generator) * .1,
            torch.randn(batch, cout, side, side, generator=generator) / (batch * side * side) ** .5)


DIRECTIONS = ((128, 64), (64, 128))


def evaluate(raw, needs, *, production=False, dtype=torch.float32, higher=False):
    data = tuple(None if value is None else value.to(device='cuda', dtype=dtype).detach().requires_grad_(need)
                 for value, need in zip(raw[:3], needs))
    x, weight, bias = data
    upstream = raw[3].to(device='cuda', dtype=dtype)
    if production:
        layer = PointwiseConv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None, backend='cuda').cuda()
        layer.weight = torch.nn.Parameter(weight, requires_grad=needs[1])
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=needs[2])
        data = x, layer.weight, layer.bias
        output = layer(x)
    else:
        output = F.conv2d(x, weight, bias)
    active = type(output.grad_fn).__name__ == ACTIVE_NODE
    targets = [(name, value) for name, value in zip(('dx', 'dweight', 'dbias'), data)
               if value is not None and value.requires_grad]
    gradients = torch.autograd.grad(output, [value for _, value in targets], upstream,
                                    create_graph=higher) if targets else ()
    result = {'output': output, **{name: value for (name, _), value in zip(targets, gradients)}}
    if higher:
        scalar = sum(value.square().sum() for value in gradients if value.requires_grad)
        second = torch.autograd.grad(scalar, [value for _, value in targets], allow_unused=True)
        result.update({'d' + name: value for (name, _), value in zip(targets, second) if value is not None})
    return result, active


def native_wgrad(x, grad_output, dtype=torch.float32):
    xd = x.to(dtype).requires_grad_(True)
    weight = torch.zeros(grad_output.shape[1], x.shape[1], 1, 1, device=x.device, dtype=dtype, requires_grad=True)
    return torch.autograd.grad(F.conv2d(xd, weight), weight, grad_output.to(dtype))[0]


class PointwiseWgrad(CUDATestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'CONVERSE2D_BACKEND': ''})
        environment.start()
        self.addCleanup(environment.stop)
        # Match the declared admission protocol. The native cuDNN FP32 weight
        # gradient otherwise uses a nondeterministic reduction: its error
        # budget changes between identical calls while the split-K kernel and
        # FP64 stay fixed.
        cudnn = torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True,
                                         allow_tf32=False)
        cudnn.__enter__()
        self.addCleanup(cudnn.__exit__, None, None, None)

    def assert_three_way(self, raw, needs=(True, True, True), *, higher=False):
        reference, _ = evaluate(raw, needs, dtype=torch.float64, higher=higher)
        baseline, _ = evaluate(raw, needs, higher=higher)
        actual, active = evaluate(raw, needs, production=True, higher=higher)
        self.assertEqual(active, needs[1])
        self.assertEqual(reference.keys(), actual.keys())
        self.assertEqual(reference.keys(), baseline.keys())
        for name in reference:
            check = comparison(actual[name], baseline[name], reference[name])
            self.assertTrue(check['passed'], f'{name}: {check}')

    def test_active_target_all_gradient_masks(self):
        for cin, cout in DIRECTIONS:
            raw = fixture(cin=cin, cout=cout)
            for bits in range(8):
                needs = tuple(bool(bits & (1 << index)) for index in range(3))
                with self.subTest(direction=(cin, cout), needs=needs):
                    self.assert_three_way(raw, needs)

    def test_no_bias_and_noncontiguous_upstream(self):
        for cin, cout in DIRECTIONS:
            x, weight, bias, upstream = fixture(4102, cin=cin, cout=cout)
            with self.subTest(direction=(cin, cout)):
                self.assert_three_way((x, weight, None, upstream), (True, True, False))
                upstream = upstream.transpose(-1, -2).contiguous().transpose(-1, -2)
                self.assertFalse(upstream.is_contiguous())
                self.assert_three_way((x, weight, bias, upstream))
                self.assert_three_way((x, weight, bias, torch._neg_view(upstream.contiguous())))

    def test_batch_one_and_other_geometries(self):
        for cin, cout, batch, side in ((64, 128, 1, 96), (128, 64, 1, 96), (64, 128, 2, 48), (128, 64, 3, 40)):
            with self.subTest(direction=(cin, cout), batch=batch, side=side):
                self.assert_three_way(fixture(4106 + batch, cin=cin, cout=cout, batch=batch, side=side))

    def test_operator_tails_bounds_and_determinism(self):
        # Raw operator: K tails, batch boundaries inside a K tile, partial and
        # multiple output tiles, against the deterministic cuDNN baseline.
        for batch, cin, h, w, cout in ((3, 64, 7, 5, 128), (3, 128, 5, 7, 64), (2, 3, 10, 10, 5),
                                       (1, 200, 17, 17, 130), (5, 130, 9, 9, 200), (1, 64, 1, 1, 128)):
            with self.subTest(geometry=(batch, cin, h, w, cout)):
                generator = torch.Generator().manual_seed(4107 + cin + cout)
                x = torch.randn(batch, cin, h, w, generator=generator).cuda()
                grad_output = (torch.randn(batch, cout, h, w, generator=generator) / (batch * h * w) ** .5).cuda()
                actual = torch.ops.converse2d._pointwise_wgrad(grad_output, x)
                self.assertEqual(tuple(actual.shape), (cout, cin, 1, 1))
                self.assertTrue(torch.equal(actual, torch.ops.converse2d._pointwise_wgrad(grad_output, x)))
                check = comparison(actual, native_wgrad(x, grad_output), native_wgrad(x, grad_output, torch.float64))
                self.assertTrue(check['passed'], f'{check}')

    def test_operator_contracts(self):
        x = torch.zeros(2, 64, 4, 4, device='cuda')
        grad_output = torch.zeros(2, 128, 4, 4, device='cuda')
        with self.assertRaisesRegex(RuntimeError, 'FP32'):
            torch.ops.converse2d._pointwise_wgrad(grad_output.double(), x.double())
        with self.assertRaisesRegex(RuntimeError, 'NCHW'):
            torch.ops.converse2d._pointwise_wgrad(grad_output[:1], x)
        with self.assertRaisesRegex(RuntimeError, 'contiguous CUDA'):
            torch.ops.converse2d._pointwise_wgrad(grad_output.cpu(), x.cpu())
        with self.assertRaisesRegex(RuntimeError, 'contiguous CUDA'):
            torch.ops.converse2d._pointwise_wgrad(grad_output.transpose(-1, -2), x.transpose(-1, -2))
        with self.assertRaisesRegex(RuntimeError, 'contiguous CUDA'):
            torch.ops.converse2d._pointwise_wgrad(torch._neg_view(grad_output), x)

    def test_higher_order_uses_native_aten(self):
        # First-order forward really selects the custom node, while
        # create_graph=True must bypass the kernel for the entire backward.
        for cin, cout in DIRECTIONS:
            raw = fixture(4103, cin=cin, cout=cout)
            with self.subTest(direction=(cin, cout)):
                self.assert_three_way(raw, higher=True)
                with patch.object(torch.ops.converse2d, '_pointwise_wgrad',
                                  side_effect=AssertionError('kernel entered higher-order backward')):
                    result, active = evaluate(raw, (True, True, True), production=True, higher=True)
                self.assertTrue(active)
                self.assertIn('ddweight', result)
                self.assertIn('ddx', result)

    def test_native_fallbacks_and_state_dict(self):
        native = torch.nn.Conv2d(128, 64, 1).cuda()
        layer = PointwiseConv2d(128, 64, 1).cuda()
        layer.load_state_dict(native.state_dict(), strict=True)
        self.assertEqual(set(layer.state_dict()), {'weight', 'bias'})
        native.load_state_dict(layer.state_dict(), strict=True)
        x = fixture(4104)[0].cuda().requires_grad_()
        self.assertEqual(type(layer(x).grad_fn).__name__, ACTIVE_NODE)
        self.assertEqual(type(layer(x[:1]).grad_fn).__name__, ACTIVE_NODE)

        def native_route(value, module=layer, reference=native):
            expected = reference(value)
            actual = module(value)
            self.assertNotEqual(type(actual.grad_fn).__name__, ACTIVE_NODE)
            self.assertTrue(torch.equal(actual, expected))

        # Both explicit reference controls must remain independent native paths.
        layer.backend = 'pytorch'
        native_route(x)
        layer.backend = 'cuda'
        with patch.dict(os.environ, {'CONVERSE2D_BACKEND': 'pytorch'}):
            native_route(x)
        with torch.no_grad():
            native_route(x)
        with torch.inference_mode():
            native_route(x)
        layer.weight.requires_grad_(False)
        native_route(x)
        layer.weight.requires_grad_(True)
        native_route(x[:, :, :95, :])
        native_route(x.transpose(-1, -2))
        native_route(x.contiguous(memory_format=torch.channels_last))
        native_route(torch._neg_view(x))
        other = torch.nn.Conv2d(128, 32, 1).cuda()
        narrow = PointwiseConv2d(128, 32, 1).cuda()
        narrow.load_state_dict(other.state_dict(), strict=True)
        native_route(x, narrow, other)
        layer = layer.cpu()
        native = native.cpu()
        native_route(x.detach().cpu())

    def test_rejects_non_fp32_public_tensors(self):
        for dtype in (torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer = PointwiseConv2d(128, 64, 1).to(dtype=dtype)
                with self.assertRaisesRegex(ValueError, 'FP32'):
                    layer(torch.zeros(1, 128, 1, 1, dtype=dtype))

    def test_usrnet_backend_propagation_and_unchanged_plain_block(self):
        from models.converse_usrnet import ConverseUSRNet
        from models.util_converse import ConverseBlock
        native_model = ConverseUSRNet(backend='pytorch').cuda()
        candidate_model = ConverseUSRNet(backend='cuda').cuda()
        candidate_model.load_state_dict(native_model.state_dict(), strict=True)
        expected = {f'p.m_body.{index}.{branch}' for index in range(7)
                    for branch in ('conv1.1', 'conv1.5', 'conv2.1', 'conv2.3')}
        native_layers = {name: layer for name, layer in native_model.named_modules()
                         if isinstance(layer, PointwiseConv2d)}
        candidate_layers = {name: layer for name, layer in candidate_model.named_modules()
                            if isinstance(layer, PointwiseConv2d)}
        self.assertEqual(set(native_layers), expected)
        self.assertEqual(set(candidate_layers), expected)
        self.assertTrue(all(layer.backend == 'pytorch' for layer in native_layers.values()))
        self.assertTrue(all(layer.backend == 'cuda' for layer in candidate_layers.values()))
        for name, cin in (('p.m_body.0.conv1.1', 64), ('p.m_body.0.conv1.5', 128)):
            x = fixture(4105, cin=cin, cout=192 - cin)[0].cuda().requires_grad_()
            before = native_layers[name](x)
            actual = candidate_layers[name](x)
            after = native_layers[name](x)
            self.assertNotEqual(type(before.grad_fn).__name__, ACTIVE_NODE)
            self.assertEqual(type(actual.grad_fn).__name__, ACTIVE_NODE)
            self.assertNotEqual(type(after.grad_fn).__name__, ACTIVE_NODE)
            self.assertTrue(torch.equal(actual, before))
            self.assertTrue(torch.equal(before, after))
        self.assertFalse(any(isinstance(layer, PointwiseConv2d) for layer in ConverseBlock(64, 64).modules()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
