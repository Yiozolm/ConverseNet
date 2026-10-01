"""Production integration gates for the measured B4 pointwise weight VJP."""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from support import CUDATestCase
from numerical_policy import comparison
from models.pointwise import PointwiseConv2d


ACTIVE_NODE = 'PointwiseWeightGradientBackward'


def fixture(seed=4101):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(4, 128, 96, 96, generator=generator),
            torch.randn(64, 128, 1, 1, generator=generator) / 128 ** .5,
            torch.randn(64, generator=generator) * .1,
            torch.randn(4, 64, 96, 96, generator=generator) / (4 * 96 * 96) ** .5)


def evaluate(raw, needs, *, production=False, dtype=torch.float32, higher=False):
    data = tuple(None if value is None else value.to(device='cuda', dtype=dtype).detach().requires_grad_(need)
                 for value, need in zip(raw[:3], needs))
    x, weight, bias = data
    upstream = raw[3].to(device='cuda', dtype=dtype)
    if production:
        layer = PointwiseConv2d(128, 64, 1, bias=bias is not None, backend='cuda').cuda()
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


class PointwiseWgrad(CUDATestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'CONVERSE2D_BACKEND': ''})
        environment.start()
        self.addCleanup(environment.stop)

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
        raw = fixture()
        for bits in range(8):
            needs = tuple(bool(bits & (1 << index)) for index in range(3))
            with self.subTest(needs=needs):
                self.assert_three_way(raw, needs)

    def test_no_bias_and_noncontiguous_upstream(self):
        x, weight, bias, upstream = fixture(4102)
        self.assert_three_way((x, weight, None, upstream), (True, True, False))
        upstream = upstream.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.assertFalse(upstream.is_contiguous())
        self.assert_three_way((x, weight, bias, upstream))

    def test_higher_order_uses_native_aten(self):
        # First-order forward really selects the custom node, while
        # create_graph=True must bypass GEMM for the entire backward.
        raw = fixture(4103)
        self.assert_three_way(raw, higher=True)
        with patch('torch.Tensor.__matmul__', side_effect=AssertionError('GEMM entered higher-order backward')):
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

        def native_route(value):
            expected = native(value)
            actual = layer(value)
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
        native_route(x[:1])
        native_route(x[:, :, :95, :])
        native_route(x.transpose(-1, -2))
        native_route(x.contiguous(memory_format=torch.channels_last))
        native_route(torch._neg_view(x))
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
                    for branch in ('conv1.5', 'conv2.3')}
        native_layers = {name: layer for name, layer in native_model.named_modules()
                         if isinstance(layer, PointwiseConv2d)}
        candidate_layers = {name: layer for name, layer in candidate_model.named_modules()
                            if isinstance(layer, PointwiseConv2d)}
        self.assertEqual(set(native_layers), expected)
        self.assertEqual(set(candidate_layers), expected)
        self.assertTrue(all(layer.backend == 'pytorch' for layer in native_layers.values()))
        self.assertTrue(all(layer.backend == 'cuda' for layer in candidate_layers.values()))
        x = fixture(4105)[0].cuda().requires_grad_()
        name = sorted(expected)[0]
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
