"""Production fused-lambda nearest k2/s2 accuracy, routing and lifetime gates."""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from support import ROOT, CUDATestCase, ExtensionTestCase, has_full_solve
from fp32_baseline import converse2d_fp32 as frozen_half
from fp32_baseline import converse2d_reference as frozen_full
from models.converse_core import converse2d_reference as reference64
from numerical_policy import comparison


OP = 'converse2d::_nearest_k2_s2'


def data(device='cuda', *, kb=1, kc=3):
    generator = torch.Generator().manual_seed(43017 + kb + kc)
    return (torch.randn(2, 3, 5, 7, generator=generator).to(device),
            (torch.randn(kb, kc, 2, 2, generator=generator) * .25).to(device),
            (torch.randn(1, 3, 1, 1, generator=generator) * .2).to(device))


def solve(fn, values, eps=1e-5):
    x, weight, bias = values
    return fn(x, F.interpolate(x, scale_factor=2, mode='nearest'), weight, bias, 2, eps)


def production(values, eps=1e-5, variant='v7'):
    return torch.ops.converse2d._nearest_k2_s2(*values, eps, variant)


def traced(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        out = call()
    return out, {event.key: event.count for event in trace.key_averages()}


def check_budget(test, actual, values, eps=1e-5):
    with torch.no_grad():
        baseline = solve(frozen_half, values, eps)
        high = solve(reference64, tuple(value.double() for value in values), eps)
    result = comparison(actual, baseline, high, regime='weak' if eps < 1e-5 else 'normal')
    test.assertTrue(result['passed'], result)
    if eps >= 1e-5:
        torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=1e-5)


class NearestK2Portable(ExtensionTestCase):
    def test_cpu_direct_api_retains_original_operator(self):
        values = data('cpu')
        with torch.no_grad():
            actual = production(values)
            check_budget(self, actual, values)
        values = tuple(value.detach().requires_grad_() for value in values)
        actual = production(values)
        baseline = solve(torch.ops.converse2d.forward, values)
        upstream = torch.ones_like(actual)
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        ag = torch.autograd.grad(actual, values, upstream)
        bg = torch.autograd.grad(baseline, values, upstream)
        for a, b in zip(ag, bg):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)

    def test_cpu_auto_module_does_not_select_cuda_helper(self):
        from models.util_converse import Converse2D
        layer = Converse2D(3, 3, 2, scale=2, padding=0, backend='auto')
        with torch.no_grad(), patch.dict(os.environ, {'CONVERSE2D_BACKEND': ''}):
            actual, counts = traced(lambda: layer(data('cpu')[0]))
            self.assertEqual(counts.get(OP, 0), 0)
            check_budget(self, actual, (data('cpu')[0], layer.weight, layer.bias))


class NearestK2CUDA(CUDATestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'CONVERSE2D_BACKEND': ''})
        environment.start()
        self.addCleanup(environment.stop)

    def test_actual_operator_all_inference_modes_and_broadcasts(self):
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            for mode, context in (('no_grad', torch.no_grad), ('inference_mode', torch.inference_mode),
                                  ('frozen', torch.enable_grad)):
                with self.subTest(kb=kb, kc=kc, mode=mode):
                    values = tuple(value.requires_grad_(mode != 'frozen') for value in data(kb=kb, kc=kc))
                    with context():
                        actual, counts = traced(lambda: production(values))
                    self.assertEqual(counts.get(OP, 0), 1)
                    self.assertNotIn('aten::fft_fft2', counts)
                    self.assertNotIn('aten::fft_rfft2', counts)
                    self.assertNotIn('aten::sigmoid', counts)
                    self.assertFalse(actual.requires_grad)
                    check_budget(self, actual, values)

    def test_each_differentiable_input_retains_full_training(self):
        from models.util_converse import Converse2D
        for requested in range(3):
            with self.subTest(requested=requested):
                layer = Converse2D(3, 3, 2, scale=2, padding=0, backend='cuda').cuda()
                layer.weight.requires_grad_(requested == 1)
                layer.bias.requires_grad_(requested == 2)
                x = data()[0].requires_grad_(requested == 0)
                values = x, layer.weight, layer.bias
                actual, counts = traced(lambda: layer(x))
                self.assertEqual(counts.get(OP, 0), 0)
                self.assertTrue(has_full_solve(actual))
                direct = production(values)
                self.assertTrue(has_full_solve(direct))
                baseline = solve(torch.ops.converse2d.forward, values)
                targets = [value for value in values if value.requires_grad]
                upstream = torch.ones_like(actual)
                for result in (actual, direct):
                    torch.testing.assert_close(result, baseline, atol=0, rtol=0)
                    gradients = torch.autograd.grad(result, targets, upstream)
                    control = torch.autograd.grad(baseline, targets, upstream, retain_graph=True)
                    for a, b in zip(gradients, control):
                        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)

    def test_layouts_negative_metadata_and_padding(self):
        from models.util_converse import Converse2D
        for layout in ('transpose', 'strided', 'channels_last', 'negative'):
            values = list(data())
            if layout == 'transpose':
                values = [value.transpose(-1, -2).contiguous().transpose(-1, -2) for value in values]
            elif layout == 'strided':
                values = [torch.stack((value, value), -1)[..., 0] for value in values]
            elif layout == 'channels_last':
                values[0] = values[0].contiguous(memory_format=torch.channels_last)
            else:
                values = [torch._neg_view(value) for value in values]
            with self.subTest(layout=layout), torch.no_grad():
                check_budget(self, production(values), values)
        for padding_mode in ('replicate', 'reflect', 'circular', 'constant'):
            with self.subTest(padding=padding_mode), torch.no_grad():
                layer = Converse2D(3, 3, 2, scale=2, padding=2,
                                   padding_mode=padding_mode, backend='cuda').cuda()
                x = data()[0]
                actual, counts = traced(lambda: layer(x))
                self.assertEqual(counts.get(OP, 0), 1)
                padded = F.pad(x, (2, 2, 2, 2), mode=padding_mode, value=0)
                values = padded, layer.weight, layer.bias
                baseline = solve(frozen_half, values)[..., 4:-4, 4:-4]
                high = solve(reference64, tuple(value.double() for value in values))[..., 4:-4, 4:-4]
                check = comparison(actual, baseline, high)
                self.assertTrue(check['passed'], check)
                torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=1e-5)

    def test_backend_and_non_target_routes_remain_native(self):
        from models.util_converse import Converse2D
        x = data()[0]
        layer = Converse2D(3, 3, 2, scale=2, padding=0, backend='pytorch').cuda()
        with torch.no_grad():
            result, counts = traced(lambda: layer(x))
            self.assertEqual(counts.get(OP, 0), 0)
            self.assertIn('aten::fft_fft2', counts)
            layer.backend = 'cuda'
            with patch.dict(os.environ, {'CONVERSE2D_BACKEND': 'pytorch'}):
                result, counts = traced(lambda: layer(x))
                self.assertEqual(counts.get(OP, 0), 0)
            for kernel_size, scale in ((2, 1), (3, 2), (2, 3)):
                other = Converse2D(3, 3, kernel_size, scale=scale, padding=0, backend='cuda').cuda()
                result, counts = traced(lambda: other(x))
                self.assertEqual(counts.get(OP, 0), 0)
            values = data()
            arbitrary_prior = torch.randn(2, 3, 10, 14, device='cuda')
            _, counts = traced(lambda: torch.ops.converse2d.forward(values[0], arbitrary_prior,
                                                                   values[1], values[2], 2))
            self.assertEqual(counts.get(OP, 0), 0)
            self.assertIn('aten::fft_rfft2', counts)

    def test_public_contracts(self):
        values = data()
        with torch.no_grad():
            for dtype in (torch.float64, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype), self.assertRaisesRegex(RuntimeError, 'FP32'):
                    production(tuple(value.to(dtype) for value in values))
            for eps in (0., -1., float('nan'), float('inf')):
                with self.subTest(eps=eps), self.assertRaisesRegex(RuntimeError, 'eps'):
                    production(values, eps)
            with self.assertRaisesRegex(RuntimeError, 'v7'):
                production(values, variant='v6')
            with torch.autocast('cuda'), self.assertRaisesRegex(RuntimeError, 'autocast'):
                production(values)

    def test_graph_replay_observes_input_weight_and_bias_updates(self):
        values = data()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                production(values)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
                actual = production(values)
            with torch.no_grad():
                for _ in range(2):
                    values[0].add_(.02)
                    values[1].mul_(.9)
                    values[2].add_(.1)
                    graph.replay()
                    torch.cuda.synchronize()
                    check_budget(self, actual, values)
                    self.assertTrue(torch.equal(actual, production(values)))
        finally:
            graph.reset()

    def test_pretrained_srresnet_selects_exactly_two_nearest_calls(self):
        from models.converse_srresnet import ConverseMSRResNet
        model = ConverseMSRResNet().cuda().eval()
        model.load_state_dict(torch.load(ROOT / 'model_zoo/converse_srresnet.pth',
                                         map_location='cuda', weights_only=True), strict=True)
        generator = torch.Generator(device='cuda').manual_seed(43043)
        x = torch.rand(1, 3, 24, 28, generator=generator, device='cuda')
        with torch.inference_mode():
            for module in model.modules():
                if hasattr(module, 'backend'):
                    module.backend = 'pytorch'
            with patch('models.util_converse.converse2d_reference', frozen_full):
                baseline = model(x)
            for module in model.modules():
                if hasattr(module, 'backend'):
                    module.backend = 'cuda'
            actual, counts = traced(lambda: model(x))
        self.assertEqual(counts.get(OP, 0), 2)
        torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
