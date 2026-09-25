"""LayerNorm keeps its original statistics and affine gradient order."""
import unittest
import torch
from support import CUDATestCase


def reference(x, weight, bias, eps=1e-5):
    u = x.mean(1, keepdim=True)
    variance = (x-u).pow(2).mean(1, keepdim=True)
    normalized = (x-u)/torch.sqrt(variance+eps)
    return weight[:, None, None]*normalized+bias[:, None, None]


class LayerNormAffineCUDA(CUDATestCase):
    def test_stats_affine_and_all_gradient_subsets_keep_bytes(self):
        from models.util_converse import LayerNorm
        for channels in (3, 64, 128):
            generator = torch.Generator().manual_seed(61237+channels)
            raw = torch.randn(2, channels, 5, 7, generator=generator).cuda()
            upstream = torch.randn(raw.shape, generator=generator).cuda()
            for layout in ('contiguous', 'transpose', 'channels_last'):
                value = raw
                if layout == 'transpose':
                    value = value.transpose(-1, -2).contiguous().transpose(-1, -2)
                if layout == 'channels_last':
                    value = value.contiguous(memory_format=torch.channels_last)
                for mask in range(1, 8):
                    results = []
                    for fused in (False, True):
                        layer = LayerNorm(channels, eps=1e-5, data_format='channels_first').cuda()
                        x = value.detach().requires_grad_(bool(mask & 1))
                        layer.weight.requires_grad_(bool(mask & 2))
                        layer.bias.requires_grad_(bool(mask & 4))
                        output = layer(x) if fused else reference(x, layer.weight, layer.bias)
                        requested = [t for t in (x, layer.weight, layer.bias) if t.requires_grad]
                        results.append((output, *torch.autograd.grad(output, requested, upstream)))
                    with self.subTest(channels=channels, layout=layout, mask=mask):
                        self.assert_results_equal(*results)

    def test_shared_ancestor_second_and_third_derivatives(self):
        from models.util_converse import _channel_affine
        torch.manual_seed(61279)
        raw = torch.randn(2, 3, 3, 5, device='cuda')
        results = []
        for fused in (False, True):
            x = raw.detach().requires_grad_()
            u = x.mean(1, keepdim=True)
            normalized = (x-u)/torch.sqrt((x-u).square().mean(1, keepdim=True)+.2)
            scale = x.mean((0, 2, 3))[:, None, None]
            bias = x.square().mean((0, 2, 3))[:, None, None]
            out = _channel_affine(scale, normalized, bias) if fused else scale*normalized+bias
            loss = (out+x).sin().sum()
            g = torch.autograd.grad(loss, x, create_graph=True)[0]
            h = torch.autograd.grad(g.square().sum(), x, create_graph=True)[0]
            j = torch.autograd.grad(h.square().sum(), x)[0]
            results.append((out, g, h, j))
        # First-order release admission remains byte-exact. Higher-order uses
        # the existing ATen-fallback tolerance from test_fp32_release, not a
        # claim of identical higher-order accumulation graphs.
        self.assert_results_equal(results[0][:2], results[1][:2])
        for actual, expected in zip(results[0][2:], results[1][2:]):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_bias_only_prunes_unneeded_saved_values(self):
        scale = torch.randn(3, 1, 1, device='cuda', requires_grad=True)
        x = torch.randn(2, 3, 5, 7, device='cuda', requires_grad=True)
        bias = torch.randn(3, 1, 1, device='cuda', requires_grad=True)
        output = torch.ops.converse2d._channel_affine(scale, x, bias)
        with torch.no_grad():
            scale.add_(1)
            x.add_(1)
        gradient = torch.autograd.grad(output.sum(), bias)[0]
        self.assert_bytes_equal(gradient, torch.full_like(bias, 70), 'bias-only VJP')

    def test_training_statistics_remain_aten_and_full_inference_precedes_affine(self):
        from models.util_converse import LayerNorm
        layer = LayerNorm(64, data_format='channels_first').cuda()
        x = torch.randn(2, 64, 128, 128, device='cuda', requires_grad=True)
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            layer(x)
        counts = {event.key:event.count for event in trace.key_averages()}
        self.assertNotIn('aten::mean', counts)
        self.assertEqual(counts.get('converse2d::_channel_layernorm'), 1)
        self.assertNotIn('converse2d::_channel_affine', counts)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            layer(x)
        training = {event.key:event.count for event in trace.key_averages()}
        self.assertEqual(training.get('aten::mean'), 2)
        self.assertNotIn('converse2d::_channel_affine', training)
        self.assertNotIn('converse2d::_channel_layernorm', training)
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            layer(x[:, :, :5, :7].contiguous())
        small = {event.key:event.count for event in trace.key_averages()}
        self.assertNotIn('aten::mean', small)
        self.assertEqual(small.get('converse2d::_channel_layernorm'), 1)
        self.assertNotIn('converse2d::_channel_affine', small)


if __name__ == '__main__':
    unittest.main(verbosity=2)
