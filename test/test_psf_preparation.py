"""Differentiable PSF movement preserves layout, gradient order and lifetime."""
import unittest

import torch

from support import CUDATestCase, fixture, leaves
from models.converse_core import converse2d_reference


class PSFPreparationCUDA(CUDATestCase):
    def test_kernel_layouts_and_padding_boundaries_keep_bytes(self):
        cases = ((1, 1, 1, 1, 1), (1, 3, 5, 3, 5), (1, 3, 6, 3, 2),
                 (1, 4, 5, 2, 5), (2, 2, 3, 4, 6), (3, 3, 4, 2, 3), (4, 2, 3, 3, 4))
        for scale, h, w, kh, kw in cases:
            generator = torch.Generator().manual_seed(103731 + scale + h + kw)
            raw = [torch.randn(2, 3, h, w, generator=generator),
                   torch.randn(2, 3, h * scale, w * scale, generator=generator),
                   torch.randn(1, 3, kh, kw, generator=generator) / (kh * kw),
                   torch.randn(1, 3, 1, 1, generator=generator)]
            up = torch.randn(raw[1].shape, generator=generator).cuda()
            for layout in ("contiguous", "transpose", "channels_last", "slice", "expand", "negative"):
                for need_kernel in (False, True):
                    with self.subTest(shape=(scale, h, w, kh, kw), layout=layout, need_kernel=need_kernel):
                        results = []
                        for call in (torch.ops.converse2d.forward, converse2d_reference):
                            data = leaves(raw, "cuda")
                            weight = data[2].detach()
                            if layout == "transpose":
                                weight = weight.transpose(-1, -2).contiguous().transpose(-1, -2)
                            elif layout == "channels_last":
                                weight = weight.contiguous(memory_format=torch.channels_last)
                            elif layout == "slice":
                                weight = torch.stack((weight, weight), -1)[..., 0]
                            elif layout == "expand":
                                weight = weight.expand(2, -1, -1, -1)
                            elif layout == "negative":
                                weight = torch._neg_view(weight)
                                self.assertTrue(weight.is_neg())
                            data[2] = weight.detach().requires_grad_(need_kernel)
                            requested = [value for value in data if value.requires_grad]
                            output = call(*data, scale, .1)
                            gradients = torch.autograd.grad(output, requested, up)
                            if need_kernel:
                                self.assertTrue(gradients[2].is_contiguous(), "upstream kernel VJP layout")
                            results.append((output, *gradients))
                        self.assert_results_equal(*results)

    def test_shared_ancestor_and_dynamic_kernel_keep_gradient_order(self):
        for scale in (1, 2, 3):
            generator = torch.Generator().manual_seed(103791 + scale)
            base_raw = torch.randn(2, 3, 5, 7, generator=generator)
            upstream = torch.randn(2, 3, 5 * scale, 7 * scale, generator=generator).cuda()
            for shared_prior in (False, True):
                with self.subTest(scale=scale, shared_prior=shared_prior):
                    results = []
                    for call in (torch.ops.converse2d.forward, converse2d_reference):
                        base = base_raw.cuda().requires_grad_()
                        x = base.sin()
                        prior = x if shared_prior else base.cos()
                        if scale > 1:
                            prior = torch.nn.functional.interpolate(prior, scale_factor=scale, mode="nearest")
                        kernel = base[..., :3, :3].reshape(2, 3, 9).softmax(-1).reshape(2, 3, 3, 3)
                        bias = base.mean((0, 2, 3), keepdim=True)
                        output = call(x, prior, kernel, bias, scale, .1)
                        # Multiple paths to the same leaf exercise accumulation
                        # order beyond independent public input VJPs.
                        loss = (output * upstream).sum() + base.square().sum() * .03
                        gradient = torch.autograd.grad(loss, base)[0]
                        results.append((output, gradient))
                    self.assert_results_equal(*results)

    def test_nonlinear_second_and_third_derivatives(self):
        for scale in (1, 2, 3):
            raw, _ = fixture(scale)
            results = []
            for call in (torch.ops.converse2d.forward, converse2d_reference):
                data = leaves(raw, "cuda")
                output = call(*data, scale, .2)
                first = torch.autograd.grad(output.sin().mean(), data, create_graph=True)
                second = torch.autograd.grad(sum(g.square().mean() for g in first), data, create_graph=True)
                third = torch.autograd.grad(sum(g.square().mean() for g in second), data)
                results.append((*second, *third))
            with self.subTest(scale=scale):
                for actual, expected in zip(*results):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_training_keeps_per_call_kernel_fft_without_psf_rolls(self):
        raw, up = fixture(1)
        data = leaves(raw, "cuda")
        gradient = up.cuda()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            for _ in range(2):
                output = torch.ops.converse2d.forward(*data, 1)
                torch.autograd.grad(output, data, gradient)
        counts = {event.key: event.count for event in trace.key_averages()}
        self.assertEqual(counts.get("aten::fft_fft2"), 6, "kernel/input/prior FFT every call")
        self.assertNotIn("aten::constant_pad_nd", counts)
        self.assertNotIn("aten::roll", counts)

    def test_nondefault_stream_and_device_guard(self):
        # Always test a side stream; also test caller-device restoration when
        # multiple devices are available without treating one-GPU as a skip.
        caller = torch.cuda.current_device()
        for device in range(min(2, torch.cuda.device_count())):
            with torch.cuda.device(device):
                raw, up = fixture(2)
                data = leaves(raw, f"cuda:{device}")
                gradient = up.to(f"cuda:{device}")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                output = torch.ops.converse2d.forward(*data, 2)
                actual = (output, *torch.autograd.grad(output, data, gradient))
                reference = converse2d_reference(*data, 2)
                expected = (reference, *torch.autograd.grad(reference, data, gradient))
            stream.synchronize()
            self.assert_results_equal(actual, expected)
            self.assertEqual(torch.cuda.current_device(), caller)


if __name__ == "__main__":
    unittest.main(verbosity=2)
