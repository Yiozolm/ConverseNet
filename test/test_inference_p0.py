"""Frozen inference routing, shared spectra and cache metadata regressions."""
import unittest

import torch

from support import CUDATestCase, ExtensionTestCase, compare_spatial, fixture, leaves, profiled
from models.converse_core import converse2d_reference
from fp32_baseline import converse2d_fp32
from numerical_policy import assert_budget


class FrozenInferenceCPU(ExtensionTestCase):
    def test_frozen_cpu_preserves_caller_grad_mode(self):
        raw, _ = fixture(2)
        data = leaves(raw, "cpu", needs=(False,) * 4)
        with torch.no_grad():
            expected = torch.ops.converse2d.forward(*data, 2)
        with torch.enable_grad():
            actual = torch.ops.converse2d.forward(*data, 2)
            self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(actual.requires_grad)
        self.assert_bytes_equal(actual, expected, "frozen CPU")


class InferenceP0(CUDATestCase):
    def tearDown(self):
        torch.ops.converse2d.clear_cache()

    def test_cache_preparation_is_shared_across_grad_modes(self):
        for scale in (1, 2, 3, 4):
            raw, _ = fixture(scale)
            data = leaves(raw, "cuda", needs=(False,) * 4, transpose=True)
            for first, second in ((torch.no_grad, torch.enable_grad),
                                  (torch.enable_grad, torch.no_grad)):
                with self.subTest(scale=scale, first=first.__name__):
                    torch.ops.converse2d.clear_cache()
                    with first():
                        torch.ops.converse2d.forward(*data, scale)
                    with second():
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                            cached = torch.ops.converse2d.forward(*data, scale)
                        counts = {event.key: event.count for event in trace.key_averages()}
                        self.assertEqual(counts.get("aten::fft_rfft2"), 2)
                        torch.ops.converse2d.clear_cache()
                        fresh = torch.ops.converse2d.forward(*data, scale)
                    self.assert_bytes_equal(cached, fresh, "cross-GradMode cache hit")

    def test_frozen_keeps_aten_results_for_broadcasts_and_layouts(self):
        for scale in (1, 2, 3, 4):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                for layout in ({}, {"strided": True}, {"transpose": True}):
                    raw, _ = fixture(scale, kb=kb, kc=kc)
                    data = leaves(raw, "cuda", needs=(False,) * 4, **layout)
                    for shared in ((False, True) if scale == 1 else (False,)):
                        with self.subTest(scale=scale, broadcast=(kb, kc), layout=layout, shared=shared):
                            args = [data[0], data[0], *data[2:]] if shared else data
                            # Frozen calls retain the original ATen half-spectrum
                            # arithmetic and public operator's contiguous inputs.
                            x = args[0].contiguous()
                            prior = x if shared else args[1].contiguous()
                            expected = converse2d_fp32(x, prior, args[2], args[3].contiguous(), scale)
                            dx = x.double()
                            dp = dx if shared else prior.double()
                            high = converse2d_reference(dx, dp, args[2].double(), args[3].double(), scale)
                            torch.ops.converse2d.clear_cache()
                            with torch.enable_grad():
                                for _ in range(2):
                                    actual = torch.ops.converse2d.forward(*args, scale)
                                    self.assertTrue(torch.is_grad_enabled())
                                    self.assertFalse(actual.requires_grad)
                                    assert_budget(self, (actual,), (expected,), (high,))

    def test_frozen_caches_preparation_and_preserves_training(self):
        raw, upstream = fixture(2)
        data = leaves(raw, "cuda", needs=(False,) * 4)
        torch.ops.converse2d.clear_cache()
        with torch.enable_grad():
            _, cold = profiled(lambda: torch.ops.converse2d.forward(*data, 2))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                torch.ops.converse2d.forward(*data, 2)
            events = {event.key: event.count for event in trace.key_averages()}
            self.assertIn("aten::constant_pad_nd", cold)
            self.assertNotIn("aten::constant_pad_nd", events)
            # Only input and independent prior FFTs remain on a hot call.
            self.assertEqual(events.get("aten::fft_rfft2"), 2)
            self.assertNotIn("aten::fft_fft2", events)
            self.assertTrue(torch.is_grad_enabled())
            data[2].requires_grad_()
            output, events = profiled(lambda: torch.ops.converse2d.forward(*data, 2))
            self.assertTrue(output.requires_grad)
            self.assertIn("aten::fft_fft2", events)
            self.assertNotIn("aten::fft_rfft2", events)
        compare_spatial(self, raw, upstream, scale=2, needs=(False, False, True, False))

    def test_cache_invalidates_layout_without_version_change(self):
        for context in (torch.no_grad, torch.enable_grad):
            for kernel_shape in ((3, 3), (3, 5)):
                with self.subTest(mode=context.__name__, kernel_shape=kernel_shape):
                    raw, _ = fixture(2)
                    data = leaves(raw, "cuda", needs=(False,) * 4)
                    count = kernel_shape[0] * kernel_shape[1]
                    kernel = torch.arange(1, count + 1, device="cuda", dtype=torch.float32)
                    data[2] = kernel.reshape(1, 1, *kernel_shape) / count
                    weight = data[2]
                    torch.ops.converse2d.clear_cache()
                    with context():
                        old = torch.ops.converse2d.forward(*data, 2)
                        pointer, version, stride = weight.data_ptr(), weight._version, weight.stride()
                        # Unlike transpose_(), set_data does not bump the version.
                        weight.data = weight.data.transpose(-1, -2)
                        self.assertEqual(weight.data_ptr(), pointer)
                        self.assertEqual(weight._version, version)
                        self.assertNotEqual(weight.stride(), stride)
                        cached = torch.ops.converse2d.forward(*data, 2)
                        torch.ops.converse2d.clear_cache()
                        fresh = torch.ops.converse2d.forward(*data, 2)
                    self.assertFalse(torch.equal(old, fresh))
                    self.assert_bytes_equal(cached, fresh, "layout replacement")

    def test_frozen_graph_survives_eager_cache_clear(self):
        raw, _ = fixture(2)
        data = leaves(raw, "cuda", needs=(False,) * 4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(3):
                torch.ops.converse2d.forward(*data, 2)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.enable_grad(), torch.cuda.graph(graph, stream=stream):
            output = torch.ops.converse2d.forward(*data, 2)
        torch.ops.converse2d.clear_cache()
        churn = torch.empty(1024 * 1024, device="cuda").fill_(123)
        data[0].add_(.125)
        graph.replay()
        with torch.enable_grad():
            expected = torch.ops.converse2d.forward(*data, 2)
        torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
        del churn


if __name__ == "__main__":
    unittest.main(verbosity=2)
