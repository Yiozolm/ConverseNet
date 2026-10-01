"""Budgeted spatial admission and layout/lifetime checks for s2 kernel VJPs."""
import unittest

import torch

from support import CUDATestCase, compare_spatial, leaves, check_operator_results


def spectral_reference(y, prior, kernel, regularizer):
    batch, channels, height, width = y.shape

    def aliases(value):
        return value.reshape(batch, channels, 2, height, 2, width).mean((2, 4))

    power = kernel.real.square() + kernel.imag.square()
    correction = (y - aliases(kernel * prior)) / (aliases(power) + regularizer)
    return prior + kernel.conj() * correction.repeat(1, 1, 2, 2)


class Scale2KernelFusionCUDA(CUDATestCase):
    def test_no_broadcast_spatial_vjps_meet_fp64_budget(self):
        for batch, channels, height, width in ((1, 1, 7, 9), (1, 5, 8, 10), (3, 5, 9, 7)):
            for weak in (None, 0.0, 1e-6):
                generator = torch.Generator().manual_seed(98391 + batch * 7 + channels)
                raw = [
                    torch.randn(batch, channels, height, width, generator=generator),
                    torch.randn(batch, channels, 2 * height, 2 * width, generator=generator),
                    torch.randn(batch, channels, 3, 5, generator=generator) / 15 ** .5,
                    torch.randn(1, channels, 1, 1, generator=generator),
                ]
                backing = torch.randn(batch, channels, 2 * height, 4 * width,
                                      generator=generator).cuda()
                if weak is not None:
                    raw[0].mul_(1e-5)
                    raw[1].mul_(1e-5)
                    raw[2].mul_(weak)
                    raw[3].fill_(-40)
                    backing.mul_(1e-5)
                upstream = backing[..., ::2]
                self.assertFalse(upstream.is_contiguous())
                for mask in (4, 5, 6, 7, 12, 13, 14, 15):
                    needs = tuple(bool(mask & (1 << i)) for i in range(4))
                    with self.subTest(batch=batch, channels=channels, weak=weak, mask=mask):
                        compare_spatial(self, raw, upstream, scale=2, needs=needs,
                                        transpose=bool(mask & 1),
                                        eps=1e-5 if weak is None else 1e-8)

    def test_nonhermitian_conjugated_spectra_and_gradient_subsets(self):
        # As in the existing internal s1 spectral regression, arbitrary complex
        # derivatives use a tolerance; public spatial admission above is exact.
        for batch in (1, 3):
            generator = torch.Generator().manual_seed(98621 + batch)
            raw = [
                torch.randn(batch, 3, 5, 7, dtype=torch.complex64, generator=generator),
                torch.randn(batch, 3, 10, 14, dtype=torch.complex64, generator=generator),
                torch.randn(batch, 3, 10, 14, dtype=torch.complex64, generator=generator),
                .25 + torch.rand(1, 3, 1, 1, generator=generator),
            ]
            upstream = torch.randn(batch, 3, 10, 28, dtype=torch.complex64,
                                   generator=generator).cuda()[..., ::2].conj()
            self.assertTrue(upstream.is_conj())
            self.assertFalse(upstream.is_contiguous())
            for mask in (4, 5, 6, 7, 12, 13, 14, 15):
                results = []
                for call in (lambda *a: torch.ops.converse2d._training_full_spectral(*a, 2),
                             spectral_reference):
                    data = leaves(raw, "cuda", transpose=True,
                                  needs=(False,) * 4)
                    for index, value in enumerate(data):
                        if value.is_complex():
                            value = value.conj()
                            self.assertTrue(value.is_conj())
                        data[index] = value.requires_grad_(bool(mask & (1 << index)))
                    output = call(*data)
                    requested = [value for value in data if value.requires_grad]
                    results.append((output, *torch.autograd.grad(output, requested, upstream)))
                with self.subTest(batch=batch, mask=mask):
                    for actual, expected in zip(*results):
                        self.assertTrue(torch.isfinite(actual).all().item())
                        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_no_broadcast_backward_omits_separate_power_division(self):
        generator = torch.Generator().manual_seed(98421)
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            for need_bias in (False, True):
                y = torch.randn(2, 3, 5, 7, dtype=torch.complex64, generator=generator).cuda()
                prior = torch.randn(2, 3, 10, 14, dtype=torch.complex64, generator=generator).cuda()
                kernel = torch.randn(kb, kc, 10, 14, dtype=torch.complex64,
                                     generator=generator).cuda().requires_grad_()
                regularizer = torch.full((1, 3, 1, 1), .1, device="cuda", requires_grad=need_bias)
                upstream = torch.randn(prior.shape, dtype=torch.complex64, generator=generator).cuda()
                output = torch.ops.converse2d._training_full_spectral(y, prior, kernel, regularizer, 2)
                requested = (kernel, regularizer) if need_bias else (kernel,)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    torch.autograd.grad(output, requested, upstream)
                events = {event.key for event in trace.key_averages()}
                with self.subTest(kb=kb, kc=kc, need_bias=need_bias):
                    self.assertEqual("aten::div" in events, (kb, kc) != (2, 3),
                                     "broadcast kernels retain the separate power division")

    def test_no_broadcast_kernel_vjp_graph_replay(self):
        from models.converse_core import converse2d_reference

        generator = torch.Generator().manual_seed(98461)
        raw = [torch.randn(2, 3, 5, 7, generator=generator),
               torch.randn(2, 3, 10, 14, generator=generator),
               torch.rand(2, 3, 3, 3, generator=generator) / 9,
               torch.zeros(1, 3, 1, 1)]
        upstream = torch.randn(2, 3, 10, 14, generator=generator).cuda()
        for needs in ((False, False, True, False), (True,) * 4):
            data = leaves(raw, "cuda", needs=needs)
            requested = [value for value in data if value.requires_grad]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            def run():
                output = torch.ops.converse2d.forward(*data, 2)
                return output, *torch.autograd.grad(output, requested, upstream)

            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = run()
            with torch.no_grad():
                for value in data:
                    value.add_(.01)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                expected_output = converse2d_reference(*data, 2)
                expected = (expected_output, *torch.autograd.grad(expected_output, requested, upstream))
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay()
            torch.cuda.synchronize()
            check_operator_results(self, actual, data, upstream, 2)
            graph.reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
