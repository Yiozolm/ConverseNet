"""Full-spectrum s3 admission: reduction order, gradient masks and fallbacks."""
import struct
import unittest

import torch

from support import CUDATestCase, compare_spatial, fixture, leaves
from numerical_policy import assert_budget


def spectral_reference(y, prior, kernel, regularizer):
    def aliases(value):
        return value.reshape(value.shape[0], value.shape[1], 3, y.shape[-2],
                             3, y.shape[-1]).mean((2, 4))

    power = kernel.real.square() + kernel.imag.square()
    correction = (y - aliases(kernel * prior)) / (aliases(power) + regularizer)
    return prior + kernel.conj() * correction.repeat(1, 1, 3, 3)


class Scale3FusionCUDA(CUDATestCase):
    def test_all_gradient_masks_and_broadcasts_meet_fp64_budget(self):
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            raw, upstream = fixture(3, kb=kb, kc=kc)
            for mask in range(1, 16):
                with self.subTest(kb=kb, kc=kc, mask=mask):
                    needs = tuple(bool(mask & (1 << i)) for i in range(4))
                    compare_spatial(self, raw, upstream, scale=3, needs=needs,
                                    transpose=bool(mask & 1), strided=not bool(mask & 1))

    def test_small_large_odd_even_shapes_and_strided_vjps(self):
        for batch, channels, height, width in ((1, 1, 1, 2), (1, 5, 8, 10),
                                                (3, 5, 9, 7), (4, 32, 33, 35)):
            broadcasts = sorted({(1, 1), (1, channels), (batch, 1), (batch, channels)})
            for kb, kc in broadcasts:
                generator = torch.Generator().manual_seed(109331 + batch * 19 + channels + kb + kc)
                raw = [
                    torch.randn(batch, channels, height, width, generator=generator),
                    torch.randn(batch, channels, 3 * height, 3 * width, generator=generator),
                    torch.randn(kb, kc, 3, 3, generator=generator) / 9 ** .5,
                    torch.randn(1, channels, 1, 1, generator=generator),
                ]
                backing = torch.randn(batch, channels, 3 * height, 6 * width,
                                      generator=generator).cuda()
                backing.div_((batch * channels * height * width * 9) ** .5)
                upstream = backing[..., ::2]
                self.assertFalse(upstream.is_contiguous())
                for transpose in (False, True):
                    with self.subTest(shape=(batch, channels, height, width),
                                      kb=kb, kc=kc, transpose=transpose):
                        compare_spatial(self, raw, upstream, scale=3, transpose=transpose)

    def test_weak_regularization_with_fused_aliases(self):
        for weak in (0.0, 1e-6, 1e-3):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                raw, upstream = fixture(3, kb=kb, kc=kc, weak=weak)
                for mask in (1, 2, 4, 8, 12, 15):
                    with self.subTest(weak=weak, kb=kb, kc=kc, mask=mask):
                        needs = tuple(bool(mask & (1 << i)) for i in range(4))
                        compare_spatial(self, raw, upstream, scale=3, needs=needs,
                                        transpose=True, eps=1e-8)

    def test_cancellation_sensitive_nine_aliases(self):
        # Large cancellation with small residuals distinguishes ATen's four
        # accumulators from a sequential nine-element sum in both directions.
        values = torch.tensor([1e8, 1, 2, 3, -1e8, 4, 5, 6, 7], dtype=torch.float32)
        prior = torch.complex(values, values.flip(0)).reshape(1, 1, 3, 3).repeat_interleave(2, -1).cuda()
        kernel = torch.ones_like(prior)
        regularizer = torch.ones(1, 1, 1, 1, device="cuda")
        upstream = prior.flip(-2).contiguous()
        results = []
        for call, dtype in ((lambda *a: torch.ops.converse2d._training_full_spectral(*a, 3), torch.complex64),
                            (spectral_reference, torch.complex64), (spectral_reference, torch.complex128)):
            y = torch.zeros(1, 1, 1, 2, dtype=dtype, device="cuda", requires_grad=True)
            output = call(y, prior.to(dtype), kernel.to(dtype), regularizer.to(y.real.dtype))
            results.append((output, *torch.autograd.grad(output, y, upstream.to(dtype))))
        assert_budget(self, *results)

    def test_large_mean_counts_keep_distinct_prediction_and_power_factors(self):
        # The first inexact integer region is inexpensive enough to exercise
        # directly: the HR prior occupies about 128 MiB, with no FFT or VJP.
        # Oracle computation below is ordinary ATen mean/solve, independently
        # of the fused kernel's host parameters and reduction implementation.
        width = 621379
        nout = 3 * width
        f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
        prediction_factor = f32(f32(nout) / f32(9 * nout))
        power_factor = f32(f32(width) / f32(9 * width))
        self.assertGreater(prediction_factor, power_factor)
        self.assertEqual(f32(9 * power_factor), 1.0)
        self.assertGreater(f32(9 * prediction_factor), 1.0)
        y = torch.zeros(3, 1, 1, width, dtype=torch.complex64, device="cuda")
        prior = torch.ones(3, 1, 3, 3 * width, dtype=torch.complex64, device="cuda")
        kernel = torch.ones(1, 1, 3, 3 * width, dtype=torch.complex64, device="cuda")
        regularizer = torch.ones(1, 1, 1, 1, device="cuda")
        actual = torch.ops.converse2d._training_full_spectral(y, prior, kernel, regularizer, 3)
        expected = spectral_reference(y, prior, kernel, regularizer)
        high = spectral_reference(y.to(torch.complex128), prior.to(torch.complex128),
                                  kernel.to(torch.complex128), regularizer.double())
        assert_budget(self, (actual,), (expected,), (high,))

    def test_alias_reduction_dispatch_and_width_one_fallback(self):
        for width in (1, 2, 7):
            generator = torch.Generator().manual_seed(109411 + width)
            y = torch.randn(2, 3, 5, width, dtype=torch.complex64,
                            generator=generator).cuda().requires_grad_()
            prior = torch.randn(2, 3, 15, 3 * width, dtype=torch.complex64,
                                generator=generator).cuda()
            kernel = torch.randn(1, 1, 15, 3 * width, dtype=torch.complex64,
                                 generator=generator).cuda()
            regularizer = torch.full((1, 3, 1, 1), .1, device="cuda")
            upstream = torch.ones_like(prior)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                output = torch.ops.converse2d._training_full_spectral(y, prior, kernel, regularizer, 3)
                torch.autograd.grad(output, y, upstream)
            counts = {event.key: event.count for event in trace.key_averages()}
            with self.subTest(width=width):
                self.assertEqual(counts.get("aten::mean", 0), 2 * int(width == 1))
                self.assertEqual(counts.get("aten::sum", 0), int(width == 1),
                                 "W=1 must retain the generic backward alias sum")

    def test_conjugated_nonhermitian_spectra(self):
        # Retain the existing internal-spectrum tolerance; spatial admission
        # and the targeted cancellation/count tests above use FP64 budgets.
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            generator = torch.Generator().manual_seed(109471 + 7 * kb + kc)
            raw = [torch.randn(2, 3, 5, 7, dtype=torch.complex64, generator=generator),
                   torch.randn(2, 3, 15, 21, dtype=torch.complex64, generator=generator),
                   torch.randn(kb, kc, 15, 21, dtype=torch.complex64, generator=generator),
                   .25 + torch.rand(1, 3, 1, 1, generator=generator)]
            upstream = torch.randn(2, 3, 15, 42, dtype=torch.complex64,
                                   generator=generator).cuda()[..., ::2].conj()
            for mask in (1, 2, 4, 8, 12, 15):
                results = []
                for call in (lambda *a: torch.ops.converse2d._training_full_spectral(*a, 3),
                             spectral_reference):
                    data = leaves(raw, "cuda", transpose=True, needs=(False,) * 4)
                    for index, value in enumerate(data):
                        if value.is_complex():
                            value = value.conj()
                        data[index] = value.requires_grad_(bool(mask & (1 << index)))
                    requested = [value for value in data if value.requires_grad]
                    output = call(*data)
                    results.append((output, *torch.autograd.grad(output, requested, upstream)))
                with self.subTest(kb=kb, kc=kc, mask=mask):
                    for actual, expected in zip(*results):
                        self.assertTrue(torch.isfinite(actual).all().item())
                        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
