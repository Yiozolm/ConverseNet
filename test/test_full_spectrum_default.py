"""Public default dispatch and exact Python FP32 training regressions.

The admission checks below require finite, byte-identical spatial outputs and
VJPs: no error tolerance is added for the switch from half to full spectra.
FP64 reference checks and higher-order comparisons live in test_fp32_release.py.
Run from the checkout with ``python -m unittest discover -s test
-p test_full_spectrum_default.py -v``. CPU-only builds skip all CUDA cases.
"""
import unittest
import torch

from support import (ExtensionTestCase, CUDATestCase, fixture, leaves,
                     has_full_solve, profiled, compare_spatial)
from models.converse_core import converse2d_reference


class PublicModule(torch.nn.Module):
    def forward(self, *args):
        return torch.ops.converse2d.forward(*args, 2, 1e-5, "v7")


class DefaultFullSpectrumCPU(ExtensionTestCase):
    def test_cpu_keeps_differentiable_reference_fallback(self):
        raw, upstream = fixture(2)
        data = leaves(raw, "cpu")
        output, events = profiled(lambda: torch.ops.converse2d.forward(*data, 2))
        self.assertFalse(has_full_solve(output))
        self.assertIn("aten::fft_fft2", events)
        self.assertIn("aten::fft_ifft2", events)
        self.assertNotIn("aten::fft_rfft2", events)
        self.assertEqual(output.dtype, torch.float32)
        gradients = torch.autograd.grad(output, data, upstream)
        for value in (output, *gradients):
            self.assertTrue(torch.isfinite(value).all().item())
        reference = converse2d_reference(*data, 2)
        reference_gradients = torch.autograd.grad(reference, data, upstream)
        for actual, expected in zip((output, *gradients), (reference, *reference_gradients)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

class DefaultFullSpectrumCUDA(CUDATestCase):
    def check_python_exact(self, scale, *, shared=False, strided=False, kb=1, kc=3, weak=None):
        raw, upstream = fixture(scale, kb=kb, kc=kc, weak=weak)
        eps = 1e-8 if weak is not None else 1e-5
        output = compare_spatial(self, raw, upstream, scale=scale, shared=shared,
                                 strided=strided, eps=eps)
        self.assertTrue(has_full_solve(output), "public v7 must use FullSolve")

    def test_grad_enabled_v7_uses_full_fft_and_full_solve(self):
        raw, _ = fixture(2)
        # Any one trainable argument must trigger the same training route.
        for index in range(4):
            with self.subTest(trainable_argument=index):
                data = leaves(raw, "cuda", needs=tuple(i == index for i in range(4)))
                output, events = profiled(lambda: torch.ops.converse2d.forward(*data, 2))
                self.assertTrue(has_full_solve(output))
                self.assertIn("aten::fft_fft2", events)
                self.assertIn("aten::fft_ifft2", events)
                self.assertNotIn("aten::fft_rfft2", events)
                self.assertNotIn("aten::fft_irfft2", events)

    def test_eval_with_grad_uses_training_and_train_no_grad_uses_inference(self):
        raw, _ = fixture(2)
        data = leaves(raw, "cuda")
        model = PublicModule().eval()
        self.assertTrue(has_full_solve(model(*data)), "eval() must not disable trainable FullSolve")
        model.train()
        for mode in (torch.no_grad, torch.inference_mode):
            with self.subTest(mode=mode.__name__):
                with mode():
                    output, events = profiled(lambda: model(*data))
                self.assertFalse(output.requires_grad)
                self.assertFalse(has_full_solve(output))
                self.assertIn("aten::fft_rfft2", events)
                self.assertIn("aten::fft_irfft2", events)
                self.assertNotIn("aten::fft_fft2", events)

    def test_grad_enabled_frozen_inputs_keep_half_spectrum(self):
        raw, _ = fixture(2)
        data = leaves(raw, "cuda", needs=(False,) * 4)
        with torch.enable_grad():
            output, events = profiled(lambda: torch.ops.converse2d.forward(*data, 2))
        self.assertFalse(output.requires_grad)
        self.assertIn("aten::fft_rfft2", events)
        self.assertIn("aten::fft_irfft2", events)
        self.assertNotIn("aten::fft_fft2", events)

    def test_non_fp32_and_legacy_variants_are_rejected(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            raw, _ = fixture(2, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "FP32"):
                torch.ops.converse2d.forward(*leaves(raw, "cuda"), 2)
        raw, _ = fixture(2)
        for variant in ("v2", "v3", "v4", "v5", "v6"):
            with self.assertRaisesRegex(RuntimeError, "only variant v7"):
                torch.ops.converse2d.forward(*leaves(raw, "cuda"), 2, 1e-5, variant)

    def test_all_scales_broadcasts_outputs_and_four_vjps_match_python_bytes(self):
        for scale in (1, 2, 3):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                with self.subTest(scale=scale, kb=kb, kc=kc):
                    self.check_python_exact(scale, kb=kb, kc=kc)

    def test_s1_shared_input_accumulation_matches_python_bytes(self):
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            with self.subTest(kb=kb, kc=kc):
                self.check_python_exact(1, shared=True, kb=kb, kc=kc)

    def test_s1_large_broadcasts_shared_inputs_and_strided_vjps_match_python_bytes(self):
        # Odd extents exercise partial launch tails; batch/channel broadcasts
        # require separate reductions before combining kernel-gradient terms.
        batch, channels, height, width = 4, 32, 65, 67
        for kb, kc in ((1, 1), (1, channels), (batch, 1), (batch, channels)):
            generator = torch.Generator(device="cpu").manual_seed(97371 + 17 * kb + kc)
            raw = [
                torch.randn(batch, channels, height, width, generator=generator),
                torch.randn(batch, channels, height, width, generator=generator),
                torch.softmax(torch.randn(kb, kc, 25, generator=generator), -1).reshape(kb, kc, 5, 5),
                torch.randn(1, channels, 1, 1, generator=generator),
            ]
            # Slice after transfer: transferring an already strided CPU view
            # can make it dense and silently stop exercising strided VJPs.
            backing = torch.randn(batch, channels, height, 2 * width, generator=generator).cuda()
            backing.div_((batch * channels * height * width) ** 0.5)
            upstream = backing[..., ::2]
            self.assertFalse(upstream.is_contiguous())
            for shared in (False, True):
                with self.subTest(kb=kb, kc=kc, shared=shared):
                    compare_spatial(self, raw, upstream, scale=1, shared=shared)


    def test_s1_kernel_broadcasts_shared_and_transposed_inputs_match_python_bytes(self):
        # Covers the no-broadcast and reduction paths without asserting any
        # kernel implementation name or launch count.
        channels, height, width = 5, 19, 23
        for batch in (1, 4):
            for kb in sorted({1, batch}):
                for kc in (1, channels):
                    generator = torch.Generator().manual_seed(97801 + batch * 31 + kb * 7 + kc)
                    raw = [
                        torch.randn(batch, channels, height, width, generator=generator),
                        torch.randn(batch, channels, height, width, generator=generator),
                        torch.randn(kb, kc, 3, 5, generator=generator) / 15 ** .5,
                        torch.randn(1, channels, 1, 1, generator=generator),
                    ]
                    upstream = torch.randn(batch, channels, height, width * 2,
                                           generator=generator).cuda()[..., ::2]
                    self.assertFalse(upstream.is_contiguous())
                    for shared in (False, True):
                        for transpose in (False, True):
                            with self.subTest(batch=batch, kb=kb, kc=kc,
                                              shared=shared, transpose=transpose):
                                compare_spatial(self, raw, upstream, scale=1,
                                                shared=shared, transpose=transpose)


    def test_s1_shared_input_gradient_subsets_match_python_bytes(self):
        for batch in (1, 4):
            generator = torch.Generator().manual_seed(97899 + batch)
            raw = [torch.randn(batch, 3, 7, 9, generator=generator),
                   torch.randn(batch, 3, 3, 3, generator=generator) / 3,
                   torch.randn(1, 3, 1, 1, generator=generator)]
            upstream = torch.randn(batch, 3, 7, 9, generator=generator).cuda()
            for mask in range(1, 8):
                needs = tuple(bool(mask & (1 << i)) for i in range(3))
                with self.subTest(batch=batch, needs=needs):
                    compare_spatial(self, raw, upstream, scale=1,
                                    shared=True, needs=needs)

    def test_internal_s1_shared_conjugated_transposed_vjps(self):
        # Arbitrary complex (not necessarily Hermitian) spectra test the layout
        # boundary directly. This derivative tolerance is separate from the
        # spatial zero-margin Python FP32 admission tests above.
        def reference(y, p, k, regularizer):
            power = k.real.square() + k.imag.square()
            q = (y - k * p) / (power + regularizer)
            return p + k.conj() * q

        for batch in (1, 4):
            for kb, kc in ((1, 1), (1, 3), (batch, 1), (batch, 3)):
                for shared in (False, True):
                    with self.subTest(batch=batch, kb=kb, kc=kc, shared=shared):
                        generator = torch.Generator().manual_seed(97941 + batch * 7 + kb + kc)
                        raw = [
                            torch.randn(batch, 3, 7, 9, dtype=torch.complex64, generator=generator),
                            torch.randn(batch, 3, 7, 9, dtype=torch.complex64, generator=generator),
                            torch.randn(kb, kc, 7, 9, dtype=torch.complex64, generator=generator),
                            .25 + torch.rand(1, 3, 1, 1, generator=generator),
                        ]
                        upstream = torch.randn(batch, 3, 7, 18, dtype=torch.complex64,
                                               generator=generator).cuda()[..., ::2].conj()
                        self.assertTrue(upstream.is_conj())
                        self.assertFalse(upstream.is_contiguous())

                        def run(call):
                            data = []
                            for value in raw:
                                value = value.cuda().transpose(-1, -2).contiguous().transpose(-1, -2)
                                if value.is_complex():
                                    value = value.conj()
                                    self.assertTrue(value.is_conj())
                                    self.assertFalse(value.is_contiguous())
                                data.append(value.detach().requires_grad_())
                            if shared:
                                data[1] = data[0]
                            requested = (data[0], data[2], data[3]) if shared else data
                            output = call(*data)
                            return (output, *torch.autograd.grad(output, requested, upstream))

                        actual = run(lambda *data: torch.ops.converse2d._training_full_spectral(*data, 1))
                        expected = run(reference)
                        for value, target in zip(actual, expected):
                            self.assertTrue(torch.isfinite(value).all().item())
                            torch.testing.assert_close(value, target, atol=3e-5, rtol=3e-5)

    def check_s2_spatial_bytes(self, *, batch, channels, height, width,
                               kh, kw, kb, kc, transpose, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scale = 2
        raw = [
            torch.randn(batch, channels, height, width, generator=generator),
            torch.randn(batch, channels, height * scale, width * scale, generator=generator),
            torch.softmax(torch.randn(kb, kc, kh * kw, generator=generator), -1).reshape(kb, kc, kh, kw),
            torch.randn(1, channels, 1, 1, generator=generator),
        ]
        # Create the slice on CUDA so transfers cannot silently densify the VJP.
        backing = torch.randn(batch, channels, height * scale, width * scale * 2,
                              generator=generator).cuda()
        backing.div_((batch * channels * height * width * scale * scale) ** 0.5)
        upstream = backing[..., ::2]
        self.assertFalse(upstream.is_contiguous())

        compare_spatial(self, raw, upstream, scale=scale, transpose=transpose)

    def test_s2_odd_rectangular_broadcasts_and_strided_vjps_match_python_bytes(self):
        # Odd LR extents cover partial launch tails with all kernel broadcast
        # combinations; both FFT layouts must preserve the Python FP32 result.
        for kb, kc in ((1, 1), (1, 32), (4, 1), (4, 32)):
            for transpose in (False, True):
                with self.subTest(kb=kb, kc=kc, transpose=transpose):
                    self.check_s2_spatial_bytes(
                        batch=4, channels=32, height=33, width=35, kh=5, kw=5,
                        kb=kb, kc=kc, transpose=transpose, seed=97401 + 17 * kb + kc)

    def test_s2_width_one_outputs_and_vjps_match_python_bytes(self):
        # A singleton LR width exercises a distinct reduction geometry. The
        # HR prior still has two columns, so its transposed layout is noncontiguous.
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            for transpose in (False, True):
                with self.subTest(kb=kb, kc=kc, transpose=transpose):
                    self.check_s2_spatial_bytes(
                        batch=2, channels=3, height=5, width=1, kh=3, kw=1,
                        kb=kb, kc=kc, transpose=transpose, seed=97483 + 17 * kb + kc)

    def test_noncontiguous_public_inputs_match_python_bytes(self):
        for scale in (1, 2, 3):
            with self.subTest(scale=scale):
                self.check_python_exact(scale, strided=True)
        self.check_python_exact(1, shared=True, strided=True)

    def test_weak_regularization_matches_python_bytes(self):
        for scale in (1, 2, 3):
            for amplitude in (0.0, 1e-6, 1e-3):
                with self.subTest(scale=scale, amplitude=amplitude):
                    self.check_python_exact(scale, weak=amplitude)
        self.check_python_exact(1, shared=True, weak=1e-6)

    def test_transposed_fft_layout_keeps_python_output_and_vjp_bits(self):
        # A transposed complex spectrum can choose a different cuFFT IFFT plan.
        # Slice-strided inputs alone do not exercise this layout distinction.
        for scale, height, width in ((1, 256, 257), (2, 31, 37), (3, 17, 19)):
            with self.subTest(scale=scale, height=height, width=width):
                generator = torch.Generator().manual_seed(97301 + scale)
                x = torch.randn(2, 3, width, height, generator=generator).transpose(-1, -2).cuda().requires_grad_()
                prior = torch.randn(2, 3, width*scale, height*scale, generator=generator).transpose(-1, -2).cuda().requires_grad_()
                weight = torch.softmax(torch.randn(1, 3, 9, generator=generator), -1).reshape(1, 3, 3, 3).cuda().requires_grad_()
                bias = torch.zeros(1, 3, 1, 1, device='cuda', requires_grad=True)
                data = (x, prior, weight, bias)
                actual = torch.ops.converse2d.forward(*data, scale, 1e-5)
                expected = converse2d_reference(*data, scale, 1e-5)
                upstream = torch.randn(actual.shape, generator=generator).cuda() / actual.numel()**.5
                actual_grads = torch.autograd.grad(actual, data, upstream)
                expected_grads = torch.autograd.grad(expected, data, upstream)
                self.assertTrue(has_full_solve(actual))
                self.assertEqual(actual.stride(), expected.stride())
                self.assert_results_equal((actual, *actual_grads), (expected, *expected_grads))


if __name__ == '__main__':
    unittest.main(verbosity=2)
