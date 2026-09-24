"""Public default dispatch and exact Python FP32 training regressions.

The admission checks below require finite, byte-identical spatial outputs and
VJPs: no error tolerance is added for the switch from half to full spectra.
Complex128 finite-difference checks exercise the separate higher-order contract.
Run from the checkout with ``python -m unittest discover -s test
-p test_full_spectrum_default.py -v``. CPU-only builds skip all CUDA cases.
"""
import os
import sys
import unittest

import torch
from torch.utils.cpp_extension import CUDA_HOME

from extension_loader import ROOT, load_extension

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference


CUDA_BUILD = (torch.cuda.is_available() and torch.version.cuda is not None
              and CUDA_HOME is not None and os.environ.get("CONVERSE2D_CPU_ONLY") != "1")


def graph_names(output):
    """Keep nodes alive while walking: Python wrapper ids can otherwise recycle."""
    pending = [output.grad_fn] if output.grad_fn is not None else []
    seen = set()
    names = []
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        names.append(node.name())
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return names


def has_full_solve(output):
    return any("FullSolve" in name for name in graph_names(output))


def profiled(call):
    # CPU dispatcher events suffice to identify which FFT API is called; this
    # is a routing assertion, not a CUDA performance measurement.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        output = call()
    return output, {event.key for event in trace.key_averages()}


def fixture(scale, *, kb=1, kc=3, dtype=torch.float32, weak=None):
    generator = torch.Generator(device="cpu").manual_seed(96013 + scale)
    batch, channels, height, width = 2, 3, 5, 6
    values = [
        torch.randn(batch, channels, height, width, generator=generator, dtype=dtype),
        torch.randn(batch, channels, height * scale, width * scale,
                    generator=generator, dtype=dtype),
        torch.rand(kb, kc, 3, 3, generator=generator, dtype=dtype) / 9,
        torch.randn(1, channels, 1, 1, generator=generator, dtype=dtype),
    ]
    if weak is not None:
        values[0].mul_(1e-5)
        values[1].mul_(1e-5)
        values[2].mul_(weak)
        values[3].fill_(-40)
    upstream = torch.randn(values[1].shape, generator=generator, dtype=dtype)
    if weak is not None:
        upstream.mul_(1e-5)
    return values, upstream


def leaves(raw, device, *, strided=False, needs=(True, True, True, True)):
    result = []
    for value, required in zip(raw, needs):
        value = value.to(device).clone()
        if strided:
            value = torch.stack((value, value), dim=-1)[..., 0]
        result.append(value.detach().requires_grad_(required))
    return result


def capture(call, raw, upstream, *, scale, shared=False, strided=False, eps=1e-5):
    data = leaves(raw, "cuda", strided=strided)
    if shared:
        data[1] = data[0]
    requested = (data[0], data[2], data[3]) if shared else tuple(data)
    output = call(*data, scale, eps)
    grads = torch.autograd.grad(output, requested, upstream.to("cuda"))
    return (output, *grads)


class PublicModule(torch.nn.Module):
    def forward(self, *args):
        return torch.ops.converse2d.forward(*args, 2, 1e-5, "v7")


class DefaultFullSpectrumCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_extension(cpu_only=not CUDA_BUILD)

    def test_cpu_keeps_differentiable_reference_fallback(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                raw, upstream = fixture(2, dtype=dtype)
                data = leaves(raw, "cpu")
                output, events = profiled(lambda: torch.ops.converse2d.forward(*data, 2))
                self.assertFalse(has_full_solve(output))
                self.assertIn("aten::fft_rfft2", events)
                self.assertIn("aten::fft_irfft2", events)
                self.assertNotIn("aten::fft_fft2", events)
                self.assertEqual(output.dtype, dtype)
                gradients = torch.autograd.grad(output, data, upstream)
                for value in (output, *gradients):
                    self.assertTrue(torch.isfinite(value).all().item())


@unittest.skipUnless(CUDA_BUILD, "CUDA extension and device required (CPU-only builds skip)")
class DefaultFullSpectrumCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_extension()

    def assert_bytes_equal(self, actual, expected, label):
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertEqual(actual.shape, expected.shape, label)
        actual = actual.detach().resolve_conj().resolve_neg().cpu().contiguous()
        expected = expected.detach().resolve_conj().resolve_neg().cpu().contiguous()
        self.assertTrue(torch.isfinite(actual).all().item(), label + ": nonfinite actual")
        self.assertTrue(torch.isfinite(expected).all().item(), label + ": nonfinite reference")
        self.assertEqual(actual.numpy().tobytes(), expected.numpy().tobytes(),
                        label + ": differs from Python FP32 at zero byte margin")

    def check_python_exact(self, scale, *, shared=False, strided=False, kb=1, kc=3, weak=None):
        raw, upstream = fixture(scale, kb=kb, kc=kc, weak=weak)
        eps = 1e-8 if weak is not None else 1e-5
        actual = capture(torch.ops.converse2d.forward, raw, upstream, scale=scale,
                         shared=shared, strided=strided, eps=eps)
        expected = capture(converse2d_reference, raw, upstream, scale=scale,
                           shared=shared, strided=strided, eps=eps)
        labels = ("output", "dshared", "dweight", "dbias") if shared else (
            "output", "dx", "dprior", "dweight", "dbias")
        self.assertTrue(has_full_solve(actual[0]), "public v7 must use FullSolve")
        for label, value, target in zip(labels, actual, expected):
            self.assert_bytes_equal(value, target, label)

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

    def test_fp64_and_non_v7_keep_reference_fallback(self):
        cases = [(torch.float64, "v7"), *[(torch.float32, f"v{version}") for version in range(2, 7)]]
        for dtype, variant in cases:
            with self.subTest(dtype=dtype, variant=variant):
                raw, upstream = fixture(2, dtype=dtype)
                data = leaves(raw, "cuda")
                output = torch.ops.converse2d.forward(*data, 2, 1e-5, variant)
                self.assertFalse(has_full_solve(output))
                self.assertEqual(output.dtype, dtype)
                gradients = torch.autograd.grad(output, data, upstream.to("cuda"))
                for value in (output, *gradients):
                    self.assertTrue(torch.isfinite(value).all().item())

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
                    actual = capture(torch.ops.converse2d.forward, raw, upstream, scale=1, shared=shared)
                    expected = capture(converse2d_reference, raw, upstream, scale=1, shared=shared)
                    labels = ("output", "dshared", "dweight", "dbias") if shared else (
                        "output", "dx", "dprior", "dweight", "dbias")
                    for label, value, target in zip(labels, actual, expected):
                        self.assert_bytes_equal(value, target, label)


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
                                def run(call):
                                    data = leaves(raw, 'cuda')
                                    if transpose:
                                        data = [v.transpose(-1, -2).contiguous().transpose(-1, -2)
                                                .detach().requires_grad_() for v in data]
                                        self.assertFalse(data[0].is_contiguous())
                                    if shared:
                                        data[1] = data[0]
                                    requested = (data[0], data[2], data[3]) if shared else data
                                    output = call(*data, 1, 1e-5)
                                    return (output, *torch.autograd.grad(output, requested, upstream))
                                actual = run(torch.ops.converse2d.forward)
                                expected = run(converse2d_reference)
                                for index, (value, target) in enumerate(zip(actual, expected)):
                                    self.assert_bytes_equal(value, target, f's1 output/VJP {index}')

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
                    def run(call):
                        data = leaves(raw, 'cuda', needs=needs)
                        x, weight, bias = data
                        output = call(x, x, weight, bias, 1, 1e-5)
                        requested = [v for v, needed in zip(data, needs) if needed]
                        return (output, *torch.autograd.grad(output, requested, upstream))
                    actual = run(torch.ops.converse2d.forward)
                    expected = run(converse2d_reference)
                    for index, (value, target) in enumerate(zip(actual, expected)):
                        self.assert_bytes_equal(value, target, f'shared subset output/VJP {index}')

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

        def run(call):
            data = leaves(raw, "cuda")
            if transpose:
                data = [value.transpose(-1, -2).contiguous().transpose(-1, -2)
                        .detach().requires_grad_(True) for value in data]
                self.assertFalse(data[1].is_contiguous())
            output = call(*data, scale, 1e-5)
            gradients = torch.autograd.grad(output, data, upstream)
            return (output, *gradients)

        actual = run(torch.ops.converse2d.forward)
        expected = run(converse2d_reference)
        for label, value, target in zip(("output", "dx", "dprior", "dweight", "dbias"),
                                        actual, expected):
            self.assert_bytes_equal(value, target, label)

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
                for value, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
                    self.assertTrue(torch.equal(value.detach().contiguous().view(torch.uint8), reference.detach().contiguous().view(torch.uint8)))

    def test_internal_full_solve_supports_small_higher_derivatives(self):
        # This fixed finite-difference contract is separate from the zero-margin
        # spatial FP32 admission above. complex128 is used for differentiation.
        generator = torch.Generator(device="cpu").manual_seed(97133)
        for scale in (1, 2, 3):
            with self.subTest(scale=scale):
                y = torch.randn(1, 1, 1, 2, generator=generator, dtype=torch.complex128).cuda().requires_grad_()
                p = torch.randn(1, 1, scale, 2 * scale, generator=generator,
                                dtype=torch.complex128).cuda().requires_grad_()
                k = torch.randn(1, 1, scale, 2 * scale, generator=generator,
                                dtype=torch.complex128).cuda().requires_grad_()
                lam = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64, device="cuda", requires_grad=True)
                op = lambda *args: torch.ops.converse2d._training_full_spectral(*args, scale)
                self.assertTrue(torch.autograd.gradcheck(op, (y, p, k, lam), fast_mode=True))
                self.assertTrue(torch.autograd.gradgradcheck(op, (y, p, k, lam), fast_mode=True))
                if scale == 1:
                    shared = lambda y, k, lam: torch.ops.converse2d._training_full_spectral(y, y, k, lam, 1)
                    self.assertTrue(torch.autograd.gradgradcheck(shared, (y, k, lam), fast_mode=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
