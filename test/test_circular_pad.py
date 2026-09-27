"""Acceptance tests for fused circular padding and complex64 preparation.

Extends the existing release suite without relaxing its numerical gates.

Assumed private interfaces:
  _training_pad_complex(x, padding) -> padded contiguous complex64 tensor
  _training_circular_s1(x, weight, bias, padding, eps) -> cropped FP32 view

The fused entry is training-only. The primitive supports no_grad probes.
No timing claims are made by these tests, and no files are written by them.
"""
import contextlib
import itertools
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT_HINT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_HINT / "test"))
sys.path.insert(0, str(ROOT_HINT))

import torch
import torch.nn.functional as F
from support import CUDATestCase, profiled
from models.converse_core import converse2d_reference, converse2d_fp32


def python_pad_complex(x, padding):
    """Exact old ATen preparation; keep dtype conversion AFTER real padding."""
    return F.pad(x, (padding,) * 4, mode="circular").to(torch.complex64)


def old_spatial(x, weight, bias, padding, eps=1e-5, solver=None):
    """Old complete s1 wrapper; x and prior MUST share the same padded tensor."""
    padded = F.pad(x, (padding,) * 4, mode="circular") if padding else x
    if solver is None:
        solver = torch.ops.converse2d.forward
    out = solver(padded, padded, weight, bias, 1, eps)
    return out[..., padding:-padding, padding:-padding] if padding else out


def python_spatial(x, weight, bias, padding, eps=1e-5):
    return old_spatial(x, weight, bias, padding, eps, converse2d_reference)


def candidate_spatial(x, weight, bias, padding, eps=1e-5):
    return torch.ops.converse2d._training_circular_s1(x, weight, bias, padding, eps)


def candidate_pad(x, padding):
    return torch.ops.converse2d._training_pad_complex(x, padding)


def fixture(batch=2, channels=3, height=5, width=7, kb=1, kc=None, weak=None):
    kc = channels if kc is None else kc
    rng = torch.Generator().manual_seed(97153 + batch + height * 13 + width)
    raw = [torch.randn(batch, channels, height, width, generator=rng),
           torch.rand(kb, kc, 3, 3, generator=rng) / 9,
           torch.randn(1, channels, 1, 1, generator=rng)]
    upstream = torch.randn(raw[0].shape, generator=rng)
    if weak is not None:
        raw[0].mul_(1e-5)
        raw[1].mul_(weak)
        raw[2].fill_(-40)
        upstream.mul_(1e-5)
    return raw, upstream


def leaves(raw, needs=(True, True, True), dtype=torch.float32):
    return [x.to(device="cuda", dtype=dtype).clone().detach().requires_grad_(need)
            for x, need in zip(raw, needs)]


def capture(fn, raw, upstream, padding=2, eps=1e-5,
            needs=(True, True, True), dtype=torch.float32):
    data = leaves(raw, needs, dtype)
    out = fn(*data, padding, eps)
    requested = [x for x in data if x.requires_grad]
    grad = torch.autograd.grad(out, requested, upstream.to(device="cuda", dtype=dtype))
    return (out, *grad)


def complex_upstream(shape, kind):
    rng = torch.Generator().manual_seed(97781)
    g = torch.randn(shape, generator=rng, dtype=torch.complex64).cuda()
    if kind == "strided":
        return torch.stack((g, g), -1)[..., 0]
    if kind == "transpose":
        return g.transpose(-1, -2).contiguous().transpose(-1, -2)
    if kind == "expand":
        return g[:1, :1, :1, :1].expand(shape)
    if kind == "negative":
        return torch._neg_view(g)
    if kind == "conjugate":
        return g.conj()
    return g


class CircularPreparationContracts(CUDATestCase):
    def setUp(self):
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.previous_fill = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False

    def tearDown(self):
        torch.utils.deterministic.fill_uninitialized_memory = self.previous_fill
        torch.use_deterministic_algorithms(self.previous_deterministic,
                                          warn_only=self.previous_warn_only)

    def test_primitive_signed_zero_subnormal_and_overlap_boundaries(self):
        # p==H or W means wrap contributions overlap; p=H=W=1 reaches corners.
        cases = [(1, 1, 1), (2, 3, 2), (2, 2, 2),
                 (3, 5, 3), (5, 7, 2), (6, 4, 1)]
        atoms = torch.tensor([0., -0., 1., -1., 2.**-149, -(2.**-149),
                              2.**-126, -(2.**-126), 2.**24, -(2.**24)])
        for h, w, p in cases:
            raw = atoms.repeat((2 * 3 * h * w + 9) // 10)[:2 * 3 * h * w]
            raw = raw.reshape(2, 3, h, w)
            for fill in (False, True):
                torch.utils.deterministic.fill_uninitialized_memory = fill
                with self.subTest(shape=(h, w), padding=p, fill=fill):
                    outputs = []
                    for fn in (candidate_pad, python_pad_complex):
                        x = raw.cuda().requires_grad_()
                        out = fn(x, p)
                        # Independently varied real/imag values check that the
                        # real-input VJP ignores arbitrary imaginary upstream.
                        g = complex_upstream(out.shape, "contiguous")
                        outputs.append((out, torch.autograd.grad(out, x, g)[0]))
                    self.assert_results_equal(*outputs)
                    self.assertEqual(outputs[0][0].dtype, torch.complex64)
                    self.assertTrue(outputs[0][0].is_contiguous())
                    with torch.no_grad():
                        self.assert_bytes_equal(candidate_pad(raw.cuda(), p),
                                                python_pad_complex(raw.cuda(), p),
                                                "no_grad primitive")

    def test_primitive_backward_signed_zero_and_strided_complex_gradients(self):
        xraw = torch.randn(2, 3, 3, 5)
        for p in (1, 2, 3):
            shape = (2, 3, 3 + 2 * p, 5 + 2 * p)
            for kind in ("contiguous", "strided", "transpose", "expand",
                         "negative", "conjugate", "signed_zero"):
                g = complex_upstream(shape, kind)
                if kind == "signed_zero":
                    # Include cancellation-sensitive magnitudes and both zero
                    # signs. A scatter/add or different grouping fails bytes.
                    a = torch.tensor([0., -0., 2.**24, 1., -(2.**24), -1.], device="cuda")
                    r = a.repeat((g.numel() + 5) // 6)[:g.numel()].reshape(shape)
                    g = torch.complex(r, r.flip(-1))
                results = []
                for fn in (candidate_pad, python_pad_complex):
                    x = xraw.cuda().requires_grad_()
                    out = fn(x, p)
                    results.append(torch.autograd.grad(out, x, g)[0])
                with self.subTest(padding=p, upstream=kind):
                    self.assert_bytes_equal(*results, "primitive dx")

    def test_primitive_higher_order_tracks_upstream_without_dummy_dependency(self):
        # Critical dummy-autograd fallback probe: arbitrary g is independent
        # of x, so dx must depend on g, and must NOT gain a fake dependency on x.
        raw = torch.randn(2, 3, 3, 5)
        directions = torch.randn_like(raw).cuda()
        for p in (1, 3):
            results = []
            for fn in (candidate_pad, python_pad_complex):
                x = raw.cuda().requires_grad_()
                out = fn(x, p)
                g = complex_upstream(out.shape, "conjugate").detach().requires_grad_()
                dx = torch.autograd.grad(out, x, g, create_graph=True, retain_graph=True)[0]
                self.assertTrue(dx.requires_grad, "higher-order VJP detached upstream")
                disconnected = torch.autograd.grad(dx.sum(), x, allow_unused=True,
                                                   retain_graph=True)[0]
                self.assertIsNone(disconnected, "linear pad VJP should not depend on x")
                dg = torch.autograd.grad(dx, g, directions, retain_graph=True)[0]
                # Keep both repeat execution and an alternate incoming gradient.
                repeated = torch.autograd.grad(out, x, g * .25, create_graph=True)[0]
                results.append((dx, dg, repeated))
            self.assert_results_equal(*results)

    def test_primitive_higher_order_nonlinear_chain_and_input_value_mutation(self):
        raw = torch.randn(1, 2, 3, 5)
        results = []
        for fn in (candidate_pad, python_pad_complex):
            x = raw.cuda().requires_grad_()
            out = fn(x, 2)
            loss = (out.real.sin() + .2 * out.imag.cos()).square().mean()
            first = torch.autograd.grad(loss, x, create_graph=True)[0]
            second = torch.autograd.grad(first.square().mean(), x, create_graph=True)[0]
            third = torch.autograd.grad(second.square().mean(), x)[0]
            results.append((first, second, third))
        for a, b in zip(*results):
            self.assertTrue(torch.isfinite(a).all().item())
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
        # The linear prep does not need input values for its backward. Saving
        # x only to recover shape would add an unnecessary version-check error.
        mutated = []
        for fn in (candidate_pad, python_pad_complex):
            x = raw.cuda().requires_grad_()
            out = fn(x, 2)
            with torch.no_grad():
                x.add_(1)
            mutated.append(torch.autograd.grad(out, x, torch.ones_like(out))[0])
        self.assert_bytes_equal(*mutated, "linear preparation after x mutation")

    def test_private_dtype_and_padding_rejections(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(RuntimeError, "FP32|float32"):
                candidate_pad(torch.ones(1, 2, 3, 5, device="cuda", dtype=dtype), 1)
        x = torch.ones(1, 2, 3, 5, device="cuda")
        for padding in (-1, 0, 4):
            with self.subTest(padding=padding), self.assertRaises(RuntimeError):
                candidate_pad(x, padding)
        raw, _ = fixture()
        for index in range(3):
            for dtype in (torch.float16, torch.bfloat16, torch.float64):
                a = leaves(raw)
                a[index] = a[index].to(dtype)
                with self.subTest(input=index, dtype=dtype), self.assertRaisesRegex(RuntimeError, "FP32|float32|dtype"):
                    candidate_spatial(*a, 2)

    def test_complete_s1_outputs_gradients_and_broadcasts_match_both_baselines(self):
        for batch in (1, 2, 4):
            for kb, kc in sorted({(1, 1), (1, 3), (batch, 1), (batch, 3)}):
                raw, g = fixture(batch=batch, kb=kb, kc=kc)
                with self.subTest(batch=batch, broadcast=(kb, kc)):
                    actual = capture(candidate_spatial, raw, g)
                    self.assert_results_equal(actual, capture(old_spatial, raw, g))
                    self.assert_results_equal(actual, capture(python_spatial, raw, g))
        # Actual prior shape without 35 model calls; memory/layout representative.
        raw, g = fixture(batch=4, channels=128, height=96, width=96)
        self.assert_results_equal(capture(candidate_spatial, raw, g),
                                  capture(old_spatial, raw, g))

    def test_gradient_masks_and_requested_subsets(self):
        raw, g = fixture(batch=4)
        for bits in itertools.product((False, True), repeat=3):
            if not any(bits):
                continue
            with self.subTest(requires_grad=bits):
                self.assert_results_equal(capture(candidate_spatial, raw, g, needs=bits),
                                          capture(python_spatial, raw, g, needs=bits))
        for requested in ((0,), (1,), (2,), (0, 2), (0, 1, 2)):
            results = []
            for fn in (candidate_spatial, python_spatial):
                a = leaves(raw)
                out = fn(*a, 2)
                results.append((out, *torch.autograd.grad(out, [a[i] for i in requested], g.cuda())))
            with self.subTest(requested=requested):
                self.assert_results_equal(*results)

    def test_independent_fp64_error_noninferiority_with_padding(self):
        # FP64 runs ONLY the independent Python expression; production input
        # stays FP32. No tolerance is added to either error comparison.
        for batch in (1, 4):
            for weak in (None, 0., 1e-6):
                raw, g = fixture(batch=batch, weak=weak)
                eps = 1e-5 if weak is None else 1e-8
                high = capture(python_spatial, raw, g, eps=eps, dtype=torch.float64)
                reference = capture(python_spatial, raw, g, eps=eps)
                actual = capture(candidate_spatial, raw, g, eps=eps)
                self.assert_results_equal(actual, reference)
                for name, a, p, r in zip(("out", "dx", "dweight", "dbias"), actual, reference, high):
                    def errors(t):
                        difference = t.double() - r
                        return (difference.abs().max().item(),
                                (difference.norm() / r.norm().clamp_min(1e-300)).item())
                    ae, pe = errors(a), errors(p)
                    with self.subTest(batch=batch, weak=weak, tensor=name):
                        self.assertLessEqual(ae[0], pe[0])
                        self.assertLessEqual(ae[1], pe[1])

    def test_shared_ancestor_residual_and_repeated_operator_accumulation(self):
        for batch in (2, 4):
            raw, up = fixture(batch=batch)
            for dynamic_kernel in (False, True):
                results = []
                for fn in (candidate_spatial, old_spatial, python_spatial):
                    base, weight, bias = leaves(raw)
                    x = base.sin()
                    if dynamic_kernel:
                        weight = base.mean(0, keepdim=True)[..., :3, :3].reshape(1, 3, 9)
                        weight = weight.softmax(-1).reshape(1, 3, 3, 3)
                        bias = base.mean((0, 2, 3), keepdim=True)
                    first = fn(x, weight, bias, 2, .1)
                    second = fn(x, weight, bias, 2, .1)
                    output = first * .3 + second * .7 + base.cos() * .03125
                    requested = (base,) if dynamic_kernel else (base, weight, bias)
                    loss = (output * up.cuda()).sum() + .03 * base.square().sum()
                    results.append((output, *torch.autograd.grad(loss, requested)))
                with self.subTest(batch=batch, dynamic=dynamic_kernel):
                    self.assert_results_equal(results[0], results[1])
                    self.assert_results_equal(results[0], results[2])

    def test_complete_second_third_derivatives_and_repeated_backward(self):
        raw, g = fixture(batch=2, channels=2, height=4, width=5)
        results = []
        for fn in (candidate_spatial, old_spatial, python_spatial):
            a = leaves(raw)
            out = fn(*a, 2, .2)
            first = torch.autograd.grad(out.sin().mean(), a, create_graph=True)
            second = torch.autograd.grad(sum(v.square().mean() for v in first), a,
                                         create_graph=True)
            third = torch.autograd.grad(sum(v.square().mean() for v in second), a)
            results.append((out, *first, *second, *third))
        for other in results[1:]:
            self.assert_bytes_equal(results[0][0], other[0], "higher-order output")
            # Existing release higher-order policy: FP32 ATen, 3e-5 tolerance.
            for a, b in zip(results[0][1:], other[1:]):
                self.assertTrue(torch.isfinite(a).all().item())
                torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
        repeated = []
        for fn in (candidate_spatial, old_spatial):
            a = leaves(raw)
            out = fn(*a, 2)
            one = torch.autograd.grad(out, a, g.cuda(), retain_graph=True)
            two = torch.autograd.grad(out, a, g.cuda() * -.3)
            repeated.append((*one, *two))
        self.assert_results_equal(*repeated)

    def test_output_view_layout_and_inplace_semantics_are_unchanged(self):
        raw, g = fixture(batch=4)
        for inplace in (False, True):
            results, layouts = [], []
            for fn in (candidate_spatial, old_spatial, python_spatial):
                a = leaves(raw)
                out = fn(*a, 2, .1)
                layouts.append((out.stride(), out.storage_offset(), out._base is not None,
                                torch._C._is_alias_of(out, a[0])))
                # Must remain legal: putting output views inside a custom
                # Function can forbid this and alter an existing public API.
                if inplace:
                    out.mul_(.875).add_(.125)
                results.append((out, *torch.autograd.grad(out, a, g.cuda())))
            with self.subTest(inplace=inplace):
                self.assertEqual(layouts[0], layouts[1])
                self.assertEqual(layouts[0], layouts[2])
                self.assertTrue(layouts[0][2])
                self.assertFalse(layouts[0][3])
                self.assert_results_equal(results[0], results[1])
                self.assert_results_equal(results[0], results[2])

    def test_kernel_fft_runs_per_call_and_parameter_updates_are_observed(self):
        raw, g = fixture(batch=2)
        a = leaves(raw)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            first = candidate_spatial(*a, 2)
            second = candidate_spatial(*a, 2)
        calls = {event.key: event.count for event in trace.key_averages()}
        self.assertEqual(calls.get("aten::fft_fft2"), 4,
                         "each call needs one shared-input FFT and one differentiable kernel FFT")
        self.assertNotIn("aten::fft_rfft2", calls)
        self.assert_bytes_equal(first, second, "repeated outputs")
        before = second.detach().clone()
        with torch.no_grad():
            a[1].add_(.02)
        actual = candidate_spatial(*a, 2)
        expected = old_spatial(*a, 2)
        self.assertFalse(torch.equal(actual, before), "kernel update was ignored")
        self.assert_bytes_equal(actual, expected, "updated kernel")
        ag = torch.autograd.grad(actual, a, g.cuda())
        eg = torch.autograd.grad(expected, a, g.cuda())
        self.assert_results_equal(ag, eg)

    def test_module_guard_routes_only_supported_training_calls(self):
        from models.util_converse import Converse2D
        cases = ("eligible", "weight_only", "input_only", "bias_only", "frozen",
                 "no_grad", "inference_mode", "scale2", "replicate", "padding0",
                 "transposed", "negative", "pytorch", "cpu")
        for name in cases:
            scale = 2 if name == "scale2" else 1
            padding = 0 if name == "padding0" else 2
            mode = "replicate" if name == "replicate" else "circular"
            device = "cpu" if name == "cpu" else "cuda"
            backend = "pytorch" if name == "pytorch" else "auto"
            torch.manual_seed(97723)
            layer = Converse2D(3, 3, 3, scale=scale, padding=padding,
                              padding_mode=mode, backend=backend).to(device)
            x = torch.randn(2, 3, 5, 7, device=device)
            if name == "transposed":
                x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
            if name == "negative":
                x = torch._neg_view(x)
            needs_x = name not in ("weight_only", "bias_only", "frozen")
            x.requires_grad_(needs_x)
            layer.weight.requires_grad_(name not in ("input_only", "bias_only", "frozen"))
            layer.bias.requires_grad_(name not in ("input_only", "weight_only", "frozen"))
            mode_context = (torch.no_grad() if name == "no_grad" else
                            torch.inference_mode() if name == "inference_mode" else
                            contextlib.nullcontext())
            with mode_context, mock.patch.dict(os.environ, {"CONVERSE2D_BACKEND": ""}):
                actual, events = profiled(lambda: layer(x))
                padded = F.pad(x, (padding,) * 4, mode=mode) if padding else x
                prior = padded if scale == 1 else F.interpolate(padded, scale_factor=scale, mode="nearest")
                solve = converse2d_fp32 if backend == "pytorch" else torch.ops.converse2d.forward
                expected = solve(padded, prior, layer.weight, layer.bias, scale, float(layer.eps))
                if padding:
                    crop = padding * scale
                    expected = expected[..., crop:-crop, crop:-crop]
            eligible = name in ("eligible", "weight_only", "input_only", "bias_only")
            with self.subTest(case=name):
                self.assertEqual("converse2d::_training_circular_s1" in events, eligible)
                self.assert_bytes_equal(actual, expected, "module dispatch output")
                if name in ("frozen", "no_grad", "inference_mode"):
                    self.assertIn("aten::fft_rfft2", events)
                    self.assertNotIn("aten::fft_fft2", events)
                elif eligible:
                    self.assertIn("aten::fft_fft2", events)
                    self.assertNotIn("aten::fft_rfft2", events)

    def test_invalid_module_configuration_still_rejected(self):
        from models.util_converse import Converse2D
        for field, value in (("padding", 2.5), ("padding", 2.0), ("scale", 1.0), ("variant", "v6")):
            layer = Converse2D(3, 3, 3, padding=2, backend="cuda").cuda()
            setattr(layer, field, value)
            x = torch.randn(2, 3, 5, 7, device="cuda", requires_grad=True)
            with self.subTest(field=field, value=value), self.assertRaises((TypeError, RuntimeError, ValueError)):
                layer(x)

    def test_nondefault_stream(self):
        raw, g = fixture(batch=2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = capture(candidate_spatial, raw, g)
            expected = capture(old_spatial, raw, g)
        torch.cuda.current_stream().wait_stream(stream)
        self.assert_results_equal(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
