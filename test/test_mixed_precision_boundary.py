"""Experimental boundary tests; the production FP32 dtype policy stays intact.

CPU-only execution does not import support.py or probe CUDA availability:
python -m unittest test.test_mixed_precision_boundary.MixedPrecisionBoundaryCPU
GPU tests load the checked extension in setUpClass, never inside the adapter.
"""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
from models import converse_core
from tools.v4_mixed_precision.adapter import mixed_converse2d


def fixture(dtype, *, scale=1, shape=(3, 5), shared=False, weight_low=False,
            needs=(False, False, False, False), device='cpu', layout='contiguous'):
    generator = torch.Generator().manual_seed(44017 + scale)
    height, width = shape
    x = torch.randn(1, 2, height, width, generator=generator).to(dtype=dtype, device=device)
    prior = x if shared else torch.randn(1, 2, height * scale, width * scale,
                                         generator=generator).to(dtype=dtype, device=device)
    kh, kw = min(3, height * scale), min(3, width * scale)
    weight = (torch.rand(1, 2, kh, kw, generator=generator) / (kh * kw)).to(
        dtype=dtype if weight_low else torch.float32, device=device)
    bias = (torch.randn(1, 2, 1, 1, generator=generator) * .1).to(device=device)
    def arranged(value):
        if layout == 'transpose':
            return value.transpose(-1, -2).contiguous().transpose(-1, -2)
        if layout == 'strided':
            return torch.stack((value, value), -1)[..., 0]
        return value
    x = arranged(x).detach().requires_grad_(needs[0])
    prior = x if shared else arranged(prior).detach().requires_grad_(needs[1])
    weight = arranged(weight).detach().requires_grad_(needs[2])
    bias = arranged(bias).detach().requires_grad_(needs[3])
    return x, prior, weight, bias


def quantized_reference(data, scale, eps=1e-5, *, output_dtype=torch.float32, backend='pytorch'):
    x, prior, weight, bias = data
    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = x.float()
        p32 = x32 if prior is x else prior.float()
        w32, b32 = weight.float(), bias.float()
        output = (torch.ops.converse2d.forward(x32, p32, w32, b32, scale, eps, 'v7') if backend == 'cuda'
                  else converse_core.converse2d_fp32(x32, p32, w32, b32, scale, eps))
        return output.to(output_dtype)


def results(function, data, scale, dtype, *, higher=False):
    output = function(*data, scale, output_dtype=dtype)
    targets = [data[0], *data[2:]] if data[1] is data[0] else list(data)
    targets = [value for value in targets if value.requires_grad]
    generator = torch.Generator().manual_seed(44029)
    upstream = (torch.randn(output.shape, generator=generator) * 1e-3).to(device=output.device, dtype=output.dtype)
    gradients = torch.autograd.grad(output, targets, upstream, create_graph=higher)
    values = [output, *gradients]
    if higher:
        scalar = sum(value.float().square().sum() for value in gradients if value.requires_grad)
        second = torch.autograd.grad(scalar, targets, allow_unused=True)
        values.extend(value for value in second if value is not None)
    return values


def overflow_diagnostic(device, backend):
    """An intentional FP16 leaf-gradient overflow, kept separate from accuracy."""
    def inputs():
        return (torch.full((1, 1, 1, 1), .25, device=device, dtype=torch.float16, requires_grad=True),
                torch.full((1, 1, 1, 1), -.2, device=device, dtype=torch.float16, requires_grad=True),
                torch.ones((1, 1, 1, 1), device=device, dtype=torch.float32, requires_grad=True),
                torch.zeros((1, 1, 1, 1), device=device, dtype=torch.float32, requires_grad=True))
    left, right = inputs(), inputs()
    actual = mixed_converse2d(*left, 1, backend=backend)
    reference = quantized_reference(right, 1, backend=backend)
    upstream = torch.full_like(actual, 1e5)
    ag = torch.autograd.grad(actual, left, upstream)
    rg = torch.autograd.grad(reference, right, upstream)
    core_data = tuple(value.detach().float().requires_grad_(True) for value in inputs())
    core_out = (torch.ops.converse2d.forward(*core_data, 1, 1e-5, 'v7') if backend == 'cuda'
                else converse_core.converse2d_fp32(*core_data, 1))
    core_grad = torch.autograd.grad(core_out, core_data, upstream)
    row = dict(kind='intentional_fp16_boundary_gradient_overflow', backend=backend,
               core_fp32_gradient_finite=[bool(torch.isfinite(value).all()) for value in core_grad],
               boundary_gradient_finite=[bool(torch.isfinite(value).all()) for value in ag],
               quantized_reference_gradient_finite=[bool(torch.isfinite(value).all()) for value in rg],
               accuracy_admitted=False, interpretation='FP16 leaf cast overflow, not FP32 solver overflow')
    return row, ag, rg


class MixedPrecisionBoundaryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_level1a_and_1b_arbitrary_shapes_and_master_weight(self):
        for dtype in (torch.float16, torch.bfloat16):
            for scale in (1, 2, 3, 4):
                for shape in ((4, 4), (3, 5), (7, 1)):
                    for low_weight in (False, True):
                        data = fixture(dtype, scale=scale, shape=shape, weight_low=low_weight)
                        for output_dtype in (torch.float32, dtype):
                            with self.subTest(dtype=dtype, scale=scale, shape=shape,
                                              low_weight=low_weight, output=output_dtype):
                                actual = mixed_converse2d(*data, scale, output_dtype=output_dtype, backend='pytorch')
                                expected = quantized_reference(data, scale, output_dtype=output_dtype)
                                self.assertEqual(actual.dtype, output_dtype)
                                self.assertEqual(tuple(actual.shape[-2:]), (shape[0] * scale, shape[1] * scale))
                                self.assertTrue(torch.isfinite(actual).all())
                                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_autocast_disabled_locally_and_alias_preserved_per_call(self):
        for dtype in (torch.float16, torch.bfloat16):
            data = fixture(dtype, shared=True, needs=(True, True, True, True), weight_low=True)
            observations = []
            def inspect(x, prior, weight, bias, scale, eps):
                observations.append((x, prior, weight, bias, torch.is_grad_enabled(), torch.is_autocast_enabled('cpu')))
                return x
            with torch.autocast('cpu', dtype=torch.bfloat16), patch.object(converse_core, 'converse2d_fp32', side_effect=inspect):
                for _ in range(2):
                    actual = mixed_converse2d(*data, 1, backend='pytorch')
                    self.assertTrue(torch.is_autocast_enabled('cpu'))
                    self.assertEqual(actual.dtype, torch.float32)
            for x, prior, weight, bias, grad, autocast in observations:
                self.assertIs(x, prior)
                self.assertTrue(grad)
                self.assertFalse(autocast)
                self.assertTrue(all(t.dtype == torch.float32 for t in (x, prior, weight, bias)))
                self.assertTrue(all(t.requires_grad for t in (x, prior, weight, bias)))
            self.assertIsNot(observations[0][0], observations[1][0])
            self.assertIsNot(observations[0][2], observations[1][2])

    def test_modes_and_any_differentiable_input_preserve_routing(self):
        original_reference = converse_core.converse2d_reference
        for dtype in (torch.float16, torch.bfloat16):
            for needed in range(4):
                data = fixture(dtype, scale=2, needs=tuple(i == needed for i in range(4)))
                with patch.object(converse_core, 'converse2d_reference', wraps=original_reference) as full:
                    actual = mixed_converse2d(*data, 2, backend='pytorch')
                self.assertEqual(full.call_count, 1)
                self.assertTrue(actual.requires_grad)
            for context in (torch.no_grad, torch.inference_mode, torch.enable_grad):
                needs = (False,) * 4 if context is torch.enable_grad else (True,) * 4
                data = fixture(dtype, scale=3, needs=needs)
                with context(), patch.object(converse_core, 'converse2d_reference', wraps=original_reference) as full:
                    actual = mixed_converse2d(*data, 3, backend='pytorch')
                self.assertEqual(full.call_count, 0)
                self.assertFalse(actual.requires_grad)

    def test_shared_noncontiguous_gradients_and_higher_derivatives(self):
        for dtype in (torch.float16, torch.bfloat16):
            for shared in (False, True):
                for low_weight in (False, True):
                    kwargs = dict(shared=shared, weight_low=low_weight, needs=(True,) * 4, layout='transpose')
                    left, right = fixture(dtype, **kwargs), fixture(dtype, **kwargs)
                    for output_dtype in (torch.float32, dtype):
                        actual = results(lambda *a, **k: mixed_converse2d(*a, **k, backend='pytorch'),
                                         left, 1, output_dtype, higher=True)
                        expected = results(lambda *a, **k: quantized_reference(a[:4], a[4], **k),
                                           right, 1, output_dtype, higher=True)
                        self.assertEqual(len(actual), len(expected))
                        for a, b in zip(actual, expected):
                            self.assertTrue(torch.isfinite(a).all())
                            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_weak_zero_kernel_keeps_regularizer_fp32(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, prior, weight, bias = fixture(dtype, scale=3, weight_low=True, layout='strided')
            weight = torch.zeros_like(weight)
            bias = torch.full_like(bias, -40)
            data = x, prior, weight, bias
            for output_dtype in (torch.float32, dtype):
                actual = mixed_converse2d(*data, 3, 1e-8, output_dtype=output_dtype, backend='pytorch')
                expected = quantized_reference(data, 3, 1e-8, output_dtype=output_dtype)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_fp16_gradient_cast_overflow_is_explicit(self):
        row, actual, expected = overflow_diagnostic('cpu', 'pytorch')
        self.assertTrue(all(row['core_fp32_gradient_finite']))
        self.assertFalse(row['boundary_gradient_finite'][0])
        self.assertEqual(row['boundary_gradient_finite'], row['quantized_reference_gradient_finite'])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
        print('Boundary overflow diagnostic:', json.dumps(row))

    def test_invalid_dtype_geometry_and_backend_contracts(self):
        data = fixture(torch.float16)
        replacements = ((0, data[0].float()), (1, data[1].bfloat16()),
                        (2, data[2].bfloat16()), (3, data[3].half()), (2, data[2].to_sparse()))
        for index, replacement in replacements:
            changed = list(data)
            changed[index] = replacement
            with self.assertRaises((TypeError, ValueError)):
                mixed_converse2d(*changed, 1, backend='pytorch')
        for scale in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                mixed_converse2d(*data, scale, backend='pytorch')
        for eps in (0, -1, float('inf'), float('nan'), True, torch.tensor(1e-5)):
            with self.assertRaises(ValueError):
                mixed_converse2d(*data, 1, eps, backend='pytorch')
        for kwargs in (dict(backend='auto'), dict(backend='cuda'), dict(backend=None),
                       dict(backend='pytorch', output_dtype=torch.bfloat16),
                       dict(backend='pytorch', output_dtype=torch.float64)):
            with self.assertRaises(ValueError):
                mixed_converse2d(*data, 1, **kwargs)
        with self.assertRaises(ValueError):
            mixed_converse2d(data[0], data[1][..., :-1], *data[2:], 1, backend='pytorch')
        with self.assertRaises(TypeError):
            mixed_converse2d(None, *data[1:], 1, backend='pytorch')
        changed = (*data[:2], torch.empty(data[2].shape, device='meta'), data[3])
        with self.assertRaises(ValueError):
            mixed_converse2d(*changed, 1, backend='pytorch')
        # Existing public Python entry still rejects low storage tensors.
        with self.assertRaisesRegex(ValueError, 'FP32'):
            converse_core.converse2d_fp32(*data, 1)


class MixedPrecisionBoundaryCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
            raise unittest.SkipTest('CPU-only run must not probe CUDA')
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None or not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA device/toolkit required')
        from extension_loader import load_extension
        load_extension()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def test_checked_boundary_matches_identical_quantized_native_chain(self):
        for dtype in (torch.float16, torch.bfloat16):
            for shared in (False, True):
                for low_weight in (False, True):
                    kwargs = dict(shared=shared, weight_low=low_weight, needs=(True,) * 4,
                                  device='cuda', layout='strided')
                    for output_dtype in (torch.float32, dtype):
                        left, right = fixture(dtype, **kwargs), fixture(dtype, **kwargs)
                        actual = results(mixed_converse2d, left, 1, output_dtype)
                        expected = results(lambda *a, **k: quantized_reference(a[:4], a[4], **k, backend='cuda'),
                                           right, 1, output_dtype)
                        for a, b in zip(actual, expected):
                            self.assertTrue(torch.isfinite(a).all())
                            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_checked_modes_and_autocast_boundary(self):
        from support import has_full_solve
        for dtype in (torch.float16, torch.bfloat16):
            for needed in range(4):
                data = fixture(dtype, scale=3, device='cuda', needs=tuple(i == needed for i in range(4)))
                with torch.autocast('cuda', dtype=dtype):
                    actual = mixed_converse2d(*data, 3)
                    self.assertTrue(torch.is_autocast_enabled('cuda'))
                self.assertTrue(has_full_solve(actual))
                self.assertEqual(actual.dtype, torch.float32)
            for context in (torch.no_grad, torch.inference_mode, torch.enable_grad):
                needs = (False,) * 4 if context is torch.enable_grad else (True,) * 4
                data = fixture(dtype, scale=2, device='cuda', needs=needs)
                with context():
                    actual = mixed_converse2d(*data, 2)
                self.assertFalse(actual.requires_grad)
                self.assertFalse(has_full_solve(actual))
            with self.assertRaisesRegex(RuntimeError, 'FP32'):
                torch.ops.converse2d.forward(*fixture(dtype, device='cuda'), 1, 1e-5, 'v7')

    def test_checked_higher_order_fallback_preserves_cast_chain(self):
        for dtype in (torch.float16, torch.bfloat16):
            kwargs = dict(shared=True, device='cuda', needs=(True,) * 4, weight_low=False)
            left, right = fixture(dtype, **kwargs), fixture(dtype, **kwargs)
            actual = results(mixed_converse2d, left, 1, torch.float32, higher=True)
            expected = results(lambda *a, **k: quantized_reference(a[:4], a[4], **k, backend='cuda'),
                               right, 1, torch.float32, higher=True)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.isfinite(a).all())
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_checked_weak_zero_kernel_keeps_regularizer_fp32(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, prior, weight, bias = fixture(dtype, scale=3, weight_low=True, device='cuda', layout='strided')
            data = x, prior, torch.zeros_like(weight), torch.full_like(bias, -40)
            for context in (torch.no_grad, torch.inference_mode, torch.enable_grad):
                for output_dtype in (torch.float32, dtype):
                    with context():
                        actual = mixed_converse2d(*data, 3, 1e-8, output_dtype=output_dtype)
                        expected = quantized_reference(data, 3, 1e-8, output_dtype=output_dtype, backend='cuda')
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_checked_fp16_gradient_cast_overflow_is_explicit(self):
        row, actual, expected = overflow_diagnostic('cuda', 'cuda')
        self.assertTrue(all(row['core_fp32_gradient_finite']))
        self.assertFalse(row['boundary_gradient_finite'][0])
        self.assertEqual(row['boundary_gradient_finite'], row['quantized_reference_gradient_finite'])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
        print('Boundary overflow diagnostic:', json.dumps(row))


if __name__ == '__main__':
    unittest.main(verbosity=2)
