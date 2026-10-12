"""The fused training kernels follow the complex-product rounding of ATen on this device."""
import unittest

import torch

from support import CUDATestCase
from models.converse_core import converse2d_fp32 as candidate
from fp32_baseline import converse2d_fp32 as baseline_fp32
from numerical_cases import evaluate, numerical_cases


class ProductPatternCUDA(CUDATestCase):
    def test_pattern_is_measured_unambiguously(self):
        report = torch.ops.converse2d._training_product_pattern().tolist()
        pattern, forced, re_a, re_b, im_a, im_b = report
        self.assertEqual(forced, 0, "CONVERSE2D_PRODUCT_PATTERN must not be set for the release suite")
        # Exactly one single-FMA rounding reproduces at::mul for each component.
        self.assertEqual(re_a + re_b, 1, f"real part: {report}")
        self.assertEqual(im_a + im_b, 1, f"imaginary part: {report}")
        self.assertEqual(pattern & 1, int(re_b == 1), report)
        self.assertEqual((pattern >> 1) & 1, int(im_b == 1), report)
        print(f"ATen complex product pattern on {torch.cuda.get_device_name(0)}: {pattern} {report}", flush=True)

    def test_small_plane_training_matches_the_python_expression_bitwise(self):
        # On planes below 16 pixels a side the fused path and the frozen Python
        # FP32 expression share every FFT kernel, so with the product pattern
        # matched they agree bitwise on output and all four gradients. This is
        # the identity the 2026-10-03 A100 release failures lacked; it is a
        # diagnostic of the pattern match, the accuracy gate is the FP64 budget.
        checked = 0
        for case in numerical_cases('fast'):
            if not case.name.endswith(('5x7', '7x6')):
                continue
            raw, up = case.fixture()
            reference = evaluate(baseline_fp32, raw, up, case, device='cuda')
            actual = evaluate(candidate, raw, up, case, device='cuda')
            for name, a, b in zip(('output', 'dx', 'dprior', 'dweight', 'dbias'), actual, reference):
                with self.subTest(case=case.name, tensor=name):
                    self.assert_bytes_equal(a, b, name)
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
