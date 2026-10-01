"""CPU-only unit checks for the independently fixed mixed-precision policy."""
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.v4_mixed_precision.policy import (comparison, output_cast_comparison,
                                           representation_status, POLICY)


class MixedPrecisionPolicy(unittest.TestCase):
    def test_kernel_extra_gate_is_independent_of_total_gate(self):
        high = torch.ones(8, dtype=torch.float64)
        rq = torch.ones(8) + .01
        candidate = rq + .003
        row = comparison(candidate, high.float(), rq, high, regime="weak")
        self.assertLess(row['total_error']['rel_l2'], row['limits']['rel_l2']['total'])
        self.assertGreater(row['kernel_extra_error']['rel_l2'], row['limits']['rel_l2']['kernel_extra'])
        self.assertFalse(row['passed'])

    def test_total_factor_and_kernel_fraction_are_frozen(self):
        self.assertEqual(POLICY['normal_total_factor'], 1.25)
        self.assertEqual(POLICY['weak_total_factor'], 1.5)
        self.assertEqual(POLICY['kernel_extra_factor'], .25)
        self.assertEqual(POLICY['rel_l2_floor'], 1e-7)
        self.assertEqual(POLICY['max_abs_floor'], 1e-6)

    def test_outlier_cannot_hide_in_relative_norm(self):
        high = torch.ones(100, dtype=torch.float64)
        rq = high.float() + .01
        candidate = rq.clone()
        candidate[0] += .004
        row = comparison(candidate, high.float(), rq, high)
        self.assertLess(row['kernel_extra_error']['rel_l2'], row['limits']['rel_l2']['kernel_extra'])
        self.assertGreater(row['kernel_extra_error']['max_abs'], row['limits']['max_abs']['kernel_extra'])
        self.assertFalse(row['passed'])

    def test_zero_quantization_error_uses_explicit_fp32_floors(self):
        high = torch.ones(8, dtype=torch.float64)
        candidate = high.float()
        candidate[0] = torch.nextafter(candidate[0], torch.tensor(2.))
        row = comparison(candidate, high.float(), high.float(), high)
        self.assertTrue(row['passed'])
        self.assertIsNone(row['kernel_to_quantized_ratio']['rel_l2'])
        json.dumps(row, allow_nan=False)
        self.assertFalse(comparison(high.float() + 1e-4, high.float(), high.float(), high)['passed'])

    def test_level1b_cannot_hide_level1a_failure(self):
        high = torch.tensor([1.0001, 2.0001], dtype=torch.float64)
        rq = high.float()
        low = rq.half()
        row = output_cast_comparison(low, rq, rq, rq, high, level1a_passed=False)
        self.assertFalse(row['passed'])
        self.assertEqual(row['kernel_extra_error']['max_abs'], 0)
        self.assertGreater(row['reference_output_cast_error']['max_abs'], 0)

    def test_output_cast_uses_matching_quantized_output_reference(self):
        high = torch.tensor([1.0001, 2.0001], dtype=torch.float64)
        rq = high.float()
        for dtype in (torch.float16, torch.bfloat16):
            row = output_cast_comparison(rq.to(dtype), rq, rq, rq, high, level1a_passed=True)
            self.assertTrue(row['passed'])
            self.assertEqual(row['kernel_extra_error']['max_abs'], 0)

    def test_overflow_is_separate_and_never_passes(self):
        high = torch.tensor([70000.], dtype=torch.float64)
        represented = high.float().half()
        self.assertEqual(representation_status(high.float(), represented), 'representation_overflow')
        row = output_cast_comparison(represented, high.float(), high.float(), high.float(), high, level1a_passed=True)
        self.assertTrue(row['representation_overflow'])
        self.assertFalse(row['passed'])
        json.dumps(row, allow_nan=False)

    def test_nonfinite_any_reference_or_candidate_fails_strict_json(self):
        for slot in range(4):
            values = [torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.float64)]
            values[slot][0] = float('nan')
            row = comparison(*values)
            self.assertFalse(row['passed'])
            json.dumps(row, allow_nan=False)

    def test_no_broadcast_or_fp64_candidate(self):
        with self.assertRaises(ValueError):
            comparison(torch.ones(2), torch.ones(1), torch.ones(2), torch.ones(2).double())
        with self.assertRaises(ValueError):
            comparison(torch.ones(2).double(), torch.ones(2), torch.ones(2), torch.ones(2).double())

    def test_reference_dtypes_cannot_be_miswired(self):
        with self.assertRaises(ValueError):
            comparison(torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(2))
        with self.assertRaises(ValueError):
            comparison(torch.ones(2), torch.ones(2).half(), torch.ones(2), torch.ones(2).double())
        with self.assertRaises(ValueError):
            comparison(torch.ones(2).half(), torch.ones(2), torch.ones(2).bfloat16(), torch.ones(2).double())
        with self.assertRaises(ValueError):
            comparison(torch.ones(2), torch.ones(2), torch.ones(2).double(), torch.ones(2).double())

    def test_sampled_matrix_has_all_required_axes_and_training_masks(self):
        from tools.v4_mixed_precision.gate import case_specs
        cases = list(case_specs())
        self.assertEqual(len(cases), 972)
        self.assertEqual(len({c['name'] for c in cases}), 972)
        primary = [c for c in cases if not c['range_probe']]
        self.assertEqual(len(primary), 960)
        self.assertEqual({c['scale'] for c in primary}, {1, 2, 3, 4})
        self.assertEqual({(c['height'], c['width']) for c in primary}, {(8, 8), (17, 19), (31, 37), (7, 1)})
        self.assertEqual({c['mode'] for c in primary}, {'training', 'no_grad', 'inference_mode', 'frozen'})
        self.assertEqual({c['kernel'] for c in primary if c['mode'] == 'training'}, {'normal', 'normalized', '1e-3', '1e-6', 'zero'})
        self.assertEqual(len({tuple(c['needs']) for c in primary if c['mode'] == 'training'}), 4)
        self.assertTrue(any(c['shared'] for c in primary))
        self.assertTrue(any(c['eps'] == 1e-8 and c['bias'] == 'normal' and c['regime'] == 'normal' for c in primary))

    def test_quantized_weak_zero_kernel_keeps_fp32_denominator(self):
        from tools.v4_mixed_precision.gate import case_specs, fixture, prepare, reference_diagnostics
        case = next(c for c in case_specs() if c['kernel'] == 'zero' and c['regime'] == 'weak')
        raw, _ = fixture(case)
        quantized = prepare(raw, (torch.float16, torch.float16, torch.float16, torch.float32),
                            dict(case, mode='frozen', needs=(False,)*4), 'cpu')
        diag = reference_diagnostics(quantized, case)
        self.assertTrue(diag['passed'])
        self.assertTrue(diag['q_finite'])
        self.assertGreater(diag['min_denominator'], 0)
        self.assertAlmostEqual(diag['min_denominator'], 1e-8, places=14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
