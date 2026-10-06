"""Admission tests: reject regressions without making bit identity a gate."""
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
from numerical_policy import (comparison, error_metrics, seeded_comparison,
                              denominator_statistics, write_report)
from numerical_cases import numerical_cases, NumericalCase


class NumericalPolicy(unittest.TestCase):
    def test_improvement_can_differ_from_baseline(self):
        ref = torch.ones(8, dtype=torch.float64)
        row = comparison(torch.ones(8), torch.ones(8) + 1e-5, ref)
        self.assertTrue(row['passed'])
        self.assertEqual(row['candidate']['rel_l2'], 0)

    def test_normal_and_weak_have_different_l2_limits(self):
        ref = torch.ones(8, dtype=torch.float64)
        baseline, candidate = torch.ones(8) + 1e-4, torch.ones(8) + 1.4e-4
        self.assertFalse(comparison(candidate, baseline, ref)['passed'])
        self.assertTrue(comparison(candidate, baseline, ref, regime='weak')['passed'])

    def test_single_outlier_cannot_hide_in_l2(self):
        ref = torch.ones(100, dtype=torch.float64)
        baseline, candidate = torch.ones(100) + 1e-4, torch.ones(100)
        candidate[0] += 2e-4
        row = comparison(candidate, baseline, ref)
        self.assertLess(row['candidate']['rel_l2'], row['limits']['rel_l2'])
        self.assertFalse(row['passed'])

    def test_zero_baseline_floor_and_strict_json(self):
        ref = torch.ones(8, dtype=torch.float64)
        baseline = ref.float()
        candidate = baseline.clone()
        candidate[0] = torch.nextafter(candidate[0], torch.tensor(2.))
        row = comparison(candidate, baseline, ref)
        self.assertTrue(row['passed'])
        self.assertIsNone(row['ratio']['rel_l2'])
        json.dumps(row, allow_nan=False)
        self.assertFalse(comparison(baseline + 1e-4, baseline, ref)['passed'])

    def test_nonfinite_candidate_baseline_or_reference_fails(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for index in range(3):
                values = [torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.float64)]
                values[index][0] = bad
                row = comparison(*values)
                self.assertFalse(row['passed'])
                json.dumps(row, allow_nan=False)

    def test_no_implicit_broadcast_or_fp64_candidate(self):
        with self.assertRaises(ValueError):
            comparison(torch.ones(2), torch.ones(1), torch.ones(2, dtype=torch.float64))
        with self.assertRaises(ValueError):
            comparison(torch.ones(2).double(), torch.ones(2), torch.ones(2).double())
        with self.assertRaises(ValueError):
            comparison(torch.ones(2, dtype=torch.complex64) * 1j,
                       torch.ones(2, dtype=torch.complex64), torch.ones(2).double())

    def _seeded_rows(self, scales, baseline_error=1e-4):
        ref = torch.ones(16, dtype=torch.float64)
        baseline = torch.ones(16) + baseline_error
        return [comparison(torch.ones(16) + baseline_error * scale, baseline, ref) for scale in scales]

    def test_seeded_budget_accepts_equal_quality_noise(self):
        rows = self._seeded_rows([0.5, 2.0, 0.8, 1.25, 1.6, 0.7, 1.0, 0.9])
        self.assertGreater(sum(not r['passed'] for r in rows), 0)
        result = seeded_comparison(rows)
        self.assertTrue(result['passed'])
        self.assertLessEqual(result['rel_l2']['geomean_ratio'], 1.25)
        json.dumps(result, allow_nan=False)

    def test_seeded_budget_rejects_systematic_regression(self):
        result = seeded_comparison(self._seeded_rows([1.4] * 8))
        self.assertFalse(result['passed'])
        self.assertFalse(result['rel_l2']['passed'])

    def test_seeded_budget_requires_seeds_finiteness_and_one_regime(self):
        with self.assertRaises(ValueError):
            seeded_comparison(self._seeded_rows([1.0] * 7))
        rows = self._seeded_rows([1.0] * 8)
        ref = torch.ones(16, dtype=torch.float64)
        bad = torch.ones(16)
        bad[0] = float('nan')
        rows[3] = comparison(bad, torch.ones(16), ref)
        self.assertFalse(seeded_comparison(rows)['passed'])
        with self.assertRaises(ValueError):
            seeded_comparison(self._seeded_rows([1.0] * 8), regime='weak')

    def test_seeded_budget_keeps_floors(self):
        ref = torch.ones(8, dtype=torch.float64)
        baseline = ref.float()
        candidate = baseline.clone()
        candidate[0] = torch.nextafter(candidate[0], torch.tensor(2.))
        result = seeded_comparison([comparison(candidate, baseline, ref) for _ in range(8)])
        self.assertTrue(result['passed'])

    def test_distribution_and_complex_components(self):
        ref = torch.zeros(4, dtype=torch.complex128)
        row = error_metrics(torch.tensor([0, 1j, 2j, 3j]), ref, distribution=True)
        self.assertEqual(row['max_abs'], 3.)
        self.assertEqual(set(row['relative_error']), {'p50', 'p90', 'p99', 'p999', 'max'})

    def test_denominator_includes_regularizer_and_broadcast(self):
        case = NumericalCase(3, kc=1, distribution='weak_0')
        raw, _ = case.fixture()
        d = denominator_statistics(raw, 3, case.eps, 'cpu')
        expected = (torch.sigmoid(torch.tensor(-49., dtype=torch.float64)) + case.eps).item()
        for value in d.values():
            self.assertAlmostEqual(value, expected, places=20)

    def test_matrix_coverage_and_unique_case_ids(self):
        cases = list(numerical_cases())
        self.assertEqual(len({c.name for c in cases}), len(cases))
        self.assertEqual({c.scale for c in cases}, {1, 2, 3, 4})
        self.assertEqual({(c.kb, c.kc) for c in cases}, {(1, 1), (1, 3), (2, 1), (2, 3)})
        self.assertEqual({c.layout for c in cases}, {'contiguous', 'strided', 'transpose'})
        self.assertTrue(any(c.width == 1 for c in cases))
        self.assertEqual(len(list(numerical_cases('fast'))), 16)

    def test_report_preserves_failure_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'CONVERSE2D_REPORT_DIR': directory}):
            row = comparison(torch.ones(2) * float('nan'), torch.ones(2), torch.ones(2).double())
            first = write_report('test', [row], device='cpu')
            second = write_report('test', [row], device='cpu')
            self.assertNotEqual(first, second)
            self.assertFalse(json.loads(first.read_text())['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
