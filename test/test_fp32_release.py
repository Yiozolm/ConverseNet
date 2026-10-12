"""FP32 release contracts and independent FP64 error measurements."""
import os
import unittest
import torch
from support import CUDATestCase, fixture, leaves, profiled
from models.converse_core import converse2d_reference, converse2d_fp32
from fp32_baseline import converse2d_fp32 as baseline_fp32
from numerical_cases import numerical_cases, evaluate
from numerical_policy import comparison, denominator_statistics, write_report


def check_numerical_matrix(test, candidate, device, level):
    rows = []
    cases = list(numerical_cases(level))
    try:
        for case in cases:
            raw, up = case.fixture()
            denominator = denominator_statistics(raw, case.scale, case.eps, device)
            for mode in ('training', 'no_grad', 'inference_mode', 'frozen'):
                with test.subTest(case=case.name, mode=mode):
                    high = evaluate(converse2d_reference, raw, up, case,
                                    device=device, dtype=torch.float64, mode=mode)
                    baseline = evaluate(baseline_fp32, raw, up, case, device=device, mode=mode)
                    actual = evaluate(candidate, raw, up, case, device=device, mode=mode)
                    test.assertEqual(len(actual), len(high))
                    test.assertEqual(len(baseline), len(high))
                    failures = []
                    for label, a, b, r in zip(('output', 'dx', 'dprior', 'dweight', 'dbias'), actual, baseline, high):
                        row = dict(case=case.name, scale=case.scale, eps=case.eps,
                                   mode=mode, tensor=label, **denominator,
                                   **comparison(a, b, r, regime='weak' if case.weak else 'normal',
                                                distribution=True))
                        # Fixed spatial tolerances target ordinary production
                        # amplitudes. Dynamic-range stress retains BOTH FP64
                        # gates; its allclose result is diagnostic, not a gate.
                        if not case.weak and mode != 'training':
                            row['spatial_smoke_required'] = case.distribution != 'dynamic'
                            try:
                                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                                row['spatial_smoke_passed'] = True
                            except AssertionError:
                                row['spatial_smoke_passed'] = False
                                if row['spatial_smoke_required']:
                                    row['passed'] = False
                        rows.append(row)
                        if not row['passed']:
                            failures.append(row)
                    test.assertFalse(failures, failures)
    finally:
        path = write_report('fp64_errors', rows, device=device,
                            extra={'matrix_level': level, 'expected_rows': len(cases) * 8,
                                   'complete': len(rows) == len(cases) * 8})
        print(f'Numerical report: {path}')


class PortableFP32(unittest.TestCase):
    def test_portable_fp64_budget(self):
        check_numerical_matrix(self, converse2d_fp32, 'cpu', 'fast')

    def test_cpu_full_training_and_half_inference(self):
        for s in (1, 2, 3, 4):
            raw, upstream = fixture(s)
            data = leaves(raw, 'cpu')
            out, events = profiled(lambda: converse2d_fp32(*data, s))
            self.assertIn('aten::fft_fft2', events)
            ref = converse2d_reference(*data, s)
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            ag = torch.autograd.grad(out, data, upstream)
            rg = torch.autograd.grad(ref, data, upstream)
            for a, r in zip(ag, rg):
                torch.testing.assert_close(a, r, atol=0, rtol=0)
            with torch.no_grad():
                inferred, events = profiled(lambda: converse2d_fp32(*data, s))
            self.assertIn('aten::fft_rfft2', events)
            self.assertNotIn('aten::fft_fft2', events)
            torch.testing.assert_close(inferred, ref, atol=3e-5, rtol=3e-5)

    def test_portable_rejects_non_fp32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            raw, _ = fixture(2, dtype=dtype)
            with self.assertRaisesRegex(ValueError, 'FP32'):
                converse2d_fp32(*raw, 2)

class ReleaseContracts(CUDATestCase):
    def test_independent_fp64_error_noninferiority(self):
        check_numerical_matrix(self, torch.ops.converse2d.forward, 'cuda',
                               os.environ.get('CONVERSE2D_NUMERICAL_LEVEL', 'full'))

    def test_higher_order_matches_python_fp32(self):
        for s in (1, 2, 3):
            torch.manual_seed(271+s)
            raw = [torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3*s, 4*s),
                   torch.rand(1, 1, 3, 3)/9, torch.zeros(1, 1, 1, 1)]
            results = []
            for fn in (torch.ops.converse2d.forward, converse2d_reference):
                a = leaves(raw, 'cuda')
                out = fn(*a, s, .1)
                g = torch.autograd.grad(out.square().mean(), a, create_graph=True)
                h = torch.autograd.grad(sum(v.square().mean() for v in g), a)
                results.append(h)
            for actual, expected in zip(*results):
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_cache_updates_and_streams(self):
        raw, _ = fixture(2)
        x, p, k, b = leaves(raw, 'cuda')
        with torch.no_grad():
            old = torch.ops.converse2d.forward(x, p, k, b, 2)
            k.add_(.02)
            cached = torch.ops.converse2d.forward(x, p, k, b, 2)
            torch.ops.converse2d.clear_cache()
            fresh = torch.ops.converse2d.forward(x, p, k, b, 2)
            self.assertFalse(torch.equal(old, fresh))
            torch.testing.assert_close(cached, fresh, atol=0, rtol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                other = torch.ops.converse2d.forward(x, p, k, b, 2)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(other, fresh, atol=0, rtol=0)

    def test_module_dtype_contract(self):
        from models.util_converse import Converse2D
        from models.converse_usrnet import ConvReverseDataNet
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            model = Converse2D(3, 3, 3, backend='pytorch').cuda().to(dtype)
            with self.assertRaisesRegex(ValueError, 'FP32'):
                model(torch.randn(1, 3, 5, 7, device='cuda', dtype=dtype))
            d = ConvReverseDataNet(backend='pytorch').cuda().to(dtype)
            with self.assertRaisesRegex(ValueError, 'FP32'):
                d(torch.randn(1, 64, 5, 7, device='cuda', dtype=dtype),
                  torch.rand(1, 64, 3, 3, device='cuda', dtype=dtype), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
