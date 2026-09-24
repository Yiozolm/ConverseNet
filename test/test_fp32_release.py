"""FP32 release contracts and independent FP64 error measurements."""
import json
import unittest
import torch
from support import ROOT, CUDATestCase, fixture, leaves, profiled
from models.converse_core import converse2d_reference, converse2d_fp32


class PortableFP32(unittest.TestCase):
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
        rows = []
        for s in (1, 2, 3, 4):
            for weak in (None, 0.0, 1e-6):
                raw, up = fixture(s, weak=weak)
                eps = 1e-5 if weak is None else 1e-8
                def run(fn, dtype):
                    a = [v.cuda().to(dtype).requires_grad_() for v in raw]
                    out = fn(*a, s, eps)
                    return (out, *torch.autograd.grad(out, a, up.cuda().to(dtype)))
                high = run(converse2d_reference, torch.float64)
                python = run(converse2d_reference, torch.float32)
                actual = run(torch.ops.converse2d.forward, torch.float32)
                for label, a, p, r in zip(('output', 'dx', 'dprior', 'dweight', 'dbias'), actual, python, high):
                    def errors(v):
                        delta = v.double()-r
                        return [delta.abs().max().item(), (delta.norm()/r.norm().clamp_min(1e-300)).item()]
                    ae, pe = errors(a), errors(p)
                    rows.append(dict(scale=s, weak=weak, tensor=label, candidate=ae, python=pe))
                    self.assertTrue(torch.isfinite(a).all())
                    self.assertLessEqual(ae[0], pe[0], rows[-1])
                    self.assertLessEqual(ae[1], pe[1], rows[-1])
        directory = ROOT/'artifacts/fp32_release'
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'fp64_errors.json').write_text(json.dumps(rows, indent=2))

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
