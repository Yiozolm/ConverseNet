"""Precision guards for spectral I/O changes, including near-underflow inputs."""
import json
import sys
import unittest

import torch
from extension_loader import ROOT, load_extension
from spectral_baseline import load_baseline
sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference


class SpectralIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_extension()
        cls.baseline = load_baseline()

    def tearDown(self):
        torch.ops.converse2d.clear_cache()
        self.baseline.clear_cache()

    def test_identity_near_underflow(self):
        torch.manual_seed(916)
        for dtype, amplitudes in ((torch.float32, (1., 1e-30, 1e-38, 1e-40)),
                                  (torch.float64, (1., 1e-280, 1e-310))):
            for amplitude in amplitudes:
                with self.subTest(dtype=dtype, amplitude=amplitude), torch.no_grad():
                    x = torch.randn(1, 2, 64, 80, device='cuda', dtype=dtype) * amplitude
                    weight = torch.zeros(1, 2, 3, 3, device='cuda', dtype=dtype)
                    weight[..., 1, 1] = 1
                    bias = torch.zeros(1, 2, 1, 1, device='cuda', dtype=dtype)
                    args = (x, x, weight, bias, 1, 1e-5)
                    baseline = self.baseline.forward(*args)
                    actual = torch.ops.converse2d.forward(*args)
                    with torch.inference_mode():
                        dynamic = torch.ops.converse2d.forward(x,x,weight.clone(),bias,1,1e-5)
                    # The exact closed form is x for an identity PSF and x0=x.
                    # Normalize before subtraction to measure subnormal errors.
                    multiplier = 1. / max(amplitude, 1e-300)
                    divisor = amplitude * multiplier
                    ref = (x.double() * multiplier) / divisor
                    old_error = ((baseline.double()*multiplier)/divisor-ref).abs().max().item()
                    error = ((actual.double()*multiplier)/divisor-ref).abs().max().item()
                    dynamic_error = ((dynamic.double()*multiplier)/divisor-ref).abs().max().item()
                    floor = 5e-6 if dtype == torch.float32 else 1e-12
                    print(json.dumps({'dtype':str(dtype),'amplitude':amplitude,
                                      'baseline_error':old_error,'optimized_error':error,
                                      'dynamic_error':dynamic_error}),flush=True)
                    self.assertLessEqual(error, max(floor, old_error * 1.5))
                    self.assertLessEqual(dynamic_error, max(floor, old_error * 1.5))

    def test_dynamic_denominator_and_rectangular_psfs(self):
        torch.manual_seed(917)
        for dtype in (torch.float32, torch.float64):
            for scale in (1, 2, 3, 4, 5):
                for kh, kw in ((1, 1), (2, 4), (3, 3)):
                    with self.subTest(dtype=dtype, scale=scale, kernel=(kh,kw)):
                        x = torch.randn(2, 3, 5, 7, device='cuda', dtype=dtype)
                        prior = torch.randn(2, 3, 5*scale, 7*scale, device='cuda', dtype=dtype)
                        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                            weight = torch.randn(kb, kc, kw, kh, device='cuda', dtype=dtype).transpose(-2,-1)
                            weight = weight / (kh*kw)
                            bias = torch.full((1,3,1,1), -20., device='cuda', dtype=dtype)
                            args = (x, prior, weight, bias, scale, 1e-8)
                            reference = converse2d_reference(*(t.double() if isinstance(t,torch.Tensor) else t for t in args))
                            with torch.no_grad():
                                old = self.baseline.forward(*args)
                                cached = torch.ops.converse2d.forward(*args)
                            with torch.inference_mode():
                                # Inference tensors bypass the fixed-kernel cache.
                                dynamic = torch.ops.converse2d.forward(x,prior,weight.clone(),bias,scale,1e-8)
                            tol = 1e-4 if dtype == torch.float32 else 1e-10
                            for result in (old, cached, dynamic):
                                torch.testing.assert_close(result.double(),reference,atol=tol,rtol=tol)

    def test_prepared_spectrum_is_bitwise_identical(self):
        for dtype in (torch.float32, torch.float64):
            for h,w,kh,kw in ((5,7,2,4),(6,8,3,3),(1,9,1,4),(7,1,2,1)):
                with self.subTest(dtype=dtype,shape=(h,w,kh,kw)), torch.no_grad():
                    x = torch.randn(1,2,h,w,device='cuda',dtype=dtype)
                    weight = torch.randn(1,2,kw,kh,device='cuda',dtype=dtype).transpose(-2,-1)
                    bias = torch.zeros(1,2,1,1,device='cuda',dtype=dtype)
                    torch.ops.converse2d.begin_graph_cache()
                    try:
                        torch.ops.converse2d.forward(x,x,weight,bias,1,1e-5)
                    finally:
                        owned = torch.ops.converse2d.end_graph_cache()
                    padded = torch.nn.functional.pad(weight,(0,w-kw,0,h-kh))
                    expected = torch.fft.rfft2(torch.roll(padded,(-(kh//2),-(kw//2)),(-2,-1)))
                    torch.testing.assert_close(owned[1],expected,atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
