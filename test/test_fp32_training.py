"""Public FP32 training against an independent full-FFT FP64 reference."""
import itertools
import sys
import unittest

import torch
from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference, converse2d_reference_nearest


class FP32Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(9214)

    def data(self, h, w, s, kb=1, kc=3):
        # Strided public inputs exercise the FFT preparation as well as solve.
        x = torch.randn(2, 3, w, h, device='cuda').transpose(-1, -2)
        p = torch.randn(2, 3, w*s, h*s, device='cuda').transpose(-1, -2)
        kh, kw = min(3, h*s), min(3, w*s)
        k = torch.rand(kb, kc, kh, kw, device='cuda') / (kh*kw)
        b = torch.randn(1, 3, 1, 1, device='cuda')
        return tuple(t.requires_grad_() for t in (x, p, k, b))

    def compare(self, args, s, nearest=False, higher=False, eps=1e-3):
        op = torch.ops.converse2d.forward_nearest if nearest else torch.ops.converse2d.forward
        ref = converse2d_reference_nearest if nearest else converse2d_reference
        expected_args = tuple(t.detach().double().requires_grad_(t.requires_grad) for t in args)
        actual = op(*args, s, eps)
        expected = ref(*expected_args, s, eps)
        upstream = torch.randn_like(actual) / actual.numel()**0.5
        inputs = [t for t in args if t.requires_grad]
        refs = [t for t in expected_args if t.requires_grad]
        grads = torch.autograd.grad(actual, inputs, upstream, create_graph=higher)
        expected_grads = torch.autograd.grad(expected, refs, upstream.double(), create_graph=higher)
        torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-5)
        for a, e in zip(grads, expected_grads):
            torch.testing.assert_close(a.double(), e, atol=5e-5, rtol=5e-5)
        if higher:
            # A directional second derivative checks the public dispatch and
            # connected recomputation, including frozen formal arguments.
            vectors = [torch.randn_like(g) for g in grads]
            aa = sum((g*v).sum() for g, v in zip(grads, vectors))
            ee = sum((g*v.double()).sum() for g, v in zip(expected_grads, vectors))
            # A solve is linear in x/prior when both parameters are frozen.
            # Its constant VJP correctly has no second-derivative graph.
            self.assertEqual(aa.requires_grad, ee.requires_grad)
            if not ee.requires_grad:
                return
            ag = torch.autograd.grad(aa, inputs, allow_unused=True)
            eg = torch.autograd.grad(ee, refs, allow_unused=True)
            for a, e in zip(ag, eg):
                if e is None:
                    self.assertIsNone(a)
                else:
                    torch.testing.assert_close(a.double(), e, atol=2e-3, rtol=2e-4)

    def test_shapes_broadcasts_and_nearest(self):
        for h, w, s in ((1,1,1), (1,5,2), (4,1,3), (5,6,1), (5,7,2), (6,8,3), (3,5,4), (3,4,5)):
            for kb, kc in ((1,1), (1,3), (2,1), (2,3)):
                for nearest in (False, True):
                    with self.subTest(h=h, w=w, s=s, kb=kb, kc=kc, nearest=nearest):
                        x, p, k, b = self.data(h, w, s, kb, kc)
                        self.compare((x,k,b) if nearest else (x,p,k,b), s, nearest)

    def test_partial_gradients_and_higher_order(self):
        for needs in itertools.product((False,True), repeat=4):
            if not any(needs):
                continue
            args = tuple(t.detach().requires_grad_(need) for t, need in zip(self.data(3,4,2), needs))
            with self.subTest(needs=needs):
                self.compare(args, 2, higher=True)

    def test_shared_prior_higher_order(self):
        x, _, k, b = self.data(3,4,1)
        # nearest(scale=1) passes a single observation through both roles.
        self.compare((x,k,b), 1, nearest=True, higher=True)
        expected_x = x.detach().double().requires_grad_()
        expected_k = k.detach().double().requires_grad_()
        expected_b = b.detach().double().requires_grad_()
        a = torch.ops.converse2d.forward(x,x,k,b,1,1e-3)
        e = converse2d_reference(expected_x,expected_x,expected_k,expected_b,1,1e-3)
        g = torch.randn_like(a)
        for aa, ee in zip(torch.autograd.grad(a,(x,k,b),g),
                          torch.autograd.grad(e,(expected_x,expected_k,expected_b),g.double())):
            torch.testing.assert_close(aa.double(),ee,atol=5e-5,rtol=5e-5)

    def test_native_policy_does_not_lower_fp32_precision(self):
        args = self.data(8,16,2)
        op = torch.ops.converse2d
        previous = op.set_native_fft(False)
        try:
            a = op.forward(*args,2,1e-3)
            g = torch.randn_like(a)
            ag = torch.autograd.grad(a,args,g)
            op.reset_native_fft_stats()
            op.set_native_fft(True)
            b = op.forward(*args,2,1e-3)
            bg = torch.autograd.grad(b,args,g)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            for x,y in zip(ag,bg):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
            stats = op.native_fft_stats()
            self.assertEqual(stats['rfft'],0)
            self.assertEqual(stats['irfft'],0)
        finally:
            op.set_native_fft(previous)

    def test_stream_and_repeated_parameter_updates(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            args = self.data(7,8,3)
            for _ in range(3):
                self.compare(args,3)
                with torch.no_grad():
                    args[2].mul_(0.98)
                    args[3].add_(0.02)
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == '__main__':
    unittest.main(verbosity=2)
