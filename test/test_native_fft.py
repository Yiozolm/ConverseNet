"""Native FP16/BF16 FFT, dynamic scaling, analytical adjoints and fallbacks."""
import concurrent.futures
import sys
import unittest

import torch
from extension_loader import ROOT, load_extension

sys.path.insert(0,str(ROOT))
from models.native_fft import native_fft


def relative_close(actual,expected,tol):
    assert torch.isfinite(actual).all()
    if expected.is_complex():
        a,e=actual.cdouble(),expected.cdouble()
    else:
        a,e=actual.double(),expected.double()
    error=(a-e).norm()/e.norm().clamp_min(1e-300)
    assert error<tol, f"relative L2 {error.item()} >= {tol}"


class NativeFFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        load_extension()
        cls.rfft=torch.ops.converse2d._native_rfft
        cls.irfft=torch.ops.converse2d._native_irfft

    def setUp(self):
        torch.manual_seed(918)
        torch.ops.converse2d.set_native_fft(False)
        torch.ops.converse2d.reset_native_fft_stats()

    def test_native_roundtrip_and_noncontiguous(self):
        for precision in (0,1):
            for h,w in ((2,2),(8,16),(64,64),(128,256)):
                x=torch.randn(2,3,w,h,device="cuda").transpose(-1,-2)
                with native_fft():
                    f=self.rfft(x,precision)
                    y=self.irfft(f,h,w,precision)
                tol=0.006 if precision==0 else 0.04
                relative_close(f,torch.fft.rfft2(x.double()),tol)
                relative_close(y,x,tol)
        stats=torch.ops.converse2d.native_fft_stats()
        self.assertEqual(stats["rfft"],8)
        self.assertEqual(stats["irfft"],8)
        self.assertEqual(stats["fallback_plan"],0)

    def test_dc_overflow_and_extreme_scaling(self):
        for precision in (0,1):
            for amplitude in (1.,10000.,1e-35,1e30):
                x=torch.full((1,1,256,256),amplitude,device="cuda")
                with native_fft():
                    f=self.rfft(x,precision)
                    y=self.irfft(f,256,256,precision)
                # Constant 256x256 has DC 65536 at unit amplitude. It must not
                # overflow internally merely because the storage is FP16.
                relative_close(f,torch.fft.rfft2(x.double()),0.004 if precision==0 else 0.02)
                relative_close(y,x,0.004 if precision==0 else 0.02)

    def test_native_adjoints_and_context_exit(self):
        for precision in (0,1):
            tol=0.007 if precision==0 else 0.05
            x=torch.randn(2,3,16,32,device="cuda",requires_grad=True)
            with native_fft():
                f=self.rfft(x,precision)
            self.assertFalse(torch.ops.converse2d.native_fft_enabled())
            # Arbitrary upstream values exercise DC/Nyquist row projection.
            g=torch.randn_like(f).conj()
            actual=torch.autograd.grad(f,x,g)[0]
            xr=x.detach().double().requires_grad_()
            expected=torch.autograd.grad(torch.fft.rfft2(xr),xr,g.cdouble())[0]
            relative_close(actual,expected,tol)
            p=torch.randn(2,3,16,17,device="cuda",dtype=torch.complex64,requires_grad=True)
            with native_fft():
                out=self.irfft(p,16,32,precision)
            upstream=torch.randn_like(out)
            actual=torch.autograd.grad(out,p,upstream)[0]
            pr=p.detach().cdouble().requires_grad_()
            ref=torch.fft.irfft2(pr,s=(16,32))
            expected=torch.autograd.grad(ref,pr,upstream.double())[0]
            relative_close(out,ref,tol)
            relative_close(actual,expected,tol)
        stats=torch.ops.converse2d.native_fft_stats()
        self.assertEqual(stats["rfft_adjoint"],2)
        self.assertEqual(stats["irfft_adjoint"],2)

    def test_higher_order_uses_fp32_analytical_adjoints(self):
        for precision in (0,1):
            x=torch.randn(1,2,8,16,device="cuda",requires_grad=True)
            with native_fft():
                out=self.irfft(self.rfft(x,precision),8,16,precision)
            gradient=torch.autograd.grad(out.square().sum(),x,create_graph=True)[0]
            second=torch.autograd.grad(gradient.sum(),x,create_graph=True)[0]
            relative_close(second,torch.full_like(x,2.),1e-5)
        stats=torch.ops.converse2d.native_fft_stats()
        self.assertEqual(stats["rfft_adjoint"],0)
        self.assertEqual(stats["irfft_adjoint"],0)

    def test_shape_fallback_matches_fp32(self):
        for h,w in ((68,76),(32,48),(1,16),(16,1)):
            x=torch.randn(1,2,h,w,device="cuda",requires_grad=True)
            with native_fft():
                f=self.rfft(x,0)
                y=self.irfft(f,h,w,1)
            ref=torch.fft.irfft2(torch.fft.rfft2(x),s=(h,w))
            torch.testing.assert_close(y,ref,atol=0,rtol=0)
            g=torch.autograd.grad(y.square().sum(),x)[0]
            eg=torch.autograd.grad(ref.square().sum(),x)[0]
            torch.testing.assert_close(g,eg,atol=0,rtol=0)
        stats=torch.ops.converse2d.native_fft_stats()
        self.assertEqual(stats["rfft"],0)
        self.assertGreaterEqual(stats["fallback_shape_device"],8)

    def test_policy_nesting_and_thread_locality(self):
        with native_fft():
            self.assertTrue(torch.ops.converse2d.native_fft_enabled())
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                self.assertFalse(pool.submit(torch.ops.converse2d.native_fft_enabled).result())
            with native_fft(False):
                self.assertFalse(torch.ops.converse2d.native_fft_enabled())
            self.assertTrue(torch.ops.converse2d.native_fft_enabled())
        self.assertFalse(torch.ops.converse2d.native_fft_enabled())

    def test_operator_training_and_inference(self):
        for dtype in (torch.float16,torch.bfloat16):
            for s in (1,2,3):
                x=torch.randn(2,3,16,16,device="cuda",dtype=dtype).requires_grad_()
                k=(torch.rand(1,3,3,3,device="cuda")/9).requires_grad_()
                b=torch.zeros(1,3,1,1,device="cuda",requires_grad=True)
                p=torch.randn(2,3,16*s,16*s,device="cuda",dtype=dtype).requires_grad_()
                for nearest in (False,True):
                    args=(x,k,b) if nearest else (x,p,k,b)
                    op=torch.ops.converse2d.forward_nearest if nearest else torch.ops.converse2d.forward
                    expected=op(*args,s,0.01)
                    with native_fft():
                        actual=op(*args,s,0.01)
                    self.assertEqual(actual.dtype,dtype)
                    tol=0.025 if dtype==torch.float16 else 0.12
                    relative_close(actual,expected,tol)
                    upstream=torch.randn_like(actual)/100
                    ag=torch.autograd.grad(actual,args,upstream)
                    eg=torch.autograd.grad(expected,args,upstream)
                    for a,e in zip(ag,eg):
                        relative_close(a,e,tol)
                    with torch.no_grad(),native_fft():
                        inference=op(*args,s,0.01)
                    relative_close(inference,expected,tol)

    def test_graph_capture_fallback_and_cache_clear(self):
        x=torch.randn(1,2,64,64,device="cuda")
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),native_fft(),torch.no_grad():
            for _ in range(3): self.rfft(x,0)
        torch.cuda.current_stream().wait_stream(stream)
        torch.ops.converse2d.reset_native_fft_stats()
        graph=torch.cuda.CUDAGraph()
        with native_fft(),torch.no_grad(),torch.cuda.graph(graph):
            out=self.rfft(x,0)
        torch.ops.converse2d.clear_cache()
        x.mul_(0.8)
        graph.replay()
        torch.testing.assert_close(out,torch.fft.rfft2(x),atol=0,rtol=0)
        self.assertEqual(torch.ops.converse2d.native_fft_stats()["fallback_capture"],1)
        torch.cuda.synchronize()

    def test_invalid_primitive_inputs(self):
        with self.assertRaises(RuntimeError): self.rfft(torch.ones(1,2,4,4,device="cuda",dtype=torch.float64),0)
        with self.assertRaises(RuntimeError): self.rfft(torch.ones(1,2,4,4,device="cuda"),2)
        with self.assertRaises(RuntimeError): self.irfft(torch.ones(1,2,4,3,device="cuda",dtype=torch.complex64),4,8,0)

    def test_amp_block_optimizer_step(self):
        from models.util_converse import ConverseBlock
        for dtype in (torch.float16,torch.bfloat16):
            block=ConverseBlock(8,8,padding=2).cuda()
            x=torch.randn(2,8,12,12,device="cuda")  # Actual FFT grid is 16x16.
            optimizer=torch.optim.SGD(block.parameters(),lr=1e-3)
            scaler=torch.amp.GradScaler("cuda",init_scale=128,enabled=dtype==torch.float16)
            before=block.conv1[-1].weight.detach().clone()
            with native_fft(),torch.autocast("cuda",dtype=dtype):
                output=block(x)
                loss=output.float().square().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for p in block.parameters():
                self.assertIsNotNone(p.grad)
                self.assertEqual(p.grad.dtype,torch.float32)
                self.assertTrue(torch.isfinite(p.grad).all())
            scaler.step(optimizer)
            scaler.update()
            self.assertFalse(torch.equal(before,block.conv1[-1].weight))
        stats=torch.ops.converse2d.native_fft_stats()
        self.assertGreater(stats["rfft_adjoint"],0)
        self.assertGreater(stats["irfft_adjoint"],0)

    def test_nondefault_stream_and_plan_eviction(self):
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),native_fft(),torch.no_grad():
            # 36 direction/precision/shape keys exceed the 32-entry plan LRU.
            for precision in (0,1):
                for h in (4,8,16):
                    for w in (4,8,16):
                        x=torch.randn(1,2,h,w,device="cuda")
                        y=self.irfft(self.rfft(x,precision),h,w,precision)
                        relative_close(y,x,0.04)
            torch.ops.converse2d.clear_cache()
            x=torch.randn(1,2,8,16,device="cuda")
            relative_close(self.irfft(self.rfft(x,0),8,16,0),x,0.006)
        torch.cuda.current_stream().wait_stream(stream)


if __name__=="__main__":
    unittest.main(verbosity=2)
