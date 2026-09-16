"""Nearest-prior fusion: independent references, gradients, tiles and ownership."""
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference

CPU = "--cpu" in sys.argv
if CPU:
    sys.argv.remove("--cpu")
DEVICE = "cpu" if CPU else "cuda"


class NearestPrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CPU and not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        load_extension(cpu_only=CPU)

    def setUp(self):
        torch.manual_seed(931)
        torch.ops.converse2d.clear_cache()

    def tearDown(self):
        torch.ops.converse2d.clear_cache()

    def data(self, b=2, c=3, h=5, w=7, s=2, kb=1, kc=3, dtype=torch.float32):
        x = torch.randn(b,c,h,w,device=DEVICE,dtype=dtype)
        weight = torch.randn(kb,kc,min(3,h*s),min(4,w*s),device=DEVICE,dtype=dtype) / 12
        bias = torch.full((1,c,1,1),-20.,device=DEVICE,dtype=dtype)
        return x,weight,bias

    def check(self, data, scale, eps=1e-8):
        x,weight,bias=data
        prior=F.interpolate(x,scale_factor=scale,mode="nearest") if scale>1 else x
        reference=converse2d_reference(x.double(),prior.double(),weight.double(),bias.double(),scale,eps)
        actual=torch.ops.converse2d.forward_nearest(*data,scale,eps)
        tol={torch.float64:1e-10,torch.float32:1e-4,torch.float16:0.008,torch.bfloat16:0.08}[x.dtype]
        torch.testing.assert_close(actual.double(),reference,atol=tol,rtol=tol)
        self.assertEqual(actual.dtype,x.dtype)
        return actual

    @torch.no_grad()
    def test_shapes_dtypes_broadcast_and_dynamic(self):
        for dtype in (torch.float32,torch.float64,torch.float16,torch.bfloat16):
            for h,w,s in ((1,1,2),(1,7,3),(8,1,4),(5,7,2),(6,8,3),(7,9,4),(5,8,5)):
                for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
                    with self.subTest(dtype=dtype,h=h,w=w,s=s,kb=kb,kc=kc):
                        data=self.data(h=h,w=w,s=s,kb=kb,kc=kc,dtype=dtype)
                        self.check(data,s)
                        with torch.inference_mode():
                            self.check(tuple(t.clone() for t in data),s)

    @torch.no_grad()
    def test_mutation_noncontiguous_and_independent_prior(self):
        for s in (1,2,3):
            data=tuple(t.transpose(-2,-1) for t in self.data(s=s,kb=2))
            x,w,b=data
            first=self.check(data,s)
            w.mul_(0.8);w.add_(0.02);b.add_(3.)
            updated=self.check(data,s)
            self.assertGreater((first-updated).abs().max().item(),1e-5)
            independent=torch.randn(x.size(0),x.size(1),x.size(2)*s,x.size(3)*s,device=DEVICE)
            # A non-negligible regularizer makes dependence on the supplied
            # prior observable even for an invertible scale=1 filter.
            actual=torch.ops.converse2d.forward(x,independent,w,b,s,0.1)
            expected=converse2d_reference(x.double(),independent.double(),w.double(),b.double(),s,0.1)
            torch.testing.assert_close(actual.double(),expected,atol=1e-4,rtol=1e-4)
            nearest=torch.ops.converse2d.forward_nearest(*data,s,0.1)
            self.assertGreater((actual-nearest).abs().max().item(),1e-4)

    def test_gradients_and_second_derivatives(self):
        data=tuple(t.requires_grad_() for t in self.data(b=1,c=2,h=3,w=4,kc=2,dtype=torch.float64))
        fn=lambda *args:torch.ops.converse2d.forward_nearest(*args,2,1e-5)
        self.assertTrue(torch.autograd.gradcheck(fn,data,fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(fn,data,fast_mode=True))
        out=fn(*data)
        x,w,b=data
        prior=F.interpolate(x,scale_factor=2,mode="nearest")
        expected=torch.ops.converse2d.forward(x,prior,w,b,2,1e-5)
        torch.testing.assert_close(out,expected,atol=1e-11,rtol=1e-11)
        grad=torch.autograd.grad(out.square().sum(),data)
        refgrad=torch.autograd.grad(expected.square().sum(),data)
        for a,r in zip(grad,refgrad):torch.testing.assert_close(a,r,atol=1e-9,rtol=1e-9)

    @torch.no_grad()
    def test_near_underflow(self):
        for dtype,amp in ((torch.float32,1e-38),(torch.float32,1e-40),(torch.float64,1e-310)):
            for s in (2,3,4):
                x,w,b=self.data(b=1,c=2,h=16,w=20,kc=2,dtype=dtype)
                x.mul_(amp);w.zero_();w[...,1,2]=1
                prior=F.interpolate(x,scale_factor=s,mode="nearest")
                args=(x,prior,w,b,s,1e-8)
                ref=converse2d_reference(*(t.double() if isinstance(t,torch.Tensor) else t for t in args))
                old=torch.ops.converse2d.forward(*args)
                actual=torch.ops.converse2d.forward_nearest(x,w,b,s,1e-8)
                factor=1/max(amp,1e-300);divisor=amp*factor
                error=(((actual.double()-ref)*factor)/divisor).abs().max().item()
                old_error=(((old.double()-ref)*factor)/divisor).abs().max().item()
                self.assertLessEqual(error,max(5e-6,old_error*1.5))

    def test_validation(self):
        x,w,b=self.data()
        for s,eps in ((0,1e-5),(2,0),(2,float('nan')),(2**62,1e-5)):
            with self.assertRaises(RuntimeError):torch.ops.converse2d.forward_nearest(x,w,b,s,eps)
        with self.assertRaises(RuntimeError):torch.ops.converse2d.forward_nearest(x,w,b.double(),2,1e-5)
        with self.assertRaises(RuntimeError):torch.ops.converse2d.forward_nearest(x,w,torch.zeros(1,2,1,1,device=DEVICE),2,1e-5)

    @unittest.skipIf(CPU,"CUDA fusion")
    @torch.no_grad()
    def test_wrappers_skip_interpolation_and_training_uses_it(self):
        from models.util_converse import Converse2D
        from models.converse_usrnet import ConvReverseDataNet
        layer=Converse2D(3,3,3,scale=2,backend="cuda").cuda().eval()
        data=ConvReverseDataNet(backend="cuda").cuda().eval()
        x=torch.randn(1,3,8,10,device=DEVICE)
        feature=torch.randn(1,64,8,10,device=DEVICE)
        kernel=torch.ones(1,1,3,3,device=DEVICE)/9
        with mock.patch.object(F,"interpolate",side_effect=AssertionError("HR interpolation in fused wrapper")):
            self.assertEqual(layer(x).shape,(1,3,16,20))
            self.assertEqual(data(feature,kernel,2).shape,(1,64,16,20))
        with torch.enable_grad():
            layer(x.requires_grad_()).sum().backward()
            self.assertIsNotNone(x.grad)

    @unittest.skipIf(CPU,"CUDA tiling")
    @torch.no_grad()
    def test_tiles_and_tail(self):
        l2=torch.cuda.get_device_properties(0).L2_cache_size
        for s in (2,3):
            c,h,w=8,128,129
            sample=c*h*w*s*s*4
            capacity=max(1,(l2//2)//sample)
            tile=1 << (capacity.bit_length()-1)
            batch=2*tile+1
            if l2<=0 or sample>l2 or batch>65:self.skipTest("unsuitable L2 geometry")
            for kb,kc in ((1,1),(1,c),(batch,1),(batch,c)):
                data=self.data(b=batch,c=c,h=h,w=w,s=s,kb=kb,kc=kc)
                x,k,b=data
                actual=torch.ops.converse2d.forward_nearest(*data,s,1e-8)
                expected=[]
                for i in range(batch):
                    wi=k if kb==1 else k[i:i+1]
                    expected.append(self.check((x[i:i+1],wi,b),s))
                torch.testing.assert_close(actual,torch.cat(expected),atol=1e-4,rtol=1e-4)
        # Exercise the same partial-tile path inside a graph-owned cache scope.
        torch.ops.converse2d.begin_graph_cache()
        try:
            torch.ops.converse2d.forward_nearest(*data,s,1e-8)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output=torch.ops.converse2d.forward_nearest(*data,s,1e-8)
        finally:
            owned=torch.ops.converse2d.end_graph_cache()
        torch.ops.converse2d.clear_cache()
        x.add_(0.1)
        graph.replay()
        expected=[self.check((x[i:i+1],k[i:i+1],b),s) for i in range(batch)]
        torch.testing.assert_close(output,torch.cat(expected),atol=1e-4,rtol=1e-4)
        torch.cuda.synchronize()
        del graph,output,owned

    @unittest.skipIf(CPU,"CUDA graphs")
    @torch.no_grad()
    def test_graph_phase_ownership_and_eviction(self):
        data=self.data(h=8,w=10,kb=2)
        for owned_scope in (False,True):
            for _ in range(3):self.check(data,3)
            if owned_scope:torch.ops.converse2d.begin_graph_cache()
            try:
                for _ in range(3):torch.ops.converse2d.forward_nearest(*data,3,1e-8)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output=torch.ops.converse2d.forward_nearest(*data,3,1e-8)
            finally:
                owned=torch.ops.converse2d.end_graph_cache() if owned_scope else []
            if owned_scope:self.assertTrue(any(t.ndim==1 and t.is_complex() for t in owned))
            torch.ops.converse2d.clear_cache()
            for h in range(1,67):
                other=self.data(b=1,c=1,h=h,w=3,kc=1)
                torch.ops.converse2d.forward_nearest(*other,2,1e-8)
            data[0].add_(0.3)
            if not owned_scope:data[1].mul_(0.9)  # Direct capture recomputes its own dynamic spectrum.
            graph.replay()
            torch.testing.assert_close(output,self.check(data,3),atol=1e-4,rtol=1e-4)
            torch.cuda.synchronize()
            del graph,output,owned

    @unittest.skipIf(CPU,"CUDA streams")
    @torch.no_grad()
    def test_stream_and_dtype_phase_keys(self):
        for dtype in (torch.float32,torch.float64):
            data=self.data(dtype=dtype)
            expected=self.check(data,3)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):actual=self.check(data,3)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(actual,expected)


if __name__=="__main__":
    unittest.main(verbosity=2)
