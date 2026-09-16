"""FP32/FP16/BF16 FFT tiles, FP64 fallback, precision and graph ownership."""
import sys
import unittest

import torch
from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference


class FFTBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()
        cls.l2 = torch.cuda.get_device_properties(0).L2_cache_size
        if cls.l2 <= 0:
            raise unittest.SkipTest('GPU does not report L2 capacity')

    def setUp(self):
        torch.manual_seed(816)
        torch.ops.converse2d.clear_cache()

    def tearDown(self):
        torch.ops.converse2d.clear_cache()

    def data(self, scale=2, dtype=torch.float32, kb=False, kc=True):
        c, h, w = 16, 128, 129
        size = 8 if dtype == torch.float64 else 4
        sample_bytes = c*h*w*scale*scale*size
        capacity = max(1, (self.l2//2)//sample_bytes)
        tile = 1 << (capacity.bit_length()-1)
        b = tile*2+1  # Two complete tiles and a final one-sample tail.
        if sample_bytes > self.l2 or b > 65:
            self.skipTest('Test geometry is unsuitable for this GPU cache size')
        x = torch.randn(b,c,h,w,device='cuda',dtype=dtype)
        prior = torch.randn(b,c,h*scale,w*scale,device='cuda',dtype=dtype)
        shape = (b if kb else 1,c if kc else 1,3,3)
        weight = torch.randn(shape,device='cuda',dtype=dtype).flatten(2).softmax(-1).reshape(shape)
        bias = torch.randn(1,c,1,1,device='cuda',dtype=dtype)
        return x,prior,weight,bias

    def reference(self, data, scale):
        x,prior,weight,bias = data
        # All samples use the independent full-FFT reference, without its
        # full-batch temporary memory footprint.
        parts=[]
        for i in range(x.size(0)):
            k = weight if weight.size(0)==1 else weight[i:i+1]
            parts.append(converse2d_reference(x[i:i+1].double(),prior[i:i+1].double(),
                                             k.double(),bias.double(),scale).to(x.dtype))
        return torch.cat(parts)

    def assert_reference(self, data, scale):
        actual = torch.ops.converse2d.forward(*data,scale,1e-5)
        reference = self.reference(data,scale)
        tol = {torch.float64:2e-10,torch.float32:1e-4,torch.float16:0.008,torch.bfloat16:0.08}[actual.dtype]
        torch.testing.assert_close(actual,reference,atol=tol,rtol=tol)
        return actual

    @torch.no_grad()
    def test_independent_prior_and_filter_broadcast(self):
        for dtype in (torch.float32,torch.float64):
            for kb,kc in ((False,False),(False,True),(True,False),(True,True)):
                with self.subTest(dtype=dtype,kb=kb,kc=kc):
                    data = self.data(dtype=dtype,kb=kb,kc=kc)
                    self.assert_reference(data,2)
                    # Mutation must invalidate the full filter cache, including
                    # all batch slices subsequently used by the tiled solver.
                    data[2].mul_(0.9)
                    data[2].add_(0.01)
                    self.assert_reference(data,2)
                    torch.ops.converse2d.clear_cache()

    @torch.no_grad()
    def test_noncontiguous_shared_prior_and_scale_three(self):
        for scale in (1,3,4):
            data = self.data(scale=scale,kb=True)
            x,prior,weight,bias = (t.transpose(-2,-1) for t in data)
            if scale==1:
                prior=x
            self.assert_reference((x,prior,weight,bias),scale)

    @torch.no_grad()
    def test_half_and_bfloat16_tiles(self):
        for dtype in (torch.float16,torch.bfloat16):
            for scale in (1,2):
                with self.subTest(dtype=dtype,scale=scale):
                    data = self.data(scale=scale,dtype=dtype)
                    self.assert_reference(data,scale)

    @torch.inference_mode()
    def test_uncached_per_sample_filters(self):
        for scale in (2,3):
            self.assert_reference(self.data(scale=scale,kb=True,kc=False),scale)

    @torch.no_grad()
    def test_identity_near_underflow(self):
        for dtype,amp in ((torch.float32,1e-40),(torch.float64,1e-310)):
            x,_,weight,bias = self.data(scale=1,dtype=dtype)
            x.mul_(amp)
            weight.zero_()
            weight[...,1,1]=1
            actual=torch.ops.converse2d.forward(x,x,weight,bias,1,1e-5)
            expected=torch.cat([torch.ops.converse2d.forward(xx,xx,weight,bias,1,1e-5)
                                for xx in x.split(1)])
            # Compare normalized errors so absolute tolerances cannot hide a
            # regression that flushes the entire subnormal signal to zero.
            factor=1/max(amp,1e-300)
            divisor=amp*factor
            a=(actual.double()*factor)/divisor
            e=(expected.double()*factor)/divisor
            torch.testing.assert_close(a,e,atol=1e-6,rtol=1e-6)

    @torch.no_grad()
    def test_graph_owns_spectra_after_eager_cache_clear(self):
        data=self.data(scale=2)
        torch.ops.converse2d.begin_graph_cache()
        try:
            for _ in range(3):
                torch.ops.converse2d.forward(*data,2,1e-5)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output=torch.ops.converse2d.forward(*data,2,1e-5)
        finally:
            owned=torch.ops.converse2d.end_graph_cache()
        self.assertTrue(all(isinstance(t,torch.Tensor) for t in owned))
        torch.ops.converse2d.clear_cache()
        data[0].add_(0.2)
        graph.replay()
        expected=self.reference(data,2)
        torch.testing.assert_close(output,expected,atol=1e-4,rtol=1e-4)
        torch.cuda.synchronize()
        del graph,output,owned


if __name__=='__main__':
    unittest.main(verbosity=2)
