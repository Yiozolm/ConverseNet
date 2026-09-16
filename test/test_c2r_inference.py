"""Direct-output C2R precision, graph ownership, cache eviction and streams."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest

import torch
from extension_loader import load_extension


class C2RInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')
        load_extension()

    def setUp(self):
        torch.manual_seed(960)
        torch.ops.converse2d.clear_cache()

    def tearDown(self):
        torch.cuda.synchronize()
        torch.ops.converse2d.clear_cache()

    @staticmethod
    def data(h=32, w=40, c=3, b=1, dtype=torch.float32):
        x=torch.randn(b,c,h,w,device='cuda',dtype=dtype)
        weight=torch.zeros(1,c,1,1,device='cuda',dtype=dtype)
        weight.fill_(1)
        bias=torch.zeros(1,c,1,1,device='cuda',dtype=dtype)
        return x,x,weight,bias,1,1e-5

    @staticmethod
    def assert_bits(a,b):
        assert a.dtype==b.dtype and a.shape==b.shape
        dtype={torch.float64:torch.int64,torch.float32:torch.int32,
               torch.float16:torch.int16,torch.bfloat16:torch.int16}[a.dtype]
        torch.testing.assert_close(a.contiguous().view(dtype),b.contiguous().view(dtype),atol=0,rtol=0)

    @torch.no_grad()
    def test_identity_matches_aten_c2r_bits(self):
        for dtype in (torch.float32,torch.float64,torch.float16,torch.bfloat16):
            for h,w in ((1,1),(1,9),(7,1),(7,9),(32,40),(64,64)):
                with self.subTest(dtype=dtype,h=h,w=w):
                    args=self.data(h,w,dtype=dtype)
                    compute=args[0].double() if dtype==torch.float64 else args[0].float()
                    expected=torch.fft.irfft2(torch.fft.rfft2(compute),s=(h,w)).to(dtype)
                    actual=torch.ops.converse2d.forward(*args)
                    self.assert_bits(actual,expected)

    @torch.no_grad()
    def test_graph_retains_more_plans_than_eager_cache(self):
        values=[self.data(8+i,19,c=2) for i in range(34)]
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.ops.converse2d.begin_graph_cache()
        try:
            with torch.cuda.stream(stream):
                for args in values:torch.ops.converse2d.forward(*args)
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                outputs=[torch.ops.converse2d.forward(*args) for args in values]
        finally:
            owners=torch.ops.converse2d.end_graph_cache()
        # Opaque CPU tokens prove that custom plans, rather than an accidental
        # cold-plan ATen fallback, are retained for every warmed shape.
        self.assertEqual(sum(t.device.type=='cpu' for t in owners),34)
        torch.ops.converse2d.clear_cache()
        for args in values:args[0].add_(0.125)
        graph.replay()
        for args,output in zip(values,outputs):
            self.assert_bits(output,torch.ops.converse2d.forward(*args))
        torch.cuda.synchronize()
        graph.reset()
        del graph,outputs,owners

    @torch.no_grad()
    def test_direct_capture_and_new_shape_fallback(self):
        args=self.data(17,21)
        for owned_scope in (False,True):
            # Prime the ATen fallback, then evict any custom plans.
            expected=torch.ops.converse2d.forward(*args)
            torch.ops.converse2d.clear_cache()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            if owned_scope:torch.ops.converse2d.begin_graph_cache()
            try:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):out=torch.ops.converse2d.forward(*args)
            finally:
                owners=torch.ops.converse2d.end_graph_cache() if owned_scope else []
            self.assertFalse(any(t.device.type=='cpu' for t in owners))
            torch.ops.converse2d.clear_cache()
            graph.replay()
            self.assert_bits(out,expected)
            torch.cuda.synchronize()
            graph.reset()
            del graph,out,owners

    @torch.no_grad()
    def test_threads_streams_and_cache_clear(self):
        values=[self.data(256,256,c=32),self.data(256,256,c=32)]
        expected=[torch.ops.converse2d.forward(*args) for args in values]
        torch.cuda.synchronize()
        barrier=threading.Barrier(2)
        def worker(index):
            with torch.cuda.device(0),torch.no_grad():
                stream=torch.cuda.Stream()
                barrier.wait()
                with torch.cuda.stream(stream):
                    outputs=[]
                    for _ in range(3):
                        outputs.append(torch.ops.converse2d.forward(*values[index]))
                        torch.ops.converse2d.clear_cache()
                stream.synchronize()
                return outputs
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs=list(pool.map(worker,range(2)))
        for actual,reference in zip(outputs,expected):
            for output in actual:self.assert_bits(output,reference)

    @torch.no_grad()
    def test_odd_batch_slice_alignment_and_tail(self):
        # Odd C,H,W with tile=1 makes alternating destination pointers aligned
        # only to float. This exercises the complex-alignment fallback.
        args=self.data(257,259,c=33,b=3)
        l2=torch.cuda.get_device_properties(0).L2_cache_size
        sample=args[0][0].numel()*4
        if not (l2//4 < sample <= l2):
            self.skipTest('geometry does not select one-sample tiles on this device')
        # On prime FFT sizes even the identity PSF's computed spectrum has
        # rounding error. Compare the identical solver on single-sample outputs.
        expected=torch.cat([torch.ops.converse2d.forward(x,x,args[2],args[3],1,1e-5)
                            for x in args[0].split(1)])
        self.assert_bits(torch.ops.converse2d.forward(*args),expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
