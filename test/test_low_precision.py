"""Mixed activations/master parameters, AMP training and low-precision stores."""
import copy
import sys
import unittest

import torch
from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))
from models.converse_core import converse2d_reference, converse2d_reference_nearest

CPU = "--cpu" in sys.argv
if CPU:
    sys.argv.remove("--cpu")
DEVICE = "cpu" if CPU else "cuda"
DTYPES = (torch.float16, torch.bfloat16)


class LowPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CPU and not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        load_extension(cpu_only=CPU)

    def setUp(self):
        torch.manual_seed(417)
        torch.ops.converse2d.clear_cache()

    def data(self, dtype, b=2, c=3, h=5, w=7, kb=1, kc=3):
        return (torch.randn(b,c,h,w,device=DEVICE,dtype=dtype),
                torch.randn(kb,kc,3,3,device=DEVICE).softmax(-1) / 3,
                torch.randn(1,c,1,1,device=DEVICE))

    def test_mixed_forward_and_gradients(self):
        for dtype in DTYPES:
            tol = 0.008 if dtype == torch.float16 else 0.06
            for s in (1,2,3):
                for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
                    with self.subTest(dtype=dtype,s=s,kb=kb,kc=kc):
                        x,w,b = (t.requires_grad_() for t in self.data(dtype,kb=kb,kc=kc))
                        prior = torch.randn(2,3,5*s,7*s,device=DEVICE,dtype=dtype,requires_grad=True)
                        for nearest in (False,True):
                            args = (x,w,b) if nearest else (x,prior,w,b)
                            ref = converse2d_reference_nearest if nearest else converse2d_reference
                            op = torch.ops.converse2d.forward_nearest if nearest else torch.ops.converse2d.forward
                            expected = ref(*args,s,1e-3)
                            actual = op(*args,s,1e-3)
                            self.assertEqual(actual.dtype,dtype)
                            torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
                            upstream = torch.randn_like(actual) / actual.numel()
                            eg = torch.autograd.grad(expected,args,upstream)
                            ag = torch.autograd.grad(actual,args,upstream)
                            for a,e,t in zip(ag,eg,args):
                                self.assertEqual(a.dtype,t.dtype)
                                torch.testing.assert_close(a,e,atol=tol/100,rtol=tol)
                            for ctx in (torch.no_grad,torch.inference_mode):
                                with ctx():
                                    out = op(*args,s,1e-3)
                                    torch.testing.assert_close(out,expected,atol=tol,rtol=tol)

    def test_second_derivatives(self):
        for dtype in DTYPES:
            x,w,b = (t.requires_grad_() for t in self.data(dtype,b=1,c=1,kc=1))
            results = []
            for fn in (converse2d_reference_nearest,torch.ops.converse2d.forward_nearest):
                out = fn(x,w,b,3,0.1)
                grad = torch.autograd.grad(out.float().square().mean(),(x,w,b),create_graph=True)
                second = torch.autograd.grad(sum(g.float().square().mean() for g in grad),(x,w,b))
                self.assertTrue(all(torch.isfinite(t).all() for t in (*grad,*second)))
                results.append(second)
            for expected,actual in zip(*results):
                torch.testing.assert_close(actual,expected,atol=0.008 if dtype==torch.float16 else 0.06,rtol=0.06)

    @torch.no_grad()
    def test_cache_dtype_switch_and_mutation(self):
        x,w,b = self.data(torch.float16)
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float16):
            for s in (1,3):
                x = x.to(dtype)
                torch.ops.converse2d.forward_nearest(x,w,b,s)
                w.add_(0.001)
                b.add_(0.1)
                out = torch.ops.converse2d.forward_nearest(x,w,b,s)
                expected = converse2d_reference_nearest(x,w,b,s)
                tol = {torch.float16:0.008,torch.bfloat16:0.06,torch.float32:5e-5}[dtype]
                torch.testing.assert_close(out,expected,atol=tol,rtol=tol)

    @torch.no_grad()
    def test_same_prior_noncontiguous_and_large_batch(self):
        for dtype in DTYPES:
            for batch,h,w in ((2,5,7),(17,16,18)):
                x,k,b = self.data(dtype,b=batch,h=h,w=w)
                x = x.transpose(-1,-2)
                expected = converse2d_reference(x,x,k,b,1)
                out = torch.ops.converse2d.forward(x,x,k,b,1)
                torch.testing.assert_close(out,expected,atol=0.06,rtol=0.06)

    def test_invalid_mixed_dtypes(self):
        x,w,b = self.data(torch.float16)
        for bad_w in (w.double(),w.bfloat16(),w.int()):
            with self.assertRaises(RuntimeError):
                torch.ops.converse2d.forward_nearest(x,bad_w,b,2)
            with self.assertRaises(ValueError):
                converse2d_reference_nearest(x,bad_w,b,2)
        with self.assertRaises(RuntimeError):
            torch.ops.converse2d.forward_nearest(x.float(),w.half(),b,2)

    def test_autocast_preserves_activation_dispatch(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            x,w,b = self.data(dtype)
            if dtype==torch.float64:
                w,b = w.double(),b.double()
            with torch.autocast(DEVICE,dtype=torch.bfloat16):
                out = torch.ops.converse2d.forward_nearest(x,w,b,2)
            self.assertEqual(out.dtype,dtype)

    def test_layernorm_fp32_statistics(self):
        from models.util_converse import LayerNorm
        for dtype in DTYPES:
            # Large variance overflows when squared in half; almost-constant
            # inputs exercise variance cancellation and eps handling.
            for source in (torch.randn(2,8,5,7,device=DEVICE)*10000,
                           torch.ones(2,8,5,7,device=DEVICE)+torch.randn(2,8,5,7,device=DEVICE)*0.01):
                x = source.to(dtype).requires_grad_()
                norm = LayerNorm(8,data_format="channels_first").to(DEVICE)
                expected = norm(x.float())
                actual = norm(x)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                ag = torch.autograd.grad(actual.square().mean(),(x,norm.weight,norm.bias))
                eg = torch.autograd.grad(expected.square().mean(),(x,norm.weight,norm.bias))
                for a,e in zip(ag,eg):
                    torch.testing.assert_close(a,e,atol=0,rtol=0)
                norm = norm.to(dtype)
                self.assertEqual(norm(x).dtype,dtype)
                self.assertTrue(torch.isfinite(norm(x)).all())

    @unittest.skipIf(CPU,"CUDA tiling")
    @torch.no_grad()
    def test_tiled_master_parameters_and_tail(self):
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        if l2 <= 0:
            self.skipTest("device does not report L2 capacity")
        for dtype in DTYPES:
            for s in (1,2,3):
                c,h,w = 16,128,129
                sample_bytes = c*h*w*s*s*4
                if sample_bytes > l2:
                    continue
                capacity = max(1,l2//2//sample_bytes)
                batch = 2*(1 << (capacity.bit_length()-1))+1
                if batch > 65:
                    continue
                x,k,b = self.data(dtype,b=batch,c=c,h=h,w=w,kb=batch,kc=1)
                x = x.transpose(-1,-2)
                prior = torch.randn(batch,c,w*s,h*s,device=DEVICE,dtype=dtype)
                for nearest in (False,True):
                    op = torch.ops.converse2d.forward_nearest if nearest else torch.ops.converse2d.forward
                    args = (x,k,b) if nearest else (x,prior,k,b)
                    out = op(*args,s)
                    ref_fn = converse2d_reference_nearest if nearest else converse2d_reference
                    ref = ref_fn(*(t.double() for t in args),s).to(dtype)
                    self.assertEqual(out.dtype,dtype)
                    torch.testing.assert_close(out,ref,atol=0.008 if dtype==torch.float16 else 0.08,rtol=0.008)

    @torch.no_grad()
    def test_fft_dynamic_range_and_small_regularizer(self):
        for dtype,amplitudes in ((torch.float16,(1e-7,10000.)),(torch.bfloat16,(1e-38,1e10))):
            for amplitude in amplitudes:
                for s in (1,2,3):
                    x = torch.full((1,1,5,7),amplitude,device=DEVICE,dtype=dtype)
                    w = torch.zeros(1,1,3,3,device=DEVICE)
                    w[...,1,1] = 1
                    b = torch.full((1,1,1,1),-40.,device=DEVICE)
                    out = torch.ops.converse2d.forward_nearest(x,w,b,s,1e-8)
                    ref = converse2d_reference_nearest(x,w,b,s,1e-8)
                    self.assertTrue(torch.isfinite(out).all())
                    torch.testing.assert_close(out,ref,atol=amplitude*0.02,rtol=0.02)

    @unittest.skipIf(CPU,"end-to-end AMP integration targets CUDA; CPU operator gradients are tested separately")
    def test_amp_usrnet_optimizer_step(self):
        from models.converse_usrnet import ConverseUSRNet
        for dtype in DTYPES:
            if CPU and dtype == torch.float16:
                continue
            py = ConverseUSRNet(num_iterations=2,num_blocks=1,backend="pytorch").to(DEVICE)
            cu = copy.deepcopy(py)
            for layer in cu.modules():
                if hasattr(layer,"backend"):
                    layer.backend = "pytorch" if CPU else "cuda"
            x = torch.randn(2,3,5,7,device=DEVICE)
            k = torch.rand(2,1,7,7,device=DEVICE)
            with torch.autocast(DEVICE,dtype=dtype):
                expected = py(x,k,2)
                actual = cu(x,k,2)
                loss = actual.float().square().mean()
            self.assertEqual(actual.dtype,dtype)
            torch.testing.assert_close(actual,expected,atol=0.015 if dtype==torch.float16 else 0.08,rtol=0.08)
            optimizer = torch.optim.SGD(cu.parameters(),lr=1e-3)
            scaler = torch.amp.GradScaler(DEVICE,init_scale=128,enabled=not CPU and dtype==torch.float16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for p in cu.parameters():
                self.assertEqual(p.dtype,torch.float32)
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            before = cu.conv2.weight.detach().clone()
            scaler.step(optimizer)
            scaler.update()
            self.assertFalse(torch.equal(before,cu.conv2.weight))
            with torch.inference_mode(),torch.autocast(DEVICE,dtype=dtype):
                self.assertTrue(torch.isfinite(cu(x,k,2)).all())

    @unittest.skipIf(CPU,"pretrained AMP inference targets CUDA")
    def test_pretrained_amp_inference(self):
        from models.converse_dncnn import ConverseDnCNN
        from models.converse_srresnet import ConverseMSRResNet
        from models.converse_usrnet import ConverseUSRNet
        for name,factory,channels in (("converse_dncnn",ConverseDnCNN,1),
                                      ("converse_srresnet",ConverseMSRResNet,3),
                                      ("converse_usrnet",ConverseUSRNet,3)):
            model = factory().to(DEVICE).eval()
            model.load_state_dict(torch.load(ROOT/"model_zoo"/f"{name}.pth",map_location=DEVICE,weights_only=True))
            x = torch.rand(1,channels,8,10,device=DEVICE)
            args = (x,torch.full((1,1,7,7),1/49,device=DEVICE),2) if name=="converse_usrnet" else (x,)
            for dtype in DTYPES:
                with self.subTest(model=name,dtype=dtype),torch.inference_mode(),torch.autocast(DEVICE,dtype=dtype):
                    for layer in model.modules():
                        if hasattr(layer,"backend"):
                            layer.backend = "pytorch"
                    expected = model(*args)
                    for layer in model.modules():
                        if hasattr(layer,"backend"):
                            layer.backend = "cuda"
                    actual = model(*args)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual,expected,atol=0.01 if dtype==torch.float16 else 0.08,rtol=0.08)

    @unittest.skipIf(CPU,"CUDA only")
    def test_low_precision_direct_capture(self):
        for dtype in DTYPES:
            x,w,b = self.data(dtype)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(3):
                    torch.ops.converse2d.forward_nearest(x,w,b,2)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(),torch.cuda.graph(graph):
                output = torch.ops.converse2d.forward_nearest(x,w,b,2)
            with torch.no_grad():
                x.mul_(0.75)
                w.add_(0.01)
                torch.ops.converse2d.clear_cache()
                graph.replay()
                expected = converse2d_reference_nearest(x,w,b,2)
                torch.testing.assert_close(output,expected,atol=0.06,rtol=0.06)
            torch.cuda.synchronize()

    @unittest.skipIf(CPU,"CUDA only")
    def test_low_precision_graph_owned_plan(self):
        for dtype in DTYPES:
            x,w,b = self.data(dtype,b=17,c=32,h=64,w=64,kc=32)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(),torch.cuda.stream(stream):
                torch.ops.converse2d.begin_graph_cache()
                try:
                    for _ in range(3):
                        torch.ops.converse2d.forward_nearest(x,w,b,2)
                    with torch.cuda.graph(graph,stream=stream):
                        output = torch.ops.converse2d.forward_nearest(x,w,b,2)
                finally:
                    owners = torch.ops.converse2d.end_graph_cache()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.no_grad():
                x.mul_(0.5)
                torch.ops.converse2d.clear_cache()
                graph.replay()
                expected = converse2d_reference_nearest(x,w,b,2)
                torch.testing.assert_close(output,expected,atol=0.06,rtol=0.06)
            torch.cuda.synchronize()
            del graph,owners


if __name__ == "__main__":
    unittest.main(verbosity=2)
