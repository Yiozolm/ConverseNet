"""Production half-spectrum VJP, boundaries and higher-order regressions."""
import itertools
import unittest

import torch
from extension_loader import load_extension


def full(t,width):
    tail=t[...,1:(width+1)//2].flip((-2,-1)).roll(1,-2)
    return torch.cat((t,tail.conj() if t.is_complex() else tail),-1)


def average(t,s):
    b,c,h,w=t.shape
    return t.reshape(b,c,s,h//s,s,w//s).mean((2,4))


def spectral_reference(y,p,k,lam,h,w,s):
    power=k.real.square()+k.imag.square()
    prediction=k*p
    if s>1:
        power=average(full(power,w*s),s)[...,:w//2+1]
        prediction=average(full(prediction,w*s),s)[...,:w//2+1]
    q=(y-prediction)/(power+lam)
    if s>1:
        q=full(q,w).repeat(1,1,s,s)[...,:w*s//2+1]
    return p+k.conj()*q


class TrainingFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        load_extension()
        cls.op=torch.ops.converse2d._training_spectral

    def setUp(self):
        torch.manual_seed(852)

    def data(self,h,w,s,kb=1,kc=2,b=2,c=2,dtype=torch.complex128):
        real=torch.float64 if dtype==torch.complex128 else torch.float32
        return (torch.randn(b,c,h,w//2+1,device="cuda",dtype=dtype).requires_grad_(),
                torch.randn(b,c,h*s,w*s//2+1,device="cuda",dtype=dtype).requires_grad_(),
                torch.randn(kb,kc,h*s,w*s//2+1,device="cuda",dtype=dtype).requires_grad_(),
                torch.rand(1,c,1,1,device="cuda",dtype=real).add_(0.2).requires_grad_())

    def test_arbitrary_complex_vjp(self):
        # Arbitrary complex half spectra intentionally violate Hermitian edge
        # constraints: this catches VJPs that only work for real FFT inputs.
        for h,w,s in ((1,1,1),(1,1,3),(1,4,2),(4,1,3),(3,4,1),(3,5,2),(4,6,3),(2,3,4),(2,3,5)):
            for kb,kc in ((1,1),(1,2),(2,1),(2,2)):
                with self.subTest(h=h,w=w,s=s,kb=kb,kc=kc):
                    data=self.data(h,w,s,kb,kc)
                    expected=spectral_reference(*data,h,w,s)
                    actual=self.op(*data,h,w,s)
                    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                    upstream=torch.randn_like(actual)
                    eg=torch.autograd.grad(expected,data,upstream)
                    ag=torch.autograd.grad(actual,data,upstream)
                    for a,e in zip(ag,eg):
                        torch.testing.assert_close(a,e,atol=1e-11,rtol=1e-11)

    def test_gradcheck_and_gradgradcheck(self):
        for h,w,s,kb,kc in ((2,3,1,1,1),(2,4,2,1,2),(3,3,3,2,1)):
            data=self.data(h,w,s,kb,kc)
            fn=lambda *t:self.op(*t,h,w,s)
            self.assertTrue(torch.autograd.gradcheck(fn,data,fast_mode=True))
            self.assertTrue(torch.autograd.gradgradcheck(fn,data,fast_mode=True))

    def test_shared_spectrum_higher_derivatives(self):
        y,_,k,lam=self.data(2,3,1)
        fn=lambda y,k,lam:self.op(y,y,k,lam,2,3,1)
        self.assertTrue(torch.autograd.gradcheck(fn,(y,k,lam),fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(fn,(y,k,lam),fast_mode=True))

    def test_partial_gradients_and_conjugate_views(self):
        for needs in itertools.product((False,True),repeat=4):
            if not any(needs):
                continue
            raw=self.data(3,5,2)
            data=tuple(t.detach().requires_grad_(need) for t,need in zip(raw,needs))
            # Slice a strided storage and pass lazy conjugate views into CUDA.
            data=tuple(torch.stack((t,t),-1)[...,0].conj() if t.is_complex() else t for t in data)
            inputs=[t for t,need in zip(data,needs) if need]
            expected=spectral_reference(*data,3,5,2)
            actual=self.op(*data,3,5,2)
            upstream=torch.randn_like(actual).conj()
            ag=torch.autograd.grad(actual,inputs,upstream,retain_graph=True)
            eg=torch.autograd.grad(expected,inputs,upstream,retain_graph=True)
            for a,e in zip(ag,eg):
                torch.testing.assert_close(a,e,atol=1e-11,rtol=1e-11)
            # Higher-order mode must also handle frozen formal arguments.
            hg=torch.autograd.grad(actual,inputs,upstream,create_graph=True)
            for a,e in zip(hg,eg):
                torch.testing.assert_close(a,e,atol=1e-11,rtol=1e-11)

    def test_fp32_vjp_and_stream(self):
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for s in (1,2,3,4):
                data=self.data(7,8,s,kb=1,kc=1,dtype=torch.complex64)
                expected=spectral_reference(*data,7,8,s)
                actual=self.op(*data,7,8,s)
                upstream=torch.randn_like(actual)
                eg=torch.autograd.grad(expected,data,upstream)
                ag=torch.autograd.grad(actual,data,upstream)
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
                for a,e in zip(ag,eg):
                    torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-5)
        torch.cuda.current_stream().wait_stream(stream)

    def test_weak_regularization_and_zero_filter(self):
        for amplitude in (0.,1e-6,1e-3):
            data=list(self.data(3,4,3,kb=1,kc=1,dtype=torch.complex64))
            with torch.no_grad():
                data[0].mul_(1e-5)
                data[1].mul_(1e-5)
                data[2].mul_(amplitude)
                data[3].fill_(1e-8)
            reference_data=tuple(t.detach().to(torch.complex128 if t.is_complex() else torch.float64)
                                 .requires_grad_() for t in data)
            expected=spectral_reference(*reference_data,3,4,3)
            actual=self.op(*data,3,4,3)
            upstream=torch.randn_like(actual)*1e-5
            eg=torch.autograd.grad(expected,reference_data,upstream.to(torch.complex128))
            ag=torch.autograd.grad(actual,data,upstream)
            torch.testing.assert_close(actual.to(expected.dtype),expected,atol=1e-6,rtol=1e-5)
            for a,e in zip(ag,eg):
                self.assertTrue(torch.isfinite(a).all())
                # Normalize the comparison so large lambda gradients do not
                # require an arbitrary, magnitude-dependent absolute tolerance.
                divisor=e.abs().max().clamp_min(1e-30)
                torch.testing.assert_close(a.to(e.dtype)/divisor,e/divisor,atol=2e-5,rtol=2e-5)

    def test_saved_input_mutation_is_rejected(self):
        data=self.data(2,3,2)
        out=self.op(*data,2,3,2)
        with torch.no_grad():
            data[2].add_(0.1)
        with self.assertRaisesRegex(RuntimeError,"modified by an inplace"):
            out.real.sum().backward()

    def test_shape_validation(self):
        y,p,k,lam=self.data(2,3,2)
        for args in ((y,p,k,lam,2,3,0),(y,p,k,lam,2,4,2),(y,p,k,lam.float(),2,3,2),
                     (y,p,k[:,:,:1],lam,2,3,2)):
            with self.assertRaises(RuntimeError):
                self.op(*args)


if __name__=="__main__":
    unittest.main(verbosity=2)
