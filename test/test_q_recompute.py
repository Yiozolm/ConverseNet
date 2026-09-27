"""Keep s1 checkpoint recomputation from retaining a full complex q tensor."""
import torch
from support import CUDATestCase


class QRecomputeCUDA(CUDATestCase):
    def test_s1_retains_only_input_spectra_and_real_denominator(self):
        for batch in (1, 2, 4):
            for shared in (False, True):
                with self.subTest(batch=batch, shared=shared):
                    torch.manual_seed(3187)
                    y = torch.randn(batch, 3, 5, 7, device="cuda", dtype=torch.complex64).requires_grad_()
                    p = y if shared else torch.randn_like(y).requires_grad_()
                    k = torch.randn(1, 3, 5, 7, device="cuda", dtype=torch.complex64).requires_grad_()
                    regularizer = torch.full((1, 3, 1, 1), .1, device="cuda", requires_grad=True)
                    inputs = (y, p, k, regularizer)
                    saved = []
                    def pack(tensor):
                        saved.append(tensor)
                        return tensor
                    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                        output = torch.ops.converse2d._training_full_spectral(*inputs, 1)
                    input_pointers = {tensor.data_ptr() for tensor in inputs}
                    extras = [tensor for tensor in saved if tensor.data_ptr() not in input_pointers]
                    self.assertEqual(len(extras), 1)
                    self.assertEqual(extras[0].dtype, torch.float32)
                    self.assertEqual(extras[0].shape, (1, 3, 5, 7))
                    requested = (y, k, regularizer) if shared else inputs
                    gradients = torch.autograd.grad(output.real.sum(), requested)
                    for value in (output, *gradients):
                        self.assertTrue(torch.isfinite(value).all().item())
