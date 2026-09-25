"""Lazy value flags are observable metadata, even without a version increment."""
import unittest
import torch

from support import ExtensionTestCase, CUDATestCase, fixture, leaves


def check_cache_toggle(test, device):
    for context in (torch.no_grad, torch.enable_grad, torch.inference_mode):
        raw, _ = fixture(2)
        data = leaves(raw, device, needs=(False,) * 4)
        weight = data[2]
        with context():
            torch.ops.converse2d.clear_cache()
            old = torch.ops.converse2d.forward(*data, 2)
            for negative in (True, False):
                identity = (weight.data_ptr(), weight._version, weight.shape,
                            weight.stride(), weight.storage_offset())
                weight.data = torch._neg_view(weight.data)
                test.assertEqual(weight.is_neg(), negative)
                test.assertEqual(identity, (weight.data_ptr(), weight._version, weight.shape,
                                           weight.stride(), weight.storage_offset()))
                cached = torch.ops.converse2d.forward(*data, 2)
                torch.ops.converse2d.clear_cache()
                fresh = torch.ops.converse2d.forward(*data, 2)
                test.assert_bytes_equal(cached, fresh, 'negative flag cache invalidation')
                test.assertFalse(torch.equal(old, fresh))
                old = fresh
        torch.ops.converse2d.clear_cache()


class NegativeMetadataCPU(ExtensionTestCase):
    def test_same_pointer_negative_flag_invalidates_cpu_cache(self):
        check_cache_toggle(self, 'cpu')


class NegativeMetadataCUDA(CUDATestCase):
    def tearDown(self):
        torch.ops.converse2d.clear_cache()

    def test_same_pointer_negative_flag_invalidates_cuda_cache(self):
        check_cache_toggle(self, 'cuda')

    def test_cold_negative_kernel_matches_materialized_values(self):
        for scale in (1, 2, 3, 4):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                raw, _ = fixture(scale, kb=kb, kc=kc)
                data = leaves(raw, 'cuda', needs=(False,) * 4)
                data[2] = torch._neg_view(data[2])
                self.assertTrue(data[2].is_neg())
                self.assertTrue(data[2].is_contiguous())
                for context in (torch.no_grad, torch.enable_grad, torch.inference_mode):
                    with self.subTest(scale=scale, kb=kb, kc=kc, context=context.__name__), context():
                        # InferenceMode's Negative dispatcher materializes the
                        # argument as an inference tensor before this custom op.
                        # Match that cache eligibility: constructing the control
                        # outside the mode compares cached and uncached arithmetic.
                        resolved = [*data[:2], data[2].resolve_neg(), data[3]]
                        torch.ops.converse2d.clear_cache()
                        expected = torch.ops.converse2d.forward(*resolved, scale)
                        torch.ops.converse2d.clear_cache()
                        actual = torch.ops.converse2d.forward(*data, scale)
                        self.assert_bytes_equal(actual, expected, 'cold lazy negative kernel')


if __name__ == '__main__':
    unittest.main(verbosity=2)
