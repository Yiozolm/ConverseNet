"""Frozen-input VJPs retain Python FP32 bits while pruning unused work."""
import unittest

import torch

from support import CUDATestCase, compare_spatial, fixture, leaves
from models.converse_core import converse2d_reference


class GradientMaskCUDA(CUDATestCase):
    def test_all_nonempty_gradient_subsets_and_kernel_broadcasts(self):
        for scale in (1, 2, 3):
            for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
                raw, upstream = fixture(scale, kb=kb, kc=kc)
                for mask in range(1, 16):
                    needs = tuple(bool(mask & (1 << i)) for i in range(4))
                    with self.subTest(scale=scale, kb=kb, kc=kc, mask=mask):
                        compare_spatial(self, raw, upstream, scale=scale,
                                        needs=needs, transpose=bool(mask & 1))

    def test_shared_input_gradient_subsets(self):
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            raw, upstream = fixture(1, kb=kb, kc=kc)
            for mask in range(1, 8):
                need_x, need_k, need_b = (bool(mask & (1 << i)) for i in range(3))
                with self.subTest(kb=kb, kc=kc, mask=mask):
                    compare_spatial(self, raw, upstream, scale=1, shared=True,
                                    needs=(need_x, need_x, need_k, need_b),
                                    strided=True)

    def test_weak_regularizer_and_singleton_width_subsets(self):
        for scale in (1, 2, 3):
            raw, upstream = fixture(scale, weak=1e-6)
            # s2/W=1 must use the generic alias-reduction implementation.
            raw[0] = raw[0][..., :1]
            raw[1] = raw[1][..., :scale]
            raw[2] = raw[2][..., :1]
            upstream = upstream[..., :scale]
            for mask in range(1, 16):
                with self.subTest(scale=scale, mask=mask):
                    needs = tuple(bool(mask & (1 << i)) for i in range(4))
                    compare_spatial(self, raw, upstream, scale=scale,
                                    needs=needs, strided=True, eps=1e-8)

    def test_input_only_backward_omits_parameter_reductions(self):
        # This observes actual dispatcher work. Returning undefined parameter
        # gradients after computing the complete VJP would fail this check.
        for scale in (1, 2, 3):
            raw, upstream = fixture(scale, kb=1, kc=1)
            for index in (0, 1):
                with self.subTest(scale=scale, trainable=index):
                    data = leaves(raw, "cuda", needs=tuple(i == index for i in range(4)))
                    output = torch.ops.converse2d.forward(*data, scale)
                    gradient = upstream.cuda()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                        torch.autograd.grad(output, data[index], gradient)
                    sums = sum(event.count for event in trace.key_averages()
                               if event.key == "aten::sum")
                    self.assertEqual(sums, int(scale > 2),
                                     "only the generic alias sum may remain")

    def test_engine_requested_subset_matches_python(self):
        # Also exercise autograd.grad pruning when every leaf is trainable.
        for scale in (1, 2, 3):
            raw, upstream = fixture(scale)
            for index in range(4):
                results = []
                for fn in (torch.ops.converse2d.forward, converse2d_reference):
                    data = leaves(raw, "cuda")
                    output = fn(*data, scale)
                    results.append(torch.autograd.grad(output, data[index], upstream.cuda())[0])
                with self.subTest(scale=scale, requested=index):
                    self.assert_bytes_equal(*results, "requested VJP")

    def test_higher_order_with_frozen_inputs(self):
        for scale in (1, 2, 3):
            raw, _ = fixture(scale)
            for needs in ((True, False, True, False), (False, True, False, True),
                          (False, False, True, False), (False, False, False, True)):
                results = []
                for fn in (torch.ops.converse2d.forward, converse2d_reference):
                    data = leaves(raw, "cuda", needs=needs)
                    requested = [value for value in data if value.requires_grad]
                    output = fn(*data, scale, .1)
                    grads = torch.autograd.grad(output.square().mean(), requested,
                                                create_graph=True)
                    results.append(torch.autograd.grad(sum(g.square().mean() for g in grads),
                                                       requested))
                with self.subTest(scale=scale, needs=needs):
                    for actual, expected in zip(*results):
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_masked_backward_capture_and_replay(self):
        for scale in (1, 2, 3):
            for index in range(4):
                with self.subTest(scale=scale, trainable=index):
                    raw, upstream = fixture(scale, kb=1, kc=1)
                    data = leaves(raw, "cuda", needs=tuple(i == index for i in range(4)))
                    gradient = upstream.cuda()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())

                    def run():
                        output = torch.ops.converse2d.forward(*data, scale)
                        return output, torch.autograd.grad(output, data[index], gradient)[0]

                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                    # The captured VJP must continue reading current frozen
                    # parameters as well as the differentiable input.
                    with torch.no_grad():
                        for value in data:
                            value.add_(.01)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        reference = converse2d_reference(*data, scale)
                        expected = (reference, torch.autograd.grad(reference, data[index], gradient)[0])
                    torch.cuda.current_stream().wait_stream(stream)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assert_results_equal(actual, expected)
                    graph.reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
