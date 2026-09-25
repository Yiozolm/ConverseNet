"""Admission tests for the isolated s1/B2/B4 batch-gradient fusion."""
import unittest

import torch

from support import CUDATestCase, compare_spatial, leaves
from models.converse_core import converse2d_reference


def fixture(batch, *, channels=3, height=5, width=7, kb=1, kc=None, weak=None):
    kc = channels if kc is None else kc
    generator = torch.Generator().manual_seed(118731 + batch + height + width)
    raw = [torch.randn(batch, channels, height, width, generator=generator),
           torch.randn(batch, channels, height, width, generator=generator),
           torch.randn(kb, kc, min(3, height), min(3, width), generator=generator) / 9,
           torch.randn(1, channels, 1, 1, generator=generator)]
    upstream = torch.randn(batch, channels, height, 2 * width, generator=generator).cuda()[..., ::2]
    upstream /= (batch * channels * height * width) ** .5
    if weak is not None:
        raw[0].mul_(1e-5)
        raw[1].mul_(1e-5)
        raw[2].mul_(weak)
        raw[3].fill_(-40)
        upstream.mul_(1e-5)
    return raw, upstream


class BatchReduceCUDA(CUDATestCase):
    def test_all_independent_masks_and_shared_masks_match_python_bytes(self):
        for batch in (2, 4):
            raw, upstream = fixture(batch)
            for mask in range(1, 16):
                needs = tuple(bool(mask & (1 << i)) for i in range(4))
                with self.subTest(batch=batch, shared=False, mask=mask):
                    compare_spatial(self, raw, upstream, scale=1, needs=needs,
                                    transpose=bool(mask & 1), strided=not bool(mask & 1))
            for mask in range(1, 8):
                nx, nk, nl = (bool(mask & (1 << i)) for i in range(3))
                with self.subTest(batch=batch, shared=True, mask=mask):
                    compare_spatial(self, raw, upstream, scale=1, shared=True,
                                    needs=(nx, nx, nk, nl), transpose=True)

    def test_weak_and_zero_kernels_keep_every_requested_gradient(self):
        for batch in (2, 4):
            for weak in (0.0, 1e-6, 1e-3):
                raw, upstream = fixture(batch, weak=weak)
                for shared in (False, True):
                    for needs in ((True, True, True, True), (True, False, True, False),
                                  (False, True, True, False), (False, False, True, False)):
                        if shared and needs[0] != needs[1]:
                            continue
                        with self.subTest(batch=batch, weak=weak, shared=shared, needs=needs):
                            compare_spatial(self, raw, upstream, scale=1, shared=shared,
                                            needs=needs, transpose=True, eps=1e-8)

    def test_real_prior_grid_and_partial_launch_tail(self):
        for batch, channels, height, width in ((2, 1, 1, 2), (4, 1, 1, 2),
                                                (4, 128, 100, 100), (4, 5, 17, 19)):
            raw, upstream = fixture(batch, channels=channels, height=height, width=width)
            for shared in (False, True):
                with self.subTest(batch=batch, height=height, width=width, channels=channels, shared=shared):
                    compare_spatial(self, raw, upstream, scale=1, shared=shared)

    def test_fallback_shapes_remain_exact(self):
        for batch, width, kb, kc in ((1, 7, 1, 3), (3, 7, 1, 3), (8, 7, 1, 3),
                                    (2, 1, 1, 3), (4, 1, 1, 3),
                                    (2, 7, 1, 1), (4, 7, 1, 1),
                                    (2, 7, 2, 3), (4, 7, 4, 3)):
            raw, upstream = fixture(batch, width=width, kb=kb, kc=kc)
            with self.subTest(batch=batch, width=width, kb=kb, kc=kc):
                compare_spatial(self, raw, upstream, scale=1, shared=True, strided=True)

    def test_backward_removes_only_the_eligible_batch_reductions(self):
        cases = ((2, 7, 1, 3, True, False, 1), (4, 7, 1, 3, True, False, 1),
                 (2, 7, 1, 3, False, True, 1), (4, 7, 1, 3, False, True, 1),
                 (2, 7, 1, 3, False, False, 3), (4, 7, 1, 3, False, False, 3),
                 (3, 7, 1, 3, True, False, 3), (8, 7, 1, 3, True, False, 3),
                 (2, 1, 1, 3, True, False, 3), (2, 7, 1, 1, True, False, 4),
                 (2, 7, 2, 3, True, False, 0))
        for batch, width, kb, kc, ny, np, expected_sums in cases:
            generator = torch.Generator().manual_seed(119071 + batch)
            y = torch.randn(batch, 3, 5, width, dtype=torch.complex64, generator=generator).cuda().requires_grad_(ny)
            p = torch.randn(batch, 3, 5, width, dtype=torch.complex64, generator=generator).cuda().requires_grad_(np)
            k = torch.randn(kb, kc, 5, width, dtype=torch.complex64, generator=generator).cuda().requires_grad_()
            regularizer = torch.full((1, 3, 1, 1), .1, device="cuda")
            incoming = torch.ones_like(p)
            out = torch.ops.converse2d._training_full_spectral(y, p, k, regularizer, 1)
            requested = [value for value in (y, p, k) if value.requires_grad]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                torch.autograd.grad(out, requested, incoming)
            sums = sum(event.count for event in trace.key_averages() if event.key == "aten::sum")
            with self.subTest(batch=batch, width=width, kb=kb, kc=kc, ny=ny, np=np):
                self.assertEqual(sums, expected_sums)

    def test_shared_ancestor_preserves_gradient_accumulation_order(self):
        for batch in (2, 4):
            generator = torch.Generator().manual_seed(119171 + batch)
            raw = torch.randn(batch, 3, 5, 7, generator=generator)
            upstream = torch.randn(batch, 3, 5, 7, generator=generator).cuda()
            for shared in (False, True):
                results = []
                for solve in (torch.ops.converse2d.forward, converse2d_reference):
                    base = raw.cuda().requires_grad_()
                    x = base.sin()
                    prior = x if shared else base.cos()
                    kernel = base.mean(0, keepdim=True)[..., :3, :3].reshape(1, 3, 9).softmax(-1).reshape(1, 3, 3, 3)
                    bias = base.mean((0, 2, 3), keepdim=True)
                    output = solve(x, prior, kernel, bias, 1, .1)
                    loss = (output * upstream).sum() + .03 * base.square().sum()
                    results.append((output, *torch.autograd.grad(loss, base)))
                with self.subTest(batch=batch, shared=shared):
                    self.assert_results_equal(*results)

    def test_conjugated_noncontiguous_spectral_inputs_and_upstream(self):
        # Match the pre-existing arbitrary-complex spectral contract; the
        # public spatial comparisons above retain their zero-byte-margin gate.
        def reference(y, p, k, regularizer):
            power = k.real.square() + k.imag.square()
            q = (y - k * p) / (power + regularizer)
            return p + k.conj() * q
        for batch in (2, 4):
            generator = torch.Generator().manual_seed(119201 + batch)
            raw = [torch.randn(batch, 3, 5, 7, dtype=torch.complex64, generator=generator),
                   torch.randn(batch, 3, 5, 7, dtype=torch.complex64, generator=generator),
                   torch.randn(1, 3, 5, 7, dtype=torch.complex64, generator=generator),
                   .25 + torch.rand(1, 3, 1, 1, generator=generator)]
            upstream = torch.randn(batch, 3, 5, 14, dtype=torch.complex64,
                                   generator=generator).cuda()[..., ::2].conj()
            for shared in (False, True):
                for needs in ((True, True, True, True), (True, False, True, False),
                              (False, True, True, False)):
                    if shared and needs[0] != needs[1]:
                        continue
                    results = []
                    for solve in (lambda *a: torch.ops.converse2d._training_full_spectral(*a, 1), reference):
                        data = leaves(raw, "cuda", transpose=True, needs=(False,) * 4)
                        for index, value in enumerate(data):
                            if value.is_complex():
                                value = value.conj()
                            data[index] = value.requires_grad_(needs[index])
                        if shared:
                            data[1] = data[0]
                        requested = (data[0], data[2], data[3]) if shared else data
                        requested = [value for value in requested if value.requires_grad]
                        output = solve(*data)
                        results.append((output, *torch.autograd.grad(output, requested, upstream)))
                    with self.subTest(batch=batch, shared=shared, needs=needs):
                        for actual, expected in zip(*results):
                            self.assertTrue(torch.isfinite(actual).all().item())
                            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_higher_order_keeps_aten_fallback(self):
        for shared in (False, True):
            raw, _ = fixture(2)
            results = []
            for solve in (torch.ops.converse2d.forward, converse2d_reference):
                data = leaves(raw, "cuda")
                if shared:
                    data[1] = data[0]
                requested = (data[0], data[2], data[3]) if shared else data
                output = solve(*data, 1, .2)
                first = torch.autograd.grad(output.sin().mean(), requested, create_graph=True)
                second = torch.autograd.grad(sum(value.square().mean() for value in first), requested,
                                             create_graph=True)
                third = torch.autograd.grad(sum(value.square().mean() for value in second), requested)
                results.append((*second, *third))
            with self.subTest(shared=shared):
                for actual, expected in zip(*results):
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_shared_backward_graph_replay_after_input_and_weight_updates(self):
        raw, upstream = fixture(4)
        data = leaves(raw, "cuda")
        data[1] = data[0]
        requested = (data[0], data[2], data[3])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        def run(solve):
            output = solve(*data, 1, .1)
            return output, *torch.autograd.grad(output, requested, upstream)
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(torch.ops.converse2d.forward)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run(torch.ops.converse2d.forward)
        with torch.no_grad():
            for value in requested:
                value.add_(.01)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = run(converse2d_reference)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        self.assert_results_equal(actual, expected)
        graph.reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
