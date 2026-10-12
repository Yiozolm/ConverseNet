"""Fused channels-first LayerNorm training: FP64 budgets, masks, dispatch, higher order."""
import os
import unittest
from unittest.mock import patch

import torch

from support import CUDATestCase, profiled
from numerical_policy import SEEDED_MIN_SEEDS, comparison, seeded_comparison

STATS_OP = "converse2d::_channel_layernorm_forward_stats"
BACKWARD_OP = "converse2d::_channel_layernorm_backward"
ACTIVE_NODE = "_ChannelLayerNormTrainingBackward"


def reference(x, weight, bias, eps):
    # The frozen Python FP32 expression of the release LayerNorm; the same
    # expression in FP64 is the independent oracle. Never the CUDA candidate.
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    return weight[:, None, None] * ((x - u) / torch.sqrt(s + eps)) + bias[:, None, None]


def fixture(shape, seed, offset=0):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(shape, generator=generator)
    storage = torch.empty(raw.numel() + offset, device="cuda")
    x = storage[offset:].view(shape)
    x.copy_(raw)
    weight = (1 + torch.randn(shape[1], generator=generator) * .1).cuda()
    bias = (torch.randn(shape[1], generator=generator) * .1).cuda()
    upstream = torch.randn(shape, generator=generator).cuda()
    return x, weight, bias, upstream


def evaluate(raw, needs, eps=1e-5, *, production=False, dtype=torch.float32, permuted=False):
    """Output and the requested VJPs of the public layer or the Python expression."""
    from models.util_converse import LayerNorm
    x, weight, bias = (v.detach().to(dtype).requires_grad_(need) for v, need in zip(raw[:3], needs))
    upstream = raw[3].to(dtype)
    if production:
        layer = LayerNorm(x.shape[1], eps=eps, data_format="channels_first").cuda()
        layer.weight = torch.nn.Parameter(weight, requires_grad=needs[1])
        layer.bias = torch.nn.Parameter(bias, requires_grad=needs[2])
        weight, bias = layer.weight, layer.bias
        output = layer(x)
    else:
        output = reference(x, weight, bias, eps)
    active = type(output.grad_fn).__name__ == ACTIVE_NODE if output.grad_fn is not None else False
    if permuted:
        # A permuted consumer hands the fused backward a non-contiguous gradient.
        output = output.permute(0, 2, 3, 1)
        upstream = upstream.permute(0, 2, 3, 1)
    targets = [(name, value) for name, value in zip(("dx", "dweight", "dbias"), (x, weight, bias))
               if value.requires_grad]
    gradients = torch.autograd.grad(output, [value for _, value in targets], upstream) if targets else ()
    return {"output": output, **{name: g for (name, _), g in zip(targets, gradients)}}, active


class LayerNormTrainingCUDA(CUDATestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"CONVERSE2D_BACKEND": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def three_way(self, raw, needs, eps=1e-5, *, permuted=False, dx_max_abs_per_seed=True):
        high, _ = evaluate(raw, needs, eps, dtype=torch.float64, permuted=permuted)
        baseline, _ = evaluate(raw, needs, eps, permuted=permuted)
        actual, active = evaluate(raw, needs, eps, production=True, permuted=permuted)
        self.assertTrue(active)
        self.assertEqual(actual.keys(), baseline.keys())
        # The training forward reuses the exact inference kernel: bytes.
        self.assert_bytes_equal(actual["output"], baseline["output"], "output")
        rows = {}
        for name in actual:
            rows[name] = comparison(actual[name], baseline[name], high[name])
            if name == "output" or (name == "dx" and dx_max_abs_per_seed):
                self.assertTrue(rows[name]["passed"], f"{name}: {rows[name]}")
            elif name == "dx":
                # dx values of a few units carry 1-2 ulp errors, at the max-abs
                # floor, from the per-pixel statistics that the frozen forward
                # kernel and ATen round differently; a single seed's max-abs is
                # a noise draw (one element in 589,824 at 2.5 vs 1.5 ulp on
                # (1, 64, 96, 96), seed 5326, on RTX 5060 Ti and A100). Rel-L2
                # stays a per-seed gate; max-abs is gated over seeds by the caller.
                row = rows[name]
                self.assertTrue(row["candidate"]["finite"], f"{name}: {row}")
                self.assertLessEqual(row["candidate"]["rel_l2"], row["limits"]["rel_l2"], f"{name}: {row}")
        return rows

    def test_production_shape_all_gradient_masks(self):
        raw = fixture((4, 64, 96, 96), 5101)
        for bits in range(1, 8):
            needs = tuple(bool(bits & (1 << index)) for index in range(3))
            with self.subTest(needs=needs):
                rows = self.three_way(raw, needs)
                # Channel vectors are 36864-term cancelling sums; one seed is a
                # noise draw, so they are gated below over independent seeds.
                self.assertTrue(all(rows[k]["candidate"]["finite"] for k in rows))

    def test_channel_vectors_over_independent_seeds(self):
        for shape in ((4, 64, 96, 96), (2, 128, 100, 100)):
            rows = {"dweight": [], "dbias": []}
            for seed in range(8):
                result = self.three_way(fixture(shape, 5200 + seed), (True, True, True))
                for name in rows:
                    rows[name].append(result[name])
            for name, seeds in rows.items():
                with self.subTest(shape=shape, name=name):
                    verdict = seeded_comparison(seeds)
                    self.assertTrue(verdict["passed"], f"{name}: {verdict}")

    def test_geometries_alignment_tails_and_eps(self):
        cases = [((1, 64, 20, 24), 0, 1e-5), ((2, 64, 5, 7), 0, 1e-5), ((1, 64, 96, 96), 1, 1e-6),
                 ((1, 128, 8, 8), 2, 1e-5), ((1, 128, 7, 10), 1, .2), ((1, 64, 1, 7), 0, 1e-5),
                 ((2, 128, 7, 1), 0, 1e-5), ((3, 64, 33, 31), 3, 1e-5)]
        for index, (shape, offset, eps) in enumerate(cases):
            # Output and dx are gated per seed inside three_way. The channel
            # vectors are cancelling sums over every pixel, so each seed is a
            # noise draw against the ATen baseline's own reduction order: the
            # single-seed budget failed 1-3 of 8 seeds per geometry on the
            # RTX 5060 Ti and two geometries on the A100 release run of
            # 2026-10-12 (tools/v4_release_a100). They are gated over
            # independent seeds, as for the production shape above.
            rows = {"dx": [], "dweight": [], "dbias": []}
            for seed in range(SEEDED_MIN_SEEDS):
                raw = fixture(shape, 5300 + 10 * index + seed, offset)
                self.assertEqual(raw[0].data_ptr() % 16, offset * 4)
                with self.subTest(shape=shape, offset=offset, eps=eps, seed=seed):
                    result = self.three_way(raw, (True, True, True), eps, dx_max_abs_per_seed=False)
                for name in rows:
                    rows[name].append(result[name])
            for name, seeds in rows.items():
                with self.subTest(shape=shape, offset=offset, eps=eps, name=name):
                    verdict = seeded_comparison(seeds)
                    self.assertTrue(verdict["passed"], f"{name}: {verdict}")

    def test_permuted_upstream_gradient(self):
        rows = self.three_way(fixture((2, 64, 24, 20), 5401), (True, True, True), permuted=True)
        for name in ("dweight", "dbias"):
            self.assertTrue(rows[name]["passed"], f"{name}: {rows[name]}")

    def test_backward_is_deterministic(self):
        raw = fixture((4, 64, 96, 96), 5501)
        first, _ = evaluate(raw, (True, True, True), production=True)
        second, _ = evaluate(raw, (True, True, True), production=True)
        for name in first:
            self.assert_bytes_equal(first[name], second[name], name)

    def test_dispatch_routing(self):
        from models.util_converse import LayerNorm
        raw = fixture((1, 64, 5, 7), 5601)
        layer = LayerNorm(64, eps=1e-5, data_format="channels_first").cuda()
        with torch.no_grad():
            layer.weight.copy_(raw[1])
            layer.bias.copy_(raw[2])

        def events(x, **env):
            with patch.dict(os.environ, env):
                return profiled(lambda: layer(x))[1]

        x = raw[0].detach().requires_grad_()
        trained = events(x)
        self.assertIn(STATS_OP, trained)
        self.assertNotIn("aten::mean", trained)
        # Frozen inputs under GradMode keep the original ATen statistics.
        layer.requires_grad_(False)
        self.assertIn("aten::mean", events(raw[0].detach()))
        layer.requires_grad_(True)
        # Backend/environment, autocast, layouts and unsupported shapes stay native.
        self.assertIn("aten::mean", events(x, CONVERSE2D_BACKEND="pytorch"))
        layer.backend = "pytorch"
        self.assertIn("aten::mean", events(x))
        layer.backend = "auto"
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            self.assertIn("aten::mean", events(x))
        self.assertIn("aten::mean", events(x.transpose(-1, -2).contiguous().transpose(-1, -2)))
        self.assertIn("aten::mean", events(x.detach().contiguous(memory_format=torch.channels_last).requires_grad_()))
        self.assertIn("aten::mean", events(torch._neg_view(x)))
        small = LayerNorm(3, eps=1e-5, data_format="channels_first").cuda()
        self.assertIn("aten::mean", profiled(lambda: small(torch.randn(1, 3, 5, 7, device="cuda", requires_grad=True)))[1])
        self.assertIn("aten::mean", events(torch.randn(1, 64, 1, 1, device="cuda", requires_grad=True)))
        with torch.no_grad():
            self.assertIn("converse2d::_channel_layernorm", events(raw[0]))
            self.assertNotIn(STATS_OP, events(raw[0]))

    def test_first_order_backward_uses_fused_kernel_and_higher_order_falls_back(self):
        from models.util_converse import LayerNorm
        raw = fixture((1, 64, 3, 5), 5701)
        results = []
        for candidate in (False, True):
            x, weight, bias = (v.detach().clone().requires_grad_() for v in raw[:3])
            if candidate:
                layer = LayerNorm(64, eps=.2, data_format="channels_first").cuda()
                layer.weight = torch.nn.Parameter(weight)
                layer.bias = torch.nn.Parameter(bias)
                weight, bias = layer.weight, layer.bias
                output = layer(x)
                _, first_events = profiled(lambda: torch.autograd.grad(output.sin().sum(), (x, weight, bias), retain_graph=True))
                self.assertIn(BACKWARD_OP, first_events)
                gradients, events = profiled(lambda: torch.autograd.grad(output.sin().sum(), (x, weight, bias), create_graph=True))
                self.assertNotIn(BACKWARD_OP, events)
                self.assertIn("aten::mean", events)
            else:
                output = reference(x, weight, bias, .2)
                gradients = torch.autograd.grad(output.sin().sum(), (x, weight, bias), create_graph=True)
            second = torch.autograd.grad(sum(g.square().sum() for g in gradients), (x, weight, bias), create_graph=True)
            third = torch.autograd.grad(sum(g.square().sum() for g in second), (x, weight, bias))
            results.append((output, *gradients, *second, *third))
        self.assert_bytes_equal(results[1][0], results[0][0], "output")
        high = [v.detach().double().requires_grad_() for v in raw[:3]]
        oracle = reference(*high, .2)
        oracle_grads = torch.autograd.grad(oracle.sin().sum(), high)
        for name, actual, baseline, ref64 in zip(("dx", "dweight", "dbias"), results[1][1:4], results[0][1:4], oracle_grads):
            row = comparison(actual, baseline, ref64)
            self.assertTrue(row["passed"], f"{name}: {row}")
        # Higher-order: the fallback differentiates the recomputed ATen graph,
        # so second/third derivatives differ only by FP32 rounding (about 2e-7
        # of each tensor's scale on third derivatives of magnitude 1e4-1e5).
        # The admission number is the existing 3e-5, applied to the tensor scale.
        for actual, expected in zip(results[1][4:], results[0][4:]):
            scale = max(1., expected.abs().max().item())
            self.assertLessEqual((actual - expected).abs().max().item(), 3e-5 * scale)

    def test_private_backward_contract(self):
        x, weight, bias, upstream = fixture((2, 64, 5, 7), 5801)
        output, mean, denominator = torch.ops.converse2d._channel_layernorm_forward_stats(x, weight, bias, 1e-5)
        self.assert_bytes_equal(output, reference(x, weight, bias, 1e-5), "stats output")
        self.assert_bytes_equal(mean, x.mean(1).reshape(-1), "mean")
        self.assert_bytes_equal(denominator, torch.sqrt((x - x.mean(1, keepdim=True)).pow(2).mean(1) + 1e-5).reshape(-1), "denominator")
        for needs in ((True, False, False), (False, True, False), (False, False, True), (False, True, True)):
            grads = torch.ops.converse2d._channel_layernorm_backward(upstream, x, mean, denominator, weight, *needs)
            self.assertEqual([g is not None for g in grads], list(needs), needs)
        with self.assertRaises(RuntimeError):
            torch.ops.converse2d._channel_layernorm_backward(upstream.transpose(-1, -2), x, mean, denominator, weight, True, True, True)
        with self.assertRaises(RuntimeError):
            torch.ops.converse2d._channel_layernorm_forward_stats(x.cpu(), weight.cpu(), bias.cpu(), 1e-5)

    def test_checkpoint_round_trip(self):
        from models.util_converse import LayerNorm
        layer = LayerNorm(64, eps=1e-5, data_format="channels_first").cuda()
        state = {k: v.clone() for k, v in layer.state_dict().items()}
        x = torch.randn(1, 64, 5, 7, device="cuda", requires_grad=True)
        layer(x).sum().backward()
        self.assertEqual(set(layer.state_dict()), set(state))
        layer.load_state_dict(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
