"""Exact full channel LayerNorm admission; no research extension is imported."""
import os
import unittest
from unittest.mock import patch

import torch

from support import CUDATestCase, ExtensionTestCase, profiled


def reference(x, weight, bias, eps=1e-5):
    # Independent ATen expression, including both subtractions and the original
    # two-pass population variance. The CUDA candidate never receives FP64.
    mean = x.mean(1, keepdim=True)
    variance = (x - mean).pow(2).mean(1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return weight[:, None, None] * normalized + bias[:, None, None]


def fixture(shape, device, offset=0, seed=7191, value_case="random"):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(shape, generator=generator)
    if value_case == "constant":
        raw.fill_(4)
    elif value_case == "large_offset":
        raw = raw * 1e-3 + 1024
    elif value_case == "subnormal":
        raw = raw * 1e-39
    # Allocate the offset on the final device: .cuda() on a sliced CPU tensor
    # would silently remove the alignment case that this test is meant to cover.
    storage = torch.empty(raw.numel() + offset, device=device)
    x = storage[offset:].view(shape)
    x.copy_(raw)
    weight = (1 + torch.randn(shape[1], generator=generator) * .1).to(device)
    bias = (torch.randn(shape[1], generator=generator) * .1).to(device)
    return x, weight, bias


def private(x, weight, bias, eps=1e-5):
    return torch.ops.converse2d._channel_layernorm(x, weight, bias, eps)


class AccuracyMixin:
    def check_output(self, actual, values, eps=1e-5, label="LayerNorm"):
        expected = reference(*values, eps)
        self.assert_bytes_equal(actual, expected, label)
        oracle = reference(*(value.detach().double() for value in values), eps)
        actual_error = actual.detach().double() - oracle
        baseline_error = expected.detach().double() - oracle
        # Each output must independently pass both metrics at zero margin.
        # No majority vote, tolerance, or candidate FP64 backend is permitted.
        self.assertLessEqual(actual_error.abs().max().item(),
                             baseline_error.abs().max().item(), label + " max_abs")
        denominator = torch.linalg.vector_norm(oracle).clamp_min(torch.finfo(torch.float64).tiny)
        self.assertLessEqual((torch.linalg.vector_norm(actual_error) / denominator).item(),
                             (torch.linalg.vector_norm(baseline_error) / denominator).item(),
                             label + " relative_l2")

    def check_training(self, device, use_public):
        from models.util_converse import LayerNorm
        raw = fixture((1, 64, 3, 5), device)
        results = []
        for candidate in (False, True):
            x, weight, bias = [value.detach().clone().requires_grad_() for value in raw]
            if candidate and use_public:
                layer = LayerNorm(64, eps=.2, data_format="channels_first").to(device)
                layer.weight = torch.nn.Parameter(weight)
                layer.bias = torch.nn.Parameter(bias)
                weight, bias = layer.weight, layer.bias
                output, events = profiled(lambda: layer(x))
                self.assertNotIn("converse2d::_channel_layernorm", events)
            else:
                output = (private if candidate else reference)(x, weight, bias, .2)
            gradients = torch.autograd.grad(output.sin().sum(), (x, weight, bias), create_graph=True)
            second = torch.autograd.grad(sum(g.square().sum() for g in gradients),
                                         (x, weight, bias), create_graph=True)
            third = torch.autograd.grad(sum(g.square().sum() for g in second), (x, weight, bias))
            results.append((output, *gradients, *second, *third))
        self.assert_results_equal(results[0][:4], results[1][:4])
        # Preserve the existing higher-order ATen admission tolerance. These
        # checks do not relax either first-order bytes or inference FP64 gates.
        for actual, expected in zip(results[0][4:], results[1][4:]):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


class LayerNormFullCPU(AccuracyMixin, ExtensionTestCase):
    def test_private_and_public_cpu_fallback(self):
        from models.util_converse import LayerNorm
        values = fixture((2, 64, 5, 7), "cpu")
        with torch.no_grad():
            self.check_output(private(*values), values)
            layer = LayerNorm(64, eps=1e-5, data_format="channels_first")
            layer.weight.copy_(values[1])
            layer.bias.copy_(values[2])
            actual, events = profiled(lambda: layer(values[0]))
            self.assertNotIn("converse2d::_channel_layernorm", events)
            self.check_output(actual, values)

    def test_cpu_training_and_higher_order(self):
        for public in (False, True):
            with self.subTest(public=public):
                self.check_training("cpu", public)


class LayerNormFullCUDA(AccuracyMixin, CUDATestCase):
    def test_all_six_reduction_pairs_alignment_and_tails(self):
        # Explicit fixtures from the independently checked ATen ReduceConfig:
        # labels give (mean input stride, variance input stride). Variance owns
        # fresh aligned storage, so unaligned x can select a different mean tree.
        cases = [
            ((1, 64, 20, 24), 0, (4, 4)),
            ((1, 128, 8, 8), 0, (8, 8)),
            ((2, 64, 5, 7), 0, (1, 1)),
            ((1, 64, 96, 96), 1, (1, 4)),
            ((1, 128, 100, 100), 2, (8, 4)),
            ((1, 128, 8, 8), 1, (1, 8)),
            ((1, 128, 7, 10), 0, (8, 8)),
            ((1, 128, 7, 10), 1, (1, 8)),
            ((1, 64, 1, 7), 0, (1, 1)),
            ((2, 128, 7, 1), 0, (1, 1)),
            ((4, 64, 96, 96), 0, (4, 4)),
            ((4, 128, 100, 100), 0, (4, 4)),
        ]
        self.assertEqual({case[2] for case in cases},
                         {(1, 1), (4, 4), (8, 8), (1, 4), (8, 4), (1, 8)})
        for index, (shape, offset, pair) in enumerate(cases):
            values = fixture(shape, "cuda", offset, seed=7191 + index)
            self.assertEqual(values[0].data_ptr() % 16, offset * 4)
            for context in (torch.no_grad, torch.inference_mode):
                with self.subTest(shape=shape, offset=offset, pair=pair, mode=context.__name__), context():
                    actual, events = profiled(lambda: private(*values))
                    self.assertNotIn("aten::mean", events)
                    self.check_output(actual, values)

    def test_constant_large_offset_subnormal_and_eps(self):
        for kind in ("constant", "large_offset", "subnormal"):
            values = fixture((1, 64, 20, 24), "cuda", value_case=kind)
            for eps in (1e-6, 1e-5, .2):
                with self.subTest(kind=kind, eps=eps), torch.no_grad():
                    self.check_output(private(*values, eps), values, eps)

    def test_private_unsupported_shape_and_layout_fallbacks(self):
        for shape in ((1, 3, 5, 7), (1, 64, 1, 1), (1, 64, 5, 7)):
            base = fixture(shape, "cuda")
            for layout in ("contiguous", "transpose", "channels_last"):
                x = base[0]
                if layout == "transpose":
                    x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
                elif layout == "channels_last":
                    x = x.contiguous(memory_format=torch.channels_last)
                if shape[1] == 64 and shape[2:] != (1, 1) and layout == "contiguous":
                    continue
                with self.subTest(shape=shape, layout=layout), torch.no_grad():
                    actual, events = profiled(lambda: private(x, *base[1:]))
                    self.assertIn("aten::mean", events)
                    self.check_output(actual, (x, *base[1:]))

    def test_public_negative_layout_backend_and_shape_bypass(self):
        from models.util_converse import LayerNorm
        cases = ("negative_x", "negative_weight", "negative_bias", "transpose",
                 "channels_last", "weight_stride", "bias_stride", "backend", "environment", "channels", "hw1")
        for case in cases:
            shape = (1, 3 if case == "channels" else 64, 1 if case == "hw1" else 5,
                     1 if case == "hw1" else 7)
            for context in (torch.no_grad, torch.inference_mode):
                with self.subTest(case=case, mode=context.__name__), context():
                    x, weight, bias = fixture(shape, "cuda")
                    if case == "negative_x": x = torch._neg_view(x)
                    if case == "negative_weight": weight = torch._neg_view(weight)
                    if case == "negative_bias": bias = torch._neg_view(bias)
                    if case == "transpose": x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
                    if case == "channels_last": x = x.contiguous(memory_format=torch.channels_last)
                    if case == "weight_stride": weight = torch.stack((weight, weight), -1)[:, 0]
                    if case == "bias_stride": bias = torch.stack((bias, bias), -1)[:, 0]
                    layer = LayerNorm(shape[1], eps=1e-5, data_format="channels_first").cuda()
                    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
                    if case == "backend": layer.backend = "pytorch"
                    with patch.dict(os.environ, {"CONVERSE2D_BACKEND": "pytorch" if case == "environment" else ""}):
                        actual, events = profiled(lambda: layer(x))
                    self.assertNotIn("converse2d::_channel_layernorm", events)
                    self.check_output(actual, (x, layer.weight, layer.bias))

    def test_grad_mode_and_higher_order_fallback(self):
        for public in (False, True):
            with self.subTest(public=public):
                self.check_training("cuda", public)

    def test_stream_and_graph_replay_with_changed_inputs(self):
        values = fixture((1, 128, 8, 8), "cuda", offset=1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3): private(*values)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
            captured = private(*values)
        with torch.cuda.stream(stream), torch.no_grad():
            for step in range(2):
                values[0].add_(.03125)
                values[1].mul_(.875)
                values[2].add_(.125)
                graph.replay()
                # The comparisons synchronize, and all reference work is queued
                # after replay on the same stream. No output survives its owner.
                self.check_output(captured, values, label=f"graph replay {step}")
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)
