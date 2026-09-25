"""FP32 residual fusion preserves arithmetic, broadcast VJPs and graph edges."""
import itertools
import unittest
from unittest import mock

import torch

from support import CUDATestCase, ExtensionTestCase, leaves, profiled


def reference(alpha, branch, residual):
    return alpha * branch + residual


def raw_fixture(shape=(2, 3, 5, 7), seed=51031):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, shape[1], 1, 1, generator=generator),
            torch.randn(shape, generator=generator),
            torch.randn(shape, generator=generator)], torch.randn(shape, generator=generator)


def output_and_gradients(call, data, upstream, *, retain_graph=False, create_graph=False):
    output = call(*data)
    requested = [value for value in data if value.requires_grad]
    gradients = torch.autograd.grad(output, requested, upstream,
                                    retain_graph=retain_graph, create_graph=create_graph) if requested else ()
    return (output, *gradients)


class PeripheralFusionCPU(ExtensionTestCase):
    def test_cpu_fallback_keeps_original_expression_and_dtype_contract(self):
        raw, upstream = raw_fixture()
        expected = output_and_gradients(reference, leaves(raw, "cpu"), upstream)
        actual = output_and_gradients(torch.ops.converse2d._alpha_residual, leaves(raw, "cpu"), upstream)
        self.assert_results_equal(actual, expected)
        self.assertNotIn("AlphaResidual", actual[0].grad_fn.name())
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(RuntimeError, "FP32"):
                torch.ops.converse2d._alpha_residual(*(value.to(dtype) for value in raw))

    def test_python_module_helper_remains_plain_aten(self):
        from models.util_converse import _alpha_residual
        raw, upstream = raw_fixture()
        actual = output_and_gradients(lambda *data: _alpha_residual(*data, backend="pytorch"), leaves(raw, "cpu"), upstream)
        expected = output_and_gradients(reference, leaves(raw, "cpu"), upstream)
        self.assert_results_equal(actual, expected)


class PeripheralFusionCUDA(CUDATestCase):
    def test_all_gradient_subsets_and_strided_residual_keep_bytes(self):
        for shape in ((1, 1, 1, 1), (2, 3, 5, 7), (1, 64, 20, 24), (1, 3, 0, 7)):
            raw, upstream = raw_fixture(shape)
            for needs in itertools.product((False, True), repeat=3):
                for strided in (False, True):
                    with self.subTest(shape=shape, needs=needs, strided=strided):
                        results = []
                        for call in (torch.ops.converse2d._alpha_residual, reference):
                            data = leaves(raw, "cuda", needs=needs)
                            if strided:
                                data[2] = torch.stack((data[2], data[2]), -1)[..., 0].detach().requires_grad_(needs[2])
                            results.append(output_and_gradients(call, data, upstream.cuda()))
                        self.assert_results_equal(*results)
                        self.assertTrue(results[0][0].is_contiguous())

    def test_forward_has_one_fused_operation_without_fma_or_ftz(self):
        # Separate rounding produces zero; a fused multiply-add produces -2^-46.
        alpha = torch.full((1, 3, 1, 1), 1 + 2**-23, device="cuda")
        branch = torch.full((2, 3, 5, 7), 1 - 2**-23, device="cuda")
        residual = torch.full_like(branch, -1)
        actual, events = profiled(lambda: torch.ops.converse2d._alpha_residual(alpha, branch, residual))
        self.assert_bytes_equal(actual, reference(alpha, branch, residual), "separate FP32 rounding")
        self.assertEqual(actual.count_nonzero().item(), 0)
        self.assertNotIn("aten::mul", events)
        self.assertNotIn("aten::add", events)
        alpha.fill_(.5)
        branch.fill_(2**-126)
        residual.fill_(2**-149)
        self.assert_bytes_equal(torch.ops.converse2d._alpha_residual(alpha, branch, residual),
                                reference(alpha, branch, residual), "subnormals")

    def test_fallback_layouts_and_other_alpha_broadcasts_keep_bytes(self):
        raw, upstream = raw_fixture()
        for kind in ("branch_transpose", "channels_last", "negative", "alpha_scalar", "alpha_batch"):
            with self.subTest(kind=kind):
                results = []
                for call in (torch.ops.converse2d._alpha_residual, reference):
                    data = leaves(raw, "cuda")
                    if kind == "branch_transpose":
                        data[1] = data[1].transpose(-1, -2).contiguous().transpose(-1, -2).detach().requires_grad_()
                    elif kind == "channels_last":
                        data[1] = data[1].contiguous(memory_format=torch.channels_last).detach().requires_grad_()
                    elif kind == "negative":
                        data[1] = torch._neg_view(data[1]).detach().requires_grad_()
                    elif kind == "alpha_scalar":
                        data[0] = data[0][..., :1, :1][:, :1].detach().requires_grad_()
                    else:
                        data[0] = data[0].expand(2, -1, -1, -1).clone().detach().requires_grad_()
                    results.append(output_and_gradients(call, data, upstream.cuda()))
                self.assert_results_equal(*results)
                self.assertNotIn("AlphaResidual", results[0][0].grad_fn.name())

    def test_alias_inputs_use_original_aten_accumulation(self):
        raw, upstream = raw_fixture()
        for view in (False, True):
            results = []
            for call in (torch.ops.converse2d._alpha_residual, reference):
                alpha, branch, _ = leaves(raw, "cuda")
                residual = branch.view_as(branch) if view else branch
                output = call(alpha, branch, residual)
                gradients = torch.autograd.grad(output, (alpha, branch), upstream.cuda())
                results.append((output, *gradients))
            self.assert_results_equal(*results)
            self.assertNotIn("AlphaResidual", results[0][0].grad_fn.name())

    def test_shared_ancestors_and_two_residual_stages_keep_gradient_order(self):
        for seed in (19, 31, 43):
            raw, upstream = raw_fixture(seed=seed)
            for dynamic_alpha in (False, True):
                results = []
                for call in (torch.ops.converse2d._alpha_residual, reference):
                    base = raw[1].cuda().requires_grad_()
                    alpha = base.mean((0, 2, 3), keepdim=True) if dynamic_alpha else raw[0].cuda().requires_grad_()
                    first = call(alpha, base.sin(), base)
                    second = call(alpha.sigmoid(), first.cos(), first)
                    loss = (second * upstream.cuda()).sum() + base.square().sum() * .03
                    requested = (base,) if dynamic_alpha else (base, alpha)
                    results.append((second, *torch.autograd.grad(loss, requested)))
                with self.subTest(seed=seed, dynamic_alpha=dynamic_alpha):
                    self.assert_results_equal(*results)

    def test_higher_order_aten_derivatives_and_repeated_backward(self):
        raw, upstream = raw_fixture()
        results = []
        for call in (torch.ops.converse2d._alpha_residual, reference):
            data = leaves(raw, "cuda")
            output = call(*data)
            first = torch.autograd.grad(output.sin().mean(), data, create_graph=True)
            second = torch.autograd.grad(sum(value.square().mean() for value in first), data, create_graph=True)
            third = torch.autograd.grad(sum(value.square().mean() for value in second), data)
            results.append((output, *first, *second, *third))
        self.assert_results_equal(results[0][:4], results[1][:4])
        for actual, expected in zip(results[0][4:], results[1][4:]):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        repeated = []
        for call in (torch.ops.converse2d._alpha_residual, reference):
            data = leaves(raw, "cuda")
            output = call(*data)
            first = torch.autograd.grad(output, data, upstream.cuda(), retain_graph=True)
            second = torch.autograd.grad(output, data, upstream.cuda() * -.3)
            repeated.append((*first, *second))
        self.assert_results_equal(*repeated)

    def test_saved_versions_match_needed_operands(self):
        raw, _ = raw_fixture()
        for need, mutate in (((True, False, False), 1), ((False, True, False), 0)):
            data = leaves(raw, "cuda", needs=need)
            output = torch.ops.converse2d._alpha_residual(*data)
            with torch.no_grad():
                data[mutate].add_(1)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                output.sum().backward()
        data = leaves(raw, "cuda", needs=(False, False, True))
        output = torch.ops.converse2d._alpha_residual(*data)
        data[0].add_(1)
        data[1].add_(1)
        gradient = torch.autograd.grad(output.sum(), data[2])[0]
        self.assert_bytes_equal(gradient, torch.ones_like(gradient), "residual-only saved operands")
        data = leaves(raw, "cuda")
        output = torch.ops.converse2d._alpha_residual(*data)
        with torch.no_grad():
            data[0].add_(1)
            data[1].add_(1)
        gradient = torch.autograd.grad(output.sum(), data[2])[0]
        self.assert_bytes_equal(gradient, torch.ones_like(gradient), "residual-only requested VJP")

    def test_actual_alpha_block_parameters_and_vjps_keep_bytes(self):
        from models.util_converse import ConverseBlockAlphaVariant
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            # CUDA replicate-pad backward is nondeterministic even without this
            # fusion. Compare that case in its explicit deterministic mode;
            # circular is the actual USRNet prior-block configuration.
            torch.use_deterministic_algorithms(True)
            for padding_mode in ("circular", "replicate"):
                with self.subTest(padding_mode=padding_mode):
                    torch.manual_seed(51037)
                    block = ConverseBlockAlphaVariant(3, 3, padding_mode=padding_mode).cuda()
                    block.conv1[3].backend = "cuda"
                    with torch.no_grad():
                        block.alpha1.normal_(0, .2)
                        block.alpha2.normal_(0, .2)
                    names = list(block.state_dict())
                    raw, upstream = raw_fixture()
                    results = []
                    for fused in (True, False):
                        x = raw[1].cuda().requires_grad_()
                        if fused:
                            output = block(x)
                        else:
                            output = block.alpha1 * block.conv1(x) + x
                            output = block.alpha2 * block.conv2(output) + output
                        results.append((output, *torch.autograd.grad(output, (x, *block.parameters()), upstream.cuda())))
                    self.assert_results_equal(*results)
                    self.assertEqual(names, list(block.state_dict()))
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_python_backend_and_missing_auto_operator_fallback(self):
        import models.util_converse as util
        raw, upstream = raw_fixture()
        namespace = torch.ops.converse2d
        original_hasattr = hasattr
        def missing_operator(obj, name):
            return False if obj is namespace and name == "_alpha_residual" else original_hasattr(obj, name)
        for backend in ("pytorch", "auto"):
            with self.subTest(backend=backend), mock.patch("builtins.hasattr", side_effect=missing_operator):
                output, events = profiled(lambda: util._alpha_residual(*leaves(raw, "cuda"), backend=backend))
                self.assertNotIn("AlphaResidual", output.grad_fn.name())
                self.assertIn("aten::mul", events)
                self.assertIn("aten::add", events)

    def test_nondefault_stream_and_graph_replay(self):
        raw, upstream = raw_fixture()
        data = leaves(raw, "cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = output_and_gradients(torch.ops.converse2d._alpha_residual, data, upstream.cuda())
            expected = output_and_gradients(reference, data, upstream.cuda())
        stream.synchronize()
        self.assert_results_equal(actual, expected)
        frozen = [value.detach().clone() for value in data]
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            for _ in range(3):
                torch.ops.converse2d._alpha_residual(*frozen)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                captured = torch.ops.converse2d._alpha_residual(*frozen)
            for delta in (0., .125, -.25):
                frozen[2].add_(delta)
                graph.replay()
                self.assert_bytes_equal(captured, reference(*frozen), "Graph replay")


if __name__ == "__main__":
    unittest.main(verbosity=2)
