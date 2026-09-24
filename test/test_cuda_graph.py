"""Graph lifetime, invalidation, input changes and stream regression tests."""
import os
import unittest
from unittest.mock import patch

import torch
from support import CUDATestCase


class CUDAGraphTests(CUDATestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from models.converse_usrnet import ConverseUSRNet
        from models.cuda_graph import USRNetCUDAGraph
        cls.model_type = ConverseUSRNet
        cls.runner_type = USRNetCUDAGraph

    def setUp(self):
        torch.manual_seed(781)
        self.model = self.model_type(num_iterations=2, num_blocks=1, backend='cuda').cuda().eval()
        self.runner = self.runner_type(self.model, warmup=2)
        self.x = torch.rand(1, 3, 8, 10, device='cuda')
        self.k = torch.rand(1, 1, 7, 7, device='cuda')
        self.k /= self.k.sum()

    def tearDown(self):
        self.runner.clear()
        torch.ops.converse2d.clear_cache()

    def compare(self, x=None, k=None, scale=2):
        x = self.x if x is None else x
        k = self.k if k is None else k
        with torch.inference_mode():
            expected = self.model(x, k, scale)
            actual = self.runner(x, k, scale)
        tol = 3e-5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        return actual

    def test_changed_inputs_and_independent_outputs(self):
        first = self.compare()
        saved = first.clone()
        self.compare(self.x + .1, self.k.flip(-1))
        # Contiguous static buffers also accept noncontiguous caller tensors.
        wide = torch.rand(1, 3, 8, 20, device='cuda')
        self.compare(wide[..., ::2], self.k.transpose(-1, -2))
        torch.testing.assert_close(first, saved, atol=0, rtol=0)
        self.assertEqual(self.runner.captures, 1)
        torch.ops.converse2d.clear_cache()
        churn = torch.empty(8 * 1024 * 1024, device='cuda').fill_(123)
        self.compare()
        del churn

    def test_optional_switch_releases_and_recaptures(self):
        self.compare()
        self.assertEqual(self.runner.cached_graphs, 1)
        self.runner.enabled = False
        self.assertEqual(self.runner.cached_graphs, 0)
        with patch.object(self.runner, '_capture', side_effect=AssertionError('unexpected capture')):
            self.compare(self.x + .1)
        self.assertEqual(self.runner.captures, 1)
        self.runner.enabled = True
        self.compare(self.x + .2)
        self.assertEqual(self.runner.captures, 2)

    def test_disabled_runner_preserves_training(self):
        self.runner.enabled = False
        self.model.train()
        x = self.x.detach().requires_grad_()
        expected = self.model(x, self.k, 2)
        actual = self.runner(x, self.k, 2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        parameters = (x, self.model.kernelnet.fc1.weight, self.model.d.alpha)
        reference_grads = torch.autograd.grad(expected.square().mean(), parameters)
        grads = torch.autograd.grad(actual.square().mean(), parameters)
        for grad, reference in zip(grads, reference_grads):
            torch.testing.assert_close(grad, reference, atol=0, rtol=0)
        self.assertEqual(self.runner.captures, 0)

    def test_disabled_constructor_runs_on_cpu(self):
        model = self.model_type(num_iterations=1, num_blocks=1, backend='pytorch').eval()
        runner = self.runner_type(model, enabled=False)
        x, kernel = self.x.cpu(), self.k.cpu()
        with patch.object(runner, '_validate', side_effect=AssertionError('graph-only validation')):
            actual = runner(x, kernel, 2)
        torch.testing.assert_close(actual, model(x, kernel, 2), atol=0, rtol=0)
        self.assertEqual(runner.cached_graphs, 0)
        self.assertEqual(runner.captures, 0)
        with self.assertRaises(TypeError):
            runner.enabled = 'false'

    def test_parameter_updates_and_configuration(self):
        self.compare()
        with torch.no_grad():
            self.model.conv2.bias.add_(.2)
        self.compare()
        self.assertEqual(self.runner.captures, 2)
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        state['conv2.bias'].add_(.1)
        self.model.load_state_dict(state)
        self.compare()
        self.assertEqual(self.runner.captures, 3)
        self.model.conv2.bias = torch.nn.Parameter(self.model.conv2.bias.detach().clone() + .1)
        self.compare()
        self.assertEqual(self.runner.captures, 4)
        self.model.conv2.bias.data = self.model.conv2.bias.detach().clone() + .1
        self.compare()
        self.assertEqual(self.runner.captures, 5)
        self.model.d.eps *= 2
        self.compare()
        self.assertEqual(self.runner.captures, 6)
        layer = next(m for m in self.model.modules() if type(m).__name__ == 'Converse2D')
        with torch.no_grad():
            layer.weight.mul_(.95)
        self.compare()
        self.assertEqual(self.runner.captures, 7)
        with patch.dict(os.environ, {'CONVERSE2D_BACKEND': 'pytorch'}):
            self.compare()
        self.assertEqual(self.runner.captures, 8)

    def test_lru_shape_scale_batch_dtype(self):
        self.runner.max_graphs = 2
        self.compare(scale=1)
        self.compare(scale=2)
        self.compare(scale=1)
        self.assertEqual(self.runner.captures, 2)
        self.compare(scale=3)
        self.assertEqual(self.runner.cached_graphs, 2)
        self.compare(scale=2)
        self.assertEqual(self.runner.captures, 4)
        self.compare(self.x.repeat(2, 1, 1, 1), self.k.repeat(2, 1, 1, 1))
        self.compare(self.x.repeat(2, 1, 1, 1))  # shared kernel
        self.runner.clear()
        self.assertEqual(self.runner.cached_graphs, 0)

    def test_sequential_calls_on_different_streams(self):
        self.compare()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outputs = []
        for i, stream in enumerate(streams):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                x = self.x + i * .1
                outputs.append((self.runner(x, self.k, 2), self.model(x, self.k, 2)))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for actual, expected in outputs:
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        self.assertEqual(self.runner.captures, 1)
        # Clear also waits for queued copies/replays before releasing their pool.
        self.runner.clear()

    def test_invalid_modes_and_training_after_inference(self):
        with self.assertRaisesRegex(RuntimeError, 'no_grad'):
            self.runner(self.x, self.k, 2)
        self.compare()
        with torch.no_grad(), torch.autocast('cuda'):
            with self.assertRaisesRegex(RuntimeError, 'autocast'):
                self.runner(self.x, self.k, 2)
        self.model.train()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'eval'):
            self.runner(self.x, self.k, 2)
        self.assertEqual(self.runner.cached_graphs, 0)
        self.model(self.x, self.k, 2).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.model.eval()
        with self.model.register_forward_hook(lambda m, a, o: o):
            with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'hooks'):
                self.runner(self.x, self.k, 2)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'CUDA device'):
            self.runner(self.x.cpu(), self.k.cpu(), 2)

    def test_operator_capture_ignores_warm_cache(self):
        # A warm eager cache on the SAME capture stream must not supply graph
        # spectra. Mutating weights after capture exposes stale cached values.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for scale in (1, 2, 3):
            with self.subTest(scale=scale):
                w = torch.rand(1, 2, 3, 3, device='cuda')
                b = torch.zeros(1, 2, 1, 1, device='cuda')
                x = torch.rand(1, 2, 8, 10, device='cuda')
                x0 = torch.nn.functional.interpolate(x, scale_factor=scale)
                stream.wait_stream(torch.cuda.current_stream())
                def forward():
                    return torch.ops.converse2d.forward(x, x0, w, b, scale, 1e-5)
                with torch.cuda.stream(stream), torch.no_grad():
                    for _ in range(3): forward()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
                    output = forward()
                with torch.no_grad():
                    w.mul_(.9)
                    torch.ops.converse2d.clear_cache()
                    expected = forward()
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)
                graph.reset()

    def test_failed_capture_releases_spectrum_scope(self):
        forward = self.model.forward
        calls = 0

        def fail_during_capture(*args):
            nonlocal calls
            calls += 1
            result = forward(*args)
            if calls > self.runner.warmup:
                raise RuntimeError('intentional capture failure')
            return result

        with patch.object(self.model, 'forward', side_effect=fail_during_capture):
            with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'intentional'):
                self.runner(self.x, self.k, 2)
        self.assertEqual(self.runner.cached_graphs, 0)
        self.compare()
        self.assertEqual(self.runner.captures, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
