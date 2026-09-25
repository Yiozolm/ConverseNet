"""CPU-only routing checks; symbolic tensors avoid allocating large fixtures."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from models import util_converse


@dataclass(frozen=True)
class ScalarField:
    """A constant-valued tensor stand-in with independently chosen numel."""

    value: float
    elements: int
    requires_grad: bool = False

    def numel(self):
        return self.elements

    def __rmul__(self, factor):
        return ScalarField(factor * self.value, self.elements, self.requires_grad)

    def __add__(self, other):
        return ScalarField(self.value + other.value, self.elements,
                           self.requires_grad or other.requires_grad)


class FakeConvolution:
    def __init__(self, factor, backend):
        self.factor = factor
        self.solver = SimpleNamespace(backend=backend)
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return self.factor * value

    def __getitem__(self, index):
        if index != 3:
            raise AssertionError(f"unexpected solver index: {index}")
        return self.solver


def block(backend="cuda"):
    # Invoke the real forward method with cheap stand-ins for its two branches.
    return SimpleNamespace(alpha1=.25, alpha2=-.5,
                           conv1=FakeConvolution(2., backend),
                           conv2=FakeConvolution(-3., backend))


class PeripheralPolicyCPU(unittest.TestCase):
    def test_alpha_threshold_and_gradmode_route_preserve_expression(self):
        for mode, context in (("grad", torch.enable_grad),
                              ("no_grad", torch.no_grad),
                              ("inference", torch.inference_mode)):
            for elements in (2**21 - 1, 2**21, 2**21 + 1):
                for requires_grad in (False, True):
                    with self.subTest(mode=mode, elements=elements, requires_grad=requires_grad):
                        model = block()
                        value = ScalarField(1.5, elements, requires_grad)
                        calls = []

                        def fused(alpha, branch, residual, backend):
                            calls.append((alpha, branch, residual, backend))
                            return alpha * branch + residual

                        with context(), mock.patch.object(util_converse, "_alpha_residual", side_effect=fused):
                            result = util_converse.ConverseBlockAlphaVariant.forward(model, value)
                        expected_first = .25 * (2. * value) + value
                        expected_final = -.5 * (-3. * expected_first) + expected_first
                        self.assertEqual(result, expected_final)
                        self.assertEqual((model.conv1.calls, model.conv2.calls), (1, 1))
                        should_fuse = mode != "grad" and elements >= 2**21
                        self.assertEqual(len(calls), 2 if should_fuse else 0)
                        if should_fuse:
                            self.assertEqual([call[3] for call in calls], ["cuda", "cuda"])
                            self.assertIs(calls[0][2], value)
                            self.assertEqual(calls[1][2], expected_first)

    def test_large_inference_forwards_backend_without_changing_global_mode(self):
        for backend in ("auto", "pytorch", "cuda"):
            model = block(backend)
            value = ScalarField(-2., 2**21)
            enabled_before = torch.is_grad_enabled()
            with torch.no_grad(), mock.patch.object(
                    util_converse, "_alpha_residual",
                    side_effect=lambda alpha, branch, residual, backend: alpha * branch + residual) as helper:
                result = util_converse.ConverseBlockAlphaVariant.forward(model, value)
                self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(torch.is_grad_enabled(), enabled_before)
            self.assertEqual(result.value, -7.5)
            self.assertEqual([call.args[3] for call in helper.call_args_list], [backend, backend])


if __name__ == "__main__":
    unittest.main(verbosity=2)
