"""Shared fixtures and execution policy for FP32 regression tests."""
import os
import sys
import unittest

import torch
from torch.utils.cpp_extension import CUDA_HOME

from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))

CUDA_BUILD = (os.environ.get("CONVERSE2D_CPU_ONLY") != "1"
              and torch.version.cuda is not None and CUDA_HOME is not None
              and torch.cuda.is_available())


class ExtensionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_extension(cpu_only=not CUDA_BUILD)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def assert_bytes_equal(self, actual, expected, label):
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertEqual(actual.shape, expected.shape, label)
        values = [v.detach().resolve_conj().resolve_neg().cpu().contiguous()
                  for v in (actual, expected)]
        for value in values:
            self.assertTrue(torch.isfinite(value).all().item(), label + ": nonfinite")
        self.assertEqual(values[0].numpy().tobytes(), values[1].numpy().tobytes(),
                         label + ": differs from Python FP32 at zero byte margin")

    def assert_results_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for index, (value, reference) in enumerate(zip(actual, expected)):
            self.assert_bytes_equal(value, reference, f"output/VJP {index}")


@unittest.skipUnless(CUDA_BUILD, "CUDA device/toolkit required; CPU-only builds skip")
class CUDATestCase(ExtensionTestCase):
    pass


def has_full_solve(output):
    # Keep wrappers alive while walking; Python node ids can otherwise recycle.
    pending = [output.grad_fn] if output.grad_fn is not None else []
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if "FullSolve" in node.name():
            return True
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return False


def profiled(call):
    """Inspect CPU dispatcher events for routing, never performance timing."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        output = call()
    return output, {event.key for event in trace.key_averages()}


def fixture(scale, *, kb=1, kc=3, dtype=torch.float32, weak=None):
    generator = torch.Generator(device="cpu").manual_seed(96013 + scale)
    batch, channels, height, width = 2, 3, 5, 6
    values = [
        torch.randn(batch, channels, height, width, generator=generator, dtype=dtype),
        torch.randn(batch, channels, height * scale, width * scale,
                    generator=generator, dtype=dtype),
        torch.rand(kb, kc, 3, 3, generator=generator, dtype=dtype) / 9,
        torch.randn(1, channels, 1, 1, generator=generator, dtype=dtype),
    ]
    if weak is not None:
        values[0].mul_(1e-5)
        values[1].mul_(1e-5)
        values[2].mul_(weak)
        values[3].fill_(-40)
    upstream = torch.randn(values[1].shape, generator=generator, dtype=dtype)
    if weak is not None:
        upstream.mul_(1e-5)
    return values, upstream


def leaves(raw, device, *, strided=False, transpose=False, needs=None):
    needs = (True,) * len(raw) if needs is None else needs
    assert len(raw) == len(needs), "fixture and gradient mask sizes differ"
    result = []
    for value, required in zip(raw, needs):
        value = value.to(device).clone()
        if strided:
            value = torch.stack((value, value), dim=-1)[..., 0]
        if transpose:
            value = value.transpose(-1, -2).contiguous().transpose(-1, -2)
        result.append(value.detach().requires_grad_(required))
    return result


def capture(call, raw, upstream, *, scale, shared=False, strided=False,
            transpose=False, needs=None, eps=1e-5):
    data = leaves(raw, "cuda", strided=strided, transpose=transpose, needs=needs)
    if shared:
        data = [data[0], data[0], *data[-2:]]
    if transpose:
        # Singleton LR widths can be contiguous; the HR prior must stay strided.
        assert not data[1].is_contiguous()
    requested = (data[0], data[2], data[3]) if shared else data
    requested = [value for value in requested if value.requires_grad]
    output = call(*data, scale, eps)
    return (output, *torch.autograd.grad(output, requested, upstream.to("cuda")))


def compare_spatial(case, raw, upstream, **kwargs):
    from models.converse_core import converse2d_reference
    actual = capture(torch.ops.converse2d.forward, raw, upstream, **kwargs)
    expected = capture(converse2d_reference, raw, upstream, **kwargs)
    case.assert_results_equal(actual, expected)
    return actual[0]
