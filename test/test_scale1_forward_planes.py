"""s1 forward plane kernel: dispatch boundary and byte parity with the fallback."""
import torch
from support import CUDATestCase


def forward_kernels(*inputs):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
        torch.ops.converse2d._training_full_spectral(*inputs, 1)
        torch.cuda.synchronize()
    return {event.name for event in trace.events() if "scale1_forward" in event.name}


def solve(y, k, regularizer, upstream):
    inputs = [t.detach().clone().requires_grad_() for t in (y, k, regularizer)]
    out = torch.ops.converse2d._training_full_spectral(inputs[0], inputs[0], inputs[1], inputs[2], 1)
    return [out, *torch.autograd.grad(out, inputs, upstream)]


class Scale1ForwardPlanesCUDA(CUDATestCase):
    def test_plane_grid_limit_selects_the_original_kernel(self):
        for batch, channels, kb, planes in ((4, 3, 1, True), (4, 3, 4, True), (1, 65535, 1, True),
                                            (1, 65536, 1, False), (2, 32768, 2, False)):
            with self.subTest(batch=batch, channels=channels, kb=kb):
                y = torch.randn(batch, channels, 2, 3, device="cuda", dtype=torch.complex64, requires_grad=True)
                k = torch.randn(kb, channels, 2, 3, device="cuda", dtype=torch.complex64, requires_grad=True)
                regularizer = torch.full((1, channels, 1, 1), .1, device="cuda", requires_grad=True)
                names = forward_kernels(y, y, k, regularizer)
                self.assertEqual(len(names), 1)
                self.assertEqual("scale1_forward_planes" in names.pop(), planes)

    def test_fallback_and_plane_kernels_are_byte_identical(self):
        # Channels are independent in s1 (B1, KB1, KC=C): one 65536-row call
        # uses the fallback, its two 32768-row halves use the plane kernel.
        generator = torch.Generator().manual_seed(65536)
        shape = (1, 65536, 2, 3)
        y, k, upstream = (torch.randn(shape, dtype=torch.complex64, generator=generator).cuda() for _ in range(3))
        k[..., 0, 0] = 0
        regularizer = torch.rand(1, 65536, 1, 1, generator=generator).add(.05).cuda()
        whole = solve(y, k, regularizer, upstream)
        halves = [solve(*(t[:, part] for t in (y, k, regularizer, upstream)))
                  for part in (slice(0, 32768), slice(32768, None))]
        for index, value in enumerate(whole):
            self.assert_bytes_equal(torch.cat([half[index] for half in halves], 1), value, f"output/VJP {index}")
