"""cuFFT callback training FFTs: selection boundary and FP64 budgets at >= 16.

Planes with a side below 16 stay on ATen, so the small fixtures elsewhere in
the suite do not exercise this path. A callback plan is also admitted only if
a random probe reproduces the replaced ATen expression bit for bit; shapes
whose callback plan uses another FFT algorithm (96x96 on the RTX 5060 Ti) keep
ATen. Every comparison here uses the frozen FP32 baseline and the independent
FP64 reference through numerical_policy.
"""
import torch
import torch.nn.functional as F

from support import CUDATestCase, compare_spatial
from fp32_baseline import converse2d_reference as baseline_reference
from models.converse_core import converse2d_reference
from numerical_policy import assert_budget


def kernel_names(call):
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    return [event.name for event in trace.events() if event.device_time > 0]


def public_raw(scale, batch, channels, height, width, seed, weak=False):
    generator = torch.Generator().manual_seed(seed)
    raw = [torch.randn(batch, channels, height, width, generator=generator),
           torch.randn(batch, channels, height * scale, width * scale, generator=generator),
           torch.softmax(torch.randn(1, channels, 9, generator=generator), -1).reshape(1, channels, 3, 3),
           torch.randn(1, channels, 1, 1, generator=generator)]
    upstream = torch.randn(batch, channels, height * scale, width * scale, generator=generator)
    if weak:
        raw[3].fill_(-40)
    return raw, upstream


def circular_reference(x, weight, bias, padding, eps=1e-5):
    reference = baseline_reference if x.dtype == torch.float32 else converse2d_reference
    padded = F.pad(x, (padding,) * 4, mode="circular")
    return reference(padded, padded, weight, bias, 1, eps)[..., padding:-padding, padding:-padding]


def circular(x, weight, bias, padding, eps=1e-5):
    return torch.ops.converse2d._training_circular_s1(x, weight, bias, padding, eps)


def run(fn, raw, upstream, dtype, *args):
    data = [value.to(device="cuda", dtype=dtype).clone().requires_grad_() for value in raw]
    out = fn(*data, *args)
    return (out, *torch.autograd.grad(out, data, upstream.to(device="cuda", dtype=dtype)))


class FFTCallbacksCUDA(CUDATestCase):
    def test_callback_transforms_are_selected_only_at_and_above_side_16(self):
        # 64x64 and 33x17 callback plans reproduced ATen bit for bit in tools/v4_cufft_callbacks.
        for height, width, selected in ((64, 64, True), (33, 17, True), (15, 40, False), (40, 15, False)):
            raw, upstream = public_raw(1, 2, 3, height, width, 7300 + height)
            data = [value.cuda().requires_grad_() for value in raw]
            names = kernel_names(lambda: torch.autograd.grad(
                torch.ops.converse2d.forward(data[0], data[0], data[2], data[3], 1, 1e-5),
                data[0], upstream.cuda()))
            with self.subTest(height=height, width=width):
                # Forward x FFT, the scaled IFFT and the crop-embed VJP each link callbacks.
                self.assertEqual(any("lto_fft" in name for name in names), selected)
                # The separate scaling passes exist exactly when callbacks are not used.
                self.assertEqual(any("AUnaryFunctor" in name for name in names), not selected)

    def test_public_forward_budgets_at_every_scale(self):
        cases = [(1, True, 2, 3, 20, 24), (1, False, 2, 3, 20, 24), (1, True, 1, 4, 96, 96),
                 (2, False, 2, 3, 10, 12), (2, False, 2, 4, 48, 48), (3, False, 2, 3, 8, 9),
                 (3, False, 1, 4, 32, 32)]
        for index, (scale, shared, batch, channels, height, width) in enumerate(cases):
            for weak in (False, True):
                raw, upstream = public_raw(scale, batch, channels, height, width, 7400 + index, weak)
                with self.subTest(scale=scale, shared=shared, shape=(batch, channels, height, width), weak=weak):
                    compare_spatial(self, raw, upstream, scale=scale, shared=shared,
                                    eps=1e-8 if weak else 1e-5)

    def test_strided_and_transposed_layouts_keep_budgets(self):
        # Noncontiguous inputs take ATen for that FFT; the IFFT and VJP still use callbacks.
        for layout in ("strided", "transpose"):
            for scale in (1, 2):
                raw, upstream = public_raw(scale, 2, 3, 16 if scale == 1 else 8, 18 if scale == 1 else 9, 7500 + scale)
                with self.subTest(layout=layout, scale=scale):
                    compare_spatial(self, raw, upstream, scale=scale, **{layout: True})

    def test_circular_s1_budgets_and_padding(self):
        for batch, channels, height, width, padding in ((2, 3, 14, 16, 2), (1, 4, 30, 17, 3), (2, 8, 96, 96, 2)):
            generator = torch.Generator().manual_seed(7600 + height)
            raw = [torch.randn(batch, channels, height, width, generator=generator),
                   torch.rand(1, channels, 3, 3, generator=generator) / 9,
                   torch.randn(1, channels, 1, 1, generator=generator)]
            upstream = torch.randn(raw[0].shape, generator=generator)
            with self.subTest(shape=(batch, channels, height, width), padding=padding):
                assert_budget(self, run(circular, raw, upstream, torch.float32, padding),
                              run(circular_reference, raw, upstream, torch.float32, padding),
                              run(circular_reference, raw, upstream, torch.float64, padding))

    def test_second_derivatives_use_the_aten_vjp_expressions(self):
        raw, upstream = public_raw(1, 2, 3, 16, 20, 7700)
        results = []
        for fn, dtype in ((torch.ops.converse2d.forward, torch.float32), (baseline_reference, torch.float32),
                          (converse2d_reference, torch.float64)):
            x, _, weight, bias = [v.to(device="cuda", dtype=dtype).clone().requires_grad_() for v in raw]
            out = fn(x, x, weight, bias, 1, 1e-5)
            gx, gw = torch.autograd.grad((out * upstream.cuda().to(dtype)).square().sum(), (x, weight),
                                         create_graph=True)
            results.append((out, gx, gw, *torch.autograd.grad(gx.square().sum() + gw.sum(), (x, weight, bias))))
        assert_budget(self, *results)

    def test_inplace_rebase_of_the_output_view(self):
        # Rebasing reconstructs the native chain through the scaled-IFFT node.
        raw, upstream = public_raw(1, 2, 3, 18, 16, 7800)
        results = []
        for fn, dtype in ((torch.ops.converse2d.forward, torch.float32), (baseline_reference, torch.float32),
                          (converse2d_reference, torch.float64)):
            x, _, weight, bias = [v.to(device="cuda", dtype=dtype).clone().requires_grad_() for v in raw]
            out = fn(x, x, weight, bias, 1, 1e-5)
            out.mul_(2.0)
            results.append((out, *torch.autograd.grad(out, (x, weight, bias), upstream.cuda().to(dtype))))
        assert_budget(self, *results)
