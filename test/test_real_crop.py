"""Acceptance tests for zero-copy real/crop with a fused first-order VJP.

The ordinary VJP is fused while native view metadata and rebasing remain intact.
"""
import contextlib
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(os.environ.get("CONVERSE_TEST_ROOT", pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT))
import torch
from support import CUDATestCase


def candidate(z, pad):
    return torch.ops.converse2d._training_real_crop(z, pad)


def reference(z, pad):
    return z.real[..., pad:-pad, pad:-pad] if pad else z.real


def raw_complex(shape=(2, 3, 7, 9)):
    return torch.randn(shape, dtype=torch.complex64, generator=torch.Generator().manual_seed(98173))


def node_names(output):
    pending = [output.grad_fn] if output.grad_fn is not None else []
    seen, names = set(), set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        names.add(node.name())
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return names


def has_real_crop_node(output):
    return any("RealCropBackward" in name for name in node_names(output))


def upstream(shape, kind):
    g = torch.randn(shape, generator=torch.Generator().manual_seed(98179)).cuda()
    if kind == "strided":
        return torch.stack((g, g), -1)[..., 0]
    if kind == "transposed":
        return g.transpose(-1, -2).contiguous().transpose(-1, -2)
    if kind == "expanded":
        return g[:1, :1, :1, :1].expand(shape)
    if kind == "negative":
        return torch._neg_view(g)
    if kind == "atoms":
        atoms = torch.tensor([0., -0., 2.**-149, -(2.**-149), 2.**-126,
                              -(2.**-126), 2.**24, -(2.**24)], device="cuda")
        return atoms.repeat((g.numel() + 7) // 8)[:g.numel()].reshape(shape)
    return g


class RealCropContracts(CUDATestCase):
    def setUp(self):
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.fill = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False

    def tearDown(self):
        torch.use_deterministic_algorithms(self.deterministic, warn_only=self.warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = self.fill

    def test_unpadded_public_forward_uses_fused_vjp_at_every_scale(self):
        for scale, shared in ((1, True), (1, False), (2, False), (3, False)):
            with self.subTest(scale=scale, shared=shared):
                generator = torch.Generator().manual_seed(98200 + scale)
                x = torch.randn(2, 3, 6, 7, generator=generator).cuda().requires_grad_()
                x0 = x if shared else torch.randn(2, 3, 6 * scale, 7 * scale, generator=generator).cuda().requires_grad_()
                weight = torch.rand(1, 3, 3, 3, generator=generator).cuda().requires_grad_()
                bias = torch.randn(1, 3, 1, 1, generator=generator).cuda()
                out = torch.ops.converse2d.forward(x, x0, weight, bias, scale, 1e-5)
                # The fused real/crop+IFFT node edges go straight to the solved spectrum.
                self.assertIn("RealCropIFFTBackward", node_names(out))
                # Native view metadata: out is the real view of the complex IFFT.
                z = out._base
                self.assertEqual(z.dtype, torch.complex64)
                self.assertEqual((out.stride(), out.storage_offset()), (z.real.stride(), z.real.storage_offset()))
                # Planes below 16 keep ATen FFTs, so the fused VJP must equal the
                # native z.real chain (through z's IFFT node) byte for byte.
                g = upstream(out.shape, "transposed")
                inputs = (x, weight) if shared else (x, x0, weight)
                fused = torch.autograd.grad(out, inputs, g, retain_graph=True)
                native = torch.autograd.grad(z.real, inputs, g)
                for index, (a, b) in enumerate(zip(fused, native)):
                    self.assert_bytes_equal(a, b, f"public forward real VJP {index}")
                with torch.no_grad():
                    frozen = torch.ops.converse2d.forward(x, x0, weight, bias, scale, 1e-5)
                    self.assertFalse(any("RealCrop" in name for name in node_names(frozen)))

    def test_exact_view_alias_layout_and_no_grad_modes(self):
        schema = torch.ops.converse2d._training_real_crop.default._schema
        self.assertIsNotNone(schema.arguments[0].alias_info)
        self.assertIsNotNone(schema.returns[0].alias_info)
        self.assertEqual(schema.arguments[0].alias_info.after_set, schema.returns[0].alias_info.after_set)
        self.assertFalse(schema.arguments[0].alias_info.is_write)
        for pad in (0, 1, 3):
            for mode in ("grad", "frozen", "no_grad", "inference"):
                context = (torch.no_grad() if mode == "no_grad" else
                           torch.inference_mode() if mode == "inference" else contextlib.nullcontext())
                with context, self.subTest(pad=pad, mode=mode):
                    z = raw_complex().cuda().requires_grad_(mode != "frozen")
                    out, expected = candidate(z, pad), reference(z, pad)
                    self.assert_bytes_equal(out, expected, "real crop view")
                    self.assertEqual(out.dtype, torch.float32)
                    self.assertEqual(out.stride(), expected.stride())
                    self.assertEqual(out.storage_offset(), expected.storage_offset())
                    self.assertEqual(out.data_ptr(), expected.data_ptr())
                    self.assertTrue(torch._C._is_alias_of(out, z))
                    self.assertEqual(out._base is z, expected._base is z)
                    self.assertEqual(out.requires_grad, expected.requires_grad)
                    self.assertEqual(has_real_crop_node(out), mode == "grad")

    def test_fused_reverse_path_is_executed(self):
        # A candidate silently falling back to native ATen everywhere would
        # pass numerical tests. Check the reverse graph and execution events.
        # CPU dispatcher events inspect routing only, never performance.
        for pad in (0, 2):
            z = raw_complex().cuda().requires_grad_()
            out = candidate(z, pad)
            self.assertTrue(has_real_crop_node(out), "eligible reverse path was not selected")
            self.assertNotIn("SliceBackward0", node_names(out))
            g = upstream(out.shape, "contiguous")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                dz = torch.autograd.grad(out, z, g)[0]
            events = {event.key for event in trace.key_averages()}
            self.assertTrue(any("RealCropBackward" in event for event in events),
                            "fused backward Node was not executed")
            for event in ("aten::slice_backward", "aten::select_backward",
                          "aten::constant_pad_nd", "aten::complex", "aten::_to_copy"):
                self.assertNotIn(event, events, "ordinary backward used ATen glue: " + event)
            expected_z = raw_complex().cuda().requires_grad_()
            expected = torch.autograd.grad(reference(expected_z, pad), expected_z, g)[0]
            self.assert_bytes_equal(dz, expected, "executed fused VJP")

    def test_native_fallback_for_cpu_input_layout_and_lazy_flags(self):
        for device in ("cpu", "cuda"):
            for kind in ("contiguous", "transposed", "strided", "expanded", "conjugate", "negative", "conjugate_negative"):
                if device == "cuda" and kind == "contiguous":
                    continue  # Covered separately as the required fused path.
                results, layouts = [], []
                for fn in (candidate, reference):
                    leaf = raw_complex().to(device).requires_grad_()
                    if kind == "transposed":
                        z = leaf.transpose(-1, -2)
                    elif kind == "strided":
                        z = torch.stack((leaf, leaf), -1)[..., 0]
                    elif kind == "expanded":
                        z = leaf[:1, :1].expand(2, 3, 7, 9)
                    elif kind == "conjugate":
                        z = leaf.conj()
                    elif kind == "negative":
                        z = torch._neg_view(leaf)
                    elif kind == "conjugate_negative":
                        z = torch._neg_view(leaf.conj())
                    else:
                        z = leaf
                    out = fn(z, 2)
                    if fn is candidate:
                        self.assertFalse(has_real_crop_node(out), "unsupported input failed to fall back")
                    self.assertTrue(torch._C._is_alias_of(out, z))
                    layouts.append((out.stride(), out.storage_offset(), out.is_neg(), out.is_conj(),
                                    out._base is not None))
                    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(98179)).to(device)
                    results.append((out, torch.autograd.grad(out, leaf, g)[0]))
                with self.subTest(device=device, kind=kind):
                    self.assertEqual(*layouts)
                    self.assert_results_equal(*results)

    def test_native_forward_ad_fallback_preserves_tangent_and_reverse_grad(self):
        for device in ("cpu", "cuda"):
            for pad in (0, 2):
                results = []
                for fn in (candidate, reference):
                    leaf = raw_complex().to(device).requires_grad_()
                    tangent = raw_complex().to(device) * .375
                    with torch.autograd.forward_ad.dual_level():
                        dual = torch.autograd.forward_ad.make_dual(leaf, tangent)
                        out = fn(dual, pad)
                        primal, output_tangent = torch.autograd.forward_ad.unpack_dual(out)
                        self.assertIsNotNone(output_tangent, "forward AD tangent was dropped")
                        if fn is candidate:
                            self.assertFalse(has_real_crop_node(primal), "forward AD must use native view")
                        gradient = torch.autograd.grad(primal.square().mean(), leaf)[0]
                        results.append((primal.detach(), output_tangent.detach(), gradient))
                with self.subTest(device=device, pad=pad):
                    self.assert_results_equal(*results)

    def test_signed_zero_subnormal_strided_upstream_and_zero_embed(self):
        for shape, pad in (((1, 1, 1, 1), 0), ((1, 2, 3, 5), 1),
                           ((2, 3, 7, 9), 0), ((2, 3, 7, 9), 3)):
            cropped = (*shape[:2], shape[2] - 2 * pad, shape[3] - 2 * pad)
            for kind in ("contiguous", "strided", "transposed", "expanded", "negative", "atoms"):
                for fill in (False, True):
                    torch.utils.deterministic.fill_uninitialized_memory = fill
                    g = upstream(cropped, kind)
                    results = []
                    for fn in (candidate, reference):
                        z = raw_complex(shape).cuda().requires_grad_()
                        out = fn(z, pad)
                        dz = torch.autograd.grad(out, z, g)[0]
                        results.append((out, dz))
                    with self.subTest(shape=shape, pad=pad, kind=kind, fill=fill):
                        self.assert_results_equal(*results)
                        self.assertTrue(results[0][1].is_contiguous())
                        self.assertEqual(results[0][1].dtype, torch.complex64)

    def test_higher_order_independent_upstream_and_repeated_backward(self):
        for pad in (0, 2):
            results = []
            for fn in (candidate, reference):
                z = raw_complex().cuda().requires_grad_()
                out = fn(z, pad)
                g = upstream(out.shape, "transposed").detach().requires_grad_()
                dz = torch.autograd.grad(out, z, g, create_graph=True, retain_graph=True)[0]
                self.assertTrue(dz.requires_grad)
                unused = torch.autograd.grad(dz.real.sum(), z, allow_unused=True, retain_graph=True)[0]
                self.assertIsNone(unused, "linear VJP must not invent dependency on input values")
                direction = raw_complex().cuda()
                dg = torch.autograd.grad(dz, g, direction, retain_graph=True)[0]
                repeated = torch.autograd.grad(out, z, g * .25, create_graph=True)[0]
                results.append((dz, dg, repeated))
            with self.subTest(pad=pad):
                self.assert_results_equal(*results)

    def test_second_third_derivatives_nonlinear_chain(self):
        for pad in (0, 2):
            results = []
            for fn in (candidate, reference):
                z = raw_complex().cuda().requires_grad_()
                out = fn(z, pad)
                first = torch.autograd.grad(out.sin().square().mean(), z, create_graph=True)[0]
                second = torch.autograd.grad(first.abs().square().mean(), z, create_graph=True)[0]
                third = torch.autograd.grad(second.abs().square().mean(), z)[0]
                results.append((first, second, third))
            self.assert_results_equal(*results)

    def test_output_base_alias_and_no_grad_mutation_semantics(self):
        # A custom Function returning a view fails all these previously legal
        # mutations. The raw primitive must behave as a normal PyTorch view.
        for pad in (0, 2):
            for mutation in ("out", "base", "alias", "nograd_out", "nograd_base"):
                results = []
                for fn in (candidate, reference):
                    leaf = raw_complex().cuda().requires_grad_()
                    base = leaf * 2
                    out = fn(base, pad)
                    if mutation == "out":
                        out.mul_(.875).add_(.125)
                    elif mutation == "base":
                        base.mul_(.875).add_(.125)
                    elif mutation == "alias":
                        base.real.mul_(.875).add_(.125)
                    else:
                        with torch.no_grad():
                            (out if mutation == "nograd_out" else base).mul_(.875).add_(.125)
                    loss = out.sin().square().mean() + base.real.square().mean() * .03125
                    first = torch.autograd.grad(loss, leaf, create_graph=True)[0]
                    second = torch.autograd.grad(first.abs().square().mean(), leaf)[0]
                    results.append((out, first, second))
                with self.subTest(pad=pad, mutation=mutation):
                    self.assert_results_equal(*results)

    def test_leaf_inplace_rejection_and_saved_view_version_check(self):
        for fn in (candidate, reference):
            leaf = raw_complex().cuda().requires_grad_()
            with self.assertRaisesRegex(RuntimeError, "view of a leaf"):
                fn(leaf, 2).add_(1)
            base = leaf * 2
            out = fn(base, 2)
            loss = out.square().sum()
            with torch.no_grad():
                base.add_(1)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                torch.autograd.grad(loss, leaf)

    def test_primitive_does_not_save_input_values_unnecessarily(self):
        results = []
        for fn in (candidate, reference):
            leaf = raw_complex().cuda().requires_grad_()
            out = fn(leaf, 2)
            with torch.no_grad():
                leaf.add_(1)
            # View version changes rebuild the history; no input-value save is
            # needed for linear crop/real. This remains a legal backward.
            results.append(torch.autograd.grad(out, leaf, torch.ones_like(out))[0])
        self.assert_bytes_equal(*results, "linear VJP after leaf value mutation")

    def test_shared_ancestor_and_output_gradient_hooks(self):
        results = []
        for fn in (candidate, reference):
            leaf = raw_complex().cuda().requires_grad_()
            base = leaf * 2
            out = fn(base, 2)
            out.retain_grad()
            handle = out.register_hook(lambda g: g * .875)
            sibling = base.imag.sin().mean() * .03125
            loss = out.sin().square().mean() + fn(base, 2).square().mean() * .25 + sibling
            loss.backward()
            results.append((out.grad, leaf.grad))
            handle.remove()
        self.assert_results_equal(*results)

    def test_private_dtype_and_metadata_rejections(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex128):
            z = torch.ones(2, 3, 7, 9, dtype=dtype, device="cuda")
            with self.subTest(dtype=dtype), self.assertRaises(RuntimeError):
                candidate(z, 1)
        z = raw_complex().cuda()
        for pad in (-1, 4, 20):
            with self.subTest(pad=pad), self.assertRaises(RuntimeError):
                candidate(z, pad)
        for shape in ((2, 3, 0, 9), (0, 3, 7, 9), (3, 7, 9), (1, 2, 3, 7, 9)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                candidate(torch.empty(shape, dtype=torch.complex64, device="cuda"), 0)

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results = []
            for fn in (candidate, reference):
                z = raw_complex().cuda().requires_grad_()
                out = fn(z, 2)
                results.append((out, torch.autograd.grad(out, z, upstream(out.shape, "atoms"))[0]))
        torch.cuda.current_stream().wait_stream(stream)
        self.assert_results_equal(*results)


if __name__ == "__main__":
    unittest.main()
