"""Regression suite for closed-form semantics, all gradients and cache lifecycle.

python test/test_correctness.py --device cuda
Uses unittest (no pytest dependency), with an independent dense spatial solve.
"""
import argparse
import json
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension_loader import load_extension
from models.converse_core import converse2d_reference

VARIANTS = tuple(f"v{i}" for i in range(2, 8))
DEVICE = "cuda"
METRICS = {"forward_cases": 0, "gradient_cases": 0, "max_abs": {}}


def inputs(s=2, h=5, w=6, c=2, b=2, dtype=torch.float64, grad=False):
    x = torch.randn(b, c, h, w, device=DEVICE, dtype=dtype)
    prior = torch.randn(b, c, h*s, w*s, device=DEVICE, dtype=dtype)
    weight = torch.randn(1, c, 3, 3, device=DEVICE, dtype=dtype).flatten(2).softmax(-1).reshape(1, c, 3, 3)
    bias = torch.randn(1, c, 1, 1, device=DEVICE, dtype=dtype)
    return tuple(t.requires_grad_(grad) for t in (x, prior, weight, bias))


def op(args, s, variant="v7", eps=1e-5):
    return torch.ops.converse2d.forward(*args, s, eps, variant)


def dense_spatial(args, s, eps):
    """Solve (A^T A + lambda I)z = A^T y + lambda x0 without FFT."""
    x, prior, weight, bias = args
    b, c, h, w = x.shape
    hs, ws = h*s, w*s
    n = hs*ws
    basis = torch.eye(n, dtype=x.dtype, device=x.device).reshape(n, hs, ws)
    kh, kw = weight.shape[-2:]
    channels = []
    for channel in range(c):
        blurred = sum(weight[0, channel, i, j] * torch.roll(basis, (i-kh//2, j-kw//2), (-2,-1))
                      for i in range(kh) for j in range(kw))
        a = blurred[:, ::s, ::s].reshape(n, h*w).T
        lam = torch.sigmoid(bias[0, channel, 0, 0] - 9) + eps
        lhs = a.T @ a + lam * torch.eye(n, dtype=x.dtype, device=x.device)
        rhs = a.T @ x[:, channel].reshape(b, -1).T + lam * prior[:, channel].reshape(b, -1).T
        channels.append(torch.linalg.solve(lhs, rhs).T.reshape(b, hs, ws))
    return torch.stack(channels, 1)


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_extension()
        torch.manual_seed(921)

    def setUp(self):
        torch.ops.converse2d.clear_cache()

    def test_forward_all_variants(self):
        for dtype in (torch.float64, torch.float32):
            for h, w in ((5, 6), (6, 5), (5, 5), (6, 6), (1, 3), (3, 1), (1, 1)):
                for s in (1, 2, 3):
                    args = inputs(s, h, w, dtype=dtype)
                    if h*s < 3 or w*s < 3:
                        args = (*args[:2], args[2][..., :h*s, :w*s], args[3])
                    reference = converse2d_reference(*args, s)
                    for variant in VARIANTS:
                        with self.subTest(dtype=dtype, h=h, w=w, s=s, variant=variant), torch.no_grad():
                            out = op(args, s, variant)
                            tol = 2e-10 if dtype == torch.float64 else 3e-5
                            torch.testing.assert_close(out, reference, atol=tol, rtol=tol)
                            error = (out-reference).abs().max().item()
                            key = f"{dtype}/{variant}"
                            METRICS["max_abs"][key] = max(error, METRICS["max_abs"].get(key, 0))
                            METRICS["forward_cases"] += 1

    def test_dense_solution_and_four_gradients(self):
        for s in (1, 2, 3):
            args = inputs(s, h=3, w=4, c=1, b=1, grad=True)
            reference = dense_spatial(args, s, eps=1e-3)
            upstream = torch.randn_like(reference)
            expected_grads = torch.autograd.grad(reference, args, upstream)
            for variant in VARIANTS:
                with self.subTest(s=s, variant=variant):
                    out = op(args, s, variant, eps=1e-3)
                    actual_grads = torch.autograd.grad(out, args, upstream)
                    torch.testing.assert_close(out, reference, atol=1e-9, rtol=1e-9)
                    for actual, expected in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
                    METRICS["gradient_cases"] += 1

    def test_gradcheck_and_gradgradcheck(self):
        for variant, s in (("v6", 2), ("v7", 2), ("v7", 3)):
            args = inputs(s, h=2, w=3, c=1, b=1, grad=True)
            f = lambda *a: op(a, s, variant, eps=1e-2)
            self.assertTrue(torch.autograd.gradcheck(f, args, fast_mode=True, atol=2e-5, rtol=2e-4))
            self.assertTrue(torch.autograd.gradgradcheck(f, args, fast_mode=True, atol=2e-5, rtol=2e-4))

    def test_cache_mutation_and_training_transition(self):
        for variant in VARIANTS:
            args = inputs(2, grad=True)
            with torch.inference_mode():
                old = op(args, 2, variant).clone()
            with torch.no_grad():
                args[2].mul_(0.7)
                args[2].add_(0.03)
                new = op(args, 2, variant)
                expected = converse2d_reference(*args, 2)
                torch.testing.assert_close(new, expected, atol=1e-10, rtol=1e-10)
                self.assertGreater((new-old).abs().max().item(), 1e-3)
            for _ in range(2):
                out = op(args, 2, variant)
                reference = converse2d_reference(*args, 2)
                actual = torch.autograd.grad(out.square().mean(), args)
                expected = torch.autograd.grad(reference.square().mean(), args)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, atol=1e-9, rtol=1e-9)
                with torch.no_grad():
                    args[2].add_(0.001)

    def test_cache_tensor_identity_and_inference_tensors(self):
        # Same-shaped new weights must not reuse another tensor's cached spectrum.
        args = inputs(2)
        with torch.no_grad():
            op(args, 2)
            for _ in range(20):
                new = (*args[:2], torch.rand_like(args[2]), args[3])
                torch.testing.assert_close(op(new, 2), converse2d_reference(*new, 2), atol=1e-10, rtol=1e-10)
        with torch.inference_mode():
            args = inputs(2)
            op(args, 2)
            args[2].add_(0.2)
            torch.testing.assert_close(op(args, 2), converse2d_reference(*args, 2), atol=1e-10, rtol=1e-10)

    def test_noncontiguous_and_rectangular_kernel(self):
        args = inputs(3, h=5, w=7)
        args = tuple(t.transpose(-2, -1) for t in args)
        args = (*args[:2], args[2][..., :2, :], args[3])
        for variant in VARIANTS:
            with torch.no_grad():
                torch.testing.assert_close(op(args, 3, variant), converse2d_reference(*args, 3), atol=1e-10, rtol=1e-10)

    def test_generic_scale_and_storage_replacement(self):
        args = inputs(4, h=3, w=5)
        for variant in VARIANTS:
            with torch.no_grad():
                op(args, 4, variant)
                replacement = torch.rand_like(args[2])
                args[2].set_(replacement)
                torch.testing.assert_close(op(args, 4, variant), converse2d_reference(*args, 4), atol=1e-10, rtol=1e-10)

    def test_half_and_bfloat16_arbitrary_sizes(self):
        for dtype, tolerance in ((torch.float16, 0.008), (torch.bfloat16, 0.08)):
            for s in (1, 2, 3):
                args = inputs(s, h=5, w=7, dtype=dtype, grad=True)
                for variant in ("v2", "v6", "v7"):
                    expected = converse2d_reference(*(t.float() for t in args), s)
                    with torch.no_grad():
                        out = op(args, s, variant)
                    self.assertEqual(out.dtype, dtype)
                    torch.testing.assert_close(out.float(), expected, atol=tolerance, rtol=tolerance)
                    grads = torch.autograd.grad(op(args, s, variant).float().square().mean(), args)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_nondefault_cuda_stream(self):
        if DEVICE != "cuda":
            self.skipTest("CUDA stream test")
        args = inputs(3)
        with torch.no_grad():
            op(args, 3)
            default = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(default)
            with torch.cuda.stream(stream):
                output = op(args, 3)
                reference = converse2d_reference(*args, 3)
            default.wait_stream(stream)
            torch.testing.assert_close(output, reference, atol=1e-10, rtol=1e-10)

    def test_invalid_inputs(self):
        args = inputs(2)
        for bad_s, bad_eps in ((0,1e-5), (2,0), (2,-1), (2,float("nan"))):
            with self.assertRaises(RuntimeError):
                op(args, bad_s, eps=bad_eps)
        with self.assertRaises(RuntimeError):
            op((args[0], args[1][..., :-1], *args[2:]), 2)
        with self.assertRaises(RuntimeError):
            op((args[0], args[1], args[2].float(), args[3]), 2)

    def test_module_padding_and_shared_prior(self):
        from models.util_converse import Converse2D
        for s in (1, 2, 3):
            py = Converse2D(2,2,3,scale=s,padding=1,backend="pytorch").to(device=DEVICE,dtype=torch.float64)
            # Module CUDA backend is intentionally GPU-only; CPU uses direct operator.
            if DEVICE != "cuda":
                self.assertEqual(py(torch.randn(1,2,5,7,dtype=torch.float64)).shape, (1,2,5*s,7*s))
                continue
            cu = Converse2D(2,2,3,scale=s,padding=1,backend="cuda").to(device=DEVICE,dtype=torch.float64)
            cu.load_state_dict(py.state_dict())
            x = torch.randn(1,2,5,7,device=DEVICE,dtype=torch.float64,requires_grad=True)
            with torch.no_grad():
                torch.testing.assert_close(cu(x), py(x), atol=1e-10, rtol=1e-10)
            for module in (py, cu):
                out = module(x)
                grads = torch.autograd.grad(out.square().mean(), (x, module.weight, module.bias))
                if module is py:
                    expected = grads
                else:
                    for a, e in zip(grads, expected):
                        torch.testing.assert_close(a, e, atol=1e-9, rtol=1e-9)


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args, remaining = parser.parse_known_args()
    DEVICE = args.device
    result = unittest.main(module=__name__, argv=[sys.argv[0], *remaining], exit=False, verbosity=2).result
    METRICS.update(device=DEVICE, torch=torch.__version__, passed=result.wasSuccessful(),
                   tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))
    output = ROOT / "artifacts" / f"correctness_{DEVICE}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(METRICS, indent=2), encoding="utf-8")
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
