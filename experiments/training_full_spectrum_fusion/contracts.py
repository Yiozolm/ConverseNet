"""CUDA full-spectrum derivative contracts; separate from precision admission.

This suite compares the isolated spectral core to a differentiable ATen
reference.  It does not replace the spatial Python FP32 zero-margin comparison.
Existing output files are never overwritten by a new invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import traceback
from pathlib import Path

import torch

from loader import OUT, load_fusion


COMPARISON_TOLERANCES = {
    "torch.complex64": {"atol": 3e-5, "rtol": 3e-5},
    "torch.complex128": {"atol": 1e-11, "rtol": 1e-11},
}
# Numerical finite differences have a separate, fixed derivative-check budget.
# These are PyTorch's default gradcheck tolerances, not accuracy admission gates.
FINITE_DIFFERENCE = {"eps": 1e-6, "atol": 1e-5, "rtol": 1e-3}
NAMES = ("y", "p", "k", "lambda")


def alias_mean(value, scale):
    if scale == 1:
        return value
    b, c, hs, ws = value.shape
    return value.reshape(b, c, scale, hs // scale, scale, ws // scale).mean((2, 4))


def reference(y, p, k, regularizer, scale):
    power = k.real.square() + k.imag.square()
    q = (y - alias_mean(k * p, scale)) / (alias_mean(power, scale) + regularizer)
    if scale != 1:
        q = q.repeat(1, 1, scale, scale)
    return p + k.conj() * q


def fixture(scale, h, w, kb, kc, dtype, *, batch=2, channels=3, seed=93011):
    """Independent arbitrary complex spectra; no Hermitian assumption."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    y = torch.randn(batch, channels, h, w, dtype=dtype, generator=rng)
    p = torch.randn(batch, channels, h * scale, w * scale, dtype=dtype, generator=rng)
    k = torch.randn(kb, kc, h * scale, w * scale, dtype=dtype, generator=rng)
    regularizer = 0.25 + torch.rand(1, channels, 1, 1, dtype=real_dtype, generator=rng)
    upstream = torch.randn(p.shape, dtype=dtype, generator=rng)
    return (y, p, k, regularizer), upstream


def variables(raw, needs=None, *, views=False):
    needs = (True,) * len(raw) if needs is None else needs
    result = []
    for value, needed in zip(raw, needs):
        value = value.to("cuda")
        if views:
            shape = list(value.shape)
            shape[-1] *= 2
            backing = torch.empty(shape, dtype=value.dtype, device=value.device)
            view = backing[..., ::2]
            view.copy_(value)
            value = view.conj() if view.is_complex() else view
        result.append(value.detach().requires_grad_(needed))
    return result


def capture(call, raw, upstream, needs=None, *, views=False):
    needs = (True,) * len(raw) if needs is None else tuple(needs)
    data = variables(raw, needs, views=views)
    output = call(data)
    requested = [value for value, needed in zip(data, needs) if needed]
    grads = torch.autograd.grad(output, requested, upstream.to("cuda"))
    return [output.detach(), *(grad.detach() for grad in grads)]


def capture_hvp(call, raw, upstream, directions, needs):
    data = variables(raw, needs)
    requested = [value for value, needed in zip(data, needs) if needed]
    selected_directions = [value.to("cuda") for value, needed in zip(directions, needs) if needed]
    output = call(data)
    grads = torch.autograd.grad(output, requested, upstream.to("cuda"), create_graph=True)
    # When k/lambda are frozen, the solve is linear in y/p: its Hessian is zero.
    differentiable = [(grad, direction) for grad, direction in zip(grads, selected_directions)
                      if grad.requires_grad]
    if differentiable:
        hvp = torch.autograd.grad(
            [pair[0] for pair in differentiable], requested,
            [pair[1] for pair in differentiable], allow_unused=True,
        )
        hvp = [torch.zeros_like(value) if grad is None else grad
               for value, grad in zip(requested, hvp)]
    else:
        hvp = [torch.zeros_like(value) for value in requested]
    return [output.detach(), *(grad.detach() for grad in grads), *(grad.detach() for grad in hvp)]


class Recorder:
    def __init__(self, path):
        self.path = path
        self.report = {
            "status": "running", "purpose": "derivative and execution contracts only",
            "precision_gate_replacement": False,
            "comparison_tolerances": COMPARISON_TOLERANCES,
            "finite_difference_tolerances": FINITE_DIFFERENCE,
            "seed": 93011, "cases": [],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        # Exclusive creation also prevents concurrent processes from overwriting a run.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.report, stream, indent=2, allow_nan=False)

    def save(self):
        rows = self.report["cases"]
        self.report["summary"] = {
            "total_cases": len(rows),
            "passed_cases": sum(row["status"] == "passed" for row in rows),
            "failed_cases": sum(row["status"] == "failed" for row in rows),
            "tensor_checks": sum(len(row.get("tensors", [])) for row in rows),
        }
        self.path.write_text(json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8")

    def case(self, category, metadata, action):
        row = {"category": category, **metadata, "status": "running", "tensors": []}
        self.report["cases"].append(row)
        self.save()
        started = time.perf_counter()
        try:
            action(row)
            row["status"] = "passed"
        except BaseException:
            row["status"] = "failed"
            row["traceback"] = traceback.format_exc()
            raise
        finally:
            row["elapsed_seconds"] = time.perf_counter() - started
            self.save()
        if len(self.report["cases"]) % 20 == 0:
            print("CONTRACT_PROGRESS", self.report["summary"], flush=True)

    def compare(self, row, actual, expected, labels, dtype):
        assert len(actual) == len(expected) == len(labels)
        tolerance = COMPARISON_TOLERANCES[str(dtype)]
        for name, value, target in zip(labels, actual, expected):
            value = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
            target = target.detach().resolve_conj().resolve_neg().cpu().contiguous()
            finite = bool(torch.isfinite(value).all() and torch.isfinite(target).all())
            high_dtype = torch.complex128 if value.is_complex() else torch.float64
            difference = (value.to(high_dtype) - target.to(high_dtype)).flatten()
            target_high = target.to(high_dtype).flatten()
            metric = {
                "name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                "finite": finite,
                "max_abs": float(difference.abs().max()) if finite else None,
                "relative_l2": (float(torch.linalg.vector_norm(difference) /
                                      torch.linalg.vector_norm(target_high).clamp_min(1e-300))
                                if finite else None),
                "bitwise": torch.equal(value.view(torch.uint8), target.view(torch.uint8)),
                "atol": tolerance["atol"], "rtol": tolerance["rtol"],
            }
            row["tensors"].append(metric)
            assert finite, f"{name}: nonfinite output or reference"
            torch.testing.assert_close(value, target, **tolerance, msg=lambda msg: f"{name}: {msg}")


def run_suite(recorder, core):
    def fused(data, scale):
        return core.spectral(*data, scale)

    def baseline(data, scale):
        return reference(*data, scale)

    # All scales, four broadcast modes, odd/even dimensions and degenerate axes.
    for dtype, scale, (h, w), (kb, kc) in itertools.product(
        (torch.complex64, torch.complex128), (1, 2, 3, 4),
        ((1, 1), (1, 4), (3, 1), (3, 4), (4, 5)), ((1, 1), (1, 3), (2, 1), (2, 3)),
    ):
        raw, upstream = fixture(scale, h, w, kb, kc, dtype)
        spec = dict(dtype=str(dtype), scale=scale, h=h, w=w, kb=kb, kc=kc)
        def check(row):
            actual = capture(lambda data: fused(data, scale), raw, upstream)
            expected = capture(lambda data: baseline(data, scale), raw, upstream)
            recorder.compare(row, actual, expected, ("output", "dy", "dp", "dk", "dlambda"), dtype)
        recorder.case("arbitrary_complex_broadcast_boundaries", spec, check)

    # Every nonempty subset of the four formal input gradients, for each scale/dtype.
    for dtype, scale, mask in itertools.product((torch.complex64, torch.complex128), (1, 2, 3, 4), range(1, 16)):
        needs = tuple(bool(mask & (1 << index)) for index in range(4))
        raw, upstream = fixture(scale, 2, 3, 1, 1, dtype)
        labels = ["output", *("d" + name for name, needed in zip(NAMES, needs) if needed)]
        def check(row):
            actual = capture(lambda data: fused(data, scale), raw, upstream, needs)
            expected = capture(lambda data: baseline(data, scale), raw, upstream, needs)
            recorder.compare(row, actual, expected, labels, dtype)
        recorder.case("selective_gradients", dict(dtype=str(dtype), scale=scale, needs=dict(zip(NAMES, needs))), check)

    # The same tensor must occupy both formal y and p slots; test VJP and HVP.
    for dtype in (torch.complex64, torch.complex128):
        full, upstream = fixture(1, 3, 4, 1, 3, dtype)
        raw = (full[0], full[2], full[3])
        def shared(fn, data):
            return fn((data[0], data[0], data[1], data[2]), 1)
        def check(row):
            actual = capture(lambda data: shared(fused, data), raw, upstream)
            expected = capture(lambda data: shared(baseline, data), raw, upstream)
            recorder.compare(row, actual, expected, ("output", "dshared_y_p", "dk", "dlambda"), dtype)
        recorder.case("shared_y_p", dict(dtype=str(dtype), scale=1), check)
        directions = [torch.full_like(value, 0.125) for value in raw]
        def check_hvp(row):
            actual = capture_hvp(lambda data: shared(fused, data), raw, upstream, directions, (True,) * 3)
            expected = capture_hvp(lambda data: shared(baseline, data), raw, upstream, directions, (True,) * 3)
            recorder.compare(row, actual, expected,
                             ("output", "dshared_y_p", "dk", "dlambda", "hshared_y_p", "hk", "hlambda"), dtype)
        recorder.case("shared_y_p_hvp", dict(dtype=str(dtype), scale=1), check_hvp)

    for dtype, scale in itertools.product((torch.complex64, torch.complex128), (1, 2, 3, 4)):
        raw, upstream = fixture(scale, 3, 4, 1, 1, dtype)
        def check(row):
            probe = variables(raw, views=True)
            row["input_layouts"] = [dict(stride=list(value.stride()), contiguous=value.is_contiguous(),
                                         conjugated=value.is_conj()) for value in probe]
            assert all(not value.is_contiguous() for value in probe)
            assert all(value.is_conj() for value in probe[:3])
            actual = capture(lambda data: fused(data, scale), raw, upstream, views=True)
            expected = capture(lambda data: baseline(data, scale), raw, upstream, views=True)
            recorder.compare(row, actual, expected, ("output", "dy", "dp", "dk", "dlambda"), dtype)
        recorder.case("noncontiguous_conjugated_views", dict(dtype=str(dtype), scale=scale), check)

    for scale in (1, 2, 3, 4):
        raw, upstream = fixture(scale, 2, 3, 1, 3, torch.complex128)
        def check(row):
            default_stream = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(default_stream)
            with torch.cuda.stream(stream):
                actual = capture(lambda data: fused(data, scale), raw, upstream)
                expected = capture(lambda data: baseline(data, scale), raw, upstream)
                event = torch.cuda.Event()
                event.record(stream)
            default_stream.wait_event(event)
            recorder.compare(row, actual, expected, ("output", "dy", "dp", "dk", "dlambda"), torch.complex128)
            row["nondefault_stream"] = True
        recorder.case("nondefault_stream", dict(dtype=str(torch.complex128), scale=scale), check)

    for changed in range(4):
        raw, upstream = fixture(2, 2, 3, 1, 1, torch.complex64)
        def check(row):
            data = variables(raw)
            output = fused(data, 2)
            with torch.no_grad():
                data[changed].add_(0.125)
            try:
                torch.autograd.grad(output, data, upstream.to("cuda"))
            except RuntimeError as error:
                row["expected_error"] = str(error)
                assert "modified by an inplace operation" in str(error), str(error)
            else:
                raise AssertionError(f"mutation of {NAMES[changed]} did not raise a version error")
        recorder.case("saved_tensor_mutation", dict(scale=2, mutated=NAMES[changed]), check)

    # Retained graph must support repeated identical VJPs without stale buffers.
    for scale in (1, 3):
        raw, upstream = fixture(scale, 2, 3, 1, 1, torch.complex128)
        def check(row):
            data = variables(raw)
            output = fused(data, scale)
            first = torch.autograd.grad(output, data, upstream.to("cuda"), retain_graph=True)
            second = torch.autograd.grad(output, data, upstream.to("cuda"))
            recorder.compare(row, second, first, ("dy", "dp", "dk", "dlambda"), torch.complex128)
        recorder.case("retained_graph_repeat", dict(scale=scale), check)

    # Frozen-input HVPs exercise requested-input handling in the ATen fallback.
    for scale, needs in itertools.product((1, 2, 3, 4),
                                          ((True, True, True, True), (True, True, False, False),
                                           (False, False, True, True), (True, False, True, False))):
        raw, upstream = fixture(scale, 1, 2, 1, 1, torch.complex128, batch=1, channels=1)
        directions = [torch.full_like(value, 0.125 + (0.25j if value.is_complex() else 0)) for value in raw]
        selected = [name for name, needed in zip(NAMES, needs) if needed]
        labels = ["output", *("d" + name for name in selected), *("h" + name for name in selected)]
        def check(row):
            actual = capture_hvp(lambda data: fused(data, scale), raw, upstream, directions, needs)
            expected = capture_hvp(lambda data: baseline(data, scale), raw, upstream, directions, needs)
            recorder.compare(row, actual, expected, labels, torch.complex128)
        recorder.case("hvp_with_frozen_inputs", dict(scale=scale, needs=dict(zip(NAMES, needs))), check)

    # Small arbitrary complex128 inputs keep numerical Jacobian checks bounded.
    for scale in (1, 2, 3, 4):
        raw, _ = fixture(scale, 1, 1, 1, 1, torch.complex128, batch=1, channels=1)
        def check(row):
            data = tuple(variables(raw))
            fn = lambda *values: fused(values, scale)
            row["gradcheck"] = bool(torch.autograd.gradcheck(fn, data, **FINITE_DIFFERENCE))
            assert row["gradcheck"]
            row["gradgradcheck"] = bool(torch.autograd.gradgradcheck(fn, data, **FINITE_DIFFERENCE))
            assert row["gradgradcheck"]
        recorder.case("numerical_gradcheck_gradgradcheck", dict(scale=scale, dtype=str(torch.complex128)), check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="contracts.json", help="Fresh JSON filename inside loader.OUT")
    parser.add_argument("--adopt", action="store_true", help="Explicitly verify/load a research manifest")
    args = parser.parse_args()
    if Path(args.name).name != args.name or any(char in args.name for char in "\\/:") or not args.name.endswith(".json"):
        parser.error("--name must be a plain filename ending in .json")
    OUT.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(OUT / args.name)
    try:
        assert torch.cuda.is_available(), "CUDA is required; this suite does not silently fall back to CPU"
        torch.manual_seed(93011)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        recorder.report["environment"] = {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "build": json.loads((OUT / "build.json").read_text()),
            "tf32": False, "amp": False,
        }
        recorder.save()
        core = load_fusion(adopt=args.adopt)
        run_suite(recorder, core)
        torch.cuda.synchronize()
        recorder.report["status"] = "passed"
    except BaseException:
        recorder.report["status"] = "failed"
        recorder.report["traceback"] = traceback.format_exc()
        raise
    finally:
        recorder.save()
        print("CONTRACT_RESULTS", str(recorder.path), recorder.report["status"], recorder.report["summary"], flush=True)


if __name__ == "__main__":
    main()
