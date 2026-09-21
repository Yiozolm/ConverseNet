"""Extra contract checks for the isolated nearest training candidate.

Run only when the owner of the shared GPU schedules this script:
    ./experiments/training_speed/run.ps1 test/check_nearest_candidate_contract.py

This loads/builds the content-addressed candidate; it does not patch production.
Seven selective-gradient higher-order checks keep nondet_tol=0. A nondefault
stream forward/VJP is compared with the independent CPU ATen spectral reference,
and modifying each saved input must raise an autograd version error.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.training_nearest.loader import load_candidate
from probe_nearest_fused_training import spectral_reference
from probe_nearest_training import phase


SEED = 18371
HEIGHT, WIDTH, SCALE = 2, 4, 3
NAMES = ("y", "kernel", "regularizer", "phaseH", "phaseW")


def cpu_fixture():
    import torch

    rng = torch.Generator().manual_seed(SEED)
    y = torch.randn(1, 2, HEIGHT, WIDTH // 2 + 1, generator=rng, dtype=torch.complex128)
    kernel = torch.randn(1, 1, HEIGHT * SCALE, WIDTH * SCALE // 2 + 1,
                         generator=rng, dtype=torch.complex128)
    regularizer = torch.full((1, 2, 1, 1), .7, dtype=torch.float64)
    ph, pw = phase(HEIGHT * SCALE, SCALE, y), phase(WIDTH * SCALE, SCALE, y)
    upstream = torch.randn(1, 2, HEIGHT * SCALE, WIDTH * SCALE // 2 + 1,
                           generator=rng, dtype=torch.complex128)
    return (y, kernel, regularizer, ph, pw), upstream


def inputs_on(device, mask=(True, True, True)):
    values, upstream = cpu_fixture()
    values = tuple(value.to(device).clone().requires_grad_(mask[index] if index < 3 else False)
                   for index, value in enumerate(values))
    return values, upstream.to(device)


def call(ops, values):
    return ops._training_nearest_spectral(*values, HEIGHT, WIDTH, SCALE)


def higher_order_mask(ops, mask):
    import torch

    torch.manual_seed(SEED)
    values, _ = inputs_on("cuda", mask)
    ph, pw = values[3:]

    def forward(y, kernel, regularizer):
        return call(ops, (y, kernel, regularizer, ph, pw))

    passed = torch.autograd.gradgradcheck(
        forward, values[:3], eps=1e-6, atol=1e-5, rtol=1e-4,
        nondet_tol=0., fast_mode=True, raise_exception=True)
    if not passed:
        raise AssertionError("gradgradcheck returned False")
    return dict(gradgradcheck=True)


def comparison(actual, expected, atol, rtol):
    import torch

    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    error = (actual - expected).abs()
    budget = atol + rtol * expected.abs()
    result = dict(max_abs=error.max().item(),
                  relative_l2=(torch.linalg.vector_norm(error) /
                               torch.linalg.vector_norm(expected).clamp_min(1e-300)).item(),
                  max_budget_ratio=(error / budget).max().item(), atol=atol, rtol=rtol)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return result


def nondefault_stream(ops):
    import torch

    reference_values, reference_upstream = inputs_on("cpu")
    reference = spectral_reference(*reference_values, HEIGHT, WIDTH, SCALE)
    reference_grads = torch.autograd.grad(reference, reference_values[:3], reference_upstream)

    stream = torch.cuda.Stream()
    default_stream = torch.cuda.default_stream()
    if stream.cuda_stream == default_stream.cuda_stream:
        raise AssertionError("Expected a distinct nondefault CUDA stream")
    # Transfers, the entire custom forward/backward, and downstream consumers
    # all run on this stream. No default-stream-created CUDA fixture is reused.
    with torch.cuda.stream(stream):
        values, upstream = inputs_on("cuda")
        output = call(ops, values)
        gradients = torch.autograd.grad(output, values[:3], upstream)
        captured = tuple(value.detach().clone() for value in (output, *gradients))
    stream.synchronize()
    expected = (reference, *reference_grads)
    errors = {name: comparison(actual, ref, 1e-9 if index == 0 else 1e-8,
                               1e-9 if index == 0 else 1e-8)
              for index, (name, actual, ref) in enumerate(
                  zip(("output", "dy", "dk", "dlambda"), captured, expected))}
    return dict(stream_id=stream.cuda_stream, default_stream_id=default_stream.cuda_stream,
                reference_device="cpu", tensors=errors)


def saved_input_version(ops, index):
    import torch

    values, upstream = inputs_on("cuda")
    output = call(ops, values)
    old_version = values[index]._version
    with torch.no_grad():
        values[index].add_(.125)
    new_version = values[index]._version
    if new_version <= old_version:
        raise AssertionError("The in-place mutation did not increment the version")
    try:
        torch.autograd.grad(output, values[:3], upstream)
    except RuntimeError as error:
        message = str(error)
        if "modified by an inplace operation" not in message or "version" not in message:
            raise AssertionError(f"Unexpected backward error: {message}") from error
        return dict(version_before=old_version, version_after=new_version,
                    expected_error_type=type(error).__name__, expected_error=message)
    raise AssertionError(f"Backward accepted a modified saved {NAMES[index]}")


def checked(name, operation, **details):
    try:
        return dict(check=name, passed=True, **details, **operation())
    except Exception as error:
        return dict(check=name, passed=False, **details,
                    error=dict(type=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/training_research/nearest_candidate_contract.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", seed=SEED,
                  fixture=dict(batch=1, channels=2, height=HEIGHT, width=WIDTH, scale=SCALE,
                               kernel_batch=1, kernel_channels=1, dtype="complex128"),
                  gradgradcheck=dict(eps=1e-6, atol=1e-5, rtol=1e-4,
                                     nondet_tol=0., fast_mode=True),
                  source_sha256={name: hashlib.sha256((ROOT / "test" / name).read_bytes()).hexdigest()
                                 for name in (Path(__file__).name, "probe_nearest_fused_training.py",
                                              "probe_nearest_training.py")}, checks=[])

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; this is not a CPU-only execution mode")
        report["environment"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                     gpu=torch.cuda.get_device_name(),
                                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        ops, report["build"] = load_candidate(verbose=args.verbose_build)
        for mask in itertools.product((False, True), repeat=3):
            if any(mask):
                report["checks"].append(checked("selective_gradgradcheck",
                    lambda mask=mask: higher_order_mask(ops, mask), needs_grad=list(mask)))
                save()
        report["checks"].append(checked("nondefault_stream_forward_vjp", lambda: nondefault_stream(ops)))
        save()
        for index, name in enumerate(NAMES):
            report["checks"].append(checked("saved_input_version",
                lambda index=index: saved_input_version(ops, index), input=name))
            save()
        report["passed"] = len(report["checks"]) == 13 and all(row["passed"] for row in report["checks"])
        report["status"] = "passed" if report["passed"] else "contract_failed"
    except Exception as error:
        report.update(status="failed", passed=False,
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    save()
    print(json.dumps(dict(status=report["status"], output=str(args.output),
                          checks=len(report["checks"]),
                          failed_checks=[{key: row[key] for key in ("check", "needs_grad", "input") if key in row}
                                         for row in report["checks"] if not row["passed"]]), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
