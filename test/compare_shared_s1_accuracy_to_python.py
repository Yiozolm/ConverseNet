"""Measure accuracy relative to the original Python FP32 baseline's error.

    ./experiments/training_speed/run.ps1 test/compare_shared_s1_accuracy_to_python.py

All routes share the frozen seed9214/full5/7/alpha=.1/LR4x5/s2 fixture and the
same independent full-model FP64 diagnostic oracle. Compare Python FP32,
production FP32, transfer32 and the isolated shared-s1 CUDA candidate.

For every output/gradient tensor, report max_abs and relative-L2 error against
FP64, then the raw candidate/Python error ratios and which metric increased.
No ratio allowance is introduced. Also record the error of rounding the FP64
oracle to FP32: the minimum error from final FP32 representation alone. This
floor is recorded separately, never added to a budget or used to excuse a raw
increase. Prior atol3e-5/rtol3e-4 pointwise failures remain diagnostic only and
do not stop collection or set this program's exit status.

This measures numerical evidence, not training quality or production readiness.
The existing 0.05dB PSNR / .001 SSIM training-quality gates remain unchanged;
no training or timing is executed here. Old files and reports stay untouched.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_shared_s1_transfer_model as frozen
import check_shared_s1_python_fp32_model as fp32_audit

FROZEN_SHA256 = "35087d37aeaa7efb58b89a85bdc10644341e830b125b99e6b04ba281446320ce"
CANDIDATES = ("transfer32", "shared_cuda")


@contextmanager
def python_model_context():
    import torch
    from models import converse_usrnet

    original = converse_usrnet.ConverseUSRNet.forward
    with fp32_audit.python_oracle_context() as stats:
        stats["model_forwards"] = 0

        def forward(model, *args, **kwargs):
            if any(value.dtype != torch.float32 for value in model.parameters()):
                raise RuntimeError("The Python error baseline must use the same FP32 model")
            backends = [(module, module.backend) for module in model.modules() if hasattr(module, "backend")]
            try:
                for module, _ in backends:
                    module.backend = "pytorch"
                stats["model_forwards"] += 1
                return original(model, *args, **kwargs)
            finally:
                for module, backend in backends:
                    module.backend = backend

        with patch.object(converse_usrnet.ConverseUSRNet, "forward", forward):
            yield stats


def verify_python(stats):
    expected = dict(python_reference_calls=40, prior_calls=35, datanet_calls=5,
                    shared_s1_calls=39, scales={"2": 1, "1": 39}, native_solver_calls=0,
                    model_forwards=1, dtype="torch.float32")
    if any(stats[key] != value for key, value in expected.items()):
        raise RuntimeError(f"The original Python FP32 error baseline was not actually used: {stats}")


def raw_ratio(candidate, baseline):
    # Preserve zero-denominator information instead of inserting an epsilon.
    if baseline == 0:
        return "0/0" if candidate == 0 else "infinity"
    return candidate / baseline


def compare_errors(report):
    baseline = report["routes"]["python_fp32"]["tensors"]
    comparisons = {}
    for route, result in report["routes"].items():
        rows = {}
        for name, tensor in result["tensors"].items():
            reference = baseline[name]
            finite = tensor["finite"] and reference["finite"]
            if finite:
                abs_worse = tensor["max_abs"] > reference["max_abs"]
                l2_worse = tensor["relative_l2"] > reference["relative_l2"]
                rows[name] = dict(
                    candidate_max_abs=tensor["max_abs"], python_max_abs=reference["max_abs"],
                    max_abs_ratio=raw_ratio(tensor["max_abs"], reference["max_abs"]),
                    max_abs_increase=tensor["max_abs"] - reference["max_abs"],
                    candidate_relative_l2=tensor["relative_l2"], python_relative_l2=reference["relative_l2"],
                    relative_l2_ratio=raw_ratio(tensor["relative_l2"], reference["relative_l2"]),
                    relative_l2_increase=tensor["relative_l2"] - reference["relative_l2"],
                    max_abs_exceeds_python=abs_worse, relative_l2_exceeds_python=l2_worse,
                    raw_both_not_higher=not abs_worse and not l2_worse, finite=True,
                    final_fp32_representation_floor=report["fp32_representation_floor"][name])
            else:
                rows[name] = dict(finite=False, raw_both_not_higher=False,
                                  candidate_diagnostic=tensor, python_diagnostic=reference)
        worse = [name for name, row in rows.items() if not row["raw_both_not_higher"]]
        comparisons[route] = dict(tensor_count=len(rows), tensors=rows,
            raw_exceeding_tensor_count=len(worse), raw_exceeding_tensor_names=worse,
            raw_max_abs_exceeding_count=sum(row.get("max_abs_exceeds_python", False) for row in rows.values()),
            raw_relative_l2_exceeding_count=sum(row.get("relative_l2_exceeds_python", False) for row in rows.values()),
            raw_no_margin_comparison_passed=not worse,
            final_acceptance_assigned=False)
    report["accuracy_relative_to_python_fp32"] = comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=CANDIDATES,
                        help="Repeat to select candidates; production and Python FP32 controls always run")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_accuracy_relative_python.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    selected = tuple(dict.fromkeys(args.mode or CANDIDATES))
    report = dict(status="running", scope=__doc__, seed=9214, alpha=.1, scale=2,
        atol=frozen.ATOL, rtol=frozen.RTOL, routes={}, timing=False, production_eligible=False,
        selected_routes=["production", "python_fp32", *selected],
        diagnostic_oracle="One frozen-fixture full-model FP64 result for every route",
        acceptance_comparator="Original backend=pytorch full-spectrum FP32 model error, measured against that same FP64 oracle",
        legacy_pointwise_budget_role="Diagnostic only; not a release veto and not the exit-status condition",
        ratio_policy="Record raw ratios with no allowance or threshold relaxation; zero denominators remain explicit",
        rounding_floor_policy="Ideal nearest FP32 rounding of each FP64 reference tensor, reported separately without adding it to baseline errors",
        training_quality=dict(executed=False, psnr_max_degradation_db=.05, ssim_max_degradation=.001,
                              policy="Existing paired-training quality gates unchanged"),
        adapter_sha256=frozen.sha256(__file__), frozen_driver_sha256=FROZEN_SHA256,
        fp32_representation_floor={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        import torch
        if frozen.sha256(ROOT / "test/check_shared_s1_transfer_model.py") != FROZEN_SHA256:
            raise RuntimeError("The frozen fixture driver changed")
        shared_ops = None
        if "shared_cuda" in selected:
            from experiments.training_shared_s1.loader import load
            shared_ops, report["shared_cuda_build"] = load(verbose=args.verbose_build)
        original_context, original_verify, original_compare = frozen.route_context, frozen.verify_hits, frozen.compare_route

        @contextmanager
        def route_context(mode):
            if mode == "python_fp32":
                context = python_model_context()
            elif mode == "shared_cuda":
                context = fp32_audit.candidate_context(mode, shared_ops)
            else:
                # Call the captured original context directly: the helper's
                # production branch resolves frozen.route_context dynamically.
                context = original_context(mode)
            with context as stats:
                yield stats

        def verify(stats, mode):
            return verify_python(stats) if mode == "python_fp32" else original_verify(stats, mode)

        def compare(values, expected, baseline, stats):
            fp32_audit.require_fp32(values)
            if not report["fp32_representation_floor"]:
                for name, value in expected.items():
                    reference = value.detach().double()
                    error = (reference.float().double() - reference).abs()
                    report["fp32_representation_floor"][name] = dict(
                        max_abs=error.max().item(),
                        relative_l2=(error.norm() / reference.norm().clamp_min(1e-30)).item())
            result = original_compare(values, expected, baseline, stats)
            result["legacy_pointwise_diagnostic_passed"] = result.pop("passed")
            result["legacy_pointwise_release_gate"] = False
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen, "MODES", ("production", "python_fp32", *selected)))
            stack.enter_context(patch.object(frozen, "route_context", route_context))
            stack.enter_context(patch.object(frozen, "verify_hits", verify))
            stack.enter_context(patch.object(frozen, "compare_route", compare))
            frozen.run(report, save)
        for name in ("test/compare_shared_s1_accuracy_to_python.py", "test/check_shared_s1_python_fp32_model.py",
                     "test/train_usrnet_python_comparison.py"):
            report["source_sha256"][name] = frozen.sha256(ROOT / name)
        compare_errors(report)
        report["status"] = "complete_accuracy_measurement"
    except Exception as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error),
                                                  traceback=traceback.format_exc()))
    save()
    print(json.dumps(dict(status=report["status"], output=str(args.output),
        routes={name: dict(legacy_failed_tensors=row["failed_tensor_count"],
                           legacy_failed_elements=row["failed_elements"], call_counts=row["call_counts"],
                           raw_exceeding_tensors=report.get("accuracy_relative_to_python_fp32", {}).get(name, {}).get("raw_exceeding_tensor_count"))
                for name, row in report["routes"].items()}), indent=2))
    return 0 if report["status"] == "complete_accuracy_measurement" else 1


if __name__ == "__main__":
    raise SystemExit(main())
