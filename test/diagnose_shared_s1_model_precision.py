"""Precision localization only; reuse the frozen full-USRNet pressure audit.

    ./experiments/training_speed/run.ps1 test/diagnose_shared_s1_model_precision.py

Default routes:
  production: unchanged FP32 control (its inherited failures stay visible).
  full64_internal: only 39 shared-input s1 calls use the existing transfer
      formula with FP64 FFT/transfer/product/IFFT, then return FP32.
  all_solver_fp64: all 40 Converse calls promote inputs/parameters to FP64,
      invoke the independent full-spectrum reference, then return FP32.

The frozen check_shared_s1_transfer_model.run owns model construction, RNG,
the archived fixture check, one full-model FP64 oracle, and all 136 output/
gradient comparisons at unchanged atol3e-5/rtol3e-4. No production or frozen
script edits, timing, optimization claim, model-wide dtype change or tolerance
relaxation occurs. Casts remain differentiable and shared x/prior identity is
preserved. The other model layers and gradient accumulation remain FP32.

Remaining error can include FP32 boundary rounding, other model operations,
and repeated-parameter accumulation; this experiment alone cannot identify a
particular Conv2d/LayerNorm bug. The two FP64 solver routes use mathematically
equivalent transfer/residual forms, not a bitwise-identical implementation.
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

FROZEN_SHA256 = "35087d37aeaa7efb58b89a85bdc10644341e830b125b99e6b04ba281446320ce"
DIAGNOSTIC_MODES = ("full64_internal", "all_solver_fp64")


@contextmanager
def all_solver_fp64_context():
    import torch
    from models import util_converse
    from models.converse_core import converse2d_reference

    stats = dict(total_calls=0, eligible_shared_s1=0, transfer_calls=0,
                 fp64_reference_calls=0, original_calls=0, scales={},
                 transferred_kernel_shapes={}, reference_kernel_shapes={})

    def dispatch(x, x0, weight, bias, scale, eps, variant="v7"):
        if not x.is_cuda or x.dtype != torch.float32 or variant != "v7":
            raise RuntimeError("The fixed model's solver route unexpectedly stopped being CUDA FP32/v7")
        stats["total_calls"] += 1
        stats["fp64_reference_calls"] += 1
        stats["scales"][str(scale)] = stats["scales"].get(str(scale), 0) + 1
        shared = x0 is x
        stats["eligible_shared_s1"] += int(scale == 1 and shared)
        kernel_key = "x".join(str(value) for value in weight.shape[-2:])
        stats["reference_kernel_shapes"][kernel_key] = stats["reference_kernel_shapes"].get(kernel_key, 0) + 1
        xx = x.double()
        prior = xx if shared else x0.double()
        output = converse2d_reference(xx, prior, weight.double(), bias.double(), scale, eps)
        return output.to(x.dtype)

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.ops.converse2d, "forward", dispatch))
        stack.enter_context(patch.object(util_converse, "converse2d_CUDA", dispatch))
        yield stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=DIAGNOSTIC_MODES,
                        help="Repeat to select diagnostics; unchanged production control always runs first")
    parser.add_argument("--control-report", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_transfer_model.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_model_precision.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    selected = tuple(dict.fromkeys(args.mode or DIAGNOSTIC_MODES))
    report = dict(status="running", scope=__doc__, seed=9214, alpha=.1, scale=2,
                  atol=frozen.ATOL, rtol=frozen.RTOL, routes={}, timing=False,
                  diagnostic_only=True, optimization_candidate=False, production_eligible=False,
                  selected_routes=["production", *selected],
                  adapter_sha256=frozen.sha256(__file__), frozen_driver_sha256=FROZEN_SHA256,
                  precision_boundary="Only solver internals change; model parameters, inter-layer activations and returned gradients remain FP32",
                  oracle="Same full-model FP64 reference and same promoted float32 upstream as the frozen audit",
                  positive_control_note="No duplicate full-model-FP64 route: the frozen driver already computes that exact oracle from the same state/input/upstream")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        frozen_path = ROOT / "test/check_shared_s1_transfer_model.py"
        if frozen.sha256(frozen_path) != FROZEN_SHA256:
            raise RuntimeError("The original model audit file changed; do not silently use a different fixture driver")
        prior_report = json.loads(args.control_report.read_text(encoding="utf-8"))
        if prior_report["source_sha256"]["test/check_shared_s1_transfer_model.py"] != FROZEN_SHA256:
            raise RuntimeError("The prior control report used a different fixture driver")
        report["control_report"] = dict(path=str(args.control_report), sha256=frozen.sha256(args.control_report))
        original_context, original_verify = frozen.route_context, frozen.verify_hits

        @contextmanager
        def selected_context(mode):
            context = all_solver_fp64_context() if mode == "all_solver_fp64" else original_context(mode)
            with context as stats:
                yield stats

        def verify(stats, mode):
            if mode != "all_solver_fp64":
                return original_verify(stats, mode)
            if (stats["total_calls"] != 40 or stats["eligible_shared_s1"] != 39
                    or stats["scales"] != {"2": 1, "1": 39}
                    or stats["fp64_reference_calls"] != 40 or stats["original_calls"] != 0
                    or stats["transfer_calls"] != 0
                    or stats["reference_kernel_shapes"] != {"7x7": 5, "3x3": 35}):
                raise RuntimeError(f"FP64 solver localization did not cover the exact full model: {stats}")

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen, "MODES", ("production", *selected)))
            stack.enter_context(patch.object(frozen, "route_context", selected_context))
            stack.enter_context(patch.object(frozen, "verify_hits", verify))
            frozen.run(report, save)

        report["source_sha256"]["test/diagnose_shared_s1_model_precision.py"] = frozen.sha256(__file__)
        report["fixture_matches_prior_control"] = report["fixture"] == prior_report["fixture"]
        if not report["fixture_matches_prior_control"]:
            raise RuntimeError("The new diagnostic did not reproduce the entire recorded fixture identity")
        report["production_metrics_match_prior_control"] = (
            report["routes"]["production"]["tensors"] == prior_report["routes"]["production"]["tensors"])
        if not report["production_metrics_match_prior_control"]:
            raise RuntimeError("The unchanged production control metrics differ from the frozen run; inspect environment before attributing errors")
        if frozen.sha256(frozen_path) != FROZEN_SHA256:
            raise RuntimeError("The frozen driver changed during the diagnostic")
        report["diagnostic_routes_all_passed"] = all(report["routes"][name]["passed"] for name in selected)
        report["all_routes_passed"] = all(row["passed"] for row in report["routes"].values())
        report["status"] = "passed" if report["all_routes_passed"] else "precision_gate_failed"
    except Exception as error:
        report.update(status="failed", all_routes_passed=False,
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    save()
    print(json.dumps(dict(status=report["status"], output=str(args.output),
        routes={name: {key: row[key] for key in ("passed", "failed_tensor_count", "failed_elements", "call_counts")}
                for name, row in report["routes"].items()}), indent=2))
    return 0 if report.get("all_routes_passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
