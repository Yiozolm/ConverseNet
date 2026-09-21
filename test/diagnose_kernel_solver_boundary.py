"""Separate learned-kernel FP64 generation from the five DataNet solvers.

    ./experiments/training_speed/run.ps1 test/diagnose_kernel_solver_boundary.py

The original full-model fixture, FP64 oracle and all 136 pointwise budgets
come from the frozen driver. The already-validated FP64 KernelNet + five
FP64 kernel projections come from diagnose_kernel_generation_precision.

kernel64_cast32_public: cast each generated kernel to FP32 at DataNet entry,
then execute the original DataNet/public solver in FP32.
kernel64_prepare64_spectral32: keep the generated kernel and its gradient in
FP64, prepare its FFT in FP64 then cast the spectrum to contiguous complex64.
Use FP32 activation FFT, FP32 lambda, original private _training_spectral,
and FP32 output IFFT. s1 shares y/p identity; s2 retains spatial nearest prior.

Both routes retain original FP32 prior solvers, ordinary model layers and
parameter storage. All conversions remain differentiable. No cache, detached
training tensor, production edit, timing or optimization claim is introduced.
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
import diagnose_kernel_generation_precision as generation

FROZEN_SHA256 = "35087d37aeaa7efb58b89a85bdc10644341e830b125b99e6b04ba281446320ce"
GENERATION_SHA256 = "2d71227f269e16c6cb9ca5a2ef81ab196c77c72a2ff11ea368c5e127c752d117"
DIAGNOSTIC_MODES = ("kernel64_cast32_public", "kernel64_prepare64_spectral32")


@contextmanager
def boundary_context(mode):
    import torch
    import torch.nn.functional as F
    from models import converse_usrnet, util_converse
    from probe_nearest_training import prepare_kernel

    original_public = torch.ops.converse2d.forward
    original_data_forward = converse_usrnet.ConvReverseDataNet.forward
    private_spectral = torch.ops.converse2d._training_spectral
    handles = []
    # This frozen context changes only the actual FP32 model's kernel branch;
    # it also supplies dtype/coverage hooks around each DataNet invocation.
    with generation.kernel_precision_context("kernel_generation_fp64") as stats:
        stats.update(data_public_fp32_calls=0, data_private_fp32_calls=0,
                     data_kernel_float_casts=0, data_kernel_fp64_preparations=0,
                     shared_activation_fft_calls=0, generated_kernel_gradient_calls=0,
                     generated_kernel_gradient_dtypes={})

        def logical_data_call(x, prior, kernel, scale):
            if not x.is_cuda or x.dtype != torch.float32 or x.shape[1] != 64:
                raise RuntimeError("Unexpected DataNet activation dtype/geometry")
            if tuple(kernel.shape[-2:]) != (7, 7) or scale not in (1, 2):
                raise RuntimeError("Unexpected generated kernel geometry/scale")
            shared = prior is x
            if (scale == 1) != shared:
                raise RuntimeError("DataNet x/prior identity no longer matches the original fixture")
            stats["total_calls"] += 1
            stats["scales"][str(scale)] = stats["scales"].get(str(scale), 0) + 1
            stats["shared_s1_calls"] += int(shared)
            stats["datanet_shared_s1_calls"] += int(shared)

        def data_public(x, prior, weight, bias, scale, eps, variant="v7"):
            if variant != "v7" or any(value.dtype != torch.float32 for value in (x, prior, weight, bias)):
                raise RuntimeError("The cast32 control did not use the original FP32 public contract")
            logical_data_call(x, prior, weight, scale)
            stats["data_public_fp32_calls"] += 1
            return original_public(x, prior, weight, bias, scale, eps, variant)

        def data_forward(module, x, kernel, scale, padding=0, padding_mode="circular"):
            if x.dtype != torch.float32:
                # Never change a full-FP64 reference model if one is invoked.
                return original_data_forward(module, x, kernel, scale, padding, padding_mode)
            if kernel.dtype != torch.float64 or not kernel.requires_grad or module.alpha.dtype != torch.float32:
                raise RuntimeError("Expected differentiable FP64 generated kernel and FP32 DataNet alpha")

            def inspect_kernel_gradient(gradient):
                if gradient.dtype != torch.float64:
                    raise RuntimeError("The generated-kernel branch did not receive an FP64 gradient")
                stats["generated_kernel_gradient_calls"] += 1
                name = str(gradient.dtype)
                stats["generated_kernel_gradient_dtypes"][name] = stats["generated_kernel_gradient_dtypes"].get(name, 0) + 1
                # Returning None leaves the automatically-computed VJP intact.
            handles.append(kernel.register_hook(inspect_kernel_gradient))

            if mode == "kernel64_cast32_public":
                stats["data_kernel_float_casts"] += 1
                # Invoke original DataNet with its own dtype promotion/nearest
                # behavior. Only its nested public call bypasses the enclosing
                # generation context's expectation of an FP64 DataNet solver.
                with ExitStack() as local:
                    local.enter_context(patch.object(torch.ops.converse2d, "forward", data_public))
                    local.enter_context(patch.object(util_converse, "converse2d_CUDA", data_public))
                    return original_data_forward(module, x, kernel.float(), scale, padding, padding_mode)

            if mode != "kernel64_prepare64_spectral32" or module.variant != "v7":
                raise RuntimeError("Unknown mixed-precision diagnostic route")
            value = F.pad(x, (padding,) * 4, mode=padding_mode, value=0) if padding else x
            value = value.contiguous()
            height, width = value.shape[-2:]
            prior = value if scale == 1 else F.interpolate(value, scale_factor=scale, mode="nearest")
            logical_data_call(value, prior, kernel, scale)
            regularizer = torch.sigmoid(module.alpha.contiguous() - 9.) + module.eps
            # The existing helper starts with weight.double(), so an already
            # double generated kernel is never rounded before its FP64 FFT.
            spectrum = prepare_kernel(kernel, height * scale, width * scale)
            stats["data_kernel_fp64_preparations"] += 1
            y = torch.fft.rfft2(value)
            p = y if scale == 1 else torch.fft.rfft2(prior)
            if scale == 1:
                stats["shared_activation_fft_calls"] += 1
                if p is not y:
                    raise RuntimeError("The s1 route created a second activation FFT")
            if (y.dtype != torch.complex64 or p.dtype != torch.complex64
                    or spectrum.dtype != torch.complex64 or not spectrum.is_contiguous()
                    or regularizer.dtype != torch.float32):
                raise RuntimeError("The private solve inputs escaped the FP32/complex64 boundary")
            # Schema: (Tensor y, Tensor p, Tensor k, Tensor regularizer,
            #          int H, int W, int scale) -> Tensor.
            corrected = private_spectral(y, p, spectrum, regularizer, height, width, scale)
            stats["data_private_fp32_calls"] += 1
            if corrected.dtype != torch.complex64:
                raise RuntimeError("The private solver changed its precision")
            output = torch.fft.irfft2(corrected, s=(height * scale, width * scale))
            crop = padding * scale
            return output[..., crop:-crop, crop:-crop] if crop else output

        try:
            with patch.object(converse_usrnet.ConvReverseDataNet, "forward", data_forward):
                yield stats
        finally:
            for handle in handles:
                handle.remove()


def verify_coverage(stats, mode):
    private = mode == "kernel64_prepare64_spectral32"
    expected = dict(total_calls=40, shared_s1_calls=39, scales={"2": 1, "1": 39}, model_forwards=1,
        kernelnet_fp64_calls=1, kernel_projection_fp64_calls=[1] * 5,
        datanet_module_calls=5, datanet_module_fp32_returns=5, datanet_fp64_operator_calls=0,
        datanet_shared_s1_calls=4, prior_fp32_input_calls=35, prior_original_fp32_calls=35,
        prior_fp64_reference_calls=0, data_public_fp32_calls=0 if private else 5,
        data_private_fp32_calls=5 if private else 0, data_kernel_float_casts=0 if private else 5,
        data_kernel_fp64_preparations=5 if private else 0,
        shared_activation_fft_calls=4 if private else 0, generated_kernel_gradient_calls=5,
        generated_kernel_gradient_dtypes={"torch.float64": 5})
    if any(stats[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Kernel/solver boundary coverage mismatch: {stats}; expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=DIAGNOSTIC_MODES)
    parser.add_argument("--control-report", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_transfer_model.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/kernel_solver_boundary.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    selected = tuple(dict.fromkeys(args.mode or DIAGNOSTIC_MODES))
    report = dict(status="running", scope=__doc__, seed=9214, alpha=.1, scale=2,
        atol=frozen.ATOL, rtol=frozen.RTOL, routes={}, timing=False, diagnostic_only=True,
        optimization_candidate=False, production_eligible=False, selected_routes=["production", *selected],
        adapter_sha256=frozen.sha256(__file__), frozen_driver_sha256=FROZEN_SHA256,
        frozen_generation_adapter_sha256=GENERATION_SHA256,
        oracle="Frozen driver's unchanged full-model FP64 reference and original archived fixture",
        logical_call_count_note="total_calls includes five DataNet computations; private spectral calls are explicitly separate from public forward calls")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        frozen_files = {"test/check_shared_s1_transfer_model.py": FROZEN_SHA256,
                        "test/diagnose_kernel_generation_precision.py": GENERATION_SHA256}
        if any(frozen.sha256(ROOT / name) != digest for name, digest in frozen_files.items()):
            raise RuntimeError("A frozen fixture/generation adapter changed")
        previous = json.loads(args.control_report.read_text(encoding="utf-8"))
        if previous["source_sha256"]["test/check_shared_s1_transfer_model.py"] != FROZEN_SHA256:
            raise RuntimeError("The prior control report used another fixture driver")
        report["control_report"] = dict(path=str(args.control_report), sha256=frozen.sha256(args.control_report))
        original_context, original_verify = frozen.route_context, frozen.verify_hits

        @contextmanager
        def route_context(mode):
            context = original_context(mode) if mode == "production" else boundary_context(mode)
            with context as stats:
                yield stats

        def verify(stats, mode):
            return original_verify(stats, mode) if mode == "production" else verify_coverage(stats, mode)

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen, "MODES", ("production", *selected)))
            stack.enter_context(patch.object(frozen, "route_context", route_context))
            stack.enter_context(patch.object(frozen, "verify_hits", verify))
            frozen.run(report, save)
        for name in (*frozen_files, "test/diagnose_kernel_solver_boundary.py", "test/probe_nearest_training.py"):
            report["source_sha256"][name] = frozen.sha256(ROOT / name)
        report["fixture_matches_prior_control"] = report["fixture"] == previous["fixture"]
        if not report["fixture_matches_prior_control"] or any(
                frozen.sha256(ROOT / name) != digest for name, digest in frozen_files.items()):
            raise RuntimeError("The original fixture or frozen files changed")
        report["production_metrics_match_prior_control"] = (
            report["routes"]["production"]["tensors"] == previous["routes"]["production"]["tensors"])
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
