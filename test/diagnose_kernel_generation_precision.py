"""Localize full-model precision through the learned-kernel generation branch.

    ./experiments/training_speed/run.ps1 test/diagnose_kernel_generation_precision.py

Reuse the frozen full-model seed9214 fixture/oracle/all-gradient gate verbatim.
kernel_generation_fp64 evaluates KernelNet's three existing linear layers and
two GELUs in FP64, and evaluates the five existing kernel-projection Conv2d
modules in FP64. Generated kernels stay double. DataNet's existing promotion
then runs its five solvers in FP64 and returns to the original FP32 activation.
The 35 prior solver calls and all remaining model operations stay unchanged.

kernel_generation_and_prior_fp64 additionally evaluates each prior solver with
the independent FP64 reference and casts its output back to FP32. It does not
change the other prior layers, model parameters, or repeated-use accumulation.

All casts are differentiable and rebuilt on each call; there is no parameter
cache, custom VJP, detached training path, timing or optimization claim. The
FP64 reference model is not patched. Original scripts remain untouched.
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
DIAGNOSTIC_MODES = ("kernel_generation_fp64", "kernel_generation_and_prior_fp64")


@contextmanager
def kernel_precision_context(mode):
    import torch
    import torch.nn.functional as F
    from models import converse_usrnet, util_converse
    from models.converse_core import converse2d_reference

    original_op = torch.ops.converse2d.forward
    original_model_forward = converse_usrnet.ConverseUSRNet.forward
    prior64 = mode == "kernel_generation_and_prior_fp64"
    stats = dict(total_calls=0, shared_s1_calls=0, scales={}, model_forwards=0,
        kernelnet_fp64_calls=0, kernel_projection_fp64_calls=[0] * 5,
        datanet_module_calls=0, datanet_module_fp32_returns=0,
        datanet_fp64_operator_calls=0, datanet_shared_s1_calls=0,
        prior_fp32_input_calls=0, prior_original_fp32_calls=0, prior_fp64_reference_calls=0,
        fp64_reference_model_patched=False, model_parameters_remain_fp32=True)
    active_datanet = [0]

    def dispatch(x, x0, weight, bias, scale, eps, variant="v7"):
        if not x.is_cuda or variant != "v7":
            raise RuntimeError("Unexpected solver backend/variant in the fixed model")
        stats["total_calls"] += 1
        stats["scales"][str(scale)] = stats["scales"].get(str(scale), 0) + 1
        shared = x0 is x
        stats["shared_s1_calls"] += int(scale == 1 and shared)
        if active_datanet[0]:
            if active_datanet[0] != 1 or any(value.dtype != torch.float64 for value in (x, x0, weight, bias)):
                raise RuntimeError("DataNet did not promote its input/kernel/alpha to FP64")
            if x.shape[1] != 64 or tuple(weight.shape[-2:]) != (7, 7):
                raise RuntimeError("Unexpected DataNet geometry")
            stats["datanet_fp64_operator_calls"] += 1
            stats["datanet_shared_s1_calls"] += int(scale == 1 and shared)
            result = original_op(x, x0, weight, bias, scale, eps, variant)
            if result.dtype != torch.float64:
                raise RuntimeError("The existing FP64 DataNet operator changed its output dtype")
            return result

        if (x.dtype != torch.float32 or x0.dtype != torch.float32 or weight.dtype != torch.float32
                or bias.dtype != torch.float32 or scale != 1 or not shared
                or x.shape[1] != 128 or tuple(weight.shape[-2:]) != (3, 3)):
            raise RuntimeError("An unexpected operator was classified as a prior solver")
        stats["prior_fp32_input_calls"] += 1
        if prior64:
            stats["prior_fp64_reference_calls"] += 1
            xx = x.double()
            return converse2d_reference(xx, xx, weight.double(), bias.double(), scale, eps).to(x.dtype)
        stats["prior_original_fp32_calls"] += 1
        return original_op(x, x0, weight, bias, scale, eps, variant)

    def model_forward(model, *args, **kwargs):
        # Patch only the actual FP32 model, never the deepcopy used as oracle.
        if next(model.parameters()).dtype != torch.float32:
            return original_model_forward(model, *args, **kwargs)
        if model.num_iterations != 5 or len(model.convs) != 5 or model.kernelnet.kernel_size != 7:
            raise RuntimeError("The original full5/7 kernel-generation architecture changed")
        stats["model_forwards"] += 1
        kernelnet = model.kernelnet

        def kernel_forward(kernel):
            if kernel.dtype != torch.float32:
                raise RuntimeError("The fixed input blur kernel must enter KernelNet in FP32")
            batch = kernel.shape[0]
            value = kernel.double().reshape(batch, -1)
            for index, layer in enumerate((kernelnet.fc1, kernelnet.fc2, kernelnet.fc3)):
                value = F.linear(value, layer.weight.double(),
                                 None if layer.bias is None else layer.bias.double())
                if index < 2:
                    value = F.gelu(value, approximate=kernelnet.gelu.approximate)
            stats["kernelnet_fp64_calls"] += 1
            if value.dtype != torch.float64:
                raise RuntimeError("KernelNet lost FP64 precision")
            return value.view(batch, 16, kernelnet.kernel_size, kernelnet.kernel_size)

        def projection_forward(index, layer):
            if (layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
                    layer.padding, layer.dilation, layer.groups, layer.padding_mode) != (
                    16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, "zeros"):
                raise RuntimeError("The kernel projection is not the original 16->64 pointwise convolution")

            def forward(value):
                if value.dtype != torch.float64:
                    raise RuntimeError("The generated kernel features were rounded before projection")
                result = F.conv2d(value.double(), layer.weight.double(),
                                  None if layer.bias is None else layer.bias.double(),
                                  layer.stride, layer.padding, layer.dilation, layer.groups)
                stats["kernel_projection_fp64_calls"][index] += 1
                if result.dtype != torch.float64:
                    raise RuntimeError("A kernel projection lost FP64 precision")
                return result
            return forward

        def before_data(_module, inputs):
            activation, kernel, _scale = inputs[:3]
            if activation.dtype != torch.float32 or kernel.dtype != torch.float64:
                raise RuntimeError("Expected FP32 activation and retained FP64 generated kernel at DataNet entry")
            active_datanet[0] += 1
            stats["datanet_module_calls"] += 1

        def after_data(_module, _inputs, output):
            active_datanet[0] -= 1
            if output is not None:
                if output.dtype != torch.float32:
                    raise RuntimeError("DataNet failed to restore the original activation dtype")
                stats["datanet_module_fp32_returns"] += 1

        with ExitStack() as local:
            local.enter_context(patch.object(kernelnet, "forward", kernel_forward))
            for index, layer in enumerate(model.convs):
                local.enter_context(patch.object(layer, "forward", projection_forward(index, layer)))
            before = model.d.register_forward_pre_hook(before_data)
            after = model.d.register_forward_hook(after_data, always_call=True)
            local.callback(before.remove)
            local.callback(after.remove)
            result = original_model_forward(model, *args, **kwargs)
        if active_datanet[0] != 0 or any(value.dtype != torch.float32 for value in model.parameters()):
            raise RuntimeError("The precision context leaked state or changed parameter storage dtype")
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.ops.converse2d, "forward", dispatch))
        stack.enter_context(patch.object(util_converse, "converse2d_CUDA", dispatch))
        stack.enter_context(patch.object(converse_usrnet.ConverseUSRNet, "forward", model_forward))
        yield stats


def verify_coverage(stats, mode):
    expected = dict(total_calls=40, shared_s1_calls=39, scales={"2": 1, "1": 39},
        model_forwards=1, kernelnet_fp64_calls=1, kernel_projection_fp64_calls=[1] * 5,
        datanet_module_calls=5, datanet_module_fp32_returns=5,
        datanet_fp64_operator_calls=5, datanet_shared_s1_calls=4, prior_fp32_input_calls=35,
        prior_original_fp32_calls=0 if mode == "kernel_generation_and_prior_fp64" else 35,
        prior_fp64_reference_calls=35 if mode == "kernel_generation_and_prior_fp64" else 0)
    if any(stats[key] != value for key, value in expected.items()):
        raise RuntimeError(f"Kernel precision diagnostic coverage mismatch: {stats}; expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=DIAGNOSTIC_MODES)
    parser.add_argument("--control-report", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_transfer_model.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/kernel_generation_precision.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    selected = tuple(dict.fromkeys(args.mode or DIAGNOSTIC_MODES))
    report = dict(status="running", scope=__doc__, seed=9214, alpha=.1, scale=2,
        atol=frozen.ATOL, rtol=frozen.RTOL, routes={}, timing=False, diagnostic_only=True,
        optimization_candidate=False, production_eligible=False, selected_routes=["production", *selected],
        adapter_sha256=frozen.sha256(__file__), frozen_driver_sha256=FROZEN_SHA256,
        oracle="Frozen driver's unchanged full-model FP64 reference; same state/input/upstream",
        limitation="Kernel generation and DataNet precision change together; remaining error includes other FP32 layers, boundaries and gradient accumulation")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        frozen_path = ROOT / "test/check_shared_s1_transfer_model.py"
        if frozen.sha256(frozen_path) != FROZEN_SHA256:
            raise RuntimeError("The original fixture driver changed")
        previous = json.loads(args.control_report.read_text(encoding="utf-8"))
        if previous["source_sha256"]["test/check_shared_s1_transfer_model.py"] != FROZEN_SHA256:
            raise RuntimeError("The control report used another fixture driver")
        report["control_report"] = dict(path=str(args.control_report), sha256=frozen.sha256(args.control_report))
        original_context, original_verify = frozen.route_context, frozen.verify_hits

        @contextmanager
        def route_context(mode):
            context = original_context(mode) if mode == "production" else kernel_precision_context(mode)
            with context as stats:
                yield stats

        def verify(stats, mode):
            return original_verify(stats, mode) if mode == "production" else verify_coverage(stats, mode)

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen, "MODES", ("production", *selected)))
            stack.enter_context(patch.object(frozen, "route_context", route_context))
            stack.enter_context(patch.object(frozen, "verify_hits", verify))
            frozen.run(report, save)
        report["source_sha256"]["test/diagnose_kernel_generation_precision.py"] = frozen.sha256(__file__)
        report["fixture_matches_prior_control"] = report["fixture"] == previous["fixture"]
        if not report["fixture_matches_prior_control"] or frozen.sha256(frozen_path) != FROZEN_SHA256:
            raise RuntimeError("The archived fixture or frozen driver changed")
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
