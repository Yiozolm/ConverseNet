"""Full-model pointwise audit against the original Python FULL-SPECTRUM FP32 path.

    ./experiments/training_speed/run.ps1 test/check_shared_s1_python_fp32_model.py

The comparison oracle is backend=pytorch / converse2d_reference in FP32,
matching the previous Python training comparison. FP64 is NOT this comparison's
oracle. Keep the archived seed9214/full5/7/alpha=.1/LR4x5/s2 pressure fixture and
the existing atol3e-5/rtol3e-4 pointwise budget for all 136 output/gradient tensors.
Double conversion inside the reused error-statistics helper only measures the
difference between two FP32 results; it does not change either computation.

Compare production, transfer32, and the isolated analytic shared-s1 CUDA path.
Only scale1 with x0 IS x uses either candidate; the first scale2 solver retains
production. Every candidate forward must record 40 calls/39 shared-s1 hits.
The FP32 oracle must make exactly 40 Python calls and zero native solver calls.
No old FP64 report or frozen script is overwritten. No timing is performed.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_shared_s1_transfer_model as frozen

FROZEN_SHA256 = "35087d37aeaa7efb58b89a85bdc10644341e830b125b99e6b04ba281446320ce"
MODES = ("production", "transfer32", "shared_cuda")


@contextmanager
def python_oracle_context():
    import torch
    from models import converse_core, converse_usrnet, util_converse

    reference = converse_core.converse2d_reference
    stats = dict(python_reference_calls=0, prior_calls=0, datanet_calls=0,
                 shared_s1_calls=0, scales={}, native_solver_calls=0, dtype="torch.float32")

    def counted(origin):
        def forward(x, prior, weight, bias, scale=1, eps=1e-5):
            if any(value.dtype != torch.float32 or not value.is_cuda for value in (x, prior, weight, bias)):
                raise RuntimeError("The acceptance oracle must remain entirely CUDA FP32")
            stats["python_reference_calls"] += 1
            stats[origin] += 1
            stats["shared_s1_calls"] += int(scale == 1 and prior is x)
            stats["scales"][str(scale)] = stats["scales"].get(str(scale), 0) + 1
            output = reference(x, prior, weight, bias, scale, eps)
            if output.dtype != torch.float32:
                raise RuntimeError("The Python oracle changed output precision")
            return output
        return forward

    def reject_native(*_args, **_kwargs):
        stats["native_solver_calls"] += 1
        raise RuntimeError("The Python FP32 oracle attempted a native solver fallback")

    with ExitStack() as stack:
        stack.enter_context(patch.object(util_converse, "converse2d_reference", counted("prior_calls")))
        stack.enter_context(patch.object(converse_usrnet, "converse2d_reference", counted("datanet_calls")))
        stack.enter_context(patch.object(torch.ops.converse2d, "forward", reject_native))
        stack.enter_context(patch.object(util_converse, "converse2d_CUDA", reject_native))
        yield stats


@contextmanager
def candidate_context(mode, shared_ops):
    if mode in ("production", "transfer32"):
        with frozen.route_context(mode) as stats:
            yield stats
        return
    if mode != "shared_cuda" or shared_ops is None:
        raise RuntimeError("The requested shared CUDA candidate was not loaded")
    import torch
    from models import util_converse
    from diagnose_boundary_precision import kernel_fft64

    original = torch.ops.converse2d.forward
    stats = dict(total_calls=0, eligible_shared_s1=0, transfer_calls=0,
                 original_calls=0, scales={}, transferred_kernel_shapes={})

    def forward(x, prior, weight, bias, scale, eps, variant="v7"):
        stats["total_calls"] += 1
        stats["scales"][str(scale)] = stats["scales"].get(str(scale), 0) + 1
        eligible = scale == 1 and prior is x and x.is_cuda and x.dtype == torch.float32 and variant == "v7"
        stats["eligible_shared_s1"] += int(eligible)
        if not eligible:
            stats["original_calls"] += 1
            return original(x, prior, weight, bias, scale, eps, variant)
        value = x.contiguous()
        y = torch.fft.rfft2(value)
        kernel = kernel_fft64(weight, *value.shape[-2:]).cfloat().contiguous()
        regularizer = torch.sigmoid(bias - 9.) + eps
        spectrum = shared_ops.shared_s1_transfer(y, kernel, regularizer)
        stats["transfer_calls"] += 1
        key = "x".join(str(size) for size in weight.shape[-2:])
        stats["transferred_kernel_shapes"][key] = stats["transferred_kernel_shapes"].get(key, 0) + 1
        return torch.fft.irfft2(spectrum, s=value.shape[-2:])

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.ops.converse2d, "forward", forward))
        stack.enter_context(patch.object(util_converse, "converse2d_CUDA", forward))
        yield stats


def require_fp32(values):
    import torch
    if any(value.dtype != torch.float32 for value in values.values()):
        raise RuntimeError("A compared output/gradient is not FP32")


def run(args, report, save):
    import torch
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    from train_usrnet_dataset import tensor_hash

    if os.environ.get("CONVERSE2D_BACKEND", "").strip():
        raise RuntimeError("Unset CONVERSE2D_BACKEND so each model's explicit backend is honored")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; the oracle is Python CUDA FP32, not CPU or FP64")
    if frozen.sha256(ROOT / "test/check_shared_s1_transfer_model.py") != FROZEN_SHA256:
        raise RuntimeError("The frozen fixture/helper source changed")
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    load_extension()
    shared_ops = None
    if "shared_cuda" in report["selected_routes"]:
        from experiments.training_shared_s1.loader import load
        shared_ops, report["shared_cuda_build"] = load(verbose=args.verbose_build)
    from models.converse_usrnet import ConverseUSRNet

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report["environment"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), tf32=False, amp=False, cudnn_deterministic=True,
        cudnn_benchmark=False, deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    report["source_sha256"] = current_manifest()
    for name in ("test/check_shared_s1_python_fp32_model.py", "test/check_shared_s1_transfer_model.py",
                 "test/probe_shared_s1_transfer.py", "test/diagnose_boundary_precision.py",
                 "test/train_usrnet_python_comparison.py", "models/converse_core.py"):
        report["source_sha256"][name] = frozen.sha256(ROOT / name)

    # Same construction/RNG order as the archived pressure fixture. The only
    # intentional oracle change is keeping its copied model/inputs FP32.
    torch.manual_seed(9214)
    model = ConverseUSRNet(backend="cuda").cuda().eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("alpha1", "alpha2")):
                value.fill_(.1)
    reference_model = copy.deepcopy(model)
    for module in reference_model.modules():
        if hasattr(module, "backend"):
            module.backend = "pytorch"
    if any(value.dtype != torch.float32 for value in reference_model.parameters()):
        raise RuntimeError("The Python reference model must stay FP32")
    if getattr(model, "reuse_training_spectra", False) or getattr(reference_model, "reuse_training_spectra", False):
        raise RuntimeError("This comparison keeps optional spectrum reuse disabled")
    x = (torch.rand(1, 3, 4, 5, device="cuda") * .2).requires_grad_()
    kernel = torch.rand(1, 1, 7, 7, device="cuda")
    kernel = (kernel / kernel.sum((-2, -1), keepdim=True)).requires_grad_()
    rx, rk = (value.detach().clone().requires_grad_() for value in (x, kernel))
    parameters = list(model.named_parameters())
    names = ["input", "input_kernel", *[name for name, _ in parameters]]
    if (len(parameters) != 133 or model.num_iterations != 5 or len(model.p.m_body) != 7
            or [name for name, _ in parameters] != [name for name, _ in reference_model.named_parameters()]):
        raise RuntimeError("The frozen full5/7 architecture or parameter ordering changed")

    with candidate_context("production", shared_ops) as production_stats:
        output = model(x, kernel, 2)
    frozen.verify_hits(production_stats, "production")
    with python_oracle_context() as oracle_stats:
        expected_output = reference_model(rx, rk, 2)
    if (oracle_stats["python_reference_calls"] != 40 or oracle_stats["prior_calls"] != 35
            or oracle_stats["datanet_calls"] != 5 or oracle_stats["shared_s1_calls"] != 39
            or oracle_stats["scales"] != {"2": 1, "1": 39} or oracle_stats["native_solver_calls"] != 0):
        raise RuntimeError(f"The requested Python FP32 oracle was not actually used: {oracle_stats}")
    report["oracle_call_counts"] = oracle_stats
    upstream = torch.randn_like(output) / output.numel() ** .5
    archive_path = ROOT / "artifacts/training_refinements/full_usrnet_failure_fixture.pt"
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    fixture = dict(x=x.detach().cpu(), kernel=kernel.detach().cpu(), upstream=upstream.detach().cpu())
    if (model.state_dict().keys() != archive["state_dict"].keys()
            or not all(torch.equal(value, archive[name]) for name, value in fixture.items()) or not all(
            torch.equal(value.detach().cpu(), archive["state_dict"][name]) for name, value in model.state_dict().items())):
        raise RuntimeError("The original model/input/kernel/upstream fixture changed")
    report["fixture"] = dict(archived_path=str(archive_path), archived_sha256=frozen.sha256(archive_path),
        archived_tensors_exact=True, tensor_sha256=tensor_hash(fixture),
        initial_state_tensor_sha256=tensor_hash(model.state_dict()), parameter_tensors=133,
        parameter_elements=sum(value.numel() for _, value in parameters), output_shape=list(output.shape))
    gradients = torch.autograd.grad(output, (x, kernel, *[value for _, value in parameters]), upstream)
    reference_gradients = torch.autograd.grad(expected_output, (rx, rk, *reference_model.parameters()), upstream)
    if not torch.count_nonzero(gradients[1]).item() or not torch.count_nonzero(reference_gradients[1]).item():
        raise RuntimeError("A compared model lost the original nonzero input-kernel gradient")
    baseline = frozen.cpu_values(output, gradients, names)
    expected = frozen.cpu_values(expected_output, reference_gradients, names)
    require_fp32(baseline)
    require_fp32(expected)
    report["oracle_tensor_sha256"] = tensor_hash(expected)
    report["routes"]["production"] = frozen.compare_route(baseline, expected, baseline, production_stats)
    del output, gradients, expected_output, reference_gradients, reference_model, rx, rk, archive
    save()

    for mode in report["selected_routes"][1:]:
        xx, kk = (value.detach().clone().requires_grad_() for value in (x, kernel))
        with candidate_context(mode, shared_ops) as stats:
            output = model(xx, kk, 2)
            gradients = torch.autograd.grad(output, (xx, kk, *[value for _, value in parameters]), upstream)
        frozen.verify_hits(stats, mode)
        if not torch.count_nonzero(gradients[1]).item():
            raise RuntimeError(f"{mode} lost the original nonzero input-kernel gradient")
        values = frozen.cpu_values(output, gradients, names)
        require_fp32(values)
        report["routes"][mode] = frozen.compare_route(values, expected, baseline, stats)
        if any(value.grad is not None for value in (xx, kk, *[value for _, value in parameters])):
            raise RuntimeError("The audit accumulated leaf gradients")
        del output, gradients, values, xx, kk
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="Repeat to select candidates; unchanged production control always runs")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_python_fp32_model.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", scope=__doc__, seed=9214, alpha=.1, scale=2,
        atol=frozen.ATOL, rtol=frozen.RTOL, routes={}, timing=False, production_eligible=False,
        selected_routes=list(dict.fromkeys(("production", *(args.mode or MODES)))),
        oracle="Original backend=pytorch full-spectrum FP32 model; FP64 is not an acceptance gate",
        oracle_dtype="torch.float32", metrics_accumulation_dtype="torch.float64",
        fixture_driver_sha256=FROZEN_SHA256, old_fp64_records_preserved=True,
        pointwise_budget_scope="Previous tolerances retained for direct alignment measurement; passing does not establish training-quality equivalence or production eligibility")

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        run(args, report, save)
        report["all_routes_passed"] = all(row["passed"] for row in report["routes"].values())
        report["status"] = "passed" if report["all_routes_passed"] else "python_fp32_gate_failed"
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
