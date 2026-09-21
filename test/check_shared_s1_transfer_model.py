"""Original full-USRNet FP64 pressure fixture with isolated shared-s1 routes.

Keep test_full_usrnet_precision.py unchanged: seed9214, default full5/7,
alpha1/alpha2=.1, LR4x5/s2, the same upstream draw and atol3e-5/rtol3e-4
for output and every gradient. No pretrained checkpoint is substituted.

Only CUDA FP32/v7 calls with scale1 and x0 IS x use transfer_core. The real
call site is torch.ops.converse2d.forward; util_converse.converse2d_CUDA is
also patched for compatibility but is currently an unused historical alias.
The context restores both attributes. No model/production files are changed.

Run only when scheduled by the owner of the shared GPU:
    ./experiments/training_speed/run.ps1 test/check_shared_s1_transfer_model.py

All failed tensor coordinates are written to JSON; no early numerical assert
hides subsequent parameters. Any failed route, including the production
control, produces exit1. There is no timing or production-eligibility claim.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ATOL, RTOL = 3e-5, 3e-4
MODES = ("production", "transfer32", "transfer64_input64_round")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def route_context(mode):
    import torch
    from models import util_converse
    from probe_shared_s1_transfer import transfer_core

    original = torch.ops.converse2d.forward
    stats = dict(total_calls=0, eligible_shared_s1=0, transfer_calls=0,
                 original_calls=0, scales={}, transferred_kernel_shapes={})

    def dispatch(x, x0, weight, bias, scale, eps, variant="v7"):
        stats["total_calls"] += 1
        scale_key = str(scale)
        stats["scales"][scale_key] = stats["scales"].get(scale_key, 0) + 1
        eligible = (scale == 1 and x0 is x and x.is_cuda
                    and x.dtype == torch.float32 and variant == "v7")
        stats["eligible_shared_s1"] += int(eligible)
        if eligible and mode != "production":
            stats["transfer_calls"] += 1
            kernel_key = "x".join(str(value) for value in weight.shape[-2:])
            stats["transferred_kernel_shapes"][kernel_key] = stats["transferred_kernel_shapes"].get(kernel_key, 0) + 1
            return transfer_core(x, weight, bias, eps, mode)
        stats["original_calls"] += 1
        return original(x, x0, weight, bias, scale, eps, variant)

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.ops.converse2d, "forward", dispatch))
        stack.enter_context(patch.object(util_converse, "converse2d_CUDA", dispatch))
        yield stats


def verify_hits(stats, mode):
    if stats["total_calls"] != 40 or stats["eligible_shared_s1"] != 39 or stats["scales"] != {"2": 1, "1": 39}:
        raise RuntimeError(f"Unexpected full5/7 solver coverage: {stats}")
    expected = 0 if mode == "production" else 39
    if stats["transfer_calls"] != expected or stats["original_calls"] != 40 - expected:
        raise RuntimeError(f"Candidate silently missed or exceeded its guard: {stats}")
    if expected and stats["transferred_kernel_shapes"] != {"3x3": 35, "7x7": 4}:
        raise RuntimeError(f"Unexpected prior/DataNet shared-s1 coverage: {stats}")


def cpu_values(output, gradients, names):
    return {"output": output.detach().cpu().clone(),
            **{"gradient/" + name: value.detach().cpu().clone()
               for name, value in zip(names, gradients)}}


def compare_route(values, expected, baseline, stats):
    import torch
    from diagnose_boundary_precision import comparison

    if values.keys() != expected.keys():
        raise RuntimeError("The candidate omitted a reference tensor or parameter gradient")
    rows = {name: comparison(value, expected[name], ATOL, RTOL, max_failures=None)
            for name, value in values.items()}
    failures = {name: row for name, row in rows.items() if not row["passed"]}
    return dict(passed=not failures, tensor_count=len(rows), call_counts=stats,
                failed_tensor_count=len(failures),
                failed_elements=sum(row["failed_elements"] for row in rows.values()),
                failing_tensor_names=list(failures), tensors=rows,
                bitwise_equal_to_production={name: torch.equal(value, baseline[name])
                                             for name, value in values.items()})


def run(report, save):
    import torch
    from extension_loader import load_extension
    from fp32_training_baseline import current_manifest
    from train_usrnet_dataset import tensor_hash

    if os.environ.get("CONVERSE2D_BACKEND", "").strip():
        raise RuntimeError("Unset CONVERSE2D_BACKEND so the FP64 model's explicit pytorch backend is honored")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; there is no skipped or substitute run")
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    load_extension()  # Verify current sources and binary; do not compile here.
    from models.converse_usrnet import ConverseUSRNet

    report["source_sha256"] = current_manifest()
    for name in ("test/check_shared_s1_transfer_model.py", "test/test_full_usrnet_precision.py",
                 "test/probe_shared_s1_transfer.py", "test/diagnose_boundary_precision.py",
                 "models/converse_core.py"):
        report["source_sha256"][name] = sha256(ROOT / name)
    report["environment"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), cudnn_allow_tf32_inside_gate=False,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

    # Match the original test's construction/RNG order exactly, including the
    # FP64 deepcopy before the CUDA input draws. No extra model is initialized.
    torch.manual_seed(9214)
    model = ConverseUSRNet(backend="cuda").cuda().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("alpha1", "alpha2")):
                parameter.fill_(.1)
    reference_model = copy.deepcopy(model).double()
    for module in reference_model.modules():
        if hasattr(module, "backend"):
            module.backend = "pytorch"
    if getattr(model, "reuse_training_spectra", False):
        raise RuntimeError("The strict fixture keeps spectrum reuse disabled")
    x = (torch.rand(1, 3, 4, 5, device="cuda") * .2).requires_grad_()
    kernel = torch.rand(1, 1, 7, 7, device="cuda")
    kernel = (kernel / kernel.sum((-2, -1), keepdim=True)).requires_grad_()
    rx, rk = (value.detach().double().requires_grad_() for value in (x, kernel))
    params = list(model.named_parameters())
    if len(params) != 133 or model.num_iterations != 5 or len(model.p.m_body) != 7:
        raise RuntimeError("The original complete model architecture changed")
    names = ["input", "input_kernel", *[name for name, _ in params]]
    if [name for name, _ in params] != [name for name, _ in reference_model.named_parameters()]:
        raise RuntimeError("FP32/FP64 parameter order differs")

    with torch.backends.cudnn.flags(allow_tf32=False):
        with route_context("production") as baseline_stats:
            output = model(x, kernel, 2)
            expected_output = reference_model(rx, rk, 2)
            if not output.requires_grad:
                raise RuntimeError("The eval-mode strict fixture lost autograd")
            upstream = torch.randn_like(output) / output.numel() ** .5

            # The archived original fixture is immutable evidence. A different
            # seed/RNG order/alpha/input/upstream must fail before comparison.
            archived_path = ROOT / "artifacts/training_refinements/full_usrnet_failure_fixture.pt"
            archived = torch.load(archived_path, map_location="cpu", weights_only=True)
            actual_fixture = dict(x=x.detach().cpu(), kernel=kernel.detach().cpu(), upstream=upstream.detach().cpu())
            if not all(torch.equal(value, archived[name]) for name, value in actual_fixture.items()):
                raise RuntimeError("Input/kernel/upstream differs from the archived seed9214 fixture")
            if model.state_dict().keys() != archived["state_dict"].keys() or not all(
                    torch.equal(value.detach().cpu(), archived["state_dict"][name])
                    for name, value in model.state_dict().items()):
                raise RuntimeError("Model parameters/buffers differ from the archived pressure fixture")
            report["fixture"] = dict(archived_path=str(archived_path), archived_sha256=sha256(archived_path),
                archived_tensors_exact=True, tensor_sha256=tensor_hash(actual_fixture),
                initial_state_tensor_sha256=tensor_hash(model.state_dict()), parameter_tensors=len(params),
                parameter_elements=sum(value.numel() for _, value in params),
                input_shape=list(x.shape), input_kernel_shape=list(kernel.shape), output_shape=list(output.shape))
            baseline_grads = torch.autograd.grad(output, (x, kernel, *[value for _, value in params]), upstream)
        verify_hits(baseline_stats, "production")
        reference_grads = torch.autograd.grad(expected_output,
            (rx, rk, *reference_model.parameters()), upstream.double())
        baseline = cpu_values(output, baseline_grads, names)
        expected = cpu_values(expected_output, reference_grads, names)
        if not torch.count_nonzero(baseline_grads[1]).item():
            raise RuntimeError("The original input-kernel gradient is unexpectedly zero")
        del output, expected_output, baseline_grads, reference_grads, reference_model, rx, rk, archived
        report["routes"]["production"] = compare_route(baseline, expected, baseline, baseline_stats)
        save()

        for mode in MODES[1:]:
            xx, kk = (value.detach().clone().requires_grad_() for value in (x, kernel))
            with route_context(mode) as stats:
                actual = model(xx, kk, 2)
                gradients = torch.autograd.grad(actual, (xx, kk, *[value for _, value in params]), upstream)
            verify_hits(stats, mode)
            if not torch.count_nonzero(gradients[1]).item():
                raise RuntimeError(f"{mode} lost the input-kernel gradient")
            values = cpu_values(actual, gradients, names)
            report["routes"][mode] = compare_route(values, expected, baseline, stats)
            if any(value.grad is not None for value in (xx, kk, *[value for _, value in params])):
                raise RuntimeError("The diagnostic unexpectedly accumulated leaf .grad")
            del actual, gradients, xx, kk, values
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "artifacts/native_deconv_target/shared_s1_transfer_model.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", scope=__doc__, seed=9214, alpha=.1, scale=2,
        atol=ATOL, rtol=RTOL, routes={}, timing=False, production_eligible=False,
        oracle="Unchanged full-model pytorch/full-FFT FP64, same FP32-rounded initial state and input values",
        shared_s1_gate="scale1, x0 is x, CUDA FP32, variant v7; all other public calls use the original op")

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        run(report, save)
        report["candidates_all_passed"] = all(report["routes"][name]["passed"] for name in MODES[1:])
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
