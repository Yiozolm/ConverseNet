"""Independent FP64 validation of two shared-input, scale-one ATen candidates.

CPU mode checks small odd/even/degenerate shapes, both kernel broadcast axes,
all input/weight/bias requires-grad masks and directional second derivatives.
CUDA mode additionally reuses probe_training_s1_shapes.cases() verbatim (seven
normal and three weak fixtures), plus the frozen pretrained padded-module
fixture. CPU fixtures are separate evidence, not a substitute for CUDA gates.
No production build, timing, parameter changes or tolerance changes are made.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("transfer32", "transfer64_input64_round")
SOURCE_FILES = (
    "test/validate_shared_s1_transfer.py", "test/probe_shared_s1_transfer.py",
    "test/diagnose_boundary_precision.py", "test/probe_training_s1_shapes.py",
    "test/probe_pointwise_training.py", "test/test_fp32_training.py",
    "models/converse_core.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def cpu_cases():
    """New CPU-only coverage; never relabel these as the original CUDA draws."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(9214)
    result = []
    for h, w, kh, kw in ((1, 1, 1, 1), (1, 5, 1, 3), (4, 1, 2, 1),
                          (5, 6, 3, 3), (5, 7, 4, 2)):
        for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
            x = torch.randn(2, 3, w, h, generator=generator).transpose(-1, -2)
            kernel = torch.rand(kb, kc, kw, kh, generator=generator).transpose(-1, -2) / (kh * kw)
            bias = torch.randn(1, 3, 1, 1, generator=generator)
            upstream = torch.randn(x.shape, generator=generator) / x.numel() ** .5
            result.append(dict(name=f"cpu_{h}x{w}_k{kh}x{kw}_kb{kb}_kc{kc}",
                               tensors=(x, kernel, bias, upstream), eps=1e-3,
                               weak=False, padding=0, selective=True))
    # Match the established weak parameter recipe, with distinct CPU RNG draws.
    for amplitude in (0., 1e-6, 1e-3):
        x = torch.randn(2, 3, 4, 3, generator=generator).transpose(-1, -2) * 1e-5
        kernel = torch.rand(1, 1, 3, 3, generator=generator) / 9 * amplitude
        bias = torch.full((1, 3, 1, 1), -40.)
        upstream = torch.randn(x.shape, generator=generator) * 1e-5
        result.append(dict(name=f"cpu_weak_amplitude_{amplitude:g}",
                           tensors=(x, kernel, bias, upstream), eps=1e-8,
                           weak=True, padding=0, selective=True))
    # Exercise differentiable native circular pad/crop in a small module case.
    x = torch.randn(2, 3, 5, 6, generator=generator)
    kernel = torch.rand(1, 3, 3, 3, generator=generator) / 9
    bias = torch.randn(1, 3, 1, 1, generator=generator)
    upstream = torch.randn(x.shape, generator=generator) / x.numel() ** .5
    result.append(dict(name="cpu_module_pad2_crop2", tensors=(x, kernel, bias, upstream),
                       eps=1e-5, weak=False, padding=2, selective=True))
    return result


def capture(case, mode, device, needs, higher):
    import torch
    from probe_shared_s1_transfer import method
    dtype = torch.float64 if mode == "reference" else torch.float32
    values = tuple(value.detach().to(device=device, dtype=dtype).requires_grad_(need)
                   for value, need in zip(case["tensors"][:3], needs))
    output = method(mode, padding=case.get("padding", 0), eps=case["eps"])(*values)
    result = dict(output=output.detach().cpu())
    inputs = [value for value in values if value.requires_grad]
    labels = [name for name, need in zip(("dx", "dw", "db"), needs) if need]
    if inputs:
        upstream = case["tensors"][3].to(device=device, dtype=dtype)
        grads = torch.autograd.grad(output, inputs, upstream, create_graph=higher)
        result.update({name: value.detach().cpu() for name, value in zip(labels, grads)})
        if higher:
            # Directions are identical FP32-rounded CPU values for both paths.
            generator = torch.Generator(device="cpu").manual_seed(31917)
            vectors = [torch.randn(value.shape, generator=generator).to(device=device, dtype=dtype)
                       for value in grads]
            dot = sum((grad * vector).sum() for grad, vector in zip(grads, vectors))
            second = (torch.autograd.grad(dot, inputs, allow_unused=True)
                      if dot.requires_grad else (None,) * len(inputs))
            result.update({"hvp_" + name: None if value is None else value.detach().cpu()
                           for name, value in zip(labels, second)})
    if any(value.grad is not None for value in values):
        raise RuntimeError("autograd.grad unexpectedly accumulated a leaf gradient")
    if mode != "reference" and any(value is not None and value.dtype != torch.float32
                                    for value in result.values()):
        raise RuntimeError("Candidate output and all derivatives must remain FP32")
    return result


def compare(actual, expected, weak):
    from diagnose_boundary_precision import comparison
    if set(actual) != set(expected):
        raise RuntimeError("Candidate/reference derivative targets differ")
    checked = {}
    for name, reference in expected.items():
        value = actual[name]
        if value is None or reference is None:
            checked[name] = dict(passed=value is None and reference is None,
                                 actual_none=value is None, reference_none=reference is None)
            continue
        if name.startswith("hvp_"):
            # Unchanged directional second-derivative gate from FP32Training.compare.
            tolerances = (2e-3, 2e-4)
        elif name == "output":
            tolerances = (1e-6, 1e-5) if weak else (3e-5, 3e-5)
        else:
            tolerances = (5e-5, 5e-5)
        checked[name] = comparison(value, reference, *tolerances)
    return dict(passed=all(row["passed"] for row in checked.values()), tensors=checked)


def validate_case(case, device):
    from probe_pointwise_training import tensor_hash
    x, kernel, bias, _ = case["tensors"]
    selective = case.get("selective", False)
    masks = list(itertools.product((False, True), repeat=3)) if selective else [(True,) * 3]
    row = dict(name=case["name"], shape=list(x.shape), kernel_shape=list(kernel.shape),
               bias_shape=list(bias.shape), eps=case["eps"], weak=case["weak"],
               padding=case.get("padding", 0), fixture_sha256=tensor_hash(case["tensors"]),
               fixture_source=case.get("fixture_source", "new independent CPU coverage"),
               input_prior_identity="same tensor x", checks=[])
    for needs in masks:
        reference = capture(case, "reference", device, needs, higher=selective)
        checked = {mode: compare(capture(case, mode, device, needs, higher=selective),
                                 reference, case["weak"]) for mode in MODES}
        row["checks"].append(dict(requires_grad=list(needs), higher_order=selective,
                                   candidates=checked))
    row["passed"] = all(value["passed"] for check in row["checks"]
                         for value in check["candidates"].values())
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite previous evidence")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA validation requested without an available CUDA device")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    before = source_hashes()
    report = dict(status="running", scope=__doc__, source_sha256=before,
                  candidates=list(MODES), device=args.device, torch=str(torch.__version__),
                  deterministic_algorithms=True, tf32=False, production_eligible=False,
                  interpretation="Numerical coverage only; no timing, quality or dispatch claim.",
                  cases=[], cuda_initialized_before=torch.cuda.is_initialized())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        # CPU derivatives are always checked, including on a CUDA validation run.
        work = [(case, "cpu") for case in cpu_cases()]
        if args.device == "cuda":
            from probe_training_s1_shapes import cases
            for case in cases():
                case["fixture_source"] = "probe_training_s1_shapes.cases() verbatim; CUDA RNG weak fixtures"
                work.append((case, "cuda"))
            from diagnose_boundary_precision import fixture
            tensors = fixture()
            from train_usrnet_dataset import tensor_hash
            frozen = ROOT / "artifacts/native_deconv_target/shared_s1_transfer.json"
            evidence = json.loads(frozen.read_text(encoding="utf-8-sig"))
            actual_hash = tensor_hash(dict(zip(("x", "weight", "bias", "upstream"), tensors)))
            if actual_hash != evidence["fixture_sha256"]:
                raise RuntimeError("The frozen pretrained screen fixture has changed")
            report["pretrained_screen"] = dict(path=str(frozen), sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                                                fixture_sha256=actual_hash)
            work.append((dict(name="pretrained_module_pad2_crop2", tensors=tensors, eps=1e-5,
                              weak=False, padding=2, fixture_source="diagnose_boundary_precision.fixture() verbatim"), "cuda"))
        for case, device in work:
            row = validate_case(case, device)
            row["device"] = device
            report["cases"].append(row)
            save()
            print(json.dumps(dict(case=row["name"], device=device, passed=row["passed"])), flush=True)
        report["source_unchanged"] = source_hashes() == before
        report["cuda_initialized_after"] = torch.cuda.is_initialized()
        no_cuda_violation = args.device == "cpu" and report["cuda_initialized_after"]
        passed = all(row["passed"] for row in report["cases"]) and report["source_unchanged"] and not no_cuda_violation
        report["status"] = ("passed_cpu_only_cuda_unverified" if args.device == "cpu" else "passed_cpu_and_cuda") if passed else "numerical_gate_failed"
        report["checks"] = sum(len(row["checks"]) * len(MODES) for row in report["cases"])
        save()
        return 0 if passed else 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
