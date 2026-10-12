"""Predeclared Level 1 representation/FP32-compute experiment policy.

This is experiment-only. Quantization quality is diagnostic, not model approval.
Floors are inherited in magnitude from the FP32 release policy, never calibrated
against this experiment's results. No 2x extreme-case exception is enabled.
"""
import math

import torch

VERSION = "level1-quantized-reference-v1"
POLICY = {
    "version": VERSION,
    "normal_total_factor": 1.25,
    "weak_total_factor": 1.50,
    "kernel_extra_factor": 0.25,
    "rel_l2_floor": 1e-7,
    "max_abs_floor": 1e-6,
    "metrics": ["rel_l2", "max_abs"],
    "weak_definition": "Only bias=-40 with eps=1e-8; eps alone does not select weak",
    "relative_l2_normalization": "Norm of the reference argument; denominator clamped to 1e-300",
    "level1b": "Requires Level1A independently, then compares against the matching RQ output cast",
    "gradient_boundary": "RQ core FP32 VJPs are rounded to each physical input dtype for matching boundary comparison; that cast error is recorded separately",
    "level1b_upstream": "All Level1B VJP references receive the same upstream quantized to the output storage dtype",
    "extra_fp32_core_gate": "Actual FP32 core on quantized inputs is also checked against RQ64 using unchanged test/numerical_policy.py budgets, before gradient storage casting",
    "nonfinite": "Always rejected; representational overflow is separately labeled, never a pass",
    "model_approval": False,
}


def metrics(value, reference):
    if value.shape != reference.shape or value.is_complex() != reference.is_complex():
        raise ValueError("Identical shape and real/complex domains are required")
    dtype = torch.complex128 if reference.is_complex() else torch.float64
    value, reference = value.detach().to(dtype), reference.detach().to(dtype)
    if not bool(torch.isfinite(value).all() and torch.isfinite(reference).all()):
        return dict(finite=False, rel_l2=None, max_abs=None)
    delta = (value - reference).abs()
    result = dict(finite=True, rel_l2=(delta.norm() / reference.norm().clamp_min(1e-300)).item(),
                  max_abs=delta.max().item())
    if not all(math.isfinite(result[key]) for key in POLICY["metrics"]):
        return dict(finite=False, rel_l2=None, max_abs=None)
    return result


def ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    value = numerator / denominator if denominator else (1.0 if numerator == 0 else None)
    return value if value is None or math.isfinite(value) else None


def comparison(candidate, original32, quantized_reference, original64, *, regime="normal"):
    """All E_total/E_Q and E_kernel/E_Q limits apply independently per tensor."""
    if regime not in ("normal", "weak"):
        raise ValueError("Unknown regularization regime")
    if candidate.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        raise ValueError("Candidate may not use FP64 or an unlisted dtype")
    if original64.dtype not in (torch.float64, torch.complex128):
        raise ValueError("Original independent reference must be FP64/complex128")
    if original32.dtype not in (torch.float32, torch.complex64):
        raise ValueError("Original frozen baseline must be FP32/complex64")
    if quantized_reference.dtype not in (torch.float32, torch.complex64, candidate.dtype):
        raise ValueError("RQ must be FP32/complex64 or match the candidate boundary dtype")
    e32, eq = metrics(original32, original64), metrics(quantized_reference, original64)
    ek, et = metrics(candidate, quantized_reference), metrics(candidate, original64)
    finite = all(row["finite"] for row in (e32, eq, ek, et))
    total_factor = POLICY[f"{regime}_total_factor"]
    limits = {key: {
        "total": max(total_factor * eq[key], POLICY[f"{key}_floor"]) if eq["finite"] else None,
        "kernel_extra": max(POLICY["kernel_extra_factor"] * eq[key], POLICY[f"{key}_floor"]) if eq["finite"] else None,
    } for key in POLICY["metrics"]}
    passed = finite and all(et[key] <= limits[key]["total"] and ek[key] <= limits[key]["kernel_extra"]
                            for key in POLICY["metrics"])
    return dict(regime=regime, passed=bool(passed), finite=finite, original_fp32_error=e32,
                quantized_reference_error=eq, kernel_extra_error=ek, total_error=et, limits=limits,
                total_to_quantized_ratio={key: ratio(et[key], eq[key]) for key in POLICY["metrics"]},
                kernel_to_quantized_ratio={key: ratio(ek[key], eq[key]) for key in POLICY["metrics"]})


def output_cast_comparison(candidate_low, candidate_fp32, rq_fp32, original32, original64,
                           *, regime="normal", level1a_passed):
    if candidate_low.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Level1B output must be FP16 or BF16")
    rq_cast = rq_fp32.to(candidate_low.dtype)
    expected_candidate_cast = candidate_fp32.to(candidate_low.dtype)
    representation_overflow = bool(torch.isfinite(rq_fp32).all() and not torch.isfinite(rq_cast).all())
    check = comparison(candidate_low, original32, rq_cast, original64, regime=regime)
    cast_matches = bool(torch.equal(candidate_low, expected_candidate_cast))
    check.update(level1a_passed=bool(level1a_passed), rq_output_dtype=str(rq_cast.dtype),
                 reference_output_cast_error=metrics(rq_cast, rq_fp32),
                 candidate_output_cast_error=metrics(candidate_low, candidate_fp32),
                 candidate_matches_its_fp32_output_cast=cast_matches,
                 representation_overflow=representation_overflow)
    check["passed"] = bool(check["passed"] and level1a_passed and cast_matches and not representation_overflow)
    return check


def representation_status(original, represented):
    """A cast-induced nonfinite value is a range failure, never numerical admission."""
    before = bool(torch.isfinite(original).all())
    after = bool(torch.isfinite(represented).all())
    if not before:
        return "nonfinite_original"
    if not after:
        return "representation_overflow"
    return "finite"
