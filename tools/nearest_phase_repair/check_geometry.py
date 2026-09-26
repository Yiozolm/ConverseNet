"""Independent exact-arithmetic and CPU-bit checks of the narrow phase repair.

This is a test oracle, not a solver/backend. It never calls CUDA or FFT.
Only final axis coefficients at DC, fourth roots, or geometric zeros may change.
All other entries must preserve the legacy complex64 bytes, including zero signs.
"""
import argparse
import ast
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
INT64_MAX = 2**63 - 1
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_value(frequency, length, scale):
    """Fraction/period oracle independent of candidate's index enumeration.

    For r=exp(-2*pi*i*f/n), S=(1-r**s)/(1-r), except DC.
    r is a fourth root iff the reduced f/n denominator divides four;
    r**s=1 iff that reduced denominator divides s. No floating comparison.
    """
    if not (0 <= frequency < length and length > 0 and scale >= 2):
        raise ValueError("Invalid exact-oracle domain")
    fraction = Fraction(frequency, length)
    if fraction == 0:
        return (scale, 0)
    if 4 % fraction.denominator == 0:
        quarter = fraction.numerator * (4 // fraction.denominator)
        roots = ((1, 0), (0, -1), (-1, 0), (0, 1))
        # Full periods cancel exactly. Summing at most three Gaussian integers
        # avoids mirroring the candidate's per-quarter coefficient table.
        terms = [roots[(quarter * a) % 4] for a in range(scale % 4)]
        return tuple(sum(term[component] for term in terms) for component in (0, 1))
    if scale % fraction.denominator == 0:
        return (0, 0)
    return None


def canonical_bytes(value):
    """Every selected value is an exact Gaussian integer for s2/s3/s4."""
    real, imag = value
    data = struct.pack("<ff", float(real), float(imag))
    decoded = struct.unpack("<ff", data)
    if decoded != (real, imag):
        raise AssertionError("Oracle value not exactly representable in complex64")
    for component, number in enumerate(value):
        if number == 0 and data[4*component:4*component+4] != b"\x00" * 4:
            raise AssertionError("Special coefficient zero must be canonical +0")
    return data


def sparse_positions(length, scale):
    """All possible selected positions for valid n=s*LR, with unbounded ints."""
    if length % scale:
        raise ValueError("Checks target the unchanged n=s*LR caller contract")
    positions = {0}
    positions.update(length * turn // 4 for turn in (1, 2, 3) if length * turn % 4 == 0)
    positions.update((length // scale) * zero for zero in range(1, scale))
    return positions


def assert_mapping(module, length, scale, exhaustive):
    actual = module.exact_axis_coefficients(length, scale)
    positions = range(length) if exhaustive else sparse_positions(length, scale)
    expected = {f: exact_value(f, length, scale) for f in positions}
    expected = {f: value for f, value in expected.items() if value is not None}
    if set(actual) != set(expected):
        raise AssertionError(f"Selected coefficient set differs for n={length}, s={scale}")
    for f, expected_value in expected.items():
        if not 0 <= f < length <= INT64_MAX:
            raise AssertionError("Selected index outside the signed int64 tensor domain")
        value = actual[f]
        if struct.pack("<ff", value.real, value.imag) != canonical_bytes(expected_value):
            raise AssertionError(f"Exact value or signed-zero mismatch at n={length}, s={scale}, f={f}")
    # Test adjacent ordinary frequencies at very large dimensions without ever
    # allocating a tensor whose size resembles the theoretical boundary.
    if not exhaustive:
        probes = {0, length - 1, length // 3, length // 5}
        probes.update(j for f in expected for j in (f - 1, f, f + 1) if 0 <= j < length)
        for f in probes:
            value = exact_value(f, length, scale)
            if (value is None) != (f not in actual):
                raise AssertionError(f"Boundary/adjacent classification error n={length}, s={scale}, f={f}")
    return expected


def old_axis(torch, length, scale):
    """Literal old FP32 polar evaluation/order, CPU only; no repaired values."""
    f = torch.arange(length, device="cpu", dtype=torch.float32)
    z = torch.zeros(length, device="cpu", dtype=torch.complex64)
    for a in range(scale):
        angle = f * (-2 * math.pi * a / length)
        z = z + torch.polar(torch.ones_like(angle), angle)
    return z


def check_cpu_axis(torch, module, length, scale, expected):
    old = old_axis(torch, length, scale)
    source = old.clone()
    repaired = module.apply_exact_coefficients(source, length, scale)
    if repaired.shape != old.shape or repaired.dtype != torch.complex64 or repaired.device.type != "cpu":
        raise AssertionError("Repair changed the public axis representation")
    a = torch.view_as_real(repaired).contiguous().view(torch.int32)
    b = torch.view_as_real(old).contiguous().view(torch.int32)
    ordinary = torch.ones(length, dtype=torch.bool)
    ordinary[list(expected)] = False
    if not torch.equal(a[ordinary], b[ordinary]):
        raise AssertionError(f"An ordinary coefficient changed bytes, n={length}, s={scale}")
    for f, value in expected.items():
        bits = struct.pack("<ii", *a[f].tolist())
        if bits != canonical_bytes(value):
            raise AssertionError(f"Selected output differs in value/zero sign, n={length}, s={scale}, f={f}")
    return int(ordinary.sum()), int((a != b).any(dim=-1).sum())


def verify_narrow_source(candidate_path, frozen_path):
    def function(path, name):
        return next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    candidate = copy.deepcopy(function(candidate_path, "nearest_spectral"))
    old = function(frozen_path, "nearest_spectral")
    insertions = []

    class StripOnlyInsertion(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "apply_exact_coefficients"):
                insertions.append(ast.dump(node, include_attributes=False))
                return None
            return self.generic_visit(node)

    stripped = StripOnlyInsertion().visit(candidate)
    intended = ast.parse("z = apply_exact_coefficients(z, n, s)").body[0]
    if insertions != [ast.dump(intended, include_attributes=False)]:
        raise AssertionError("Expected exactly the one declared final-coefficient insertion")
    if ast.dump(stripped, include_attributes=False) != ast.dump(old, include_attributes=False):
        raise AssertionError("Candidate changed other FFT/solver/cache/phase-order source")
    return dict(only_final_axis_assignment_added=True, original_other_operations_ast_equal=True,
                frozen_helper_sha256=sha(frozen_path), candidate_sha256=sha(candidate_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).with_name("candidate.py"))
    parser.add_argument("--max-lr", type=int, default=129)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.max_lr <= 1024:
        parser.error("Use a bounded CPU geometry check, max-lr in [2,1024]")
    if args.output and args.output.exists():
        parser.error("Choose a fresh report; preserve earlier evidence")
    candidate_path = args.candidate.resolve()
    frozen = ROOT / "tools/training_followup/route_b_nearest.py"
    source = verify_narrow_source(candidate_path, frozen)
    spec = importlib.util.spec_from_file_location("checked_nearest_phase_candidate", candidate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU geometry check must not initialize CUDA")
    torch.set_num_threads(1)
    counts = dict(axes=0, frequencies=0, selected=0, ordinary_bytes_checked=0, actually_changed=0)
    with torch.no_grad():
        for scale in (2, 3, 4):
            for lr in range(1, args.max_lr + 1):
                length = scale * lr
                expected = assert_mapping(module, length, scale, True)
                ordinary, changed = check_cpu_axis(torch, module, length, scale, expected)
                counts["axes"] += 1
                counts["frequencies"] += length
                counts["selected"] += len(expected)
                counts["ordinary_bytes_checked"] += ordinary
                counts["actually_changed"] += changed
    fixtures = {
        (4, 2): {0: (2, 0), 1: (1, -1), 2: (0, 0), 3: (1, 1)},
        (6, 2): {0: (2, 0), 3: (0, 0)},
        (9, 3): {0: (3, 0), 3: (0, 0), 6: (0, 0)},
        (12, 3): {0: (3, 0), 3: (0, -1), 4: (0, 0), 6: (1, 0), 8: (0, 0), 9: (0, 1)},
        (8, 4): {0: (4, 0), 2: (0, 0), 4: (0, 0), 6: (0, 0)},
    }
    for (length, scale), expected in fixtures.items():
        if assert_mapping(module, length, scale, True) != expected:
            raise AssertionError("Named independent minimal/odd/ordinary fixture differs")
    boundaries = []
    for scale in (2, 3, 4):
        lrs = {2**23 - 1, 2**23, 2**23 + 1, 2**24 - 1, 2**24, 2**24 + 1,
               (2**31 - 1) // scale, (2**53 - 1) // scale, INT64_MAX // scale}
        for lr in sorted(lrs):
            length = scale * lr
            exact = assert_mapping(module, length, scale, False)
            boundaries.append(dict(scale=scale, lr=lr, n=length, selected_indices=sorted(exact),
                                   n_times_four_exceeds_int64=4 * length > INT64_MAX,
                                   f32_cannot_represent_every_frequency=length > 2**24))
    # This is only the helper's conservative claim boundary, not a request to
    # execute the old enormous scale loop or an expansion of solver admission.
    if module.exact_axis_coefficients(4, 2**24).get(0) != complex(2**24, 0):
        raise AssertionError("Exactly representable DC upper boundary missing")
    if 0 in module.exact_axis_coefficients(4, 2**24 + 1):
        raise AssertionError("Inexact DC was advertised as exact")
    if torch.cuda.is_initialized() or sha(candidate_path) != source["candidate_sha256"] or sha(frozen) != source["frozen_helper_sha256"]:
        raise RuntimeError("CUDA initialized or source changed during CPU check")
    report = dict(kind="independent_nearest_phase_geometry_check", status="passed", cpu_only=True,
                  cuda_initialized=False, source=source, checker_sha256=sha(__file__), torch=str(torch.__version__),
                  supported_test_scales=[2, 3, 4], lr_exhaustive_range=[1, args.max_lr], counts=counts,
                  theory="Fraction denominator divides4 => exact Gaussian-integer root sum; non-DC denominator divides scale => geometric zero.",
                  final_axis_scope_only=True, ordinary_complex64_bytes_preserved=True,
                  selected_signed_zeros_canonical_positive=True, minimal_n4_s2=fixtures[(4, 2)],
                  sparse_large_integer_checks=boundaries,
                  limits=["No CUDA, FFT, solver, VJP, cache lifecycle, performance or FP64-output admission is established here.",
                          "Large dimensions are sparse Python-integer checks only; no large tensor is allocated.",
                          "Canonical signed zeros are a declared coefficient representation, not a claim of original FFT byte equality.",
                          "The exact DC upper-bound helper check does not expand solver scale support beyond tested s2/s3/s4."])
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
