"""CPU/stdlib-only audit of the two frozen nearest_spectral diagnostic reports."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {name: root / "artifacts/training_followup" / f"route_b_{name}.json"
             for name in ("cpu", "cuda")}
    old_path = root / "artifacts/fp32_roadmap/research_numeric.json"
    runner = root / "tools/training_followup/route_b_nearest.py"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    original = old["candidates"]["nearest_spectral"]["cases"][0]["tensors"]["output0"]
    errors, historical_comparison, stage_rows = [], {}, {}
    for device, report in reports.items():
        if report["status"] != "counterexample_reproduced" or report["candidate_admitted"]:
            errors.append(f"{device}: counterexample not reproduced")
        if report["script_sha256"] != digest(runner):
            errors.append(f"{device}: current runner differs from measured source")
        if not report["frozen_sources_unchanged"]:
            errors.append(f"{device}: frozen source changed")
        for name, case in report["cases"].items():
            if not all(case["instrumentation"].values()) or not all(case["single_variable_interventions"].values()):
                errors.append(f"{device}/{name}: instrumentation or intervention failed")
            if case["first_different_common_stage"] != "P" or case["output_gate"]["passed"]:
                errors.append(f"{device}/{name}: expected P divergence/output failure absent")
        observed = report["cases"]["historical_seed41191"]["output_gate"]
        historical_comparison[device] = {
            route: {metric: dict(original=original[route][metric], observed=observed[route][metric],
                                  exact_equal=original[route][metric] == observed[route][metric])
                    for metric in ("max_abs", "relative_l2")}
            for route in ("candidate", "python_fp32")}
        minimal = report["cases"]["minimal_identity_impulse"]
        stage_rows[device] = {
            name: dict(max_abs_vs_control=row["candidate_vs_python"]["max_abs"],
                       component_ulp_max=row["candidate_vs_python"]["component_ulp_max"],
                       different_components=row["candidate_vs_python"]["different_components"],
                       l2_vs_fp64=row["candidate_vs_fp64"]["l2"],
                       # The raw diagnostic uses a 1e-300 denominator floor.
                       # A zero-reference stage has no meaningful relative norm.
                       zero_reference_norm=name in ("numerator", "q", "q_tiled", "correction"))
            for name, row in minimal["stages"].items()}
    if not all(row["exact_equal"] for route in historical_comparison["cuda"].values() for row in route.values()):
        errors.append("GPU original case metrics do not exactly reproduce the old report")
    for device, report in reports.items():
        algorithm = next(item for path, item in report["old_source"].items() if path.endswith("algorithms.py"))
        if algorithm["sha256"] != old["source_sha256"]["algorithms.py"]:
            errors.append(f"{device}: algorithm hash differs from historical study")
    packets = {name: paths[name].with_name(paths[name].stem + "_inputs.json") for name in paths}
    if packets["cpu"].read_bytes() != packets["cuda"].read_bytes():
        errors.append("CPU and GPU input packets differ")
    result = dict(kind="single_route_b_diagnostic_integrity", status="passed" if not errors else "failed",
                  errors=errors, cpu_only=True, candidate_admitted=False,
                  source_sha256=digest(runner), summarizer_sha256=digest(Path(__file__)),
                  input_sha256={str(p.relative_to(root)): digest(p) for p in (*paths.values(), old_path, *packets.values())},
                  historical_candidate_status=old["candidates"]["nearest_spectral"]["status"],
                  historical_failures=old["candidates"]["nearest_spectral"]["failures"],
                  historical_tensor_count=old["candidates"]["nearest_spectral"]["tensor_count"],
                  historical_output_metrics=historical_comparison, minimal_stage_summary=stage_rows,
                  interpretation=["A passing audit proves faithful failure reproduction, not candidate accuracy.",
                                  "CPU and CUDA outputs need not be byte-identical; each uses its own independent FP64 reference.",
                                  "Minimality is only for nonempty B/C and spatial dimensions in the fixed k3/s2 contract.",
                                  "For mathematically zero-reference stages use absolute/L2 error, not denominator-floor relative ratios.",
                                  "One candidate, two diagnostic inputs. No VJP, performance, repair or production admission is implied."])
    if args.output.exists():
        raise FileExistsError("Keep old audit outputs; choose a fresh path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], errors=errors, output=str(args.output))))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
