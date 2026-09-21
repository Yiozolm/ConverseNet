"""CPU-only paired quality audit: shared-s1 vs Python FP32 and matched current.

Reuse the original completed-run validator and unchanged per-seed RGB/Y gates:
PSNR drop <= .05dB, SSIM drop <= .001. Compare data/init/recipe/250 batch hashes,
133 gradients at every update, evaluation payloads, checkpoints and dispatch.
Parameter bitwise equality between trained methods is NOT required.

No PyTorch/GPU import, training, checkpoint mutation or historical speed ratio.
Default expects seeds17/29/43; --seeds 17 clearly yields single-seed evidence.
Exit 0 complete passing requested pairs, 2 missing, 3 invalid, 4 quality failed.
"""
import argparse
import datetime
import json
from pathlib import Path

import summarize_usrnet_training as shared
from summarize_python_training import validate_python_backend

ROOT = shared.ROOT
ADAPTER = "test/train_usrnet_shared_s1_candidate.py"
ADAPTER_SHA256 = "bfa56ccdbacd609a5a1a6f3196064b648d4d2d44fa22a85e17b4995a397912df"
SHARED_ID = "shared_s1_cuda_training_catpad_cropview_c128_96_v1"
IDENTITY_KEYS = ("comparison_recipe_sha256", "split_sha256", "input_checkpoint_sha256",
                 "initial_state_tensor_sha256", "manifest_sha256", "kernels_sha256",
                 "validation_payload_sha256", "shared_source_sha256")


def candidate_validator(candidate):
    def validate(run, require):
        expected_id = SHARED_ID if candidate == "shared_cuda" else "unchanged_current_control"
        adapter, proof = run["candidate_adapter"], run["candidate_verification"]
        require(run.get("comparison_candidate") == candidate and run.get("candidate_id") == expected_id,
                "Candidate label/ID does not match its requested slot")
        require(adapter.get("sha256") == run["source_sha256"].get(ADAPTER) == ADAPTER_SHA256,
                "Training adapter is not the frozen reviewed revision")
        require(shared.file_hash(ROOT / ADAPTER) == ADAPTER_SHA256, "Frozen training adapter changed on disk")
        for name, digest in {**adapter["shared_worker_files_sha256"], **adapter["helper_source_sha256"]}.items():
            require(shared.digest_valid(digest) and digest == run["source_sha256"].get(name),
                    f"Worker/helper identity mismatch: {name}")
        require(proof.get("candidate") == candidate and proof.get("candidate_id") == expected_id,
                "Dispatch counters belong to another candidate")
        require(adapter.get("production_eligible") is False, "Research adapter claimed production eligibility")
        if run.get("status") == "complete":
            expected = dict(solver_calls=22000, grad_solver_calls=10000, no_grad_solver_calls=12000,
                eligible_grad_solver_calls=9750, candidate_solver_calls=9750 if candidate == "shared_cuda" else 0,
                candidate_no_grad_solver_calls=0, original_solver_calls=12250 if candidate == "shared_cuda" else 22000,
                prior_module_calls=19250, boundary_geometry_calls=19250, eligible_grad_boundary_calls=8750,
                candidate_boundary_calls=8750 if candidate == "shared_cuda" else 0,
                candidate_no_grad_boundary_calls=0, model_forwards=550, grad_enabled_forwards=250,
                no_grad_forwards=300, first_training_graph_checked=True,
                first_training_graph_has_shared_transfer=candidate == "shared_cuda",
                first_training_graph_has_spectral_solve=True)
            for key, value in expected.items():
                require(proof.get(key) == value, f"Dispatch count mismatch: {key}={proof.get(key)!r}")
            frames = proof["forward_checks"]
            require(len(frames) == 550, "Per-forward coverage records are incomplete")
            expected_modes = [False] * 100 + [True] * 125 + [False] * 100 + [True] * 125 + [False] * 100
            for index, (frame, grad) in enumerate(zip(frames, expected_modes), 1):
                require(frame.get("index") == index and frame.get("grad_enabled") is grad
                        and frame.get("scale") == 3 and frame.get("solver_calls") == 40,
                        f"Invalid full-forward coverage at index {index}")
                require(frame.get("candidate_solver_calls") == (39 if candidate == "shared_cuda" and grad else 0)
                        and frame.get("boundary_calls") == (35 if candidate == "shared_cuda" and grad else 0),
                        f"Training/no_grad route escaped its guard at forward {index}")
        backend = run["backend"]
        if candidate == "shared_cuda":
            require(backend.get("candidate_id") == SHARED_ID and backend.get("no_grad_inference") == "unchanged production",
                    "Shared candidate did not retain original inference")
            build = backend["shared_training_build"]
            for name, digest in build["source_sha256"].items():
                require(run["source_sha256"].get("experiments/training_shared_s1/" + name) == digest,
                        f"Shared extension source mismatch: {name}")
            binary = Path(build["binary_path"])
            require(binary.is_file() and shared.file_hash(binary) == build["binary_sha256"],
                    "The recorded shared extension binary is missing or changed")
            production = backend["production_backend"]
        else:
            build = None
            production = backend
        require(production.get("kind") == "current production", "Unexpected no_grad production backend")
        sources = production["build_manifest"]["inputs"]["sources"]
        for name, digest in sources.items():
            require(run["source_sha256"].get("Converse2D/torch_converse2d/" + name) == digest,
                    f"Production source/build mismatch: {name}")
        return dict(candidate=candidate, adapter_sha256=adapter["sha256"],
                    helper_source_sha256=adapter["helper_source_sha256"], production_sources=sources,
                    shared_build=build, model_source_sha256={key: value for key, value in run["source_sha256"].items()
                                                           if key.startswith("models/")})
    return validate


def compare_pair(candidate, baseline, baseline_name):
    errors = []
    for key in IDENTITY_KEYS:
        if candidate["identity"][key] != baseline["identity"][key]:
            errors.append(f"Paired {key} differs")
    if candidate["batch_hashes"] != baseline["batch_hashes"]:
        errors.append("The 250 training tensor hashes differ")
    if candidate["effective_backend"]["model_source_sha256"] != baseline["effective_backend"]["model_source_sha256"]:
        errors.append("The model source differs")
    gates = {}
    for space in ("rgb", "y"):
        for metric, limit in (("psnr_db", shared.PSNR_DROP_DB), ("ssim", shared.SSIM_DROP)):
            delta = shared.metric_difference(shared.metric(candidate["quality"]["final"][space][metric]),
                                             shared.metric(baseline["quality"]["final"][space][metric]))
            gates[space + "_" + metric] = dict(candidate_minus_baseline=delta,
                                              minimum_allowed_delta=-limit, passed=delta >= -limit)
    result = dict(seed=candidate["seed"], baseline=baseline_name, candidate="shared_cuda",
        comparable=not errors, comparison_errors=errors, matched_batch_hashes=250,
        matched_validation_images=100, gates=gates,
        quality_passed=all(row["passed"] for row in gates.values()),
        parameter_bitwise_equality_required=False)
    if baseline_name == "matched_current":
        result["observed_sequential_run_time_ratio"] = {
            key: baseline["timing"][key] / candidate["timing"][key]
            for key in ("warm_step_wall_median_ms", "data_plus_step_total_s", "loop_including_evaluation_io_s")}
        result["timing_scope"] = "Descriptive fresh sequential pair, not a stable rotated benchmark or time-to-convergence claim"
        result["evaluation_routes"] = "Both final parameter sets evaluated with unchanged production no_grad solver/boundaries"
    else:
        result["evaluation_routes"] = "Historical Python-trained baseline evaluates with Python FP32; candidate evaluates with unchanged production no_grad. Candidate training-only routes are disabled for evaluation."
        result["timing_scope"] = "Quality-only historical Python comparison; no cross-report speed ratio"
    return result


def audit(args):
    runs, pairs, missing, errors = [], [], [], []
    for seed in args.seeds:
        slots = (
            ("shared_cuda", args.root / f"shared_cuda_seed{seed}", candidate_validator("shared_cuda")),
            ("matched_current", args.root / f"shared_control_seed{seed}", candidate_validator("current")),
            ("python_fp32", args.python_root / f"pytorch_seed{seed}", validate_python_backend),
        )
        values = {}
        for label, directory, validator in slots:
            result = shared.inspect_run(directory, seed, "current", backend_validator=validator)
            result["comparison_route"] = label
            if result["validated"]:
                training = shared.read_jsonl(directory / "training.jsonl")
                if any(row.get("gradient_tensor_count") != 133 for row in training):
                    result["errors"].append("Not all 133 parameter gradients were present at each update")
                    result["validated"] = False
            runs.append(result)
            values[label] = result
            missing.extend(result["missing"])
            errors.extend(f"{label}/seed{seed}: {message}" for message in result["errors"])
        if values["shared_cuda"]["validated"]:
            for name in ("python_fp32", "matched_current"):
                if values[name]["validated"]:
                    pair = compare_pair(values["shared_cuda"], values[name], name)
                    pairs.append(pair)
                    errors.extend(f"{name}/seed{seed}: {message}" for message in pair["comparison_errors"])
    valid = [run for run in runs if run["validated"]]
    for key in IDENTITY_KEYS[1:]:
        if len({shared.hash_json(run["identity"][key]) for run in valid}) > 1:
            errors.append(f"Cross-seed identity differs: {key}")
    for label in ("shared_cuda", "matched_current", "python_fp32"):
        if len({run["effective_backend_sha256"] for run in valid if run["comparison_route"] == label}) > 1:
            errors.append(f"Effective {label} source/build differs between seeds")
    failed_quality = any(not pair["quality_passed"] for pair in pairs)
    if errors:
        status, code = "invalid", 3
    elif failed_quality:
        status, code = "quality_failed", 4
    elif missing or len(pairs) != 2 * len(args.seeds):
        status, code = "incomplete", 2
    else:
        status, code = "passed_requested_seed_pairs", 0
    for run in runs:
        run.pop("batch_hashes", None)
    return dict(status=status, exit_code=code, requested_seeds=args.seeds, complete_pairs=len(pairs),
        required_pairs=2 * len(args.seeds), single_seed_evidence=len(args.seeds) == 1,
        production_eligible=False, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        missing=missing, integrity_errors=sorted(set(errors)), runs=runs, pairs=pairs,
        scope="CPU-only completed-run/data/init/checkpoint/dispatch/quality audit. No parameter-equality requirement, no GPU recomputation, no full convergence or global release conclusion.")


def seeds(value):
    result = [int(item) for item in value.split(",")]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Use unique comma-separated seeds")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/native_deconv_target")
    parser.add_argument("--python-root", type=Path, default=ROOT / "artifacts/python_training_comparison")
    parser.add_argument("--seeds", type=seeds, default=[17, 29, 43])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/shared_s1_quality_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new audit output; prior evidence is not overwritten")
    protected = [args.root / f"{prefix}_seed{seed}" for seed in args.seeds for prefix in ("shared_cuda", "shared_control")]
    protected += [args.python_root / f"pytorch_seed{seed}" for seed in args.seeds]
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in protected):
        parser.error("Write the audit outside the read-only run directories")
    report = audit(args)
    report["auditor_sha256"] = shared.file_hash(__file__)
    report["shared_validator_sha256"] = shared.file_hash(shared.__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(shared.clean(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], complete_pairs=report["complete_pairs"],
                         required_pairs=report["required_pairs"], integrity_errors=report["integrity_errors"],
                         missing=report["missing"], pairs=report["pairs"]), indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
