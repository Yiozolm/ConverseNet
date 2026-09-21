"""CPU-only mixed-shared training quality audit against Python and fresh current.

Reuse the original completed-run/data/recipe/checkpoint/metric validator and
unchanged per-seed RGB/Y PSNR .05dB / SSIM .001 gates. Validate all 550 forward
records, FP64 generated-kernel VJPs, FP32 parameter storage/final gradients and
250 pre-Adam checks. No parameter bitwise equality is required between methods.

Default requires seeds17/29/43 and BOTH baselines. --python-only can record an
initial quality comparison while a matched current control is still running;
that report explicitly does not claim completion of the unrequested comparison.
No GPU/PyTorch import, training or modification of run records occurs.
"""
import argparse
import datetime
import json
from pathlib import Path

import summarize_usrnet_training as shared
import summarize_shared_s1_training as earlier
from summarize_python_training import validate_python_backend

ROOT = shared.ROOT
ADAPTER = "test/train_usrnet_mixed_shared_candidate.py"
ADAPTER_SHA256 = "45735b92b57b91cce9e39467e38cd2bf9c87498432bc49da03c6ccf6046101f9"
MIXED_ID = "kernel64_mixeddata_sharedprior_training_boundary_v1"


def candidate_validator(candidate):
    def validate(run, require):
        expected_id = MIXED_ID if candidate == "mixed_shared" else "unchanged_current_control"
        adapter, proof = run["candidate_adapter"], run["candidate_verification"]
        require(run.get("comparison_candidate") == candidate and run.get("candidate_id") == expected_id,
                "Candidate label/ID does not match the requested slot")
        require(adapter.get("sha256") == run["source_sha256"].get(ADAPTER) == ADAPTER_SHA256
                and shared.file_hash(ROOT / ADAPTER) == ADAPTER_SHA256,
                "The training adapter is not the frozen reviewed version")
        for name, digest in {**adapter["shared_worker_files_sha256"], **adapter["helper_source_sha256"]}.items():
            require(shared.digest_valid(digest) and digest == run["source_sha256"].get(name),
                    f"Worker/helper identity mismatch: {name}")
        require(adapter.get("production_eligible") is False, "Research adapter claimed production readiness")
        require(proof.get("candidate") == candidate and proof.get("candidate_id") == expected_id,
                "Counters describe a different candidate")
        if run.get("status") == "complete":
            expected = dict(model_forwards=550, grad_enabled_forwards=250, no_grad_forwards=300,
                mixed_grad_forwards=250 if candidate == "mixed_shared" else 0,
                optimizer_gradient_dtype_checks=250, parameter_tensor_count=133,
                first_training_graph_checked=True, first_training_graph_has_shared_transfer=candidate == "mixed_shared",
                first_training_graph_has_spectral_solve=True)
            for key, value in expected.items():
                require(proof.get(key) == value, f"Aggregate route/dtype count mismatch: {key}")
            frames = proof["forward_checks"]
            require(len(frames) == 550, "Per-forward coverage records are incomplete")
            modes = [False] * 100 + [True] * 125 + [False] * 100 + [True] * 125 + [False] * 100
            for index, (frame, grad) in enumerate(zip(frames, modes), 1):
                mixed = candidate == "mixed_shared" and grad
                expected_frame = dict(index=index, grad_enabled=grad, model_training=grad,
                    candidate_active=mixed, scale=3, public_solver_calls=35 if mixed else 40,
                    original_public_calls=0 if mixed else 40, mixed_data_calls=5 if mixed else 0,
                    shared_prior_calls=35 if mixed else 0, private_spectral_calls=5 if mixed else 0,
                    logical_solver_calls=40, parameter_storage_dtype="torch.float32", prior_module_calls=35,
                    boundary_calls=35 if mixed else 0, kernelnet_fp64_calls=1 if mixed else 0,
                    projection_fp64_calls=[1 if mixed else 0] * 5,
                    generated_kernel_gradient_calls=5 if mixed else 0,
                    generated_kernel_gradient_dtypes={"torch.float64": 5} if mixed else {},
                    shared_data_fft_calls=4 if mixed else 0, fp32_solver_outputs=40,
                    output_dtype="torch.float32", forward_verified=True, backward_verified=grad)
                if grad:
                    expected_frame.update(parameter_gradient_dtype="torch.float32", parameter_gradient_tensors=133)
                for key, value in expected_frame.items():
                    require(frame.get(key) == value, f"Forward {index}: {key}={frame.get(key)!r}, expected {value!r}")
        backend = run["backend"]
        if candidate == "mixed_shared":
            require(backend.get("candidate_id") == MIXED_ID and backend.get("no_grad_inference") == "unchanged production",
                    "Mixed candidate did not keep original inference")
            build = backend["shared_training_build"]
            for name, digest in build["source_sha256"].items():
                require(run["source_sha256"].get("experiments/training_shared_s1/" + name) == digest,
                        f"Shared extension source mismatch: {name}")
            binary = Path(build["binary_path"])
            require(binary.is_file() and shared.file_hash(binary) == build["binary_sha256"],
                    "Recorded shared extension binary is missing or changed")
            production = backend["production_backend"]
        else:
            build, production = None, backend
        require(production.get("kind") == "current production", "Unexpected original inference backend")
        sources = production["build_manifest"]["inputs"]["sources"]
        for name, digest in sources.items():
            require(run["source_sha256"].get("Converse2D/torch_converse2d/" + name) == digest,
                    f"Production source/build mismatch: {name}")
        return dict(candidate=candidate, adapter_sha256=adapter["sha256"],
            helper_source_sha256=adapter["helper_source_sha256"], production_sources=sources,
            shared_build=build, model_source_sha256={key: value for key, value in run["source_sha256"].items()
                                                   if key.startswith("models/")})
    return validate


def compare_pair(candidate, baseline, label):
    # This helper only compares identities, batches, metrics and descriptive
    # matched-control times. It does not require candidate parameter equality.
    result = earlier.compare_pair(candidate, baseline, label)
    result["candidate"] = "mixed_shared"
    if label == "matched_current":
        for key in ("adapter_sha256", "helper_source_sha256", "production_sources"):
            if candidate["effective_backend"][key] != baseline["effective_backend"][key]:
                result["comparison_errors"].append(f"Matched current/candidate {key} differs")
        result["comparable"] = not result["comparison_errors"]
    return result


def audit(args):
    runs, pairs, missing, errors = [], [], [], []
    labels = ("python_fp32",) if args.python_only else ("python_fp32", "matched_current")
    for seed in args.seeds:
        slots = [("mixed_shared", args.root / f"mixed_shared_seed{seed}", candidate_validator("mixed_shared")),
                 ("python_fp32", args.python_root / f"pytorch_seed{seed}", validate_python_backend)]
        if not args.python_only:
            slots.append(("matched_current", args.root / f"mixed_control_seed{seed}", candidate_validator("current")))
        values = {}
        for label, directory, validator in slots:
            result = shared.inspect_run(directory, seed, "current", backend_validator=validator)
            result["comparison_route"] = label
            if result["validated"]:
                if any(row.get("gradient_tensor_count") != 133 for row in shared.read_jsonl(directory / "training.jsonl")):
                    result["errors"].append("Not all 133 parameter gradients were recorded at every update")
                    result["validated"] = False
            values[label] = result
            runs.append(result)
            missing.extend(f"{label}/seed{seed}: {message}" for message in result["missing"])
            errors.extend(f"{label}/seed{seed}: {message}" for message in result["errors"])
        if values["mixed_shared"]["validated"]:
            for label in labels:
                if values[label]["validated"]:
                    pair = compare_pair(values["mixed_shared"], values[label], label)
                    pairs.append(pair)
                    errors.extend(f"{label}/seed{seed}: {message}" for message in pair["comparison_errors"])
    valid = [run for run in runs if run["validated"]]
    for key in earlier.IDENTITY_KEYS[1:]:
        if len({shared.hash_json(run["identity"][key]) for run in valid}) > 1:
            errors.append(f"Cross-seed identity differs: {key}")
    for label in ("mixed_shared", *labels):
        if len({run["effective_backend_sha256"] for run in valid if run["comparison_route"] == label}) > 1:
            errors.append(f"Effective {label} source/build differs between seeds")
    required = len(labels) * len(args.seeds)
    if errors:
        status, code = "invalid", 3
    elif any(not pair["quality_passed"] for pair in pairs):
        status, code = "quality_failed", 4
    elif missing or len(pairs) != required:
        status, code = "incomplete", 2
    else:
        status, code = "passed_requested_comparisons", 0
    for run in runs:
        run.pop("batch_hashes", None)
    return dict(status=status, exit_code=code, requested_seeds=args.seeds, requested_baselines=list(labels),
        complete_pairs=len(pairs), required_pairs=required, single_seed_evidence=len(args.seeds) == 1,
        production_eligible=False, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        missing=missing, integrity_errors=sorted(set(errors)), runs=runs, pairs=pairs,
        scope="CPU-only completed-run/data/init/checkpoint/dispatch/dtype/quality audit. Unfinished or unrequested comparisons are not declared passed. No parameter bitwise-equality requirement, GPU recomputation, convergence or global-release conclusion.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/native_deconv_target")
    parser.add_argument("--python-root", type=Path, default=ROOT / "artifacts/python_training_comparison")
    parser.add_argument("--seeds", type=earlier.seeds, default=[17, 29, 43])
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/mixed_shared_quality_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; old evidence is not overwritten")
    protected = [args.root / f"{prefix}_seed{seed}" for seed in args.seeds for prefix in ("mixed_shared", "mixed_control")]
    protected += [args.python_root / f"pytorch_seed{seed}" for seed in args.seeds]
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in protected):
        parser.error("Audit output must remain outside read-only run directories")
    report = audit(args)
    report["auditor_sha256"] = shared.file_hash(__file__)
    report["reused_validator_sha256"] = {Path(module.__file__).name: shared.file_hash(module.__file__)
                                         for module in (shared, earlier)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(shared.clean(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], complete_pairs=report["complete_pairs"], required_pairs=report["required_pairs"],
                         integrity_errors=report["integrity_errors"], missing=report["missing"], pairs=report["pairs"]), indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
