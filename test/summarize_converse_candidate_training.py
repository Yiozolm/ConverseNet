"""CPU-only audit of current-control/combined full-model training pairs.

Default --seeds 17; use --seeds 17,29,43 when those new pairs are complete.
No GPU/PyTorch import and no modification of workers, adapters or run records.
Incomplete evidence returns 2, invalid evidence 3, failed quality gates 4.
Passing a requested single pair is not multi-seed or production approval.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import summarize_usrnet_training as shared

ROOT = shared.ROOT
ADAPTER = "test/train_usrnet_converse_candidate.py"
HELPERS = ("probe_training_s1_shapes.py", "probe_converse_boundaries.py", "probe_pointwise_training.py")
COMBINED_ID = "isolated_forced_s1_catpad_cropview_c128_96_v1"
SOURCE = "Converse2D/torch_converse2d/"
SOURCE_NAMES = ("converse2d.cpp", "converse2d_kernels.cu", "converse2d_training.cu", "converse2d_training.h")
SELECTOR = "bool use_scale1(I H,I W,I s) { return s==1 && H*W>=65536; }"
FORCED = "bool use_scale1(I H,I W,I s) { return s==1; }"
PAIR_KEYS = ("comparison_recipe_sha256", "split_sha256", "input_checkpoint_sha256",
             "initial_state_tensor_sha256", "manifest_sha256", "kernels_sha256",
             "validation_payload_sha256", "shared_source_sha256")


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def forced_manifest_audit(run, require):
    backend = run["backend"]
    require(backend.get("kind") == "isolated combined Converse candidate", "Combined is not labelled as an isolated candidate")
    require(backend.get("production_source_unchanged") is True, "Combined did not attest unchanged production source")
    forced = backend["forced_build"]
    expected_patch = dict(file="converse2d_training.cu", count=1, before=SELECTOR, after=FORCED)
    require(forced.get("patch") == expected_patch, "Forced manifest is not the one permitted selector change")
    current_sources = backend["verified_current_build"]["build_manifest"]["inputs"]["sources"]
    originals = {name: (ROOT / SOURCE / name).read_bytes() for name in SOURCE_NAMES}
    original_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in originals.items()}
    require(original_hashes == forced["source_sha256"] == current_sources, "Forced/current/on-disk source hashes disagree")
    for name, value in original_hashes.items():
        require(run["source_sha256"].get(SOURCE + name) == value, f"Production source changed since capture: {name}")
    texts = {name: value.decode("utf-8").replace("\r\n", "\n") for name, value in originals.items()}
    require(texts["converse2d_training.cu"].count(SELECTOR) == 1, "Selector is not unique in original CUDA source")
    texts["converse2d_training.cu"] = texts["converse2d_training.cu"].replace(SELECTOR, FORCED, 1)
    patched = {name: text_hash(value) for name, value in texts.items()}
    require(patched == forced["patched_source_sha256"], "Patched source contains changes beyond the selector")
    fingerprint = text_hash(json.dumps(dict(original=original_hashes, patched=patched), sort_keys=True))[:16]
    prefix = "s1_forced_" + fingerprint + "_converse"
    require(forced["namespace"] == prefix + "2d", "Forced namespace fingerprint mismatch")
    namespaced = {name.replace("converse", prefix): text_hash(value.replace("converse", prefix))
                  for name, value in texts.items()}
    require(namespaced == forced["namespaced_source_sha256"], "Namespaced source manifest mismatch")
    return dict(kind=backend["kind"], namespace=forced["namespace"], original_source_sha256=original_hashes,
                patched_source_sha256=patched, unique_selector_patch=expected_patch,
                production_source_hashes_recomputed=True)


def backend_validator(candidate):
    def validate(run, require):
        expected_id = COMBINED_ID if candidate == "combined" else "unchanged_current_control"
        require(run.get("comparison_candidate") == candidate and run.get("candidate_id") == expected_id,
                "Candidate label or ID differs from the requested slot")
        adapter, proof = run["candidate_adapter"], run["candidate_verification"]
        require(adapter.get("sha256") == run["source_sha256"].get(ADAPTER)
                and shared.digest_valid(adapter.get("sha256")), "Adapter hash is missing or inconsistent")
        for name, value in adapter["shared_worker_files_sha256"].items():
            require(value == run["source_sha256"].get("test/" + name), f"Shared worker file mismatch: {name}")
        for name in HELPERS:
            value = adapter["helper_source_sha256"].get(name)
            require(shared.digest_valid(value) and value == run["source_sha256"].get("test/" + name),
                    f"Candidate helper hash mismatch: {name}")
        require(proof.get("candidate") == candidate and proof.get("candidate_id") == expected_id,
                "Verification counters belong to a different candidate")
        if run.get("status") == "complete":
            expected = dict(model_forwards=550, grad_enabled_forwards=250, inference_forwards=300,
                            prior_calls=19250, prior_training_calls=8750, prior_inference_calls=10500,
                            guarded_prior_calls=19250, guarded_prior_training_calls=8750,
                            guarded_prior_inference_calls=10500, expected_prior_calls_per_model_forward=35,
                            prior_modules=7, model_iterations=5,
                            first_training_graph_checked=True, first_training_graph_has_spectral_solve=True)
            expected.update(candidate_boundary_calls=19250 if candidate == "combined" else 0,
                            candidate_boundary_training_calls=8750 if candidate == "combined" else 0,
                            candidate_boundary_inference_calls=10500 if candidate == "combined" else 0)
            for key, value in expected.items():
                require(proof.get(key) == value, f"Prior coverage/route mismatch: {key}={proof.get(key)!r}, expected {value!r}")
        if candidate == "combined":
            backend = forced_manifest_audit(run, require)
        else:
            backend = run["backend"]
            require(backend.get("kind") == "current production", "Current control did not use the original production backend")
            sources = backend["build_manifest"]["inputs"]["sources"]
            for name, value in sources.items():
                require(run["source_sha256"].get(SOURCE + name) == value == shared.file_hash(ROOT / SOURCE / name),
                        f"Current build/capture/on-disk source mismatch: {name}")
            backend = dict(kind=backend["kind"], source_sha256=sources)
        return dict(candidate=candidate, backend=backend, adapter_sha256=adapter["sha256"],
                    helper_source_sha256=adapter["helper_source_sha256"],
                    model_source_sha256={key: value for key, value in run["source_sha256"].items() if key.startswith("models/")})
    return validate


def compare_pair(control, combined):
    errors = []
    for key in PAIR_KEYS:
        if control["identity"][key] != combined["identity"][key]:
            errors.append(f"Paired {key} differs")
    if control["batch_hashes"] != combined["batch_hashes"]:
        errors.append("Paired 250 training batches differ")
    for key in ("adapter_sha256", "helper_source_sha256", "model_source_sha256"):
        if control["effective_backend"][key] != combined["effective_backend"][key]:
            errors.append(f"Paired {key} differs")
    gates = {}
    for space in ("rgb", "y"):
        for name, budget in (("psnr_db", shared.PSNR_DROP_DB), ("ssim", shared.SSIM_DROP)):
            delta = shared.metric_difference(shared.metric(combined["quality"]["final"][space][name]),
                                             shared.metric(control["quality"]["final"][space][name]))
            gates[space + "_" + name] = dict(combined_minus_control=delta, minimum_delta=-budget, passed=delta >= -budget)
    return dict(seed=control["seed"], comparison_errors=errors, comparable=not errors,
                matched_batch_hashes=250, validation_images=100,
                final_quality_gates=gates, quality_passed=all(value["passed"] for value in gates.values()),
                observational_time_ratio={key: control["timing"][key] / combined["timing"][key]
                                          for key in ("warm_step_wall_median_ms", "data_plus_step_total_s", "loop_including_evaluation_io_s")},
                timing_caveat="These two sequential training runs are descriptive; not the independent rotated native/candidate timing benchmark, not a historical cross-run ratio, and not a causal stable-speed claim")


def audit(args):
    runs, pairs, missing, errors = [], [], [], []
    for seed in args.seeds:
        pair = {}
        for candidate, prefix in (("current", "current_control"), ("combined", "combined")):
            directory = args.root / f"{prefix}_seed{seed}"
            result = shared.inspect_run(directory, seed, "current", backend_validator=backend_validator(candidate))
            result["candidate"] = candidate
            if (directory / "run.json").is_file():
                raw = shared.read_json(directory / "run.json")
                result["candidate_verification"] = raw.get("candidate_verification")
            if result["validated"]:
                rows = shared.read_jsonl(directory / "training.jsonl")
                if any(row.get("gradient_tensor_count") != 133 for row in rows):
                    result["errors"].append("Not all 133 parameter gradients were present at every update")
                    result["validated"] = False
            runs.append(result)
            pair[candidate] = result
            missing.extend(result["missing"])
            errors.extend(f"{candidate}/seed{seed}: {item}" for item in result["errors"])
        if all(value["validated"] for value in pair.values()):
            value = compare_pair(pair["current"], pair["combined"])
            pairs.append(value)
            errors.extend(f"seed{seed}: {item}" for item in value["comparison_errors"])
    valid = [run for run in runs if run["validated"]]
    for key in PAIR_KEYS[1:]:
        if len({shared.hash_json(run["identity"][key]) for run in valid}) > 1:
            errors.append(f"Cross-seed {key} differs")
    for candidate in ("current", "combined"):
        if len({run["effective_backend_sha256"] for run in valid if run["candidate"] == candidate}) > 1:
            errors.append(f"Effective {candidate} source differs across seeds")
    if missing:
        status, code = "incomplete", 2
    elif errors or len(pairs) != len(args.seeds):
        status, code = "invalid", 3
    elif not all(pair["quality_passed"] for pair in pairs):
        status, code = "quality_failed", 4
    else:
        status, code = "passed_requested_pairs", 0
    for run in runs:
        run.pop("batch_hashes", None)
    return dict(status=status, exit_code=code, requested_seeds=args.seeds, complete_pairs=len(pairs),
                single_seed_evidence=len(args.seeds) == 1, production_eligible=False,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                missing=missing, integrity_errors=sorted(set(errors)), runs=runs, pairs=pairs,
                scope=["Only these new current-control/combined runs are compared; no old native/depthwise/Python speed ratios are multiplied or substituted.",
                       "Shared completed-run validation checks strict full model, all 250 updates/batches, all three full 100-image evaluations, checkpoint file hashes, initial/final state changes and unchanged recipe.",
                       "Validation payload equality is the recorded aggregate over ordered IDs and all LR/kernel/HR tensors; complete per-image ID sets and reported metric means are also checked.",
                       "The forced selector, normalized/namespaced source hashes and source immutability are independently reconstructed without loading or compiling the extension.",
                       "Warm step median discards the first five updates. Full data+step and end-to-end loop totals retain all 250 steps; loop includes evaluations and I/O. Peak allocated/reserved are separate totals.",
                       "Single-pair timing is descriptive and sensitive to temporal drift. Passing requested short fine-tuning pairs is not full convergence, broader FP64 regression, or production eligibility."])


def markdown(report):
    lines = ["# Combined Converse candidate training audit", "", f"Status: **{report['status']}**; completed pairs: {report['complete_pairs']}/{len(report['requested_seeds'])}; seeds: {report['requested_seeds']}.", "",
             "Single-seed evidence: " + str(report["single_seed_evidence"]) + ". Production eligible: false.", ""]
    for label, values in (("Missing/unfinished", report["missing"]), ("Integrity failures", report["integrity_errors"])):
        if values:
            lines += [label + ":", "", *("- " + value for value in values), ""]
    lines += ["| Seed | Candidate | RGB PSNR initial -> final | Y PSNR initial -> final | RGB SSIM initial -> final | Y SSIM initial -> final |",
              "| --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        if "quality" not in run or "final" not in run["quality"]:
            continue
        values = [f"{shared.fmt(run['quality']['initial'][space][key])} -> {shared.fmt(run['quality']['final'][space][key])}"
                  for space, key in (("rgb", "psnr_db"), ("y", "psnr_db"), ("rgb", "ssim"), ("y", "ssim"))]
        lines.append("| " + " | ".join([str(run["seed"]), run["candidate"], *values]) + " |")
    lines += ["", "Final differences are combined minus current control; PSNR >= -0.05 dB and SSIM >= -0.001 are required per space and seed.", "",
              "| Seed | RGB PSNR delta | Y PSNR delta | RGB SSIM delta | Y SSIM delta | Passed |",
              "| --- | --- | --- | --- | --- | --- |"]
    for pair in report["pairs"]:
        values = [shared.fmt(pair["final_quality_gates"][key]["combined_minus_control"], 6)
                  for key in ("rgb_psnr_db", "y_psnr_db", "rgb_ssim", "y_ssim")]
        lines.append("| " + " | ".join([str(pair["seed"]), *values, str(pair["quality_passed"])]) + " |")
    lines += ["", "Observed run timings (not a stabilized independent A/B speed claim):", "",
              "| Seed | Candidate | All-step median ms | Warm median ms | Data+steps s | Loop incl. eval/I/O s | Allocated GiB | Reserved GiB |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        if "timing" not in run:
            continue
        timing, memory = run["timing"], run["peak_memory"]["overall_peak_memory"]
        lines.append(f"| {run['seed']} | {run['candidate']} | {timing['step_wall_median_ms']:.3f} | {timing['warm_step_wall_median_ms']:.3f} | {timing['data_plus_step_total_s']:.3f} | {timing['loop_including_evaluation_io_s']:.3f} | {memory['allocated_bytes']/2**30:.3f} | {memory['reserved_bytes']/2**30:.3f} |")
    lines += ["", "All source/checkpoint/coverage details are in JSON. No historical ratios are multiplied; short fine-tuning is not convergence or production approval.", ""]
    return "\n".join(lines)


def seeds(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integer seeds") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Seeds must be nonempty and unique")
    return values


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/native_deconv_target")
    parser.add_argument("--seeds", type=seeds, default=[17])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/training_summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "artifacts/native_deconv_target/training_summary.md")
    args = parser.parse_args()
    protected = [args.root / f"{prefix}_seed{seed}" for seed in args.seeds for prefix in ("current_control", "combined")]
    for output in (args.output.resolve(), args.markdown.resolve()):
        if any(output.is_relative_to(path.resolve()) for path in protected):
            parser.error("Outputs must remain outside read-only run directories")
    report = audit(args)
    report["summarizer_sha256"] = shared.file_hash(Path(__file__))
    report["shared_validator_sha256"] = shared.file_hash(Path(shared.__file__))
    for path, content in ((args.output, json.dumps(shared.clean(report), indent=2, allow_nan=False)),
                          (args.markdown, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print("Saved", path)
    print(f"Status: {report['status']}; complete pairs: {report['complete_pairs']}; integrity errors: {len(report['integrity_errors'])}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
