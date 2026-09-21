"""Offline Python/ATen versus production timing and three-seed quality audit.

Shares the original completed-run validator. Speed comes ONLY from the new
alternating warmed/reset timing benchmark; historical 250-step current runs are
used only for quality. Exit 2 denotes missing/incomplete evidence, 3 invalid
evidence, 4 a failed predeclared quality gate, and 0 a complete passing audit.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics

import summarize_usrnet_training as shared

ROOT = shared.ROOT
ADAPTER = "test/train_usrnet_python_comparison.py"
IDENTITY_KEYS = ("comparison_recipe_sha256", "split_sha256", "input_checkpoint_sha256",
                 "initial_state_tensor_sha256", "manifest_sha256", "kernels_sha256",
                 "validation_payload_sha256", "shared_source_sha256")


def validate_python_backend(run, require):
    backend, proof, adapter = run["backend"], run["backend_verification"], run["comparison_adapter"]
    require(run.get("comparison_backend") == "pytorch" and backend.get("comparison_backend") == "pytorch"
            and proof.get("comparison_backend") == "pytorch", "Python backend labels disagree")
    require(backend.get("native_extension_build") is False, "Python run unexpectedly built a native backend")
    require(shared.digest_valid(adapter.get("sha256")) and adapter["sha256"] == run["source_sha256"].get(ADAPTER),
            "Python adapter source hash is missing or inconsistent")
    for name in ("train_usrnet_dataset.py", "usrnet_training_data.py", "evaluate_usrnet_quality.py"):
        require(adapter["shared_worker_files_sha256"].get(name) == run["source_sha256"].get("test/" + name),
                f"Adapter changed or misreported shared source: {name}")
    if run.get("status") == "complete":
        expected = dict(first_training_graph_checked=True, first_training_graph_has_spectral_solve=False,
                        prior_solvers_per_iteration=7, expected_reference_calls_per_forward=40,
                        configured_backend_layers=8, grad_enabled_forwards=250, inference_forwards=300,
                        model_forwards=550, python_reference_calls=22000,
                        reference_calls_by_module={"DataNet": 2750, "Converse2D": 19250})
        for key, value in expected.items():
            require(proof.get(key) == value, f"Python route verification mismatch: {key}={proof.get(key)!r}")
    return dict(backend="pytorch full-FFT reference", adapter_sha256=adapter["sha256"],
                model_source_sha256={key: value for key, value in run["source_sha256"].items()
                                     if key.startswith("models/")})


def audit_timing(path):
    result = dict(path=str(path), status="missing", missing=[], errors=[], validated=False)
    if not path.is_file():
        result["missing"].append(f"Missing alternating timing report: {path}")
        return result
    data = shared.read_json(path)
    result["status"] = data.get("status")
    if data.get("status") != "complete":
        result["missing"].append(f"Alternating timing report is {data.get('status')}")
        return result

    def require(condition, message):
        if not condition:
            result["errors"].append(message)

    try:
        cfg = data["config"]
        for key, value in dict(rounds=4, warmup=5, batch_size=4, microbatch_size=4, seed=17,
                               patch_size=96, scale=3, noise_std=0.01, loss="mse", lr=1e-5).items():
            require(cfg.get(key) == value, f"Timing recipe mismatch: {key}")
        require(cfg["steps"] in (8, 10), "Timing requires 8 or 10 measured steps per fixture")
        require(data["environment"].get("tf32") is False and data["environment"].get("amp") is False
                and data["environment"].get("cudnn_deterministic") is True, "Timing precision/determinism settings differ")
        expected_batches = data["cached_batch_sha256"][:cfg["steps"]]
        require(len(expected_batches) == cfg["steps"] and all(shared.digest_valid(value) for value in expected_batches),
                "Timing input batch hashes are incomplete")
        require(shared.hash_json(expected_batches) == data["timed_batch_sequence_sha256"], "Timing batch sequence digest mismatch")
        require(len(data["rounds"]) == 4, "Timing does not contain four rounds")
        numerical = data["numerical_check"]
        require(numerical["routes"]["pytorch"]["spectral_solve"] is False
                and numerical["routes"]["current"]["spectral_solve"] is True, "Numerical-check graph routing is incorrect")
        require(len(numerical["current_vs_pytorch"]) == 2, "Expected two diagnostic initial updates")
        for row in numerical["current_vs_pytorch"]:
            for group in ("outputs", "parameters", "gradients", "adam"):
                require(row[group].get("finite") is True, "Nonfinite two-step diagnostic snapshot")
        rounds, adam_hashes = [], set()
        for index, row in enumerate(data["rounds"]):
            order = ["pytorch", "current"] if index % 2 == 0 else ["current", "pytorch"]
            require(row["round"] == index + 1 and row["order"] == order, f"Round {index + 1} order is not alternating")
            compact = dict(round=index + 1, order=order, variants={})
            for name in ("pytorch", "current"):
                value = row["variants"][name]
                samples = value["rows"]
                require(value["initial_state_tensor_sha256"] == data["initial_state_tensor_sha256"], "Timing fixture initial model differs")
                require(value["warmup_steps"] == 5 and value["parameters_finite"].get("finite") is True,
                        "Timing fixture warmup/finite verification failed")
                require(shared.digest_valid(value["initial_adam_state_sha256"]), "Invalid initial Adam state hash")
                adam_hashes.add(value["initial_adam_state_sha256"])
                require(len(samples) == cfg["steps"] and [sample["batch_sha256"] for sample in samples] == expected_batches,
                        f"Round {index + 1}/{name}: measured batches differ")
                for number, sample in enumerate(samples, 1):
                    require(sample["step"] == number and sample["optimizer_applied"] is True
                            and sample["loss_and_grad_finite"] is True
                            and shared.finite(sample["loss"]) and shared.finite(sample["grad_l2_norm"]),
                            f"Round {index + 1}/{name}: invalid update {number}")
                medians = {}
                for key in ("training_step_wall_ms", "training_step_cuda_span_ms", "forward_backward_wall_ms",
                            "forward_backward_cuda_event_ms"):
                    values = [sample[key] for sample in samples]
                    require(all(shared.finite(item) and item > 0 for item in values), "Invalid timing samples")
                    medians[key] = statistics.median(values)
                    require(math.isclose(medians[key], value["medians"][key], rel_tol=1e-12), "Stored timing median differs from raw samples")
                memory = {key: max(sample["peak_memory"][key] for sample in samples)
                          for key in ("allocated_bytes", "reserved_bytes")}
                require(memory == value["peak_memory"], "Timing memory summary differs from raw samples")
                compact["variants"][name] = dict(medians=medians, peak_memory=memory,
                                                 initial_state_tensor_sha256=value["initial_state_tensor_sha256"],
                                                 initial_adam_state_sha256=value["initial_adam_state_sha256"])
            compact["speedup_pytorch_over_current"] = {
                key: compact["variants"]["pytorch"]["medians"][key] / compact["variants"]["current"]["medians"][key]
                for key in compact["variants"]["current"]["medians"]}
            for key, value in compact["speedup_pytorch_over_current"].items():
                require(math.isclose(value, row["speedup_pytorch_over_current"][key], rel_tol=1e-12), "Stored paired round ratio differs")
            rounds.append(compact)
        require(len(adam_hashes) == 1, "Timing fixtures start from different Adam states")
        result.update(config=cfg, rounds=rounds, numerical_diagnostics=numerical,
                      identity={key: data[key] for key in ("input_checkpoint_sha256", "initial_state_tensor_sha256",
                                                         "source_sha256", "dataset", "cached_batch_sha256")},
                      speedup_summary={key: dict(median=statistics.median(values), min=min(values), max=max(values),
                                                 per_round=values)
                                       for key in rounds[0]["speedup_pytorch_over_current"]
                                       for values in [[row["speedup_pytorch_over_current"][key] for row in rounds]]})
        result["validated"] = not result["errors"]
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        result["errors"].append(f"Timing audit could not complete: {error}")
    return result


def quality_pair(python, current):
    errors = []
    for key in IDENTITY_KEYS:
        if python["identity"][key] != current["identity"][key]:
            errors.append(f"Paired {key} differs")
    if python["batch_hashes"] != current["batch_hashes"]:
        errors.append("Paired 250 training batches differ")
    gates = {}
    for space in ("rgb", "y"):
        for name, limit in (("psnr_db", shared.PSNR_DROP_DB), ("ssim", shared.SSIM_DROP)):
            delta = shared.metric_difference(shared.metric(current["quality"]["final"][space][name]),
                                             shared.metric(python["quality"]["final"][space][name]))
            gates[space + "_" + name] = dict(current_minus_python=delta, minimum_allowed_delta=-limit, passed=delta >= -limit)
    return dict(seed=python["seed"], comparison_errors=errors, comparable=not errors,
                checked_batch_hashes=250, gates=gates, quality_passed=all(value["passed"] for value in gates.values()))


def audit(args):
    timing = audit_timing(args.timing)
    runs, pairs, missing, errors = [], [], list(timing["missing"]), list(timing["errors"])
    for seed in shared.SEEDS:
        current = shared.inspect_run(args.current_root / f"current_seed{seed}", seed, "current")
        python_dir = args.python_root / f"pytorch_seed{seed}"
        python = shared.inspect_run(python_dir, seed, "current", backend_validator=validate_python_backend)
        python["variant"] = "pytorch"  # Adapter preserves worker config.variant=current; explicit proof supplies its real backend.
        if (python_dir / "run.json").is_file():
            raw = shared.read_json(python_dir / "run.json")
            python["backend_verification"] = raw.get("backend_verification")
        for value in (python, current):
            missing.extend(value["missing"])
            errors.extend(f"{value['variant']}/seed{seed}: {item}" for item in value["errors"])
            runs.append(value)
        if python["validated"] and current["validated"]:
            pair = quality_pair(python, current)
            pairs.append(pair)
            errors.extend(f"seed{seed}: {item}" for item in pair["comparison_errors"])
    valid = [run for run in runs if run["validated"]]
    for key in ("split_sha256", "input_checkpoint_sha256", "initial_state_tensor_sha256", "manifest_sha256",
                "kernels_sha256", "validation_payload_sha256", "shared_source_sha256"):
        if len({shared.hash_json(run["identity"][key]) for run in valid}) > 1:
            errors.append(f"Cross-seed {key} differs")
    for variant in ("pytorch", "current"):
        if len({run["effective_backend_sha256"] for run in valid if run["variant"] == variant}) > 1:
            errors.append(f"Effective {variant} source changed across quality seeds")
    if timing["validated"]:
        identity = timing["identity"]
        for run in valid:
            for key in ("input_checkpoint_sha256", "initial_state_tensor_sha256"):
                if run["identity"][key] != identity[key]:
                    errors.append(f"Timing/quality {key} differs")
            for key in shared.SHARED_SOURCE_KEYS:
                if run["identity"]["shared_source_sha256"][key] != identity["source_sha256"].get(key):
                    errors.append(f"Timing/quality shared source differs: {key}")
            if run["identity"]["manifest_sha256"] != identity["dataset"]["manifest_sha256"]:
                errors.append("Timing/quality dataset manifest differs")
            if run["seed"] == 17 and run["batch_hashes"][:timing["config"]["steps"]] != identity["cached_batch_sha256"][:timing["config"]["steps"]]:
                errors.append("Timing does not replay the same initial seed-17 training batches")
    if missing:
        status, code = "partial", 2
    elif errors or not timing["validated"] or len(pairs) != 3:
        status, code = "invalid", 3
    elif not all(pair["quality_passed"] for pair in pairs):
        status, code = "quality_failed", 4
    else:
        status, code = "passed", 0
    # Compact quality output deliberately excludes historical training timings.
    compact = []
    for run in runs:
        compact.append({key: run[key] for key in
                        ("seed", "variant", "directory", "run_status", "validated", "errors", "missing", "identity",
                         "quality", "parameter_state_changed", "checkpoints", "backend_verification", "batch_sequence_sha256")
                        if key in run})
    return dict(status=status, exit_code=code, final_success=status == "passed",
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), missing=missing,
                integrity_errors=sorted(set(errors)), timing=timing, quality_runs=compact, quality_pairs=pairs,
                scope=[
                    "Speed uses ONLY the new same-process alternating four-round benchmark: five warmup updates, then exact pretrained/zero-Adam reset, then eight measured updates per fixture (or configured ten).",
                    "Its warmed/reset workload differs from the historical 250-step training segments. Historical current runs are reused for matched quality only; their approximately 501 ms training medians are NEVER divided into the new Python timings.",
                    "Full-step timing retains the unchanged worker's synchronization, H2D, finite/norm checks, microbatch accumulation and Adam. Allocated/reserved are separate PyTorch allocator totals.",
                    "Every seed must independently pass final RGB/Y PSNR drop <=0.05 dB and SSIM drop <=0.001, with 250 identical input batches, matching initialization/data/source provenance and finite training.",
                    "Two initial-update output/gradient/state differences are diagnostic, not a new tolerance or replacement for independent FP64 tests. Short fine-tuning does not establish convergence.",
                ])


def markdown(report):
    lines = ["# Python/ATen versus production training", "", f"Status: **{report['status']}**. Quality pairs completed: {len(report['quality_pairs'])}/3.", "",
             "Speed below uses the new alternating warmed/reset benchmark only. Historical 250-step current runs supply quality, not this timing baseline.", ""]
    if report["missing"]:
        lines += ["Missing or unfinished:", "", *("- " + item for item in report["missing"]), ""]
    if report["integrity_errors"]:
        lines += ["Integrity failures:", "", *("- " + item for item in report["integrity_errors"]), ""]
    timing = report["timing"]
    if timing.get("rounds"):
        lines += ["## Alternating timing", "", "5 warmup updates, reset model and allocated Adam states, then 8 measured steps per fixture. Full-step includes H2D and finite checks.", "",
                  "| Round | Python full ms | Current full ms | Ratio | Python F+B ms | Current F+B ms |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in timing["rounds"]:
            python, current = (row["variants"][key]["medians"] for key in ("pytorch", "current"))
            lines.append(f"| {row['round']} | {python['training_step_wall_ms']:.3f} | {current['training_step_wall_ms']:.3f} | {row['speedup_pytorch_over_current']['training_step_wall_ms']:.4f}x | {python['forward_backward_wall_ms']:.3f} | {current['forward_backward_wall_ms']:.3f} |")
        for key, label in (("training_step_wall_ms", "Full step"), ("forward_backward_wall_ms", "Forward + backward")):
            row = timing["speedup_summary"][key]
            lines += ["", f"{label}: median paired ratio {row['median']:.4f}x; round range [{row['min']:.4f}, {row['max']:.4f}]."]
        lines += ["", "| Backend | Maximum allocated GiB | Maximum reserved GiB |", "| --- | --- | --- |"]
        for backend in ("pytorch", "current"):
            memory = {key: max(row["variants"][backend]["peak_memory"][key] for row in timing["rounds"])
                      for key in ("allocated_bytes", "reserved_bytes")}
            lines.append(f"| {backend} | {memory['allocated_bytes'] / 2**30:.3f} | {memory['reserved_bytes'] / 2**30:.3f} |")
    lines += ["", "## Quality", "", "Each cell is initial -> final over all 100 fixed held-out crops.", "",
              "| Seed | Backend | RGB PSNR dB | Y PSNR dB | RGB SSIM | Y SSIM | State changed |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["quality_runs"]:
        if "quality" not in run or "final" not in run["quality"]:
            continue
        values = [f"{shared.fmt(run['quality']['initial'][space][key])} -> {shared.fmt(run['quality']['final'][space][key])}"
                  for space, key in (("rgb", "psnr_db"), ("y", "psnr_db"), ("rgb", "ssim"), ("y", "ssim"))]
        lines.append("| " + " | ".join([str(run["seed"]), run["variant"], *values, str(run["parameter_state_changed"])]) + " |")
    lines += ["", "Final deltas are current minus Python. All four gates must pass for each seed.", "",
              "| Seed | RGB PSNR delta | Y PSNR delta | RGB SSIM delta | Y SSIM delta | Passed |",
              "| --- | --- | --- | --- | --- | --- |"]
    for pair in report["quality_pairs"]:
        values = [shared.fmt(pair["gates"][key]["current_minus_python"], 6)
                  for key in ("rgb_psnr_db", "y_psnr_db", "rgb_ssim", "y_ssim")]
        lines.append("| " + " | ".join([str(pair["seed"]), *values, str(pair["quality_passed"])]) + " |")
    lines += ["", "The two-step numerical differences in JSON are diagnostics only. Independent FP64 gates remain unchanged; this is not convergence evidence.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--timing", type=Path, default=ROOT / "artifacts/python_training_comparison/paired_timing_b4_m4.json")
    parser.add_argument("--python-root", type=Path, default=ROOT / "artifacts/python_training_comparison")
    parser.add_argument("--current-root", type=Path, default=ROOT / "artifacts/dataset_training")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/python_training_comparison/summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "artifacts/python_training_comparison/summary.md")
    args = parser.parse_args()
    protected = [args.current_root / f"current_seed{seed}" for seed in shared.SEEDS]
    protected += [args.python_root / f"pytorch_seed{seed}" for seed in shared.SEEDS]
    for output in (args.output.resolve(), args.markdown.resolve()):
        if output == args.timing.resolve() or any(output.is_relative_to(path.resolve()) for path in protected):
            parser.error("Summary outputs must not overwrite timing evidence or live run directories")
    report = audit(args)
    report["summarizer_sha256"] = shared.file_hash(Path(__file__))
    report["shared_validator_sha256"] = shared.file_hash(Path(shared.__file__))
    for path, content in ((args.output, json.dumps(shared.clean(report), indent=2, allow_nan=False)),
                          (args.markdown, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print("Saved", path)
    print(f"Status: {report['status']}; quality pairs: {len(report['quality_pairs'])}/3; integrity errors: {len(report['integrity_errors'])}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
