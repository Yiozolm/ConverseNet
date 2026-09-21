"""Audit six completed paired USRNet fine-tuning runs, entirely offline.

Exit codes: 0 = complete, comparable and all predeclared quality gates passed;
2 = missing/incomplete runs; 3 = integrity/comparability failures; 4 = quality
gate failure. Partial reports are written with their missing evidence explicitly
listed. No PyTorch import, CUDA operation, checkpoint mutation or training occurs.
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
VARIANTS = ("before", "current")
STEPS = 250
EVAL_STEPS = [0, 125, 250]
PSNR_DROP_DB = 0.05
SSIM_DROP = 0.001
SHARED_SOURCE_KEYS = (
    "test/train_usrnet_dataset.py", "test/usrnet_training_data.py",
    "test/evaluate_usrnet_quality.py", "utils/utils_image.py",
    "test/extension_loader.py", "test/training_refinement_baseline.py",
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}:{index}: invalid JSONL (possibly an active partial write)") from error
    return rows


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest_valid(value):
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def metric(value):
    # The existing quality helper permits a perfect-reconstruction PSNR of +inf.
    result = float(value)
    if math.isnan(result):
        raise ValueError("NaN quality metric")
    return result


def metric_difference(current, before):
    return 0.0 if current == before else current - before


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def inspect_run(directory, seed, variant, backend_validator=None):
    directory = Path(directory).resolve()
    result = dict(seed=seed, variant=variant, directory=str(directory), run_status="missing",
                  errors=[], missing=[], validated=False)
    report_path = directory / "run.json"
    if not report_path.is_file():
        result["missing"].append(f"Missing {report_path}")
        return result
    try:
        run = read_json(report_path)
    except (OSError, ValueError) as error:
        result["missing"].append(str(error))
        return result
    result["run_status"] = run.get("status", "unknown")
    result["optimizer_steps"] = run.get("optimizer_steps", 0)
    if result["run_status"] != "complete":
        result["missing"].append(f"{variant}/seed{seed} is {result['run_status']}; completed run required")
    if run.get("error"):
        result["worker_error"] = run["error"]

    def require(condition, message):
        if not condition:
            result["errors"].append(message)

    try:
        config = run["config"]
        require(config["seed"] == seed and config["variant"] == variant, "Run seed/variant differs from its requested slot")
        required_config = dict(purpose="formal", init="pretrained", steps=STEPS, eval_every=125,
                               batch_size=4, microbatch_size=4, patch_size=96, scale=3,
                               noise_std=0.01, lr=1e-5, loss="mse", crop_border=3)
        for key, value in required_config.items():
            require(config.get(key) == value, f"Declared formal recipe mismatch: {key}={config.get(key)!r}, expected {value!r}")
        require(run["config_sha256"] == hash_json(config), "config_sha256 does not match config")
        recipe = {key: value for key, value in config.items()
                  if key not in ("run_dir", "variant", "verbose_build", "snapshot", "purpose")}
        require(run["comparison_recipe_sha256"] == hash_json(recipe), "comparison_recipe_sha256 does not match recipe")
        dataset = run["dataset"]
        require(dataset.get("train_images") == 900 and dataset.get("validation_images") == 100,
                "Dataset split must contain 900 training and 100 validation images")
        require(dataset.get("image_hashes_verified") is True, "Worker did not verify dataset image hashes")
        require(dataset.get("source_sha256") == run["source_sha256"].get("test/usrnet_training_data.py"),
                "Dataset implementation hash differs between source metadata and dataset metadata")
        for key in ("split_sha256", "input_checkpoint_sha256"):
            require(digest_valid(run.get(key)), f"Missing or invalid {key}")
        require(digest_valid(dataset.get("manifest_sha256")), "Missing or invalid dataset manifest hash")
        for key in SHARED_SOURCE_KEYS:
            require(digest_valid(run["source_sha256"].get(key)), f"Missing source hash: {key}")
        protocol = run["protocol"]
        require(protocol.get("optimizer") == "Adam" and protocol.get("betas") == [0.9, 0.999]
                and protocol.get("eps") == 1e-8 and protocol.get("weight_decay") == 0,
                "Adam parameters differ from the declared protocol")
        require(protocol.get("amp") is False and protocol.get("scheduler") is False
                and protocol.get("gradient_clipping") is False, "AMP/scheduler/clipping is enabled or undocumented")
        expected_gate = dict(max_psnr_drop_db=PSNR_DROP_DB, max_ssim_drop=SSIM_DROP,
                             spaces=["rgb", "y"], each_seed=True, all_steps_finite=True)
        require(protocol.get("preregistered_paired_final_quality_gate") == expected_gate,
                "Worker's recorded quality gate differs from the predeclared gate")
        require(run.get("environment", {}).get("tf32") is False, "TF32 must be disabled")
        require(run.get("architecture") == dict(num_iterations=5, num_blocks=7, in_channels=64,
                                                 kernel_size=7, strict=True), "Model is not the declared full strict architecture")
        require(run.get("parameter_count") == 307987 and run.get("parameter_tensor_count") == 133,
                "Model parameter count differs from the audited full checkpoint")
        result["identity"] = dict(
            comparison_recipe_sha256=run["comparison_recipe_sha256"], split_sha256=run["split_sha256"],
            input_checkpoint_sha256=run["input_checkpoint_sha256"],
            initial_state_tensor_sha256=run.get("initial_state_tensor_sha256"),
            manifest_sha256=dataset["manifest_sha256"], kernels_sha256=dataset["kernels_sha256"],
            validation_payload_sha256=run.get("validation_payload_sha256"),
            shared_source_sha256={key: run["source_sha256"].get(key) for key in SHARED_SOURCE_KEYS},
        )
        backend = run["backend"]
        if backend_validator is not None:
            effective = backend_validator(run, require)
        elif variant == "current":
            actual_sources = backend["build_manifest"]["inputs"]["sources"]
            for name, value in actual_sources.items():
                require(run["source_sha256"].get("Converse2D/torch_converse2d/" + name) == value,
                        f"Current build and recorded source disagree: {name}")
            effective = dict(cpp=actual_sources, models={key: value for key, value in run["source_sha256"].items()
                                                       if key.startswith("models/")})
        else:
            require(backend.get("ref") == "frozen-refinement-before:bc3926dc5ddc32e1", "Unexpected frozen baseline revision")
            effective = dict(ref=backend["ref"], source_sha256=backend["source_sha256"],
                             manifest_sha256=backend["manifest_sha256"])
        result["effective_backend"] = effective
        result["effective_backend_sha256"] = hash_json(effective)

        # Reading the small manifest verifies split identity without re-reading
        # the 1000 photographs already hashed by each training worker.
        manifest_path = Path(config["manifest"])
        expected_ids = None
        if manifest_path.is_file():
            document = read_json(manifest_path)
            require(file_hash(manifest_path) == dataset["manifest_sha256"], "Manifest bytes changed since training")
            records = document["images"]
            split = dict(train=sorted(row["relative_path"] for row in records if row["split"] == "train"),
                         validation=sorted(row["relative_path"] for row in records
                                           if row["split"] in ("validation", "val", "holdout")))
            require(hash_json(split) == run["split_sha256"], "Manifest split differs from split_sha256")
            expected_ids = set(split["validation"])

        training = read_jsonl(directory / "training.jsonl")
        evaluations = read_jsonl(directory / "evaluations.jsonl")
        result["training_rows"] = len(training)
        result["evaluation_steps"] = [row.get("optimizer_steps") for row in evaluations]
        if result["run_status"] != "complete":
            result["missing"].append(f"{variant}/seed{seed}: {len(training)}/{STEPS} training rows, evaluations {result['evaluation_steps']}")
            return result
        require(run["optimizer_steps"] == STEPS and run["samples_seen"] == 1000,
                "Final optimizer/sample counts differ from 250 steps / 1000 patches")
        require(run.get("all_loss_and_grad_finite") is True, "Run did not remain finite")
        require(run.get("final_parameter_check", {}).get("finite") is True, "Final parameters were not verified finite")
        require(len(training) == STEPS, f"Expected {STEPS} training rows, got {len(training)}")
        batch_hashes = []
        for index, row in enumerate(training, 1):
            require(row.get("step") == index and row.get("data_step") == index - 1
                    and row.get("optimizer_steps") == index, f"Non-contiguous training counters at row {index}")
            require(row.get("samples") == 4 and row.get("samples_seen") == index * 4
                    and row.get("microbatches") == 1, f"Effective batch/microbatch mismatch at step {index}")
            require(row.get("optimizer_applied") is True and row.get("loss_and_grad_finite") is True,
                    f"Missing optimizer step or nonfinite loss/gradient at step {index}")
            require(finite(row.get("loss")) and finite(row.get("grad_l2_norm")), f"Nonfinite recorded scalar at step {index}")
            require(digest_valid(row.get("batch_sha256")), f"Invalid batch hash at step {index}")
            batch_hashes.append(row.get("batch_sha256"))
            for key in ("training_step_wall_ms", "data_prepare_and_hash_wall_ms", "h2d_wall_ms"):
                require(finite(row.get(key)) and row[key] >= 0, f"Invalid timing {key} at step {index}")
            require(finite(row.get("training_step_wall_ms")) and row["training_step_wall_ms"] > 0,
                    f"Training step wall time must be positive at step {index}")
        result["batch_hashes"] = batch_hashes
        result["batch_sequence_sha256"] = hash_json(batch_hashes)
        require(result["evaluation_steps"] == EVAL_STEPS and run.get("evaluation_steps") == EVAL_STEPS,
                f"Expected complete evaluations at {EVAL_STEPS}")
        evaluation_by_step = {}
        for row in evaluations:
            step = row["optimizer_steps"]
            require(step not in evaluation_by_step, f"Duplicate evaluation at step {step}")
            evaluation_by_step[step] = row
            require(row.get("images") == 100 and len(row.get("per_image", [])) == 100,
                    f"Evaluation {step} did not include all 100 images")
            ids = [item["id"] for item in row["per_image"]]
            require(len(set(ids)) == 100 and (expected_ids is None or set(ids) == expected_ids),
                    f"Evaluation {step} has duplicate or incorrect held-out IDs")
            require(row.get("all_outputs_finite") is True and row.get("parameter_check", {}).get("finite") is True,
                    f"Evaluation {step} outputs/parameters were not verified finite")
            require(row.get("validation_payload_sha256") == run.get("validation_payload_sha256")
                    and digest_valid(row.get("validation_payload_sha256")), f"Evaluation {step} validation data changed")
            for space in ("rgb", "y"):
                for name in ("psnr_db", "ssim"):
                    mean = statistics.mean(metric(item[space][name]) for item in row["per_image"])
                    recorded = metric(row[space][name])
                    require(mean == recorded or math.isclose(mean, recorded, rel_tol=0, abs_tol=1e-10),
                            f"Evaluation {step} {space}/{name} mean differs from per-image metrics")
                    require(math.isfinite(recorded) or (name == "psnr_db" and recorded == math.inf),
                            f"Invalid quality metric at evaluation {step}")
        result["quality"] = {name: {space: evaluation_by_step[step][space] for space in ("rgb", "y")}
                             for name, step in (("initial", 0), ("final", STEPS)) if step in evaluation_by_step}

        checkpoints = run["checkpoints"]
        result["checkpoints"] = {}
        for label in ("initial", "final", "best"):
            record = checkpoints[label]
            path = directory / (label + ".pth")
            require(path.is_file(), f"Missing checkpoint: {path}")
            actual = file_hash(path) if path.is_file() else None
            require(actual == record["file_sha256"], f"Checkpoint bytes disagree with recorded hash: {label}")
            require(digest_valid(record.get("state_tensor_sha256")), f"Invalid state tensor hash: {label}")
            result["checkpoints"][label] = dict(file_sha256=actual, state_tensor_sha256=record.get("state_tensor_sha256"),
                                                optimizer_steps=record.get("optimizer_steps"))
        require(checkpoints["initial"]["optimizer_steps"] == 0 and checkpoints["final"]["optimizer_steps"] == STEPS,
                "Initial/final checkpoint optimizer counts are incorrect")
        require(checkpoints["initial"]["state_tensor_sha256"] == run["initial_state_tensor_sha256"],
                "Initial tensor hashes disagree")
        changed = checkpoints["initial"]["state_tensor_sha256"] != checkpoints["final"]["state_tensor_sha256"]
        require(changed, "Model state did not change between initial and final checkpoints")
        result["parameter_state_changed"] = changed
        result["parameter_change_evidence"] = "Canonical initial/final model tensor hashes recorded by worker; checkpoint file SHA256 independently verified; optimizer-only changes cannot satisfy this check"
        step_ms = [row["training_step_wall_ms"] for row in training]
        data_ms = sum(row["data_prepare_and_hash_wall_ms"] for row in training)
        h2d_ms = sum(row["h2d_wall_ms"] for row in training)
        for key, measured in (("total_training_step_wall_ms", sum(step_ms)),
                              ("data_prepare_and_hash_wall_ms", data_ms), ("total_h2d_wall_ms", h2d_ms)):
            require(math.isclose(run["timing"][key], measured, rel_tol=1e-10, abs_tol=1e-6),
                    f"run.json aggregate disagrees with training JSONL: {key}")
        result["timing"] = dict(step_wall_median_ms=statistics.median(step_ms), step_wall_mean_ms=statistics.mean(step_ms),
                                warm_step_wall_median_ms=statistics.median(step_ms[5:]), warmup_discarded_steps=5,
                                step_total_s=sum(step_ms) / 1000, data_prepare_and_hash_total_s=data_ms / 1000,
                                h2d_total_s=h2d_ms / 1000, data_plus_step_total_s=(data_ms + sum(step_ms)) / 1000,
                                loop_including_evaluation_io_s=run["timing"]["training_loop_end_to_end_wall_s"],
                                evaluation_wall_s=run["timing"]["evaluation_wall_s"],
                                checkpoint_wall_s=run["timing"]["checkpoint_wall_s"],
                                setup_wall_s=run["timing"]["setup_wall_s"])
        require(finite(result["timing"]["loop_including_evaluation_io_s"])
                and result["timing"]["loop_including_evaluation_io_s"] > 0, "Invalid loop wall time")
        result["peak_memory"] = {key: run[key] for key in
                                 ("training_peak_memory", "evaluation_peak_memory", "overall_peak_memory")}
        for scope, values in result["peak_memory"].items():
            require(all(finite(values.get(key)) and values[key] >= 0 for key in ("allocated_bytes", "reserved_bytes")),
                    f"Invalid memory statistics: {scope}")
        result["validated"] = not result["errors"] and not result["missing"]
    except (OSError, ValueError, KeyError, TypeError, statistics.StatisticsError) as error:
        target = result["missing"] if result["run_status"] != "complete" else result["errors"]
        target.append(f"Could not fully audit run: {type(error).__name__}: {error}")
    return result


def paired(before, current):
    errors = []
    for key in ("comparison_recipe_sha256", "split_sha256", "input_checkpoint_sha256", "initial_state_tensor_sha256",
                "manifest_sha256", "kernels_sha256", "validation_payload_sha256", "shared_source_sha256"):
        if before["identity"].get(key) != current["identity"].get(key):
            errors.append(f"Paired {key} differs")
    if before["batch_hashes"] != current["batch_hashes"]:
        mismatches = [index + 1 for index, (a, b) in enumerate(zip(before["batch_hashes"], current["batch_hashes"])) if a != b]
        errors.append(f"Training batches differ at steps {mismatches[:20]}")
    gates = {}
    for space in ("rgb", "y"):
        for name, tolerance in (("psnr_db", PSNR_DROP_DB), ("ssim", SSIM_DROP)):
            delta = metric_difference(metric(current["quality"]["final"][space][name]),
                                      metric(before["quality"]["final"][space][name]))
            gates[space + "_" + name] = dict(current_minus_before=delta, minimum_allowed_delta=-tolerance,
                                            passed=delta >= -tolerance)
    ratios = {key: before["timing"][key] / current["timing"][key]
              for key in ("step_wall_median_ms", "warm_step_wall_median_ms", "data_plus_step_total_s", "loop_including_evaluation_io_s")}
    return dict(seed=before["seed"], comparability_errors=errors, comparable=not errors,
                checked_training_batches=STEPS, batch_sequence_sha256=current["batch_sequence_sha256"],
                final_quality_gates=gates, all_quality_gates_passed=all(row["passed"] for row in gates.values()),
                speedup_before_over_current=ratios)


def audit(directories):
    runs = [inspect_run(directories[(seed, variant)], seed, variant) for seed in SEEDS for variant in VARIANTS]
    missing = [item for run in runs for item in run["missing"]]
    errors = [f"{run['variant']}/seed{run['seed']}: {item}" for run in runs for item in run["errors"]]
    pairs = []
    for seed in SEEDS:
        before, current = [next(run for run in runs if run["seed"] == seed and run["variant"] == variant) for variant in VARIANTS]
        if before["validated"] and current["validated"]:
            pair = paired(before, current)
            pairs.append(pair)
            errors.extend(f"seed{seed}: {item}" for item in pair["comparability_errors"])
    valid = [run for run in runs if run["validated"]]
    # Validation is fixed independently of training seed. The worker/data code,
    # input checkpoint and split must also be common to the whole experiment.
    for key in ("split_sha256", "input_checkpoint_sha256", "initial_state_tensor_sha256", "manifest_sha256",
                "kernels_sha256", "validation_payload_sha256", "shared_source_sha256"):
        if len({hash_json(run["identity"][key]) for run in valid}) > 1:
            errors.append(f"Cross-seed {key} differs")
    for variant in VARIANTS:
        if len({run["effective_backend_sha256"] for run in valid if run["variant"] == variant}) > 1:
            errors.append(f"Effective {variant} backend source changed across seeds")
    if missing or len(pairs) < len(SEEDS) and not errors:
        status, exit_code = "incomplete", 2
    elif errors:
        status, exit_code = "invalid", 3
    elif not all(pair["all_quality_gates_passed"] for pair in pairs):
        status, exit_code = "quality_failed", 4
    else:
        status, exit_code = "passed", 0
    speedups = {}
    if len(pairs) == len(SEEDS) and not errors and not missing:
        for key in pairs[0]["speedup_before_over_current"]:
            values = [pair["speedup_before_over_current"][key] for pair in pairs]
            speedups[key] = dict(median=statistics.median(values), min=min(values), max=max(values), per_seed=values)
    for run in runs:
        run.pop("batch_hashes", None)  # Their complete sequence was compared; retain its digest.
    return dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status=status, exit_code=exit_code, final_success=status == "passed",
                expected_seeds=list(SEEDS), expected_variants=list(VARIANTS), expected_runs=6,
                completed_comparable_pairs=sum(pair["comparable"] for pair in pairs),
                missing=missing, integrity_errors=errors, runs=runs, pairs=pairs,
                paired_speedup_summary=speedups,
                interpretation=[
                    "Four predeclared quality gates apply separately to every seed: current final RGB/Y PSNR may fall at most 0.05 dB and RGB/Y SSIM at most 0.001 relative to before. No averaging can rescue a failed seed.",
                    "Model changes are evidenced by canonical tensor-state hashes, independently of optimizer state. Checkpoint file hashes are verified without loading PyTorch.",
                    "Each paired run must share all 250 exact CPU batch hashes, validation payloads, initialization/checkpoint/split hashes and worker/data/metric source hashes. Effective C++/model source is checked within each variant, not equated between different backends.",
                    "Step wall includes H2D, microbatch accumulation, forward/backward, finite/norm checks and Adam. Data+step totals also include CPU decode/degradation/hash time, excluding evaluation/checkpoint/logging I/O.",
                    "Warm step median discards exactly the first five steps and uses steps 6-250. The all-250 median remains in JSON; data+step and loop totals retain all 250 steps including startup.",
                    "Loop wall separately includes evaluation and run I/O. Setup is separate and includes initial checkpoint writing; reported checkpoint wall includes that initial write and is not a term to subtract blindly from loop wall.",
                    "All timing is unprofiled. Speedups are per-seed before/current ratios; summary ranges span the three paired ratios, not confidence intervals. No ratios are multiplied.",
                    "This is 250-step fine-tuning, about 1.11 sampled training epochs at effective batch four, not proof of convergence or an external benchmark. Previous independent FP64/synthetic gates remain separate.",
                ])


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}" if value is not None else "missing"


def markdown(report):
    lines = ["# Paired USRNet short fine-tuning audit", "", f"Status: **{report['status']}**. Expected six runs, seeds 17/29/43, 250 optimizer steps each.", ""]
    if report["missing"]:
        lines += ["Missing or unfinished evidence:", "", *("- " + item for item in report["missing"]), ""]
    if report["integrity_errors"]:
        lines += ["Integrity/comparability failures:", "", *("- " + item for item in report["integrity_errors"]), ""]
    lines += ["## Initial and final quality", "", "All 100 held-out fixed crops; each cell is initial -> final.", "",
              "| Seed | Variant | RGB PSNR dB | Y PSNR dB | RGB SSIM | Y SSIM | Model changed |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        if "quality" not in run or "final" not in run["quality"]:
            continue
        cells = [f"{fmt(run['quality']['initial'][space][name])} -> {fmt(run['quality']['final'][space][name])}"
                 for space, name in (("rgb", "psnr_db"), ("y", "psnr_db"), ("rgb", "ssim"), ("y", "ssim"))]
        lines.append("| " + " | ".join([str(run["seed"]), run["variant"], *cells, str(run.get("parameter_state_changed"))]) + " |")
    lines += ["", "## Per-seed final quality gates", "", "Deltas are current minus before. PSNR >= -0.05 dB and SSIM >= -0.001 are required for each metric and seed.", "",
              "| Seed | RGB PSNR delta | Y PSNR delta | RGB SSIM delta | Y SSIM delta | All four pass | Comparable |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for pair in report["pairs"]:
        cells = [fmt(pair["final_quality_gates"][name]["current_minus_before"], 6)
                 for name in ("rgb_psnr_db", "y_psnr_db", "rgb_ssim", "y_ssim")]
        lines.append("| " + " | ".join([str(pair["seed"]), *cells, str(pair["all_quality_gates_passed"]), str(pair["comparable"])]) + " |")
    lines += ["", "## Time and memory", "", "Warm step median uses steps 6-250 and includes checks, H2D and optimizer. Data+step and loop totals include all 250 steps; loop wall also includes evaluation and I/O. Allocated/reserved are separate totals, not additive.", "",
              "| Seed | Variant | Warm step median ms | Data+steps s | Loop incl. eval/I/O s | Overall allocated GiB | Overall reserved GiB |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        if "timing" not in run:
            continue
        timing, memory = run["timing"], run["peak_memory"]["overall_peak_memory"]
        lines.append(f"| {run['seed']} | {run['variant']} | {fmt(timing['warm_step_wall_median_ms'], 3)} | {fmt(timing['data_plus_step_total_s'], 3)} | {fmt(timing['loop_including_evaluation_io_s'], 3)} | {memory['allocated_bytes'] / 2**30:.3f} | {memory['reserved_bytes'] / 2**30:.3f} |")
    lines += ["", "## Paired speed ratios", "", "Before/current; >1 means less current time. These do not override failed quality gates.", "",
              "| Seed | Warm step median | Data+steps total | Loop incl. eval/I/O |", "| --- | --- | --- | --- |"]
    for pair in report["pairs"]:
        values = pair["speedup_before_over_current"]
        lines.append(f"| {pair['seed']} | {values['warm_step_wall_median_ms']:.4f}x | {values['data_plus_step_total_s']:.4f}x | {values['loop_including_evaluation_io_s']:.4f}x |")
    for name, row in report["paired_speedup_summary"].items():
        if name == "step_wall_median_ms":
            continue
        lines += ["", f"{name}: median {row['median']:.4f}x; per-seed range [{row['min']:.4f}, {row['max']:.4f}]."]
    lines += ["", "This short real-photo fine-tuning comparison does not establish convergence or replace existing FP64/stress tests.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", nargs="*", help="Optional six run directories/globs; default uses root/{variant}_seed{seed}")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/dataset_training")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dataset_training/paired_summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "artifacts/dataset_training/paired_summary.md")
    args = parser.parse_args()
    directories = {(seed, variant): (args.root / f"{variant}_seed{seed}").resolve() for seed in SEEDS for variant in VARIANTS}
    if args.runs:
        occupied = set()
        for pattern in args.runs:
            matches = sorted(glob.glob(pattern))
            if not matches:
                parser.error(f"No run directories match {pattern}")
            for item in matches:
                path = Path(item).resolve()
                if path.name == "run.json":
                    path = path.parent
                try:
                    config = read_json(path / "run.json")["config"]
                    slot = (config["seed"], config["variant"])
                except (OSError, ValueError, KeyError) as error:
                    parser.error(f"Cannot identify run {path}: {error}")
                if slot not in directories or slot in occupied:
                    parser.error(f"Unexpected or duplicate run slot: {slot}")
                occupied.add(slot)
                directories[slot] = path
    for output in (args.output.resolve(), args.markdown.resolve()):
        if any(output.is_relative_to(directory) for directory in directories.values()):
            parser.error("Summary outputs must be outside the read-only individual run directories")
    report = audit(directories)
    report["summarizer_sha256"] = file_hash(Path(__file__))
    for path, content in ((args.output, json.dumps(clean(report), indent=2, allow_nan=False)),
                          (args.markdown, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print("Saved", path)
    print(f"Status: {report['status']}; complete comparable pairs: {report['completed_comparable_pairs']}/3; missing items: {len(report['missing'])}; integrity errors: {len(report['integrity_errors'])}")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
