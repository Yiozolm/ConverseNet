"""Offline audit of three paired long/resumed USRNet training trajectories.

No CUDA APIs, compilation, training, or checkpoint mutation occurs. Optional
checkpoint comparisons use torch.load(map_location='cpu', weights_only=True).
Exit: 0 quality gates passed, 2 incomplete evidence, 3 integrity/comparability
failure, 4 quality gate failure. A passed audit never asserts convergence.
"""
import argparse
from collections import defaultdict
import datetime
import glob
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import statistics


SEEDS = (17, 29, 43)
VARIANTS = ("before", "current")
CLOSED = {"complete", "stable", "max_steps_reached", "budget_stopped"}
SHARED_SOURCES = ("tools/roadmap_quality/train_usrnet_dataset.py",
                  "tools/roadmap_quality/usrnet_training_data.py",
                  "tools/roadmap_quality/evaluate_usrnet_quality.py",
                  "tools/roadmap_quality/run_state.py", "utils/utils_image.py",
                  "test/extension_loader.py")
IGNORED_RECIPE = {"run_dir", "variant", "verbose_build", "root", "build", "purpose",
                  "resume", "max_wall_seconds", "deadline_utc", "steps"}
HEX = re.compile(r"^[0-9a-f]{64}$")


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def normalized_recipe(config, *, omit_seed=False):
    ignored = IGNORED_RECIPE | ({"seed"} if omit_seed else set())
    return {key: value for key, value in config.items() if key not in ignored}


def metrics(row):
    result = {}
    for space in ("rgb", "y"):
        result[space] = {}
        for name in ("psnr_db", "ssim"):
            value = float(row[space][name])
            if not math.isfinite(value) and not (name == "psnr_db" and value == math.inf):
                raise ValueError("invalid metric: " + space + "/" + name)
            result[space][name] = value
    return result


def stability(history, endpoint):
    window = history[-5:]
    expected = list(range(endpoint - 1000, endpoint + 1, 250))
    eligible = endpoint >= 1000 and endpoint % 250 == 0 and [row["optimizer_steps"] for row in window] == expected
    spans = {}
    if eligible:
        for space in ("rgb", "y"):
            for name in ("psnr_db", "ssim"):
                values = [row[space][name] for row in window]
                spans[space + "_" + name] = max(values) - min(values) if all(math.isfinite(v) for v in values) else None
    satisfied = bool(eligible and all(value is not None and value < (.02 if key.endswith("psnr_db") else .0005)
                                     for key, value in spans.items()))
    return {"eligible": eligible, "satisfied": satisfied, "window_steps": [row["optimizer_steps"] for row in window],
            "spans": spans, "interpretation": "Observed four-metric window only; not proof of convergence"}


def inspect_session(directory):
    directory = Path(directory).resolve()
    out = dict(directory=str(directory), status="missing", seed=None, variant=None, start_step=0, end_step=0,
               errors=[], missing=[], warnings=[], data_steps=[], evaluations=[], timings={}, checkpoints={},
               parent=None, identity={}, _run={}, _rows={}, _metrics={})
    def require(condition, message):
        if not condition:
            out["errors"].append(message)
    if not (directory / "run.json").is_file():
        out["missing"].append("Missing parent/session run.json")
        return out
    try:
        run = read_json(directory / "run.json")
        out["_run"] = run
        config, dataset = run["config"], run["dataset"]
        out.update(status=run["status"], seed=config["seed"], variant=config["variant"],
                   start_step=run.get("session_start_step", 0), end_step=run["optimizer_steps"],
                   stop_reason=run.get("stop_reason"), config=config,
                   created_utc=run.get("created_utc"), completed_utc=run.get("completed_utc"),
                   worker_error=run.get("error"))
        if out["status"] not in CLOSED:
            out["missing"].append("Session is not a closed successful/budget/stability boundary: " + out["status"])
        require(config.get("purpose") in ("formal", "pilot"), "Unknown run purpose")
        require(out["variant"] in VARIANTS and out["seed"] in SEEDS, "Unexpected seed or variant")
        require(run["config_sha256"] == hash_json(config), "Config hash mismatch")
        require(run["resume_recipe_sha256"] == hash_json(normalized_recipe(config)), "Resume recipe hash mismatch")
        comparison_recipe = {key: value for key, value in config.items() if key not in IGNORED_RECIPE - {"steps"}}
        require(run["comparison_recipe_sha256"] == hash_json(comparison_recipe), "Comparison recipe hash mismatch")
        start, end = out["start_step"], out["end_step"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end or end > 10_000_000:
            raise ValueError("Invalid or impractically large session update range")
        require(run.get("next_data_step") == end, "next_data_step differs from completed updates")
        require(run.get("session_optimizer_steps", end - start) == end - start, "Session update count mismatch")
        require(run.get("samples_seen") == end * config["batch_size"], "Global sample count mismatch")
        require(run.get("all_loss_and_grad_finite") is True, "Nonfinite loss/gradient flag")
        require(dataset.get("train_images") == 900 and dataset.get("validation_images") == 100
                and dataset.get("image_hashes_verified") is True, "Unverified or unexpected 900/100 split")
        sources = run["source_sha256"]
        for name in SHARED_SOURCES:
            require(digest(sources.get(name)), "Missing shared source identity: " + name)
        require(dataset.get("source_sha256") == sources.get("tools/roadmap_quality/usrnet_training_data.py"),
                "Dataset source identity mismatch")
        for name in ("split_sha256", "input_checkpoint_sha256", "origin_initial_state_tensor_sha256",
                     "initial_state_tensor_sha256"):
            require(digest(run.get(name)), "Invalid/missing " + name)
        require(digest(dataset.get("manifest_sha256")) and all(digest(value) for value in dataset["kernels_sha256"].values()),
                "Invalid data manifest/kernel hashes")
        backend = run["backend"]
        require(backend.get("kind") == "checked production checkout", "Unchecked backend")
        build = backend["build_manifest"]
        require(digest(build.get("binary_sha256")), "Missing compiled binary SHA256")
        for name, value in build["inputs"]["sources"].items():
            source_name = posixpath.normpath("Converse2D/torch_converse2d/" + name)
            require(sources.get(source_name) == value, "Build/source identity mismatch: " + source_name)
        environment = run["environment"]
        require(environment.get("tf32") is False and environment.get("amp") is False, "TF32/AMP not disabled")
        require(environment.get("deterministic_algorithms") == config.get("deterministic_algorithms"),
                "Algorithm lane differs from the recipe")
        protocol = run["protocol"]
        require(protocol.get("optimizer") == "Adam" and protocol.get("betas") == [.9, .999]
                and protocol.get("eps") == 1e-8 and protocol.get("weight_decay") == 0,
                "Unexpected optimizer settings")
        require(protocol.get("amp") is False and protocol.get("scheduler") is False
                and protocol.get("gradient_clipping") is False, "Unexpected AMP/scheduler/clipping")
        require(protocol.get("preregistered_paired_final_quality_gate") ==
                dict(max_psnr_drop_db=.05, max_ssim_drop=.001, spaces=["rgb", "y"], each_seed=True, all_steps_finite=True),
                "Recorded quality thresholds differ from the four fixed per-seed gates")
        out["identity"] = dict(source_sha256=sources, build_manifest=build, environment=environment,
            resume_recipe_sha256=run["resume_recipe_sha256"], recipe=normalized_recipe(config),
            split_sha256=run["split_sha256"], input_checkpoint_sha256=run["input_checkpoint_sha256"],
            origin_initial_state_tensor_sha256=run["origin_initial_state_tensor_sha256"],
            manifest_sha256=dataset["manifest_sha256"], kernels_sha256=dataset["kernels_sha256"],
            shared_source_sha256={name: sources[name] for name in SHARED_SOURCES},
            architecture=run.get("architecture"), parameter_count=run.get("parameter_count"),
            parameter_tensor_count=run.get("parameter_tensor_count"))
        parent = run.get("resume_parent")
        if parent is not None and not isinstance(parent, dict):
            raise ValueError("resume_parent must be an object or null")
        out["parent"] = parent
        if parent:
            require(start == parent.get("optimizer_steps"), "Parent checkpoint/update boundary mismatch")
            require(digest(parent.get("file_sha256")), "Missing resume checkpoint SHA256")
            valid_paths = isinstance(parent.get("path"), str) and isinstance(parent.get("run_dir"), str)
            require(valid_paths, "Missing/invalid resume parent path")
            if not valid_paths:
                out["invalid_parent_link"] = parent
                out["parent"] = None
        else:
            require(start == 0 and config.get("resume") is None, "Missing parent link for a resumed/nonzero-start session")
            require(run["origin_initial_state_tensor_sha256"] == run["initial_state_tensor_sha256"],
                    "Fresh session origin differs from its initial state")
        training = read_jsonl(directory / "training.jsonl")
        grouped = defaultdict(list)
        for row in training:
            step = row.get("data_step")
            if type(step) is not int or not start <= step < end:
                require(False, "Training row outside declared session data range")
                continue
            grouped[step].append(row)
            require(row.get("step") == step + 1 and row.get("optimizer_steps") == step + 1,
                    f"Training counters disagree at data step {step}")
            require(row.get("optimizer_applied") is True and row.get("loss_and_grad_finite") is True,
                    f"Skipped update or nonfinite gradient at data step {step}")
            require(finite(row.get("loss")) and finite(row.get("grad_l2_norm")), f"Invalid scalar at data step {step}")
            require(digest(row.get("batch_sha256")), f"Missing batch hash at data step {step}")
            require(row.get("gradient_tensor_count") == run.get("parameter_tensor_count"),
                    f"Incomplete parameter gradients at data step {step}")
            for key in ("training_step_wall_ms", "data_prepare_and_hash_wall_ms"):
                require(finite(row.get(key)) and row[key] >= 0, f"Invalid {key} at data step {step}")
        holes = sorted(set(range(start, end)) - grouped.keys())
        overlaps = {step: [row.get("batch_sha256") for row in values] for step, values in grouped.items() if len(values) != 1}
        if holes:
            out["missing"].append(f"Missing {len(holes)} training data steps")
        require(not overlaps, "Duplicate data steps within this session")
        out.update(holes=holes, overlaps=overlaps, _rows={step: values[0] for step, values in grouped.items()},
                   data_steps=[{"data_step": step, "batch_sha256": values[0].get("batch_sha256")} for step, values in sorted(grouped.items())])
        history = []
        for row in run.get("metric_history", []):
            step = row["optimizer_steps"]
            require(type(step) is int and 0 <= step <= end, "Invalid metric-history step")
            history.append({"optimizer_steps": step, **metrics(row)})
        history_steps = [row["optimizer_steps"] for row in history]
        require(history_steps == sorted(set(history_steps)), "Metric history is not strictly ordered")
        out["_metrics"] = {row["optimizer_steps"]: row for row in history}
        out["metric_history"] = history
        for row in read_jsonl(directory / "evaluations.jsonl"):
            step = row["optimizer_steps"]
            require(start <= step <= end, "Local evaluation outside session bounds")
            require(row.get("images") == 100 and len(row.get("per_image", [])) == 100,
                    f"Incomplete held-out evaluation at update {step}")
            ids = [item["id"] for item in row["per_image"]]
            require(len(set(ids)) == 100, f"Duplicate held-out IDs at update {step}")
            require(row.get("all_outputs_finite") is True and row.get("parameter_check", {}).get("finite") is True,
                    f"Nonfinite/unverified evaluation at update {step}")
            require(digest(row.get("validation_payload_sha256")), "Missing validation payload identity")
            value = metrics(row)
            for space in ("rgb", "y"):
                for name in ("psnr_db", "ssim"):
                    mean = statistics.mean(metrics(item)[space][name] for item in row["per_image"])
                    require(mean == value[space][name] or math.isclose(mean, value[space][name], rel_tol=0, abs_tol=1e-10),
                            f"Per-image mean mismatch at update {step}: {space}/{name}")
            require(out["_metrics"].get(step) == {"optimizer_steps": step, **value}, "History/evaluation metrics disagree")
            out["evaluations"].append({"optimizer_steps": step, **value,
                "validation_payload_sha256": row["validation_payload_sha256"], "image_ids": ids})
        require([row["optimizer_steps"] for row in out["evaluations"]] == run.get("evaluation_steps", []),
                "Local evaluation log/count mismatch")
        for name, row in run.get("checkpoints", {}).items():
            path = directory / (name + ".pth")
            entry = {**row, "path": str(path), "available": path.is_file()}
            require(digest(row.get("file_sha256")) and digest(row.get("state_tensor_sha256")), "Invalid checkpoint identity: " + name)
            if path.is_file():
                entry["actual_file_sha256"] = file_hash(path)
                require(entry["actual_file_sha256"] == row["file_sha256"], "Checkpoint file hash mismatch: " + name)
            else:
                out["warnings"].append("Checkpoint unavailable for independent CPU inspection: " + str(path))
            out["checkpoints"][name] = entry
        initial = out["checkpoints"].get("initial")
        final = out["checkpoints"].get("final")
        require(initial is not None and initial.get("optimizer_steps") == start
                and initial.get("state_tensor_sha256") == run["initial_state_tensor_sha256"], "Initial checkpoint boundary/hash mismatch")
        require(final is not None and final.get("optimizer_steps") == end, "Final checkpoint boundary missing/mismatched")
        timing = run["timing"]
        for source, target in (("total_training_step_wall_ms", "training_step_wall_s"),
                               ("data_prepare_and_hash_wall_ms", "data_prepare_wall_s")):
            values = [row.get("training_step_wall_ms" if source.startswith("total") else source, 0) for row in training]
            total = sum(values)
            require(finite(timing.get(source)) and math.isclose(total, timing[source], rel_tol=1e-9, abs_tol=1e-6),
                    "Aggregate/JSONL timing mismatch: " + source)
            out["timings"][target] = timing[source] / 1000
        for name in ("evaluation_wall_s", "checkpoint_wall_s", "setup_wall_s", "training_loop_end_to_end_wall_s"):
            require(finite(timing.get(name)) and timing[name] >= 0, "Missing/invalid timing: " + name)
            out["timings"][name] = timing[name]
        require(finite(run.get("process_elapsed_wall_s")) and run["process_elapsed_wall_s"] >= 0, "Invalid process elapsed wall time")
        out["timings"]["process_elapsed_wall_s"] = run["process_elapsed_wall_s"]
        for scope in ("training_peak_memory", "evaluation_peak_memory", "overall_peak_memory"):
            values = run.get(scope, {})
            if values:
                require(isinstance(values, dict) and all(finite(values.get(key)) and values[key] >= 0
                        for key in ("allocated_bytes", "reserved_bytes")), "Invalid measured memory scope: " + scope)
                out[scope] = values
        out["stability"] = stability(history, end)
        if out["status"] == "stable":
            require(out["stability"]["satisfied"] and config.get("stop_when_stable") is True,
                    "Stable status does not satisfy the fixed four-metric window")
    except (OSError, ValueError, KeyError, TypeError, statistics.StatisticsError) as error:
        target = out["missing"] if out["status"] not in CLOSED else out["errors"]
        target.append(f"Cannot fully audit session: {type(error).__name__}: {error}")
    if type(out["end_step"]) is not int or not 0 <= out["end_step"] <= 10_000_000:
        out["start_step"] = out["end_step"] = 0
    if out["status"] in ("initializing", "running") and out["errors"]:
        # JSONL is appended before the worker atomically replaces run.json.
        # A live snapshot can therefore be inconsistent without corruption.
        out["missing"].extend("Provisional live snapshot: " + value for value in out["errors"])
        out["errors"].clear()
    return out


def discover(paths, root=None):
    selected = set()
    candidates = [] if root is None else list(Path(root).resolve().rglob("run.json"))
    for value in paths:
        matches = glob.glob(str(value), recursive=True) or [str(value)]
        for match in matches:
            path = Path(match).resolve()
            if path.name == "run.json":
                candidates.append(path)
            elif (path / "run.json").is_file():
                candidates.append(path / "run.json")
            elif path.is_dir():
                candidates.extend(path.rglob("run.json"))
            else:
                candidates.append(path / "run.json")
    selected.update(str(path.parent.resolve()) for path in candidates)
    return sorted(selected)


def load_sessions(directories):
    sessions, queue = {}, list(directories)
    while queue:
        key = str(Path(queue.pop()).resolve())
        if key in sessions:
            continue
        session = inspect_session(key)
        sessions[key] = session
        parent = session.get("parent")
        if parent and parent.get("run_dir"):
            queue.append(parent["run_dir"])
        elif parent:
            session["errors"].append("Resume link lacks parent run_dir")
    return sessions


def merge_chain(leaf, sessions):
    path, seen, current = [], set(), leaf
    errors, missing, warnings = [], [], []
    while current is not None:
        if current in seen:
            errors.append("Resume-parent cycle: " + current)
            break
        seen.add(current)
        session = sessions[current]
        path.append(session)
        parent = session.get("parent")
        current = str(Path(parent["run_dir"]).resolve()) if parent and parent.get("run_dir") else None
    path.reverse()
    end = path[-1]["end_step"]
    rows, metric_map, eval_map, all_hashes = {}, {}, {}, defaultdict(list)
    identity = next((session["identity"] for session in path if session["identity"]), {})
    for index, session in enumerate(path):
        prefix = session["directory"] + ": "
        errors.extend(prefix + value for value in session["errors"])
        missing.extend(prefix + value for value in session["missing"])
        warnings.extend(prefix + value for value in session["warnings"])
        if session["identity"] and session["identity"] != identity:
            errors.append(prefix + "Source/build/data/recipe/environment/origin changed across sessions")
        if index:
            parent = path[index - 1]
            link = session["parent"]
            if parent["identity"] and parent["end_step"] != session["start_step"]:
                errors.append(prefix + "Parent endpoint and resumed start differ (gap or rewind)")
            name = Path(link.get("path", "")).stem
            record = parent["checkpoints"].get(name)
            if record is None:
                missing.append(prefix + "Parent checkpoint record unavailable: " + name)
            elif (record["file_sha256"] != link["file_sha256"] or record["optimizer_steps"] != session["start_step"]
                  or record["state_tensor_sha256"] != session["_run"].get("initial_state_tensor_sha256")):
                errors.append(prefix + "Parent checkpoint/child initial-state identity mismatch")
            if parent["identity"] and (parent["variant"] != session["variant"] or parent["seed"] != session["seed"]):
                errors.append(prefix + "Resume changed variant or seed")
        for step, row in session["_rows"].items():
            all_hashes[step].append({"session": session["directory"], "batch_sha256": row.get("batch_sha256")})
            rows.setdefault(step, row)
        for step, metric in session["_metrics"].items():
            if step in metric_map and metric_map[step] != metric:
                errors.append(prefix + f"Conflicting metric history at update {step}")
            metric_map[step] = metric
        for metric in session["evaluations"]:
            step = metric["optimizer_steps"]
            if step in eval_map and eval_map[step] != metric:
                errors.append(prefix + f"Conflicting completed evaluations at update {step}")
            eval_map[step] = metric
    holes = sorted(set(range(end)) - rows.keys())
    overlaps = [{"data_step": step, "occurrences": values,
                 "all_hashes_equal": len({value["batch_sha256"] for value in values}) == 1}
                for step, values in sorted(all_hashes.items()) if len(values) > 1]
    if holes:
        missing.append(f"Chain has {len(holes)} holes in data steps [0,{end})")
    if overlaps:
        errors.append(f"Chain has {len(overlaps)} overlapping data steps; no silent truncation/deduplication is admitted")
    for step in metric_map:
        if step not in eval_map:
            missing.append(f"Metric at update {step} lacks its completed 100-image evaluation record")
    interval = path[-1].get("config", {}).get("eval_every")
    expected_evaluations = ([0, *range(interval, end + 1, interval)]
                            if type(interval) is int and interval > 0 else [])
    if not expected_evaluations:
        target = missing if not path[-1].get("config") else errors
        target.append("Missing/invalid evaluation interval")
    absent_evaluations = sorted(set(expected_evaluations) - eval_map.keys())
    if absent_evaluations:
        missing.append("Missing scheduled evaluations at updates: " + str(absent_evaluations))
    payloads = {row["validation_payload_sha256"] for row in eval_map.values()}
    if len(payloads) > 1:
        errors.append("Validation payload changed across sessions/evaluations")
    ids = {hash_json(sorted(row["image_ids"])) for row in eval_map.values()}
    if len(ids) > 1:
        errors.append("Held-out image IDs changed across sessions")
    latest = eval_map[max(eval_map)] if eval_map else None
    if latest is None or latest["optimizer_steps"] != end:
        missing.append("Endpoint has no completed evaluation; latest earlier metrics are reported but cannot pass the final gate")
    totals = {key: sum(session["timings"].get(key, 0) for session in path)
              for key in {key for session in path for key in session["timings"]}}
    metadata_gaps = []
    for parent, child in zip(path, path[1:]):
        row = {"parent": parent["directory"], "child": child["directory"],
               "parent_completed_utc": parent.get("completed_utc"), "child_created_utc": child.get("created_utc")}
        try:
            before = datetime.datetime.fromisoformat(row["parent_completed_utc"].replace("Z", "+00:00"))
            after = datetime.datetime.fromisoformat(row["child_created_utc"].replace("Z", "+00:00"))
            if before.tzinfo is None or after.tzinfo is None:
                raise ValueError("naive timestamp")
            row["metadata_interval_seconds"] = (after - before).total_seconds()
        except (AttributeError, TypeError, ValueError):
            row["metadata_interval_seconds"] = None
            warnings.append("Intersession timestamp interval unavailable: " + child["directory"])
        metadata_gaps.append(row)
    return {"seed": path[-1]["seed"], "variant": path[-1]["variant"], "leaf": leaf,
            "sessions": [session["directory"] for session in path], "endpoint_updates": end,
            "status": path[-1]["status"], "errors": errors, "missing": missing, "warnings": warnings,
            "identity": identity, "holes": holes, "overlaps": overlaps,
            "expected_scheduled_evaluations": expected_evaluations,
            "missing_scheduled_evaluations": absent_evaluations,
            "completed_evaluations": [eval_map[key] for key in sorted(eval_map)],
            "latest_evaluation": latest, "evaluation_lag_updates": end - latest["optimizer_steps"] if latest else None,
            "stability": stability([metric_map[key] for key in sorted(metric_map)], end),
            "summed_session_timings": totals,
            "intersession_metadata_intervals": metadata_gaps,
            "timing_interpretation": "Arithmetic sum of recorded process sessions excludes waits between sessions and includes repeated setup/checkpoint work. Timestamp intervals are metadata intervals, not exact downtime: created_utc is written after some initial preparation. Checkpoint time overlaps setup/loop/process totals; these categories must not be added together.",
            "_rows": rows, "_evals": eval_map, "_leaf": path[-1]}


def quality_gates(before, current):
    gates = {}
    for space in ("rgb", "y"):
        for name, tolerance in (("psnr_db", .05), ("ssim", .001)):
            b, c = before[space][name], current[space][name]
            delta = 0.0 if b == c else c - b
            gates[space + "_" + name] = {"before": b, "current": c, "current_minus_before": delta,
                                           "minimum_allowed_delta": -tolerance, "passed": delta >= -tolerance}
    return {"before_evaluated_update": before["optimizer_steps"],
            "current_evaluated_update": current["optimizer_steps"], "metrics": gates,
            "passed": all(value["passed"] for value in gates.values())}


def state_tensor_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(cpu.dtype), list(cpu.shape)]).encode())
        value.update(cpu.numpy().tobytes())
    return value.hexdigest()


def load_final_checkpoint(chain):
    result = {"status": "unavailable", "errors": [], "warnings": [], "_payload": None}
    leaf = chain["_leaf"]
    record = leaf["checkpoints"].get("final")
    if record is None or not record.get("available"):
        result["warnings"].append("Final checkpoint unavailable; no state/optimizer equality claim")
        return result
    result["path"] = record["path"]
    if record.get("actual_file_sha256") != record["file_sha256"]:
        result["status"] = "invalid"
        result["errors"].append("Final checkpoint SHA256 mismatch")
        return result
    try:
        import torch
        payload = torch.load(record["path"], map_location="cpu", weights_only=True)
        checks = {"optimizer_steps": chain["endpoint_updates"], "next_data_step": chain["endpoint_updates"],
                  "source_sha256": chain["identity"]["source_sha256"],
                  "backend_manifest": chain["identity"]["build_manifest"],
                  "origin_initial_state_tensor_sha256": chain["identity"]["origin_initial_state_tensor_sha256"],
                  "config_sha256": leaf["_run"]["config_sha256"]}
        for key, expected in checks.items():
            if payload.get(key) != expected:
                result["errors"].append("Checkpoint/report mismatch: " + key)
        actual_state_hash = state_tensor_hash(payload["state_dict"])
        if actual_state_hash != record["state_tensor_sha256"]:
            result["errors"].append("Checkpoint model tensor hash differs from its recorded canonical hash")
        if "optimizer_state_dict" not in payload:
            result["errors"].append("Missing optimizer state")
        if any(not bool(torch.isfinite(value).all()) for value in payload["state_dict"].values()):
            result["errors"].append("Nonfinite final model state")
        result.update(status="invalid" if result["errors"] else "loaded_on_cpu", model_tensor_sha256=actual_state_hash,
                      torch_version=str(torch.__version__), _payload=payload)
    except Exception as error:
        result["status"] = "unreadable"
        result["warnings"].append(f"CPU weights_only load/audit unavailable: {type(error).__name__}: {error}")
    return result


def compare_state_trees(before, current):
    import torch
    tensors, metadata, structure = [], [], []
    def visit(left, right, name):
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
                structure.append(name + ": tensor type/shape/dtype differs")
                return
            def raw(value):
                return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            a, b = raw(left), raw(right)
            row = {"path": name, "shape": list(left.shape), "dtype": str(left.dtype),
                   "before_sha256": hashlib.sha256(a).hexdigest(), "current_sha256": hashlib.sha256(b).hexdigest(),
                   "bitwise_equal": a == b}
            finite_values = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            row["finite"] = finite_values
            if finite_values:
                dtype = torch.complex128 if left.is_complex() else torch.float64
                high_left, high_right = left.to(dtype), right.to(dtype)
                difference = high_right - high_left
                row["max_abs_difference"] = difference.abs().max().item() if difference.numel() else 0.0
                row["relative_l2_difference"] = (difference.norm() / high_left.norm().clamp_min(1e-300)).item()
            tensors.append(row)
        elif isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                structure.append(name + ": mapping keys differ")
            for key in sorted(left.keys() & right.keys(), key=lambda key: (str(type(key)), repr(key))):
                visit(left[key], right[key], name + "[" + repr(key) + "]")
        elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                structure.append(name + ": sequence lengths differ")
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{name}[{index}]")
        elif type(left) is not type(right) or left != right:
            metadata.append({"path": name, "before": repr(left), "current": repr(right)})
    visit(before, current, "state")
    return {"tensor_count": len(tensors), "different_tensors": sum(not row["bitwise_equal"] for row in tensors),
            "all_tensors_bitwise_equal": all(row["bitwise_equal"] for row in tensors),
            "structure_equal": not structure, "metadata_equal": not metadata,
            "structure_differences": structure, "metadata_differences": metadata, "tensors": tensors}


def pair_chains(before, current, *, inspect_checkpoints=True):
    errors, warnings = [], []
    keys = ("resume_recipe_sha256", "split_sha256", "input_checkpoint_sha256", "origin_initial_state_tensor_sha256",
            "manifest_sha256", "kernels_sha256", "shared_source_sha256", "environment", "architecture",
            "parameter_count", "parameter_tensor_count")
    for key in keys:
        if before["identity"].get(key) != current["identity"].get(key):
            errors.append("Paired identity mismatch: " + key)
    b_rows, c_rows = before["_rows"], current["_rows"]
    shared = sorted(b_rows.keys() & c_rows.keys())
    compared = [{"data_step": step, "before_sha256": b_rows[step].get("batch_sha256"),
                 "current_sha256": c_rows[step].get("batch_sha256"),
                 "equal": b_rows[step].get("batch_sha256") == c_rows[step].get("batch_sha256")} for step in shared]
    mismatches = [row["data_step"] for row in compared if not row["equal"]]
    if mismatches:
        errors.append("Different exact data hashes at shared steps: " + str(mismatches))
    prefix = 0
    while prefix in b_rows and prefix in c_rows and b_rows[prefix].get("batch_sha256") == c_rows[prefix].get("batch_sha256"):
        prefix += 1
    common_evaluations = sorted(before["_evals"].keys() & current["_evals"].keys())
    payloads_before = {row["validation_payload_sha256"] for row in before["_evals"].values()}
    payloads_current = {row["validation_payload_sha256"] for row in current["_evals"].values()}
    if payloads_before != payloads_current:
        errors.append("Paired validation payload identity differs")
    final_available = before["evaluation_lag_updates"] == 0 and current["evaluation_lag_updates"] == 0
    final = quality_gates(before["latest_evaluation"], current["latest_evaluation"]) if final_available else None
    latest_common = quality_gates(before["_evals"][common_evaluations[-1]], current["_evals"][common_evaluations[-1]]) if common_evaluations else None
    state_audit = {"status": "not_requested", "interpretation": "State equality is diagnostic, never a substitute for quality gates"}
    if inspect_checkpoints:
        b_state, c_state = load_final_checkpoint(before), load_final_checkpoint(current)
        state_audit.update(before=clean(b_state), current=clean(c_state), status="unavailable")
        errors.extend("before: " + value for value in b_state["errors"])
        errors.extend("current: " + value for value in c_state["errors"])
        warnings.extend(b_state["warnings"] + c_state["warnings"])
        if b_state["status"] == c_state["status"] == "loaded_on_cpu":
            state_audit.update(status="compared_on_cpu", same_update_count=before["endpoint_updates"] == current["endpoint_updates"],
                model=compare_state_trees(b_state["_payload"]["state_dict"], c_state["_payload"]["state_dict"]),
                optimizer=compare_state_trees(b_state["_payload"]["optimizer_state_dict"], c_state["_payload"]["optimizer_state_dict"]))
    prefix_times = {}
    for name, rows in (("before", b_rows), ("current", c_rows)):
        prefix_times[name] = {"training_step_wall_s": sum(rows[i]["training_step_wall_ms"] for i in range(prefix)) / 1000,
                              "data_prepare_wall_s": sum(rows[i]["data_prepare_and_hash_wall_ms"] for i in range(prefix)) / 1000}
    return {"seed": before["seed"], "errors": errors, "warnings": warnings,
            "before_endpoint_updates": before["endpoint_updates"], "current_endpoint_updates": current["endpoint_updates"],
            "matching_data_prefix_steps": prefix, "shared_data_steps": compared,
            "unshared_before_steps": sorted(b_rows.keys() - c_rows.keys()),
            "unshared_current_steps": sorted(c_rows.keys() - b_rows.keys()),
            "final_quality_gate": final, "latest_common_evaluation_gate": latest_common,
            "quality_interpretation": "Final endpoints may have different update counts. The common-evaluation comparison is separate; no averaging rescues a failed seed and earlier metrics cannot replace an unevaluated endpoint.",
            "common_prefix_timings": prefix_times, "state_comparison": state_audit,
            "convergence_claim": False}


def audit(directories, *, inspect_checkpoints=True):
    sessions = load_sessions(directories)
    errors, missing, warnings = [], [], []
    # Root discovery may encounter unrelated capacity/resume pilots. Keep them
    # visible, but only formal endpoints (and every linked ancestor) contribute
    # to the six-trajectory quality audit. A pilot ancestor is not discarded.
    relevant = {key for key, session in sessions.items() if session.get("config", {}).get("purpose") != "pilot"}
    queue = list(relevant)
    while queue:
        session = sessions[queue.pop()]
        parent = session.get("parent")
        if parent and parent.get("run_dir"):
            key = str(Path(parent["run_dir"]).resolve())
            if key not in relevant:
                relevant.add(key)
                queue.append(key)
    parents = {str(Path(sessions[key]["parent"]["run_dir"]).resolve()) for key in relevant
               if sessions[key].get("parent") and sessions[key]["parent"].get("run_dir")}
    leaves = [key for key in relevant if key not in parents]
    if relevant and not leaves:
        errors.append("No leaf session: resume graph contains a cycle")
    chains = [merge_chain(key, sessions) for key in leaves]
    covered = {path for chain in chains for path in chain["sessions"]}
    if relevant - covered:
        errors.append("Unresolved/cyclic sessions without an auditable leaf: " + str(sorted(relevant - covered)))
    slots = defaultdict(list)
    for chain in chains:
        slots[(chain["seed"], chain["variant"])].append(chain)
        errors.extend(chain["errors"])
        missing.extend(chain["missing"])
        warnings.extend(chain["warnings"])
    selected = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            values = slots[(seed, variant)]
            if not values:
                missing.append(f"Missing trajectory: {variant}/seed{seed}")
            elif len(values) > 1:
                errors.append(f"Ambiguous competing leaf trajectories: {variant}/seed{seed}; no best/latest branch is selected")
            else:
                selected[(seed, variant)] = values[0]
    # Across seeds, each implementation must remain fixed. Data order changes
    # with seed, while the optimizer/data recipe and validation payload do not.
    for variant in VARIANTS:
        values = [selected[(seed, variant)] for seed in SEEDS if (seed, variant) in selected]
        for key in ("source_sha256", "build_manifest", "environment", "shared_source_sha256"):
            if len({hash_json(chain["identity"].get(key)) for chain in values}) > 1:
                errors.append(f"{variant}: {key} changed across seeds")
        recipes = {hash_json({k: v for k, v in chain["identity"].get("recipe", {}).items() if k != "seed"}) for chain in values}
        if len(recipes) > 1:
            errors.append(variant + ": optimizer/data recipe changed across seeds")
    all_selected = list(selected.values())
    for key in ("split_sha256", "input_checkpoint_sha256", "manifest_sha256", "kernels_sha256", "shared_source_sha256", "environment"):
        if len({hash_json(chain["identity"].get(key)) for chain in all_selected}) > 1:
            errors.append("Shared experimental identity differs across seeds/variants: " + key)
    pairs = []
    for seed in SEEDS:
        if all((seed, variant) in selected for variant in VARIANTS):
            before, current = (selected[(seed, variant)] for variant in VARIANTS)
            if not before["errors"] and not current["errors"]:
                pair = pair_chains(before, current, inspect_checkpoints=inspect_checkpoints)
                pairs.append(pair)
                errors.extend(f"seed{seed}: " + value for value in pair["errors"])
                warnings.extend(f"seed{seed}: " + value for value in pair["warnings"])
    if errors:
        status, exit_code = "invalid", 3
    elif missing or len(pairs) != len(SEEDS) or any(pair["final_quality_gate"] is None for pair in pairs):
        status, exit_code = "incomplete", 2
    elif not all(pair["final_quality_gate"]["passed"] for pair in pairs):
        status, exit_code = "quality_failed", 4
    else:
        status, exit_code = "quality_gates_passed", 0
    return clean({"kind": "paired_long_training_audit", "status": status, "exit_code": exit_code,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "auditor_sha256": file_hash(__file__),
        "expected_seeds": SEEDS, "expected_variants": VARIANTS,
        "errors": errors, "missing": missing, "warnings": sorted(set(warnings)),
        "sessions": list(sessions.values()), "trajectories": chains, "pairs": pairs,
        "excluded_sessions": [{"directory": key, "reason": "Unlinked pilot; not a formal endpoint or an ancestor of one"}
                              for key in sorted(set(sessions) - relevant)],
        "convergence_claim": False,
        "interpretation": ["Quality gates are per seed and per RGB/Y metric: PSNR drop <=0.05 dB and SSIM drop <=0.001.",
            "Source/build/data/recipe/environment identity must remain fixed within each variant's resumed chain; different production sources are allowed between before/current.",
            "State and optimizer equality are independent diagnostics. Permitted rounding differences never establish or override quality admission.",
            "All shared data-step hashes are shown. Holes, rewinds, overlaps, missing parent sessions and competing branches are not silently repaired.",
            "Endpoint update counts can differ. Their extra steps and the common exact-data prefix are reported separately; full elapsed ratios would not be equal-work speedups.",
            "Session process elapsed times are summed honestly, including repeated setup. Training-step and data preparation are separate; evaluation/checkpoint/setup/loop are overlapping scopes and must not be added to process totals.",
            "stable is only the declared five-evaluation four-metric window; max_steps_reached, budget_stopped and unstable trajectories are not convergence evidence."]})


def markdown(report):
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    lines = ["# Paired long-training audit", "", f"Status: **{report['status']}**. This report makes no convergence claim.", ""]
    for title, key in (("Integrity/comparability errors", "errors"), ("Missing evidence", "missing"), ("Evidence limitations", "warnings")):
        if report[key]:
            lines += ["## " + title, "", *("- " + value for value in report[key]), ""]
    lines += ["## Trajectories", "", "| Seed | Variant | Sessions | Updates | Last evaluated | Status | Window satisfied | Process seconds |",
              "|---:|---|---:|---:|---:|---|---|---:|"]
    for chain in report["trajectories"]:
        last = chain["latest_evaluation"]
        lines.append(f"| {chain['seed']} | {chain['variant']} | {len(chain['sessions'])} | {chain['endpoint_updates']} | "
                     f"{last['optimizer_steps'] if last else 'missing'} | {chain['status']} | {chain['stability']['satisfied']} | "
                     f"{fmt(chain['summed_session_timings'].get('process_elapsed_wall_s'))} |")
    lines += ["", "## Per-seed output-quality gates", "", "Each of the four metrics must pass independently; terminal updates may differ.", "",
              "| Seed | Before/current evaluated updates | RGB PSNR delta | RGB SSIM delta | Y PSNR delta | Y SSIM delta | Passed |",
              "|---:|---|---:|---:|---:|---:|---|"]
    for pair in report["pairs"]:
        gate = pair["final_quality_gate"]
        if gate is None:
            lines.append(f"| {pair['seed']} | endpoint evaluation missing | — | — | — | — | unknown |")
        else:
            values = [fmt(gate["metrics"][key]["current_minus_before"]) for key in ("rgb_psnr_db", "rgb_ssim", "y_psnr_db", "y_ssim")]
            lines.append("| " + " | ".join([str(pair["seed"]), f"{gate['before_evaluated_update']}/{gate['current_evaluated_update']}", *values, str(gate["passed"])]) + " |")
    lines += ["", "## Data and checkpoint comparisons", ""]
    for pair in report["pairs"]:
        state = pair["state_comparison"]
        lines.append(f"- Seed {pair['seed']}: exact matching data prefix {pair['matching_data_prefix_steps']} updates; "
                     f"before-only/current-only updates {len(pair['unshared_before_steps'])}/{len(pair['unshared_current_steps'])}; "
                     f"checkpoint comparison: {state['status']}.")
        if state["status"] == "compared_on_cpu":
            lines.append(f"  Model tensors differing: {state['model']['different_tensors']}/{state['model']['tensor_count']}; "
                         f"optimizer tensors differing: {state['optimizer']['different_tensors']}/{state['optimizer']['tensor_count']}. "
                         "These counts do not decide quality admission.")
    lines += ["", "## Session timing", "", "| Session | Training step s | Data s | Evaluation s | Checkpoint s | Setup s | Loop s | Process s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for session in report["sessions"]:
        values = [fmt(session["timings"].get(key)) for key in ("training_step_wall_s", "data_prepare_wall_s", "evaluation_wall_s",
                  "checkpoint_wall_s", "setup_wall_s", "training_loop_end_to_end_wall_s", "process_elapsed_wall_s")]
        lines.append("| " + " | ".join([session["directory"], *values]) + " |")
    lines += ["", "Setup/checkpoint work repeats across resumed sessions. These scopes overlap; only the recorded process times are summed as process elapsed time.",
              "Summed process times exclude waits between sessions. Inter-session timestamp gaps in JSON are metadata intervals, not exact downtime, because created_utc is recorded after initial preparation.",
              "", "## Interpretation", "", *("- " + value for value in report["interpretation"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Run directories, run.json paths, or glob patterns")
    parser.add_argument("--root", type=Path, help="Recursively discover run.json beneath this directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, help="Defaults to --output with .md suffix")
    parser.add_argument("--skip-checkpoint-comparison", action="store_true", help="Keep checkpoint comparison explicitly unavailable")
    args = parser.parse_args()
    if not args.runs and args.root is None:
        parser.error("provide run paths and/or --root")
    md = args.markdown or args.output.with_suffix(".md")
    if args.output.resolve() == md.resolve() or args.output.exists() or md.exists():
        parser.error("choose distinct fresh JSON/Markdown outputs; previous reports are preserved")
    report = audit(discover(args.runs, args.root), inspect_checkpoints=not args.skip_checkpoint_comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "sessions": len(report["sessions"]),
                      "trajectories": len(report["trajectories"]), "pairs": len(report["pairs"]),
                      "errors": len(report["errors"]), "missing": len(report["missing"])}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
