"""Stopping and checkpoint state for auditable, opt-in long training runs."""
import datetime
import hashlib
import json
import math
import random
import time


CHECKPOINT_VERSION = 1
STABILITY_MIN_UPDATES = 1000
STABILITY_INTERVAL = 250
STABILITY_WINDOW = 5


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def recipe_hash(config, *, resume=False):
    ignored = {"run_dir", "variant", "verbose_build", "root", "build", "purpose",
               "resume", "max_wall_seconds", "deadline_utc"}
    if resume:
        # The maximum is a stopping budget, not an optimizer or data parameter.
        ignored.add("steps")
    return hash_json({key: value for key, value in config.items() if key not in ignored})


def parse_deadline(value):
    if value is None:
        return None
    deadline = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline-utc needs an explicit timezone, for example 2026-09-25T12:02:31Z")
    return deadline.astimezone(datetime.timezone.utc)


def budget_reason(started, max_wall_seconds, deadline, *, monotonic_now=None, utc_now=None):
    if deadline is not None:
        now = datetime.datetime.now(datetime.timezone.utc) if utc_now is None else utc_now
        if now >= deadline:
            return "deadline_utc"
    if max_wall_seconds is not None:
        now = time.perf_counter() if monotonic_now is None else monotonic_now
        if now - started >= max_wall_seconds:
            return "max_wall_seconds"
    return None


def stability(history, updates):
    expected_steps = list(range(updates - STABILITY_INTERVAL * (STABILITY_WINDOW - 1),
                                updates + 1, STABILITY_INTERVAL))
    window = history[-STABILITY_WINDOW:]
    eligible = (updates >= STABILITY_MIN_UPDATES and updates % STABILITY_INTERVAL == 0
                and [row["optimizer_steps"] for row in window] == expected_steps)
    result = dict(satisfied=False, eligible=eligible, minimum_updates=STABILITY_MIN_UPDATES,
                  evaluation_interval=STABILITY_INTERVAL, window_evaluations=STABILITY_WINDOW,
                  max_psnr_span_db=0.02, max_ssim_span=0.0005,
                  window_steps=[row["optimizer_steps"] for row in window], spans={})
    if not eligible:
        return result
    checks = []
    for space in ("rgb", "y"):
        for metric, limit in (("psnr_db", .02), ("ssim", .0005)):
            values = [float(row[space][metric]) for row in window]
            finite = all(math.isfinite(value) for value in values)
            span = max(values) - min(values) if finite else None
            result["spans"][f"{space}_{metric}"] = span
            checks.append(finite and span < limit)
    result["satisfied"] = all(checks)
    return result


def capture_rng(*, cuda=True):
    import numpy as np
    import torch
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if cuda else [],
                python=random.getstate(),
                numpy=dict(name=name, keys=keys.tolist(), position=position,
                           has_gauss=has_gauss, cached_gaussian=cached_gaussian))


def restore_rng(state, *, cuda=True):
    import numpy as np
    import torch
    if cuda and len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG device count differs from this process")
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))


def validate_resume(payload, report):
    if payload.get("resume_state_version") != CHECKPOINT_VERSION:
        raise ValueError("Resume requires a roadmap checkpoint with complete RNG/data state")
    if payload.get("config_sha256") != hash_json(payload["config"]):
        raise ValueError("Checkpoint config hash is inconsistent")
    if payload.get("resume_recipe_sha256") != recipe_hash(payload["config"], resume=True):
        raise ValueError("Checkpoint resume recipe hash is inconsistent")
    for key in ("resume_recipe_sha256", "source_sha256", "split_sha256", "input_checkpoint_sha256"):
        if payload.get(key) != report[key]:
            raise ValueError(f"Resume identity mismatch: {key}")
    for key in ("manifest_sha256", "kernels_sha256"):
        if payload.get("dataset", {}).get(key) != report["dataset"][key]:
            raise ValueError(f"Resume data mismatch: {key}")
    if "environment" in report and payload.get("environment") != report["environment"]:
        raise ValueError("Resume environment or algorithm lane differs")
    if "backend" in report and payload.get("backend_manifest") != report["backend"]["build_manifest"]:
        raise ValueError("Resume checked build inputs or binary identity differs")
    if payload["config"].get("variant") != report["config"]["variant"]:
        raise ValueError("Resume must keep the same comparison variant")
    updates = payload.get("optimizer_steps")
    if (type(updates) is not int or updates < 0 or type(payload.get("next_data_step")) is not int
            or payload["next_data_step"] != updates):
        raise ValueError("Checkpoint optimizer and next-data counters are inconsistent")
    if not all(key in payload for key in ("state_dict", "optimizer_state_dict", "rng_state", "metric_history",
                                         "origin_initial_state_tensor_sha256", "parent_run_dir", "run_status")):
        raise ValueError("Checkpoint lacks complete optimizer/RNG/metric state")
    steps = [row["optimizer_steps"] for row in payload["metric_history"]]
    if any(type(step) is not int or step < 0 or step > updates for step in steps):
        raise ValueError("Checkpoint metric history contains invalid step counters")
    if steps != sorted(set(steps)):
        raise ValueError("Checkpoint metric history is not strictly ordered")
    if (updates > 0 and not steps) or steps and steps[0] != 0:
        raise ValueError("Checkpoint lost the beginning of its metric trajectory")
    return updates
