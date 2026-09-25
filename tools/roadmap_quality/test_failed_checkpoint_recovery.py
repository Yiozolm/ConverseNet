"""CPU-only adversarial recovery fixtures; none of these are training evidence."""
import copy
import json
from pathlib import Path
import random
import unittest

import torch

import summarize_long_training as audit
from test_summarize_long_training import session, temporary_reports, write_json, write_rows


def update_config_hashes(run):
    run["config_sha256"] = audit.hash_json(run["config"])
    run["resume_recipe_sha256"] = audit.hash_json(audit.normalized_recipe(run["config"]))
    run["comparison_recipe_sha256"] = audit.hash_json({
        key: value for key, value in run["config"].items()
        if key not in audit.IGNORED_RECIPE - {"steps"}
    })


def checkpoint_payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(directory, name, payload):
    path = directory / (name + ".pth")
    torch.save(payload, path)
    run = audit.read_json(directory / "run.json")
    run["checkpoints"][name] = {
        "path": str(path), "optimizer_steps": payload["optimizer_steps"],
        "file_sha256": audit.file_hash(path),
        "state_tensor_sha256": audit.state_tensor_hash(payload["state_dict"]),
    }
    write_json(directory / "run.json", run)


def resume_payload(payload, run, directory):
    step = payload["optimizer_steps"]
    history = [row for row in run["metric_history"] if row["optimizer_steps"] <= step]
    payload.update(
        resume_state_version=1, config=copy.deepcopy(run["config"]),
        config_sha256=run["config_sha256"], resume_recipe_sha256=run["resume_recipe_sha256"],
        split_sha256=run["split_sha256"], dataset=copy.deepcopy(run["dataset"]),
        environment=copy.deepcopy(run["environment"]),
        input_checkpoint_sha256=run["input_checkpoint_sha256"],
        origin_initial_state_tensor_sha256=run["origin_initial_state_tensor_sha256"],
        source_sha256=copy.deepcopy(run["source_sha256"]),
        backend_manifest=copy.deepcopy(run["backend"]["build_manifest"]),
        next_data_step=step, parent_run_dir=str(directory),
        session_start_step=run["session_start_step"], run_status="running", stop_reason=None,
        metric_history=copy.deepcopy(history),
        metrics={key: copy.deepcopy(history[-1][key]) for key in ("rgb", "y")},
        rng_state={
            "torch_cpu": torch.Generator(device="cpu").manual_seed(17).get_state(),
            "torch_cuda": [torch.tensor([17] + [0] * 15, dtype=torch.uint8)],
            "python": random.Random(17).getstate(),
            "numpy": {"name": "MT19937", "keys": list(range(624)), "position": 624,
                      "has_gauss": 0, "cached_gaussian": 0.0},
        },
    )
    for state in payload["optimizer_state_dict"]["state"].values():
        state["exp_avg_sq"] = torch.tensor([step * .01])
    for group in payload["optimizer_state_dict"]["param_groups"]:
        group.update(betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False,
                     maximize=False, foreach=False, capturable=False, differentiable=False, fused=False,
                     decoupled_weight_decay=False)
    return payload


def write_manifest(parent, child, target):
    run = audit.read_json(parent / "run.json")
    payload = checkpoint_payload(parent / "latest.pth")
    authorization = {
        "parent_run_dir": str(parent), "child_run_dir": str(child),
        "seed": run["config"]["seed"], "variant": run["config"]["variant"],
        "optimizer_steps": run["optimizer_steps"], "checkpoint_name": "latest",
        "files_sha256": {name: audit.file_hash(parent / name) for name in (
            "run.json", "training.jsonl", "evaluations.jsonl", "latest.pth", "initial.pth")},
        "error_sha256": audit.hash_json(run["error"]),
        "identity_sha256": audit.hash_json(audit.inspect_session(parent)["identity"]),
        "process_elapsed_wall_s": run["process_elapsed_wall_s"],
        "model_state_sha256": audit.state_tensor_hash(payload["state_dict"]),
        "optimizer_state_sha256": audit.tree_hash(payload["optimizer_state_dict"]),
        "rng_state_sha256": audit.tree_hash(payload["rng_state"]),
    }
    write_json(target, {"kind": "explicit_failed_checkpoint_recovery", "version": 1,
                        "authorizations": [authorization]})
    return target


def recovery_fixture(root, *, child_end=4):
    parent = session(root, 17, "before", 0, 2)
    child = session(root, 17, "before", 2, child_end, parent=parent)
    others = [session(root, seed, variant, 0, 4)
              for seed in audit.SEEDS for variant in audit.VARIANTS
              if (seed, variant) != (17, "before")]
    run = audit.read_json(parent / "run.json")
    run["config"]["steps"] = 4
    update_config_hashes(run)
    run["status"] = "failed"
    run["completed_utc"] = None
    run["last_evaluation"] = copy.deepcopy(run["metric_history"][-1])
    message = f"[WinError 5] Access is denied: {str(parent / 'run.json.tmp')!r} -> {str(parent / 'run.json')!r}"
    run["error"] = {
        "type": "PermissionError", "message": message,
        "traceback": 'Traceback (most recent call last):\n'
                     '  File "tools/roadmap_quality/train_usrnet_dataset.py", line 538, in execute\n'
                     '    write_json(args.run_dir / "run.json", report)\n'
                     '  File "tools/roadmap_quality/train_usrnet_dataset.py", line 72, in write_json\n'
                     '    temporary.replace(path)\n'
                     '  File "pathlib.py", line 1376, in replace\n'
                     '    os.replace(self, target)\n'
                     'PermissionError: ' + message + '\n',
    }
    parent_payload = resume_payload(checkpoint_payload(parent / "final.pth"), run, parent)
    del run["checkpoints"]["final"]
    write_json(parent / "run.json", run)
    save_checkpoint(parent, "latest", parent_payload)
    (parent / "final.pth").unlink()
    # The original initial checkpoint remains an immutable bound input.
    run = audit.read_json(child / "run.json")
    run["config"]["resume"] = str(parent / "latest.pth")
    update_config_hashes(run)
    run["resume_parent"].update(path=str(parent / "latest.pth"), previous_status="running",
                                file_sha256=audit.file_hash(parent / "latest.pth"))
    write_json(child / "run.json", run)
    initial = resume_payload(checkpoint_payload(child / "initial.pth"), run, child)
    initial["rng_state"] = copy.deepcopy(parent_payload["rng_state"])
    initial["optimizer_state_dict"] = copy.deepcopy(parent_payload["optimizer_state_dict"])
    save_checkpoint(child, "initial", initial)
    final = resume_payload(checkpoint_payload(child / "final.pth"), run, child)
    save_checkpoint(child, "final", final)
    manifest = write_manifest(parent, child, root / "recovery.json")
    return parent, child, [child, *others], manifest


class FailedCheckpointRecoveryCPU(unittest.TestCase):
    def assert_rejected(self, paths, manifest):
        result = audit.audit(paths, inspect_checkpoints=False, recovery_manifests=[manifest])
        diagnostics = {key: result[key] for key in ("status", "errors", "missing", "failure_recoveries")}
        self.assertNotEqual(result["status"], "quality_gates_passed", diagnostics)
        self.assertTrue(result["errors"] or result["missing"], diagnostics)
        json.dumps(result, allow_nan=False)
        return result

    def test_explicit_recovery_preserves_failed_record_and_full_process_time(self):
        with temporary_reports() as temporary:
            parent, child, paths, manifest = recovery_fixture(Path(temporary))
            before = {name: audit.file_hash(parent / name) for name in
                      ("run.json", "training.jsonl", "evaluations.jsonl", "latest.pth", "initial.pth")}
            result = audit.audit(paths, inspect_checkpoints=False, recovery_manifests=[manifest])
            self.assertEqual(result["status"], "quality_gates_passed", result["errors"] + result["missing"])
            failed = next(row for row in result["sessions"] if row["directory"] == str(parent))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["worker_error"], audit.read_json(parent / "run.json")["error"])
            chain = next(row for row in result["trajectories"] if row["seed"] == 17 and row["variant"] == "before")
            total = sum(audit.read_json(path / "run.json")["process_elapsed_wall_s"] for path in (parent, child))
            self.assertEqual(chain["summed_session_timings"]["process_elapsed_wall_s"], total)
            self.assertEqual(chain["sessions"], [str(parent), str(child)])
            self.assertEqual(chain["holes"], [])
            self.assertEqual(chain["overlaps"], [])
            self.assertFalse(result["convergence_claim"])
            self.assertEqual(before, {name: audit.file_hash(parent / name) for name in before})

    def test_unapproved_or_terminal_failed_session_cannot_pass(self):
        with temporary_reports() as temporary:
            parent, child, paths, manifest = recovery_fixture(Path(temporary))
            result = audit.audit(paths, inspect_checkpoints=False)
            self.assertNotEqual(result["status"], "quality_gates_passed")
            self.assert_rejected([parent, *paths[1:]], manifest)

    def test_closed_child_without_additional_updates_cannot_launder_failed_endpoint(self):
        with temporary_reports() as temporary:
            parent, child, paths, manifest = recovery_fixture(Path(temporary), child_end=2)
            self.assert_rejected(paths, manifest)

    def test_manifest_bound_files_and_fields_cannot_change(self):
        mutations = {
            "run": lambda p, c, m: (p / "run.json").write_text((p / "run.json").read_text() + "\n"),
            "training": lambda p, c, m: (p / "training.jsonl").write_text((p / "training.jsonl").read_text() + "\n"),
            "evaluations": lambda p, c, m: (p / "evaluations.jsonl").write_text((p / "evaluations.jsonl").read_text() + "\n"),
            "child_path": lambda p, c, m: m["authorizations"][0].update(child_run_dir=str(c.parent / "other_child")),
            "step": lambda p, c, m: m["authorizations"][0].update(optimizer_steps=1),
            "error_hash": lambda p, c, m: m["authorizations"][0].update(error_sha256="0" * 64),
            "identity_hash": lambda p, c, m: m["authorizations"][0].update(identity_sha256="0" * 64),
            "process_total": lambda p, c, m: m["authorizations"][0].update(process_elapsed_wall_s=0),
            "duplicate": lambda p, c, m: m["authorizations"].append(copy.deepcopy(m["authorizations"][0])),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), temporary_reports() as temporary:
                parent, child, paths, manifest = recovery_fixture(Path(temporary))
                document = audit.read_json(manifest)
                mutate(parent, child, document)
                write_json(manifest, document)
                self.assert_rejected(paths, manifest)

    def test_rebound_wrong_error_operation_or_code_is_rejected(self):
        cases = (("path", "latest.pth"), ("code", "[WinError 32]"), ("type", "RuntimeError"),
                 ("path_suffix", None), ("reverse_paths", None))
        for name, replacement in cases:
            with self.subTest(name=name), temporary_reports() as temporary:
                parent, child, paths, manifest = recovery_fixture(Path(temporary))
                run = audit.read_json(parent / "run.json")
                if name == "type":
                    run["error"]["type"] = replacement
                elif name in ("path_suffix", "reverse_paths"):
                    source, destination = parent / "run.json.tmp", parent / "run.json"
                    if name == "path_suffix":
                        source, destination = Path(str(source) + ".backup"), Path(str(destination) + ".backup")
                    else:
                        source, destination = destination, source
                    run["error"]["message"] = f"[WinError 5] Access is denied: {str(source)!r} -> {str(destination)!r}"
                    run["error"]["traceback"] = "in write_json\n    temporary.replace(path)\n" + run["error"]["message"]
                else:
                    old = "run.json" if name == "path" else "[WinError 5]"
                    run["error"]["message"] = run["error"]["message"].replace(old, replacement)
                    run["error"]["traceback"] = run["error"]["traceback"].replace(old, replacement)
                write_json(parent / "run.json", run)
                write_manifest(parent, child, manifest)
                self.assert_rejected(paths, manifest)

    def test_malformed_manifest_is_rejected_without_crashing(self):
        for value in (None, [], "not a manifest"):
            with self.subTest(value=value), temporary_reports() as temporary:
                parent, child, paths, manifest = recovery_fixture(Path(temporary))
                write_json(manifest, value)
                self.assert_rejected(paths, manifest)

    def test_corrupt_bound_checkpoint_is_rejected_without_crashing(self):
        with temporary_reports() as temporary:
            parent, child, paths, manifest = recovery_fixture(Path(temporary))
            (parent / "latest.pth").write_bytes(b"This is not a PyTorch checkpoint.\n")
            digest = audit.file_hash(parent / "latest.pth")
            run = audit.read_json(parent / "run.json")
            run["checkpoints"]["latest"]["file_sha256"] = digest
            write_json(parent / "run.json", run)
            run = audit.read_json(child / "run.json")
            run["resume_parent"]["file_sha256"] = digest
            write_json(child / "run.json", run)
            document = audit.read_json(manifest)
            document["authorizations"][0]["files_sha256"].update({
                "latest.pth": digest, "run.json": audit.file_hash(parent / "run.json")})
            write_json(manifest, document)
            self.assert_rejected(paths, manifest)

    def test_checkpoint_validation_is_mandatory_with_comparison_disabled(self):
        for name in ("missing_latest", "missing_initial", "next_data_step", "source", "recipe", "rng", "adam"):
            with self.subTest(name=name), temporary_reports() as temporary:
                parent, child, paths, manifest = recovery_fixture(Path(temporary))
                if name.startswith("missing_"):
                    target = parent / "latest.pth" if name == "missing_latest" else child / "initial.pth"
                    target.unlink()
                else:
                    payload = checkpoint_payload(child / "initial.pth")
                    if name == "next_data_step":
                        payload["next_data_step"] += 1
                    elif name == "source":
                        payload["source_sha256"][audit.SHARED_SOURCES[0]] = "0" * 64
                    elif name == "recipe":
                        payload["resume_recipe_sha256"] = "0" * 64
                    elif name == "rng":
                        payload["rng_state"]["torch_cpu"][0] ^= 1
                    elif name == "adam":
                        payload["optimizer_state_dict"]["state"][0]["exp_avg"].add_(.5)
                    save_checkpoint(child, "initial", payload)
                self.assert_rejected(paths, manifest)

    def test_rebound_parent_checkpoint_payload_counter_mismatch_is_rejected(self):
        with temporary_reports() as temporary:
            parent, child, paths, manifest = recovery_fixture(Path(temporary))
            payload = checkpoint_payload(parent / "latest.pth")
            payload["next_data_step"] += 1
            save_checkpoint(parent, "latest", payload)
            run = audit.read_json(child / "run.json")
            run["resume_parent"]["file_sha256"] = audit.file_hash(parent / "latest.pth")
            write_json(child / "run.json", run)
            write_manifest(parent, child, manifest)
            self.assert_rejected(paths, manifest)

    def test_matching_but_incomplete_or_stale_adam_states_are_rejected(self):
        for name in ("empty", "missing_second_moment", "stale_step", "moment_shape", "moment_dtype",
                     "duplicate_parameter", "maximize"):
            with self.subTest(name=name), temporary_reports() as temporary:
                parent, child, paths, manifest = recovery_fixture(Path(temporary))
                payload = checkpoint_payload(parent / "latest.pth")
                optimizer = payload["optimizer_state_dict"]
                if name == "empty":
                    optimizer["state"] = {}
                elif name == "missing_second_moment":
                    del optimizer["state"][0]["exp_avg_sq"]
                elif name == "stale_step":
                    optimizer["state"][0]["step"] -= 1
                elif name == "moment_shape":
                    optimizer["state"][0]["exp_avg"] = torch.zeros(2)
                elif name == "moment_dtype":
                    optimizer["state"][0]["exp_avg"] = optimizer["state"][0]["exp_avg"].double()
                elif name == "duplicate_parameter":
                    optimizer["param_groups"][0]["params"].append(0)
                else:
                    optimizer["param_groups"][0]["maximize"] = True
                save_checkpoint(parent, "latest", payload)
                initial = checkpoint_payload(child / "initial.pth")
                initial["optimizer_state_dict"] = copy.deepcopy(optimizer)
                save_checkpoint(child, "initial", initial)
                run = audit.read_json(child / "run.json")
                run["resume_parent"]["file_sha256"] = audit.file_hash(parent / "latest.pth")
                write_json(child / "run.json", run)
                write_manifest(parent, child, manifest)
                self.assert_rejected(paths, manifest)

    def test_child_gap_replay_and_unclosed_status_are_rejected(self):
        for name in ("gap", "rewind", "replay", "running", "failed"):
            with self.subTest(name=name), temporary_reports() as temporary:
                parent, child, paths, manifest = recovery_fixture(Path(temporary))
                run = audit.read_json(child / "run.json")
                if name in ("gap", "rewind"):
                    run["session_start_step"] += 1 if name == "gap" else -1
                    run["resume_parent"]["optimizer_steps"] = run["session_start_step"]
                    write_json(child / "run.json", run)
                elif name == "replay":
                    rows = audit.read_jsonl(child / "training.jsonl")
                    write_rows(child / "training.jsonl", [rows[0], *rows])
                else:
                    run["status"] = name
                    write_json(child / "run.json", run)
                self.assert_rejected(paths, manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
