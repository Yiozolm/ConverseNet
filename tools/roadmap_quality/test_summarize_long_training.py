"""CPU-only synthetic-report tests; these fixtures are not training evidence."""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

import summarize_long_training as audit


def h(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


@contextmanager
def temporary_reports():
    # Verify the generated directory before TemporaryDirectory recursively
    # removes its own synthetic fixtures on exit (including on Windows).
    with tempfile.TemporaryDirectory(prefix="converse-long-audit-") as directory:
        resolved = Path(directory).resolve()
        if resolved.parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError("temporary audit reports escaped the intended temporary directory")
        yield directory


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def session(root, seed, variant, start, end, *, parent=None, quality_delta=0., state_delta=0., purpose="formal"):
    directory = root / f"{variant}_{seed}_{start}_{end}"
    directory.mkdir()
    config = dict(variant=variant, seed=seed, run_dir=str(directory), root=str(root / (variant + "_source")),
                  data_root="fixed_data", checkpoint="initial.pth", manifest="split.json", init="pretrained",
                  steps=end, batch_size=4, microbatch_size=4, eval_every=2, eval_batch_size=1,
                  patch_size=96, scale=3, noise_std=.01, lr=1e-5, loss="mse", crop_border=3,
                  deterministic_algorithms=True, stop_when_stable=False, purpose=purpose,
                  resume=str(parent / "final.pth") if parent else None, max_wall_seconds=None,
                  deadline_utc=None, build=False, verbose_build=False)
    sources = {name: h(name) for name in audit.SHARED_SOURCES}
    sources["Converse2D/torch_converse2d/unit.cpp"] = h(variant)
    build = dict(library="unit.pyd", binary_sha256=h("binary" + variant),
                 inputs=dict(sources={"unit.cpp": h(variant)}))
    dataset = dict(train_images=900, validation_images=100, image_hashes_verified=True,
                   manifest_sha256=h("manifest"), kernels_sha256={"kernel": h("kernel")},
                   source_sha256=sources["tools/roadmap_quality/usrnet_training_data.py"])
    environment = dict(tf32=False, amp=False, deterministic_algorithms=True, torch="synthetic CPU fixture")
    def state(step):
        return {"weight": torch.tensor([seed + step * .001 + (state_delta if step == end else 0)], dtype=torch.float32)}
    origin = audit.state_tensor_hash({"weight": torch.tensor([float(seed)])})
    initial_hash = audit.state_tensor_hash(state(start))
    parent_run = audit.read_json(parent / "run.json") if parent else None
    local_steps = ([] if parent else [0]) + [step for step in range(start + 1, end + 1) if step % 2 == 0 or step == end]
    def values(step):
        return dict(rgb=dict(psnr_db=30 + step * .01 + quality_delta, ssim=.9 + quality_delta * .0001),
                    y=dict(psnr_db=32 + step * .01 + quality_delta, ssim=.92 + quality_delta * .0001))
    history = copy.deepcopy(parent_run["metric_history"]) if parent else []
    history += [dict(optimizer_steps=step, **values(step)) for step in local_steps]
    rows = [dict(step=step + 1, data_step=step, optimizer_steps=step + 1,
                 batch_sha256=h(f"batch:{seed}:{step}"), optimizer_applied=True, loss_and_grad_finite=True,
                 loss=.01, grad_l2_norm=.1, gradient_tensor_count=1,
                 training_step_wall_ms=1000., data_prepare_and_hash_wall_ms=10.)
            for step in range(start, end)]
    evaluations = [dict(optimizer_steps=step, images=100, all_outputs_finite=True,
                        parameter_check=dict(finite=True), validation_payload_sha256=h("validation"),
                        per_image=[dict(id=f"image_{index:03d}", **values(step)) for index in range(100)],
                        **values(step)) for step in local_steps]
    report = dict(status="complete", config=config, config_sha256=audit.hash_json(config),
                  resume_recipe_sha256=audit.hash_json(audit.normalized_recipe(config)),
                  comparison_recipe_sha256=audit.hash_json({key: value for key, value in config.items()
                    if key not in audit.IGNORED_RECIPE - {"steps"}}),
                  dataset=dataset, source_sha256=sources, split_sha256=h("split"), input_checkpoint_sha256=h("checkpoint"),
                  initial_state_tensor_sha256=initial_hash, origin_initial_state_tensor_sha256=origin,
                  optimizer_steps=end, next_data_step=end, session_start_step=start, session_optimizer_steps=end - start,
                  samples_seen=end * 4, all_loss_and_grad_finite=True,
                  backend=dict(kind="checked production checkout", root=config["root"], build_manifest=build),
                  environment=environment, architecture=dict(kind="CPU synthetic report fixture"),
                  parameter_count=1, parameter_tensor_count=1,
                  protocol=dict(optimizer="Adam", betas=[.9, .999], eps=1e-8, weight_decay=0,
                                amp=False, scheduler=False, gradient_clipping=False,
                                preregistered_paired_final_quality_gate=dict(max_psnr_drop_db=.05, max_ssim_drop=.001,
                                                                           spaces=["rgb", "y"], each_seed=True, all_steps_finite=True)),
                  metric_history=history, evaluation_steps=local_steps, validation_payload_sha256=h("validation"),
                  created_utc=f"2026-09-25T04:{start:02d}:00+00:00", completed_utc=f"2026-09-25T04:{end:02d}:00+00:00",
                  process_elapsed_wall_s=(end - start) * 1.01 + len(local_steps) * .1 + .25,
                  timing=dict(total_training_step_wall_ms=(end - start) * 1000.,
                              data_prepare_and_hash_wall_ms=(end - start) * 10., evaluation_wall_s=len(local_steps) * .1,
                              checkpoint_wall_s=.03, setup_wall_s=.25,
                              training_loop_end_to_end_wall_s=(end - start) * 1.01 + len(local_steps) * .1),
                  resume_parent=None, checkpoints={})
    if parent:
        record = parent_run["checkpoints"]["final"]
        report["resume_parent"] = dict(path=str(parent / "final.pth"), run_dir=str(parent),
                                       file_sha256=record["file_sha256"], optimizer_steps=start, previous_status="budget_stopped")
    for name, step in (("initial", start), ("final", end)):
        model = state(step)
        payload = dict(state_dict=model,
                       optimizer_state_dict=dict(state={0: dict(step=torch.tensor(float(step)), exp_avg=torch.tensor([step * .1]))},
                                                 param_groups=[dict(lr=1e-5, params=[0])]),
                       optimizer_steps=step, next_data_step=step, source_sha256=sources, backend_manifest=build,
                       origin_initial_state_tensor_sha256=origin, config_sha256=report["config_sha256"])
        path = directory / (name + ".pth")
        torch.save(payload, path)
        report["checkpoints"][name] = dict(path=str(path), optimizer_steps=step,
                                           file_sha256=audit.file_hash(path), state_tensor_sha256=audit.state_tensor_hash(model))
    write_rows(directory / "training.jsonl", rows)
    write_rows(directory / "evaluations.jsonl", evaluations)
    write_json(directory / "run.json", report)
    return directory


class LongSummaryCPU(unittest.TestCase):
    def test_six_pairs_different_endpoints_and_rounding_do_not_replace_quality(self):
        with temporary_reports() as temporary:
            root = Path(temporary)
            paths = [session(root, seed, variant, 0, 4 if variant == "before" else 6,
                             quality_delta=0 if variant == "before" else -.01,
                             state_delta=0 if variant == "before" else .0001)
                     for seed in audit.SEEDS for variant in audit.VARIANTS]
            result = audit.audit(paths)
            self.assertEqual(result["status"], "quality_gates_passed", result["errors"])
            self.assertFalse(result["convergence_claim"])
            for pair in result["pairs"]:
                self.assertEqual(pair["matching_data_prefix_steps"], 4)
                self.assertEqual(pair["unshared_current_steps"], [4, 5])
                self.assertTrue(pair["final_quality_gate"]["passed"])
                self.assertEqual(pair["state_comparison"]["status"], "compared_on_cpu")
                self.assertFalse(pair["state_comparison"]["same_update_count"])
                self.assertGreater(pair["state_comparison"]["model"]["different_tensors"], 0)
            json.dumps(result, allow_nan=False)
            self.assertIn("no convergence claim", audit.markdown(result))
            current = root / "current_17_0_6"
            run = audit.read_json(current / "run.json")
            evals = audit.read_jsonl(current / "evaluations.jsonl")
            run["metric_history"][-1]["y"]["psnr_db"] -= 1
            evals[-1]["y"]["psnr_db"] -= 1
            for item in evals[-1]["per_image"]:
                item["y"]["psnr_db"] -= 1
            write_json(current / "run.json", run)
            write_rows(current / "evaluations.jsonl", evals)
            failed = audit.audit(paths)
            self.assertEqual(failed["status"], "quality_failed")
            self.assertFalse(failed["pairs"][0]["final_quality_gate"]["passed"])

    def test_resume_parent_autoload_and_exact_coverage(self):
        with temporary_reports() as temporary:
            root = Path(temporary)
            parent = session(root, 17, "before", 0, 2)
            child = session(root, 17, "before", 2, 4, parent=parent)
            control = session(root, 17, "current", 0, 4)
            result = audit.audit([child, control])
            self.assertEqual(result["status"], "incomplete")  # Other two seeds are intentionally absent.
            self.assertFalse(result["errors"], result["errors"])
            chain = next(row for row in result["trajectories"] if row["variant"] == "before")
            self.assertEqual(chain["sessions"], [str(parent), str(child)])
            self.assertEqual(chain["summed_session_timings"]["training_step_wall_s"], 4.)
            self.assertEqual(chain["holes"], [])
            self.assertEqual(chain["overlaps"], [])
            self.assertEqual(chain["intersession_metadata_intervals"][0]["metadata_interval_seconds"], 0.)

    def test_holes_overlaps_and_missing_scheduled_evaluation_never_pass(self):
        with temporary_reports() as temporary:
            root = Path(temporary)
            path = session(root, 17, "before", 0, 4)
            rows = audit.read_jsonl(path / "training.jsonl")
            write_rows(path / "training.jsonl", [rows[0], rows[1], rows[3], rows[3]])
            result = audit.audit([path], inspect_checkpoints=False)
            chain = result["trajectories"][0]
            self.assertEqual(chain["holes"], [2])
            self.assertTrue(result["errors"])
            self.assertTrue(result["missing"])
            write_rows(path / "training.jsonl", rows)
            run = audit.read_json(path / "run.json")
            run["metric_history"] = [row for row in run["metric_history"] if row["optimizer_steps"] != 2]
            run["evaluation_steps"].remove(2)
            evaluations = [row for row in audit.read_jsonl(path / "evaluations.jsonl") if row["optimizer_steps"] != 2]
            write_json(path / "run.json", run)
            write_rows(path / "evaluations.jsonl", evaluations)
            chain = audit.audit([path], inspect_checkpoints=False)["trajectories"][0]
            self.assertEqual(chain["missing_scheduled_evaluations"], [2])

    def test_invalid_parent_path_is_reported_without_crashing(self):
        with temporary_reports() as temporary:
            path = session(Path(temporary), 17, "before", 0, 4)
            run = audit.read_json(path / "run.json")
            run["resume_parent"] = dict(run_dir=23, path=[], file_sha256=h("x"), optimizer_steps=0)
            write_json(path / "run.json", run)
            result = audit.audit([path], inspect_checkpoints=False)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("parent path" in error for error in result["errors"]))
            json.dumps(result, allow_nan=False)

    def test_active_snapshot_is_incomplete_when_jsonl_is_ahead(self):
        with temporary_reports() as temporary:
            path = session(Path(temporary), 17, "before", 0, 4)
            run = audit.read_json(path / "run.json")
            run["status"] = "running"
            rows = audit.read_jsonl(path / "training.jsonl")
            extra = dict(rows[-1], data_step=4, step=5, optimizer_steps=5)
            write_json(path / "run.json", run)
            write_rows(path / "training.jsonl", rows + [extra])
            result = audit.audit([path], inspect_checkpoints=False)
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["errors"], result["errors"])
            self.assertTrue(any("Provisional live snapshot" in item for item in result["missing"]))

    def test_missing_requested_session_stays_incomplete(self):
        with temporary_reports() as temporary:
            result = audit.audit([Path(temporary) / "not_started"], inspect_checkpoints=False)
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["errors"])
            self.assertTrue(result["missing"])

    def test_build_change_and_exact_batch_hash_mismatch_are_rejected(self):
        with temporary_reports() as temporary:
            root = Path(temporary)
            parent = session(root, 17, "before", 0, 2)
            child = session(root, 17, "before", 2, 4, parent=parent)
            run = audit.read_json(child / "run.json")
            run["backend"]["build_manifest"]["binary_sha256"] = h("different binary")
            write_json(child / "run.json", run)
            result = audit.audit([child], inspect_checkpoints=False)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("changed across sessions" in error for error in result["errors"]))
            before = session(root, 29, "before", 0, 4)
            current = session(root, 29, "current", 0, 4)
            rows = audit.read_jsonl(current / "training.jsonl")
            rows[2]["batch_sha256"] = h("other batch")
            write_rows(current / "training.jsonl", rows)
            result = audit.audit([before, current], inspect_checkpoints=False)
            self.assertTrue(any("Different exact data hashes" in error for error in result["errors"]))

    def test_discovery_lists_unlinked_pilots_and_stability_needs_four_metrics(self):
        with temporary_reports() as temporary:
            root = Path(temporary)
            pilot = session(root, 17, "before", 0, 2, purpose="pilot")
            formal = session(root, 17, "before", 0, 4)
            result = audit.audit(audit.discover([], root), inspect_checkpoints=False)
            self.assertEqual([row["directory"] for row in result["excluded_sessions"]], [str(pilot)])
            self.assertEqual(result["trajectories"][0]["leaf"], str(formal))
        history = [dict(optimizer_steps=step, rgb=dict(psnr_db=30., ssim=.9),
                        y=dict(psnr_db=32., ssim=.92)) for step in range(0, 1001, 250)]
        self.assertTrue(audit.stability(history, 1000)["satisfied"])
        history[-1]["rgb"]["ssim"] += .0006
        self.assertFalse(audit.stability(history, 1000)["satisfied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
