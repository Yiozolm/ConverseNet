"""Fixed-native-fixture s1 module candidates; isolated source/tensor boundaries.

Compare current pad/slice, forced-s1, cat-pad/as_strided-crop, their combination,
and native dense ConvTranspose2d. Native is a different mathematical operator.
The exact seed17 s1_module fixture from benchmark_native_deconv is reused and
hashed against the saved baseline; eps, amplitudes and upstream are not tunable.

Normal timing requires every Converse path to pass the unchanged independent
FP64 budgets: output atol/rtol=3e-5/3e-5, each VJP=5e-5/5e-5. If current already
fails, --diagnostic-only permits timing current plus only candidates whose
output AND dx/dw/db bit patterns equal current. Such results preserve the
precision failure and are explicitly not production-eligible.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONVERSE_NAMES = ("current", "forced_s1", "catpad_viewcrop", "combined")


def audit_native_baseline(path):
    """Pure CPU/standard-library integrity audit; no torch import or GPU use."""
    from train_usrnet_dataset import file_hash
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["settings"]["seed"] == 17
    assert report["settings"]["warmup"] == 5 and report["settings"]["rounds"] == 4 and report["settings"]["iters"] == 20
    assert report["mathematical_equivalence"] is False and report["quality_comparison"] is False
    source_checks = {name: file_hash(ROOT / name) == value for name, value in report["source_sha256"].items()}
    assert all(source_checks.values()), "Native baseline sources changed"
    summaries = []
    for case in report["cases"]:
        config = case["config"]
        batch, channels, height, width = config["shape"]
        scale, kernel = config["scale"], config["kernel"]
        for native_name, comparison in case["comparisons"].items():
            for kind, fixture in comparison["fixtures"].items():
                assert fixture["input_shape"] == [batch, channels, height, width]
                assert fixture["output_shape"] == [batch, channels, height * scale, width * scale]
                counts = fixture["gradient_elements"]
                assert counts == dict(x=batch * channels * height * width,
                                      weight=math.prod(fixture["weight_shape"]), bias=channels)
                expected_weights = (batch if config["dynamic"] else 1) * channels * kernel * kernel
                if kind == "native_dense":
                    expected_weights *= channels
                assert counts["weight"] == expected_weights
                assert fixture["gradient_targets"] == ["x", "weight", "bias"] and fixture["gradients_present"] == 3
                assert fixture["route"]["spectral_solve"] == (kind == "converse")
                if kind != "converse":
                    geometry = fixture["native_geometry"]
                    assert geometry["groups"] if "groups" in geometry else True
                    assert fixture["route"]["groups"] == (channels if kind == "native_depthwise" else 1)
                    assert fixture["route"]["native_calls_per_forward"] == (batch if config["dynamic"] else 1)
                    computed = [(size - 1) * geometry["stride"] - 2 * geometry["padding"]
                                + geometry["dilation"] * (kernel - 1) + geometry["output_padding"] + 1
                                for size in (height, width)]
                    assert computed == geometry["output_hw"] == [height * scale, width * scale]
            for index, order in enumerate(comparison["order"]):
                assert order == (["converse", native_name] if index % 2 == 0 else [native_name, "converse"])
            for key in ("wall_ms", "event_ms"):
                ratios = [comparison["rounds"]["converse"][index][key] /
                          comparison["rounds"][native_name][index][key] for index in range(4)]
                assert ratios == comparison["converse_over_this_native"][key]["per_round"]
                assert statistics.median(ratios) == comparison["converse_over_this_native"][key]["median"]
            for kind, rows in comparison["rounds"].items():
                assert len(rows) == 4
                for key, value in comparison["medians"][kind].items():
                    assert value == statistics.median(row[key] for row in rows)
            summaries.append(dict(case=case["name"], native=native_name,
                                  converse_wall_ms=comparison["medians"]["converse"]["wall_ms"],
                                  native_wall_ms=comparison["medians"][native_name]["wall_ms"],
                                  paired_ratio=comparison["converse_over_this_native"]["wall_ms"],
                                  weight_elements={kind: fixture["gradient_elements"]["weight"]
                                                   for kind, fixture in comparison["fixtures"].items()}))
    return dict(passed=True, path=str(Path(path).resolve()), file_sha256=file_hash(path),
                source_checks=source_checks, shape_and_vjp_counts_verified=True,
                independent_denominator_ratios_recomputed=True, comparisons=summaries)


def methods(current, forced, eps):
    import torch.nn.functional as F
    import probe_training_s1_shapes as s1probe
    from probe_converse_boundaries import pad_cat, crop_view
    from models.converse_core import converse2d_reference

    def wrap(core, cat=False):
        def forward(x, weight, bias):
            padded = pad_cat(x, 2) if cat else F.pad(x, (2, 2, 2, 2), mode="circular")
            output = core(padded, weight, bias)
            return crop_view(output, 2) if cat else output[..., 2:-2, 2:-2]
        return forward

    return dict(current=wrap(s1probe.method(current, eps)),
                forced_s1=wrap(s1probe.method(forced, eps)),
                catpad_viewcrop=wrap(s1probe.method(current, eps), cat=True),
                combined=wrap(s1probe.method(forced, eps), cat=True),
                reference_fp64=wrap(lambda x, weight, bias: converse2d_reference(x, x, weight, bias, 1, eps)),
                native_dense=lambda x, weight, bias: F.conv_transpose2d(x, weight, bias, stride=1, padding=1, groups=1))


def validate(tensors, paths):
    import torch
    from probe_pointwise_training import capture
    import probe_training_s1_shapes as s1probe
    reference = capture(tensors, torch.float64, paths["reference_fp64"])
    outputs = {name: capture(tensors, torch.float32, paths[name]) for name in CONVERSE_NAMES}
    result = {}
    for name, actual in outputs.items():
        if any(value.dtype != torch.float32 for value in actual.values()):
            raise RuntimeError("Candidates must keep FP32 output and all gradients")
        stats = {key: s1probe.metrics(actual[key], expected, output=key == "output", weak=False)
                 for key, expected in reference.items()}
        same = {key: torch.equal(value.contiguous().view(torch.int32),
                                 outputs["current"][key].contiguous().view(torch.int32))
                for key, value in actual.items()}
        result[name] = dict(passed=all(row["passed"] for row in stats.values()), fp64=stats,
                            bitwise_equal_to_current=same, all_bits_equal_to_current=all(same.values()))
    return result


def summarize(rounds, names):
    medians = {name: {key: statistics.median(row["variants"][name][key] for row in rounds)
                      for key in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
               for name in names}
    ratios = {}
    for name in names:
        ratios[name] = {}
        for key in ("wall_ms", "cuda_event_ms"):
            against_current = [row["variants"]["current"][key] / row["variants"][name][key] for row in rounds]
            against_native = [row["variants"][name][key] / row["variants"]["native_dense"][key] for row in rounds]
            ratios[name][key] = dict(current_over_candidate=against_current,
                                     median_current_over_candidate=statistics.median(against_current),
                                     candidate_over_native_dense=against_native,
                                     median_candidate_over_native_dense=statistics.median(against_native),
                                     min_candidate_over_native_dense=min(against_native),
                                     max_candidate_over_native_dense=max(against_native))
    return dict(medians=medians, paired_ratios=ratios)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/native_deconv_target/native_baseline.json")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Only if current fails FP64: time bitwise-identical candidates, never label production eligible")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv_target/candidates.json")
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("Require a new output path and positive counts")
    import train_usrnet_dataset as worker
    baseline_audit = audit_native_baseline(args.baseline)
    saved_baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    saved_case = next(row for row in saved_baseline["cases"] if row["name"] == "s1_module")
    import torch
    import benchmark_native_deconv as native
    import probe_training_s1_shapes as s1probe
    from probe_pointwise_training import timed_fixture
    sys.path.insert(0, str(ROOT))
    config = native.CASES["s1_module"]
    if worker.json_safe(config) != saved_case["config"] or config["eps"] != 1e-5:
        raise RuntimeError("The fixed baseline fixture configuration changed")
    cpu = native.cpu_data(config, seed=17)
    fixture_hash = worker.tensor_hash(cpu)
    input_hash = worker.tensor_hash({key: cpu[key] for key in ("x", "upstream")})
    if fixture_hash != saved_case["complete_fixture_sha256"] or input_hash != saved_case["input_upstream_sha256"]:
        raise RuntimeError("Regenerated seed17 fixture does not exactly match the original native baseline")
    args.variant = "current"
    before = s1probe.source_hashes()
    current, current_build = worker.load_backend(args)
    forced, forced_build = s1probe.load_forced(args.verbose_build)
    if before != s1probe.source_hashes() or before != forced_build["source_sha256"]:
        raise RuntimeError("Sources changed while building the isolated candidate")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tensors = (cpu["x"], cpu["weight"], cpu["bias"].reshape(1, 128, 1, 1), cpu["upstream"])
    native_tensors = (cpu["x"], cpu["dense_weight"], cpu["bias"], cpu["upstream"])
    paths = methods(current, forced, config["eps"])
    hashes = worker.source_hashes()
    for name in ("benchmark_native_deconv.py", "benchmark_converse_candidates.py", "probe_training_s1_shapes.py",
                 "probe_converse_boundaries.py", "probe_pointwise_training.py"):
        hashes["test/" + name] = worker.file_hash(ROOT / "test" / name)
    report = dict(status="validating", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  settings=worker.json_safe(vars(args)), fixed_config=config, fixed_seed=17,
                  complete_fixture_sha256=fixture_hash, input_upstream_sha256=input_hash,
                  baseline_audit=baseline_audit, source_sha256=hashes, current_build=current_build,
                  forced_build=forced_build, production_eligible=False,
                  eligibility_scope="This isolated one-fixture experiment alone never grants production eligibility; broader regressions remain required",
                  precision_reference="Independent full-FFT FP64 with original native circular pad/slice crop and unmodified fixture",
                  native_scope="Dense ConvTranspose2d is a non-equivalent timing target; excluded from the Converse FP64 equation gate",
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                   tf32=False, amp=False, cudnn_benchmark=False, cudnn_deterministic=True),
                  timing_scope="Full pad/operator/crop forward + dx/dw/db VJP, one GPU fixture; no loss/optimizer; no profiler",
                  stability_note="Longer 6x50 defaults reflect drift in a previous same-logic already-specialized namespace control; retain individual rounds and never multiply historical ratios",
                  rounds=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    try:
        report["validation"] = validate(tensors, paths)
        baseline_passed = report["validation"]["current"]["passed"]
        all_passed = all(row["passed"] for row in report["validation"].values())
        diagnostic = args.diagnostic_only and not baseline_passed
        report["all_fixed_fixture_precision_gates_passed"] = all_passed
        report["diagnostic_only"] = diagnostic
        if not all_passed and not diagnostic:
            report["status"] = "precision_gate_failed_no_timing"
            worker.write_json(args.output, report)
            return 2
        selected = ([name for name in CONVERSE_NAMES if report["validation"][name]["all_bits_equal_to_current"]]
                    if diagnostic else list(CONVERSE_NAMES))
        report["excluded_candidates"] = [name for name in CONVERSE_NAMES if name not in selected]
        selected.append("native_dense")
        report["timed_variants"] = selected
        report["status"] = "diagnostic_timing_precision_failed" if diagnostic else "timing_local_gate_passed"
        worker.write_json(args.output, report)
        for index in range(args.rounds):
            order = selected[index % len(selected):] + selected[:index % len(selected)]
            values = {name: timed_fixture(native_tensors if name == "native_dense" else tensors, paths[name], args)
                      for name in order}
            report["rounds"].append(dict(round=index + 1, order=order, variants=values))
            report["summary"] = summarize(report["rounds"], selected)
            worker.write_json(args.output, report)
            print(json.dumps(dict(round=index + 1, diagnostic_only=diagnostic, values=values)), flush=True)
        report["status"] = "complete_diagnostic_only_precision_failed" if diagnostic else "complete_local_candidate_measurement"
        if before != s1probe.source_hashes():
            raise RuntimeError("Production source changed during measurement")
        report["production_source_unchanged"] = True
        worker.write_json(args.output, report)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        worker.write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
