"""CPU-only immutable source archival and compact v4 campaign index.

Reads existing reports and four pinned files through read-only Git commands.
Never imports Torch, builds, runs a GPU, stages files, or changes Git refs.
Outputs are created exclusively below this new directory and must not exist.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/v4_campaign"
CC244E3 = "cc244e3e49dcd896962e14c98095243b41ee578a"
PINNED = {
    "research/run_algorithms.py": "68958153d88b9326a4641d6e9b8f38eee7dbd6a30994f60a7e1968e59c27164d",
    "research/algorithms.py": "2ca5f02ee9581d5693f2c1d361a2fbc199bc7a689573df3f1562aea1d0b82286",
    "research/training_candidates.py": "f601f90bdd907bdc65c785d33a64cd871784164b964e5e2ce5f11a8973bc5df6",
    "models/converse_core.py": "8b31f77ae03fafad69f6e8d3f696fe02166aa0d8fe258a2937de3ab619041ccd",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=digest(raw))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create(path, raw):
    path = Path(path)
    if not path.resolve().is_relative_to(HERE.resolve()):
        raise RuntimeError("Archive destination escaped tools/v4_campaign")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def flags(manifest):
    identity = manifest.get("identity", manifest.get("inputs", {}))
    return dict(cxx_flags=identity.get("cxx_flags"), cuda_flags=identity.get("cuda_flags"),
                torch=identity.get("torch"), cuda=identity.get("torch_cuda", identity.get("cuda")),
                architecture=identity.get("toolchain", {}).get("environment", {}).get("TORCH_CUDA_ARCH_LIST"),
                binary_sha256=manifest.get("binary_sha256"))


def model_runs(report, key="model_performance"):
    return [dict(median_speedup=run["median_speedup"], passed=run["passed"],
                 paired_speedups=run["paired_speedups"])
            for run in report.get(key, {}).get("independent_runs", [])]


def main():
    if (HERE / "index.json").exists():
        raise FileExistsError("Retain the current archive/index; this one-shot archiver does not overwrite")
    copies = []
    selections = {
        "power_fma_rejected": ("candidate.patch", "decision.json", "source_manifest.json"),
        "div_vjp_rejected": ("candidate.patch", "decision.json", "source_manifest.json"),
        "fftfree_output_fma_rejected": ("kernel.cu", "decision.json"),
        "fftfree_production_padded_rejected": ("decision.json", "source_manifest.json"),
    }
    for directory, names in selections.items():
        for name in names:
            source = ART / directory / name
            target = HERE / "rejected" / directory / name
            create(target, source.read_bytes())
            copies.append(dict(source=record(source), archived=record(target)))
    # Complete changed-source snapshot, excluding its binary and 15 MB report.
    source_root = ART / "fftfree_production_padded_rejected/source"
    for source in sorted(source_root.rglob("*")):
        if source.is_file():
            target = HERE / "rejected/fftfree_production_padded_rejected/source" / source.relative_to(source_root)
            create(target, source.read_bytes())
            copies.append(dict(source=record(source), archived=record(target)))
    create(HERE / "protocol.json", (ART / "protocol.json").read_bytes())

    pinned = []
    resolved = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", CC244E3], text=True).strip()
    if resolved != CC244E3:
        raise RuntimeError("Pinned commit mismatch")
    for name, expected in PINNED.items():
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{CC244E3}:{name}"])
        if digest(raw) != expected:
            raise RuntimeError("Pinned historical file hash mismatch: " + name)
        target = HERE / "pinned_cc244e3" / name
        create(target, raw)
        pinned.append(dict(git_source=f"{CC244E3}:{name}", archived=record(target)))

    experiments = []
    wgrad_path = ROOT / "tools/v4_experiments/results_c128_to64.json"
    wgrad = read(wgrad_path)
    experiments.append(dict(id="pointwise_wgrad_c128_to64", status="accepted_production",
        accepted_commit="09f0f12", compact_evidence=record(wgrad_path),
        rejected_broad_scope=wgrad["broad_candidate_rejected"],
        model_speedups=wgrad["production"]["B4_Adam_speedups"], convergence_evidence=False))
    for directory, identifier in (("power_fma_rejected", "power_fma"), ("div_vjp_rejected", "division_vjp_reuse")):
        original = ART / directory / "decision.json"
        decision = read(original)
        experiments.append(dict(id=identifier, status=decision["status"],
            baseline_commit=decision["baseline_commit"],
            failed_test_subcases=decision["failed_test_subcases"],
            fp64_matrix_total=decision["fp64_matrix_total"], fp64_matrix_failed=decision["fp64_matrix_failed"],
            performance=decision["performance"], original_decision=record(original),
            archived_decision=record(HERE / "rejected" / directory / "decision.json"),
            source_flags=flags(read(ART / directory / "source_manifest.json"))))
    for name, identifier, status in (
        ("fftfree_gate_001.json", "fftfree_residual", "rejected_numeric"),
        ("fftfree_output_fma_gate_002.json", "fftfree_output_fma", "rejected_numeric"),
        ("fftfree_compensated_gate_003.json", "fftfree_compensated", "rejected_whole_model_performance"),
        ("fftfree_fused_lambda_gate_004.json", "fftfree_fused_lambda_prototype", "passed_prototype_only"),
    ):
        path = ART / name
        gate = read(path)
        row = dict(id=identifier, status=status, raw_gate=record(path),
                   gate_status=gate["status"], gate_passed=gate["passed"],
                   cases=len(gate["cases"]), failed_cases=gate["failed_cases"],
                   contracts_passed=gate["contracts"]["passed"],
                   source_flags=flags(gate["checked_research_manifest"]))
        if identifier in ("fftfree_compensated", "fftfree_fused_lambda_prototype"):
            perf_path = ART / ("fftfree_compensated_perf_003.json" if identifier == "fftfree_compensated"
                               else "fftfree_fused_lambda_perf_004.json")
            perf = read(perf_path)
            row.update(raw_performance=record(perf_path), performance_passed=perf["passed"],
                       model_quality_passed=perf["model_quality"]["passed"],
                       complete_call_performance_passed=perf["operator_performance"]["passed"],
                       model_runs=model_runs(perf), unchanged_model_threshold=1.03)
        experiments.append(row)
    path = ART / "fftfree_production_padded_rejected/fftfree_production_all_001.json"
    padded = read(path)
    experiments.append(dict(id="fftfree_initial_padded_production", status="rejected_whole_model_performance",
        raw_report=record(path), report_status=padded["status"], report_passed=padded["passed"],
        gate_passed=padded["gate"]["passed"], gate_cases=len(padded["gate"]["cases"]),
        model_quality_passed=padded["model"]["passed"],
        model_runs=model_runs(padded["perf"], "model"), unchanged_model_threshold=1.03,
        original_decision=record(ART / "fftfree_production_padded_rejected/decision.json"),
        source_flags=flags(padded["identity"]["checked_production_manifest"])))
    path = ART / "nearest_v4_recheck.json"
    nearest = read(path)
    experiments.append(dict(id="nearest_spectral_recheck", status="rejected_numeric", raw_report=record(path),
        routes={name: {key: route[key] for key in ("passed", "tensor_count", "failed_tensors")}
                for name, route in nearest["routes"].items()}, performance="not run"))
    experiments.append(dict(id="fftfree_overflow_guard_and_pad_cancellation", status="PENDING",
        accepted=False, note="Root is running fresh validation; no pass or commit is inferred from earlier candidates.",
        accuracy_scope="Fresh actual production direct-op and pad/crop module gates plus release tests",
        performance_scope="Fresh actual production cold/warm callers and complete pretrained model",
        unchanged_model_threshold=1.03))

    sass_path = ART / "power_fma_sass_comparison.json"
    sass = read(sass_path)
    profile = dict(status="measured_baseline_diagnostic", kernel="scale1_forward",
        measured_input_shape=[4,128,100,100], shared_x_prior=True, kernel_shape=[1,128,3,3],
        dram_throughput_percent=88.33652, dram_GB_per_second=395.358185,
        sm_throughput_percent=31.234921,
        interpretation="This measured s1 kernel is close to its DRAM throughput ceiling; this does not prove all shapes are memory-bound.",
        profiler_is_not_benchmark=True, raw_report=record(ART / "power_fma_baseline_ncu/capture.ncu-rep"),
        raw_metrics_csv=record(ART / "power_fma_baseline_ncu/metrics.csv"),
        launcher=record(ART / "power_fma_baseline_ncu/launcher.json"),
        static_comparison=record(sass_path), binary_identity=sass["inputs"],
        limits=sass["limits"])

    small_groups = []
    for directory in ("tools/v4_arithmetic", "tools/v4_fftfree", "tools/v4_fftfree_compensated",
                      "tools/v4_fftfree_compensated_fused_lambda"):
        files = [record(p) for p in sorted((ROOT / directory).rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
        small_groups.append(dict(directory=directory, total_bytes=sum(p["bytes"] for p in files), files=files))
    helpers = [record(ROOT / "tools" / name) for name in
               ("v4_fftfree_perf.py", "v4_fftfree_fused_lambda_perf.py", "v4_fftfree_production.py")]
    index = dict(schema_version=1, kind="compact_v4_campaign_history", generated_by=record(Path(__file__)),
        scope="Source and decision archive only; not a replacement gate report or a new acceptance decision.",
        old_records_preserved_verbatim=True, protocol=record(HERE / "protocol.json"),
        archived_files=copies, pinned_git_sources=dict(commit=CC244E3, files=pinned,
            note="Files are preserved independently of Git ref reachability; legacy helper still names cc244e3 explicitly."),
        experiments=experiments, baseline_s1_profile=profile, small_source_groups=small_groups,
        standalone_helpers=helpers,
        reproducibility="After a checked build, the current production helper supports --stage all --output NEW.json "
            "without prototype arguments and records fresh actual-production admission. Optional --prototype-gate and "
            "--prototype-perf must be supplied together and must be authentic full reports, regenerated against the "
            "pinned baseline or retained verbatim. Never fabricate compact stubs. This index is descriptive provenance, "
            "not a replacement numerical/performance gate report.",
        excluded_from_archive=["binaries", "full numerical/performance reports", "Nsight captures", "SASS dumps", "tensor snapshots"])
    create(HERE / "index.json", (json.dumps(index, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(dict(index=record(HERE / "index.json"), copied_sources=len(copies), pinned_sources=len(pinned),
                          newest_candidate_status="PENDING"), indent=2))


if __name__ == "__main__":
    main()
