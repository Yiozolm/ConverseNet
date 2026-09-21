"""Audit the fixed 2026-09-21 small-s1 experiment without using a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ("solve_alias", "solve_output", "adjoint_q", "adjoint_inputs", "adjoint_filter",
           "forward_scale1", "backward_scale1", "filter_scale1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def profile(folder):
    assert read(folder / "launcher.json")["passed"]
    assert read(folder / "metadata.json")["status"] == "complete"
    with sqlite3.connect(str(folder / "full.sqlite")) as connection:
        rows = connection.execute("SELECT s.value,count(*),sum(k.end-k.start)/1e6 FROM "
            "CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value").fetchall()
        intervals = connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    left, right = intervals[0]
    busy = 0
    for start, end in intervals[1:]:
        if start > right:
            busy += right-left
            left, right = start, end
        else:
            right = max(right, end)
    busy += right-left
    counts = {key: sum(count for name, count, _ in rows if key + "<" in name) for key in KERNELS}
    return dict(kernel_count=sum(c for _, c, _ in rows), kernel_sum_ms=sum(t for _, _, t in rows),
        gpu_kernel_span_ms=(max(e for _, e in intervals)-intervals[0][0])/1e6,
        kernel_busy_union_ms=busy/1e6, spectral_kernel_counts=counts,
        spectral_kernel_sum_ms=sum(t for n, _, t in rows if any(k + "<" in n for k in KERNELS)),
        scope="Two profiled training steps; diagnostic, not formal speed")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/small_s1_20260921")
    args = parser.parse_args()
    base = args.directory
    output = base / "summary.json"
    if output.exists():
        parser.error("Refusing to overwrite evidence")
    op = read(base / "operators.json")
    assert op["status"] == "complete"
    operator_rows, failures = [], []
    for row in op["cases"]:
        assert all(all(v["bitwise_before"].values()) for v in row["validation"].values())
        assert all(v["finite"] for route in row["validation"].values() for v in route["error"].values())
        assert row["validation"]["before"]["error"] == row["validation"]["combined"]["error"]
        operator_rows.append(dict(batch=row["batch"], median_ms=row["median_ms"], paired_speedup=row["paired_speedup"]))
        for key, passed in row["validation"]["combined"]["python_noninferior"].items():
            if not passed:
                failures.append(dict(batch=row["batch"], tensor=key, inherited_bitwise=True,
                    before_and_candidate=row["validation"]["combined"]["error"][key],
                    python=row["python_fp32_error"][key]))
    models = []
    for filename in ("model_b1.json", "model_b4_retry.json"):
        row = read(base / filename)
        assert row["status"] == "complete" and len(row["rounds"]) == 4
        assert all(all(v.values()) for v in row["validation"].values())
        ratios = [r["values"]["before"]["wall_ms"] / r["values"]["combined"]["wall_ms"] for r in row["rounds"]]
        models.append(dict(batch=row["settings"]["batch"], bitwise_state_checks=sum(len(v) for v in row["validation"].values()),
            median_ms={name: statistics.median(r["values"][name]["wall_ms"] for r in row["rounds"])
                       for name in ("before", "combined")}, paired_speedups=ratios,
            paired_speedup_median=statistics.median(ratios),
            allocated_bytes={name: [r["values"][name]["allocated"] for r in row["rounds"]]
                             for name in ("before", "combined")}))
    assert read(base / "model_finite_b4.json")["status"] == "complete"
    assert read(base / "contracts.json")["passed"]
    checks = dict(cuda_contracts=read(base / "contracts.json")["tests_run"])
    for kind in ("cpu", "cuda"):
        report = read(base / f"helpers_{kind}.json")
        assert report["status"] == "passed_extraction_parity"
        checks["helpers_" + kind] = len(report["checks"])
    profiles = {name: profile(base / f"nsys_{name}_verified") for name in ("before", "combined")}
    assert profiles["before"]["spectral_kernel_counts"]["solve_alias"] == 80
    assert profiles["combined"]["spectral_kernel_counts"]["forward_scale1"] == 78
    assert profiles["combined"]["spectral_kernel_counts"]["backward_scale1"] == 78
    assert profiles["combined"]["spectral_kernel_counts"]["solve_alias"] == 2
    frozen = read(base / "before_manifest.json")["files"]
    unchanged = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n, h in frozen.items()}
    assert all(unchanged.values()), "Baseline production/model inputs changed"
    summary = dict(status="complete_experimental_not_promoted", operators=operator_rows, models=models,
        inherited_python_noninferiority_failures=failures, checks=checks, nsight=profiles,
        frozen_inputs_unchanged=unchanged, production_default_changed=False,
        limitation="No native parity, convergence, full-image or time-to-quality claim; five inherited max_abs failures retain the Python noninferiority gate")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=summary["status"], models=models, checks=checks,
                         inherited_failures=len(failures)), indent=2))


if __name__ == "__main__":
    main()
