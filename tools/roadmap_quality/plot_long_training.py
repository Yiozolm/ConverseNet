"""CPU-only PNG/SVG figures from a self-contained long-training audit JSON.

Plots validation against optimizer updates, never invented evaluation wall
times. Terminal process totals are optional; memory is drawn only when the
audit itself contains measured peak fields. Frozen worker/helpers are unused.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
COLORS = {17: "#2563eb", 29: "#d97706", 43: "#15956b"}
CLOSED = {"complete", "stable", "max_steps_reached", "budget_stopped"}
STATUS_LABELS = {"incomplete": "INCOMPLETE - progress snapshot", "invalid": "INVALID - inspect audit errors",
                 "quality_failed": "COMPLETE - quality gate failed",
                 "quality_gates_passed": "COMPLETE - paired quality gates passed"}
METRICS = (("rgb", "psnr_db", "RGB PSNR", "dB"), ("y", "psnr_db", "Y PSNR", "dB"),
           ("rgb", "ssim", "RGB SSIM", "SSIM"), ("y", "ssim", "Y SSIM", "SSIM"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def plot_inputs(report):
    traces, omitted = [], []
    for chain in report.get("trajectories", []):
        points = sorted(chain.get("completed_evaluations", []), key=lambda row: row["optimizer_steps"])
        trace = {"seed": chain.get("seed"), "variant": chain.get("variant"), "status": chain.get("status", "unknown"),
                 "leaf": chain.get("leaf", "unknown"), "endpoint_updates": chain.get("endpoint_updates"),
                 "points": points, "metrics": {}}
        for space, name, _, _ in METRICS:
            values = []
            for point in points:
                value = numeric(point.get(space, {}).get(name))
                if value is None:
                    omitted.append({"leaf": trace["leaf"], "optimizer_steps": point["optimizer_steps"],
                                    "metric": space + "_" + name, "original": point.get(space, {}).get(name)})
                values.append(value)
            trace["metrics"][space + "_" + name] = values
        traces.append(trace)
    return traces, omitted


def measured_peak_bytes(chain, report):
    # Current audits may omit peak-memory fields. Never infer them from state
    # sizes, CUDA reserved bytes or a different run's report.
    values = []
    for field in ("overall_peak_memory", "training_peak_memory"):
        value = numeric(chain.get(field, {}).get("allocated_bytes"))
        if value is not None and value >= 0:
            values.append(value)
    selected = set(chain.get("sessions", []))
    for session in report.get("sessions", []):
        if session.get("directory") in selected:
            for field in ("overall_peak_memory", "training_peak_memory"):
                value = numeric(session.get(field, {}).get("allocated_bytes"))
                if value is not None and value >= 0:
                    values.append(value)
    return max(values) if values else None


def terminal_inputs(report):
    rows = []
    for chain in report.get("trajectories", []):
        seconds = numeric(chain.get("summed_session_timings", {}).get("process_elapsed_wall_s"))
        if chain.get("status") not in CLOSED or seconds is None:
            continue
        rows.append({"seed": chain.get("seed"), "variant": chain.get("variant"),
                     "status": chain["status"], "updates": chain.get("endpoint_updates"),
                     "leaf": chain.get("leaf", "unknown"), "process_seconds": seconds,
                     "peak_allocated_bytes": measured_peak_bytes(chain, report)})
    counts = Counter((row["seed"], row["variant"]) for row in rows)
    ambiguous = [key for key, count in counts.items() if count > 1]
    differences = []
    for seed in COLORS:
        before = [row for row in rows if row["seed"] == seed and row["variant"] == "before"]
        current = [row for row in rows if row["seed"] == seed and row["variant"] == "current"]
        if len(before) == len(current) == 1 and before[0]["updates"] != current[0]["updates"]:
            differences.append({"seed": seed, "before_updates": before[0]["updates"], "current_updates": current[0]["updates"]})
    return rows, ambiguous, differences


def style_axes(axes):
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.grid(True, color="#e5e7eb", linewidth=.7, alpha=.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#9ca3af")
        axis.tick_params(colors="#374151", labelsize=9)


def validation_figure(report, traces, plt, Line2D):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), sharex=True)
    flat = axes.flatten()
    style_axes(flat)
    groups = Counter((trace["seed"], trace["variant"]) for trace in traces)
    handles, labels = [], []
    maximum = 0
    for trace in sorted(traces, key=lambda row: (str(row["seed"]), str(row["variant"]), row["leaf"])):
        seed, variant = trace["seed"], trace["variant"]
        color = COLORS.get(seed, "#6b7280")
        linestyle = "--" if variant == "current" else "-"
        marker = "s" if variant == "current" else "o"
        updates = [point["optimizer_steps"] for point in trace["points"]]
        maximum = max(maximum, *(updates or [0]), numeric(trace["endpoint_updates"]) or 0)
        for axis, (space, name, title, unit) in zip(flat, METRICS):
            values = [math.nan if value is None else value for value in trace["metrics"][space + "_" + name]]
            if updates:
                axis.plot(updates, values, color=color, linestyle=linestyle, linewidth=1.8,
                          marker=marker, markersize=3.6, markerfacecolor="white" if variant == "current" else color,
                          alpha=.95 if variant == "current" else .75)
            axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#111827")
            axis.set_ylabel(unit, fontsize=10)
        last = updates[-1] if updates else "none"
        label = f"seed {seed} / {variant} | {trace['status']} | eval {last}, updates {trace['endpoint_updates']}"
        if groups[(seed, variant)] > 1:
            label += " | " + Path(trace["leaf"]).name
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, markersize=4))
        labels.append(label)
    for axis, (_, _, title, unit) in zip(flat, METRICS):
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel(unit)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        if not axis.lines:
            axis.text(.5, .5, "No completed evaluation recorded", transform=axis.transAxes,
                      ha="center", va="center", color="#6b7280")
        axis.set_xlim(0, max(1, maximum * 1.025))
    for axis in axes[-1]:
        axis.set_xlabel("Optimizer updates", fontsize=10)
    state = STATUS_LABELS.get(report.get("status"), "UNVERIFIED AUDIT STATUS")
    figure.suptitle("USRNet validation - 100 fixed held-out crops", fontsize=16, fontweight="bold", y=.985)
    figure.text(.5, .94, state, ha="center", color="#b45309" if report.get("status") == "incomplete" else "#374151", fontsize=11)
    if handles:
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=2,
                      frameon=False, fontsize=8.2, handlelength=3, columnspacing=1.8)
    figure.text(.5, .025, "Before: solid. Current: dashed. Curves use completed evaluations only; no interpolation in wall time.\n"
                "Fixed validation crops with declared synthetic degradation. Not an external benchmark or proof of convergence.",
                ha="center", va="center", fontsize=8.4, color="#4b5563")
    figure.subplots_adjust(left=.085, right=.97, top=.88, bottom=.23, hspace=.3, wspace=.23)
    return figure


def terminal_figure(report, rows, ambiguous, differences, plt, Patch):
    has_memory = any(row["peak_allocated_bytes"] is not None for row in rows)
    endpoints = {(row["updates"], row["status"]) for row in rows}
    common_endpoint = next(iter(endpoints)) if len(endpoints) == 1 else None
    figure, axes = plt.subplots(1, 2 if has_memory else 1, figsize=(12 if has_memory else 9, 5.8), squeeze=False)
    flat = axes.flatten()
    style_axes(flat)
    if ambiguous:
        for axis in flat:
            axis.text(.5, .5, "Competing terminal branches: no selection or resource comparison", transform=axis.transAxes,
                      ha="center", va="center", wrap=True, color="#991b1b")
    elif not rows:
        flat[0].text(.5, .5, "No closed terminal session yet", transform=flat[0].transAxes,
                     ha="center", va="center", color="#6b7280")
    else:
        for index, seed in enumerate(COLORS):
            for variant, offset in (("before", -.2), ("current", .2)):
                selected = [row for row in rows if row["seed"] == seed and row["variant"] == variant]
                if not selected:
                    continue
                row = selected[0]
                values = [row["process_seconds"] / 60]
                if has_memory:
                    values.append(None if row["peak_allocated_bytes"] is None else row["peak_allocated_bytes"] / 2 ** 30)
                for axis, value in zip(flat, values):
                    if value is None:
                        continue
                    axis.bar(index + offset, value, width=.35, color=COLORS[seed] if variant == "before" else "white",
                             edgecolor=COLORS[seed], linewidth=1.4, hatch="///" if variant == "current" else None, zorder=3)
                    label = f"{value:.2f}" if common_endpoint else f"{row['updates']} updates\n{row['status']}"
                    axis.annotate(label, (index + offset, value),
                                  xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    flat[0].set_title("Sum of terminal trajectory process times", loc="left", fontsize=11)
    flat[0].set_ylabel("Minutes")
    if has_memory:
        flat[1].set_title("Measured allocated peak (maximum across sessions)", loc="left", fontsize=11)
        flat[1].set_ylabel("GiB; allocated, not reserved")
    for panel, axis in enumerate(flat):
        axis.set_xticks(range(3), [f"seed {seed}" for seed in COLORS])
        values = ([row["process_seconds"] / 60 for row in rows] if panel == 0 else
                  [row["peak_allocated_bytes"] / 2 ** 30 for row in rows if row["peak_allocated_bytes"] is not None])
        upper = max(values, default=0)
        axis.set_ylim(0, upper * 1.35 if upper > 0 else 1)
    figure.suptitle("USRNet recorded terminal resources", fontsize=15, fontweight="bold", y=.98)
    figure.text(.5, .92, STATUS_LABELS.get(report.get("status"), "UNVERIFIED"), ha="center", fontsize=10)
    if common_endpoint:
        figure.text(.5, .87, f"All selected trajectories: {common_endpoint[0]} updates / {common_endpoint[1]}",
                    ha="center", fontsize=9, color="#4b5563")
    handles = [Patch(facecolor="#64748b", edgecolor="#64748b", label="before"),
               Patch(facecolor="white", edgecolor="#64748b", hatch="///", label="current")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .13), ncol=2, frameon=False)
    if differences:
        note = "Endpoint updates differ: these elapsed bars are NOT equal-work speedups."
    else:
        note = "No speedup ratio is inferred from these terminal totals."
    recovery_parents = {row["parent_run_dir"] for row in report.get("failure_recoveries", [])
                        if row.get("status") == "verified_checkpoint_prefix"}
    recovered = [f"seed {row['seed']} {row['variant']}" for row in report.get("trajectories", [])
                 if recovery_parents.intersection(row.get("sessions", []))]
    recovery_note = ("\nRecovered failed prefix: " + ", ".join(recovered) + ". Its process cost is included; restart waits are excluded.") if recovered else ""
    figure.text(.5, .025, note + "\nProcess sums include repeated setup and exclude waits between sessions. Overlapping scopes are not added." + recovery_note,
                ha="center", fontsize=8.5, color="#4b5563")
    figure.subplots_adjust(left=.09, right=.97, bottom=.28, top=.80 if common_endpoint else .84, wspace=.25)
    return figure, has_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="long_training")
    parser.add_argument("--terminal-resources", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.prefix):
        parser.error("prefix must be a simple filename component")
    audit_path = args.audit.resolve()
    audit_digest = sha(audit_path)
    report = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if report.get("kind") != "paired_long_training_audit":
        parser.error("input must be the self-contained long-training audit JSON")
    output_dir = args.output_dir.resolve()
    kinds = ["validation"] + (["terminal"] if args.terminal_resources else [])
    outputs = [output_dir / f"{args.prefix}_{kind}.{suffix}" for kind in kinds for suffix in ("png", "svg")]
    manifest_path = output_dir / f"{args.prefix}_manifest.json"
    if any(path.exists() for path in [*outputs, manifest_path]):
        parser.error("choose fresh figure filenames; prior artifacts are preserved")
    config = (ROOT / ".build/matplotlib_long_training").resolve()
    if not config.is_relative_to(ROOT.resolve()):
        raise RuntimeError("MPLCONFIGDIR escaped the workspace")
    os.environ["MPLCONFIGDIR"] = str(config)
    config.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": audit_digest,
                         "axes.labelcolor": "#374151", "text.color": "#111827", "savefig.facecolor": "white"})
    traces, omitted = plot_inputs(report)
    figure = validation_figure(report, traces, plt, Line2D)
    figures = {"validation": figure}
    terminal_rows, ambiguous, differences = terminal_inputs(report)
    has_memory = False
    if args.terminal_resources:
        figures["terminal"], has_memory = terminal_figure(report, terminal_rows, ambiguous, differences, plt, Patch)
    try:
        for kind, value in figures.items():
            value.savefig(output_dir / f"{args.prefix}_{kind}.png", dpi=170,
                          metadata={"Software": "ConverseNet long-training plotter", "AuditSHA256": audit_digest,
                                    "AuditStatus": str(report.get("status"))})
            value.savefig(output_dir / f"{args.prefix}_{kind}.svg", metadata={"Date": None, "Creator": "ConverseNet long-training plotter"})
    finally:
        for value in figures.values():
            plt.close(value)
    if sha(audit_path) != audit_digest:
        raise RuntimeError("Audit JSON changed while rendering")
    manifest = {"kind": "long_training_figures", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "audit": str(audit_path), "audit_sha256": audit_digest, "audit_status": report.get("status"),
                "plotter_sha256": sha(__file__), "matplotlib": matplotlib.__version__, "python": sys.version,
                "backend": "Agg", "MPLCONFIGDIR": str(config), "gpu_execution": False,
                "traces": [{key: value for key, value in trace.items() if key not in ("points", "metrics")} for trace in traces],
                "nonfinite_points_omitted_as_gaps": omitted, "terminal_rows": terminal_rows,
                "ambiguous_terminal_groups": ambiguous, "unequal_endpoint_updates": differences,
                "memory_plotted": has_memory, "memory_note": "Only fields present in the audit are eligible; absent memory data is not estimated.",
                "outputs": {str(path): sha(path) for path in outputs},
                "interpretation": "Updates-axis validation and recorded terminal process totals only; no invented per-evaluation wall trajectory, equal-work speedup, external-test or convergence claim."}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"audit_status": report.get("status"), "outputs": [str(path) for path in outputs],
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
