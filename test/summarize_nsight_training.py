"""Read Nsight Systems SQLite exports and summarize training without a GPU.

CUDA kernel durations are summed work, not complete-step wall latency. NVTX
projects GPU launches to CPU ranges through runtime correlation IDs. Autograd
worker launches without a same-thread manual range use a separately labelled,
unique same-process temporal containment; ambiguous assignments stay unknown.
"""
import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import re
import sqlite3
import statistics

PROCESS_MASK = 0xFFFFFFFFFF000000
PHASES = {"prior", "solver_forward", "solver_backward", "zero_grad",
          "model_forward", "loss", "model_backward", "optimizer"}
SPECTRAL = re.compile(r"\b(?:forward_scale1|backward_scale1|filter_scale1|solve_alias|solve_output|adjoint_q|adjoint_inputs|adjoint_filter)\s*<")
MOVEMENT = re.compile(r"roll(?:_cuda)?_kernel|roll_cuda|constant_pad|pad_kernel|copy_kernel|direct_copy|copy_device_to_device", re.I)
SEQUENCE = re.compile(r"\b(?:stashed )?seq\s*=\s*(\d+)")
FFT_FORWARD = re.compile(r"^aten::(?:_fft_|fft_)")
FFT_BACKWARD = re.compile(r"\bFft\w*Backward\w*")


def base_name(text):
    return text.split(",", 1)[0].strip()


def process_id(global_tid):
    return global_tid & PROCESS_MASK if global_tid is not None else None


def durations(values):
    values = list(values)
    return dict(count=len(values), sum_ns=sum(values),
                median_ns=statistics.median(values) if values else None,
                min_ns=min(values) if values else None,
                max_ns=max(values) if values else None)


def activity(intervals):
    intervals = sorted((start, end) for start, end in intervals if end > start)
    if not intervals:
        return dict(sum_ns=0, union_ns=0, span_ns=0, overlap_ns=0,
                    kernel_free_inside_span_ns=0, union_fraction_of_span=None), []
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    total = sum(end - start for start, end in intervals)
    union = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    gaps = [(merged[i - 1][1], merged[i][0]) for i in range(1, len(merged))]
    return dict(sum_ns=total, union_ns=union, span_ns=span, overlap_ns=total - union,
                kernel_free_inside_span_ns=span - union,
                union_fraction_of_span=union / span if span else None), gaps


def attach_ranges(events, ranges, event_key, range_key, field):
    """Sweep nested ranges, avoiding an all-events x all-ranges join."""
    event_groups, range_groups = defaultdict(list), defaultdict(list)
    for event in events:
        event_groups[event_key(event)].append(event)
    for item in ranges:
        range_groups[range_key(item)].append(item)
    for key, rows in event_groups.items():
        candidates = sorted(range_groups[key], key=lambda row: row["start"])
        index, active = 0, []
        for event in sorted(rows, key=lambda row: row["start"]):
            while index < len(candidates) and candidates[index]["start"] <= event["start"]:
                active.append(candidates[index])
                index += 1
            active = [row for row in active if row["end"] >= event["start"]]
            event[field] = sorted((row for row in active if row["end"] >= event["end"]),
                                  key=lambda row: (row["end"] - row["start"], -row["start"]))


def manual_assignment(api, kind):
    def matches(row):
        name = base_name(row["name"])
        return name.startswith("NsightStep/") if kind == "step" else name in PHASES
    direct = [row for row in api["_thread_ranges"] if matches(row)]
    if direct:
        return direct[0], "same_thread_nvtx"
    candidates = [row for row in api["_process_manual_ranges"] if matches(row)]
    if len(candidates) == 1:
        return candidates[0], "unique_same_process_temporal"
    return None, "ambiguous" if candidates else "unassigned"


def category(name):
    # This partition is intentionally conservative and uses full kernel symbols.
    # Generic elementwise kernels are not guessed from their tensor shapes.
    if SPECTRAL.search(name):
        return "custom_spectral"
    if re.search(r"fft|cufft", name, re.I):
        return "FFT"
    if MOVEMENT.search(name):
        return "roll_copy_pad"
    return "other"


def aggregate(rows, key, total_ns):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row["end"] - row["start"])
    return sorted((dict(name=name, **durations(values),
                        fraction_of_kernel_sum=sum(values) / total_ns if total_ns else None)
                   for name, values in groups.items()), key=lambda row: -row["sum_ns"])


def table_rows(connection, table):
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if not exists:
        return []
    return [dict(row) for row in connection.execute(f'SELECT rowid AS _rowid, * FROM "{table}"')]


def summarize(path, top=20):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        names = {row["id"]: row["value"] for row in table_rows(connection, "StringIds")}
        kernels = table_rows(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        apis = table_rows(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
        raw_nvtx = table_rows(connection, "NVTX_EVENTS")
        memories = []
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            memories += [dict(row, activity_type=table.removeprefix("CUPTI_ACTIVITY_KIND_"))
                         for row in table_rows(connection, table)]
    if not kernels:
        raise ValueError(f"{path}: no CUDA kernels")
    ranges = []
    skipped_ranges = Counter()
    for row in raw_nvtx:
        if row["eventType"] not in (59, 60, 70, 71):
            continue
        if row["end"] is None:
            skipped_ranges["open_range"] += 1
            continue
        if row.get("endGlobalTid") not in (None, row["globalTid"]):
            skipped_ranges["cross_thread_range"] += 1
            continue
        row["name"] = names.get(row.get("textId"), row.get("text") or "")
        row["process"] = process_id(row["globalTid"])
        ranges.append(row)
    steps = [row for row in ranges if base_name(row["name"]).startswith("NsightStep/")]
    manual = [row for row in ranges if row in steps or base_name(row["name"]) in PHASES]
    for api in apis:
        api["name"] = names.get(api.get("nameId"), "unknown CUDA API")
        api["process"] = process_id(api["globalTid"])
    attach_ranges(apis, ranges, lambda row: row["globalTid"], lambda row: row["globalTid"], "_thread_ranges")
    attach_ranges(apis, manual, lambda row: row["process"], lambda row: row["process"], "_process_manual_ranges")
    runtime_by_correlation = defaultdict(list)
    for api in apis:
        runtime_by_correlation[(api["process"], api["correlationId"])].append(api)
        api["_step"], api["_step_method"] = manual_assignment(api, "step")
        api["_phase"], api["_phase_method"] = manual_assignment(api, "phase")

    prepare_ranges = [row for row in ranges if base_name(row["name"]).endswith("::prepare_training_kernel")]
    fft_forward_ranges = [row for row in ranges if FFT_FORWARD.match(row["name"])]
    attach_ranges(fft_forward_ranges, prepare_ranges, lambda row: row["globalTid"],
                  lambda row: row["globalTid"], "_prepare_ancestors")
    prepare_fft_sequences = set()
    for row in fft_forward_ranges:
        match = SEQUENCE.search(row["name"])
        if row["_prepare_ancestors"] and match:
            prepare_fft_sequences.add((row["process"], int(match.group(1))))

    correlation_status, phase_methods, step_methods = Counter(), Counter(), Counter()
    for kernel in kernels:
        name_id = kernel.get("demangledName", kernel.get("shortName", kernel.get("nameId")))
        kernel["name"] = names.get(name_id, str(name_id))
        kernel["category"] = category(kernel["name"])
        kernel["phase"] = "unassigned"
        kernel["step_id"] = None
        kernel["tags"] = set()
        candidates = runtime_by_correlation[(kernel.get("globalPid"), kernel.get("correlationId"))]
        if len(candidates) != 1:
            correlation_status["ambiguous" if candidates else "unmatched"] += 1
            kernel["_api"] = None
            continue
        correlation_status["unique"] += 1
        api = candidates[0]
        kernel["_api"] = api
        step_methods[api["_step_method"]] += 1
        phase_methods[api["_phase_method"]] += 1
        if api["_step"]:
            kernel["step_id"] = api["_step"]["_rowid"]
        if api["_phase"]:
            kernel["phase"] = base_name(api["_phase"]["name"])
        labels = [row["name"] for row in api["_thread_ranges"]]
        if any(base_name(label).endswith("::prepare_training_kernel") for label in labels):
            kernel["tags"].add("prepare_forward_all_ops")
            if any(FFT_FORWARD.match(label) for label in labels):
                kernel["tags"].add("prepare_fft_forward")
        if any(FFT_FORWARD.match(label) or FFT_BACKWARD.search(label) for label in labels):
            kernel["tags"].add("fft_nvtx_observed")
        if any(base_name(label) == "aten::roll" or "RollBackward" in label for label in labels):
            kernel["tags"].add("roll_nvtx_observed")
        for label in labels:
            match = SEQUENCE.search(label)
            if FFT_BACKWARD.search(label) and match and (api["process"], int(match.group(1))) in prepare_fft_sequences:
                kernel["tags"].add("prepare_fft_backward_sequence_matched")

    total_ns = sum(row["end"] - row["start"] for row in kernels)
    categories = aggregate(kernels, lambda row: row["category"], total_ns)
    phases = aggregate(kernels, lambda row: row["phase"], total_ns)
    grouped_kernels = aggregate(kernels, lambda row: row["name"], total_ns)
    tags = sorted({tag for row in kernels for tag in row["tags"]})
    tag_groups = []
    for tag in tags:
        values = [row["end"] - row["start"] for row in kernels if tag in row["tags"]]
        tag_groups.append(dict(name=tag, **durations(values),
                               fraction_of_kernel_sum=sum(values) / total_ns))
    per_step = []
    for step in sorted(steps, key=lambda row: row["start"]):
        rows = [row for row in kernels if row["step_id"] == step["_rowid"]]
        subtotal = sum(row["end"] - row["start"] for row in rows)
        gpu, _ = activity((row["start"], row["end"]) for row in rows)
        per_step.append(dict(name=step["name"], process=step["process"], nvtx_rowid=step["_rowid"],
                             host_nvtx_duration_ns=step["end"] - step["start"], kernel_count=len(rows),
                             gpu=gpu, categories=aggregate(rows, lambda row: row["category"], subtotal),
                             phases=aggregate(rows, lambda row: row["phase"], subtotal),
                             attributed_tags=[dict(name=tag,
                                                   **durations(row["end"] - row["start"] for row in rows if tag in row["tags"]),
                                                   fraction_of_kernel_sum=(sum(row["end"] - row["start"] for row in rows if tag in row["tags"]) / subtotal if subtotal else None))
                                              for tag in tags]))
    devices = []
    for device in sorted({row["deviceId"] for row in kernels}):
        rows = [row for row in kernels if row["deviceId"] == device]
        memory = [row for row in memories if row["deviceId"] == device]
        gpu, gaps = activity((row["start"], row["end"]) for row in rows)
        mem, _ = activity((row["start"], row["end"]) for row in memory)
        combined, combined_gaps = activity((row["start"], row["end"]) for row in [*rows, *memory])
        next_at_start = {}
        for row in sorted(rows, key=lambda item: item["start"]):
            next_at_start.setdefault(row["start"], row)
        top_gaps = []
        for start, end in sorted(gaps, key=lambda item: item[1] - item[0], reverse=True)[:top]:
            following = next_at_start[end]
            api = following["_api"]
            top_gaps.append(dict(start_ns=start, end_ns=end, duration_ns=end - start,
                                 next_kernel=following["name"], next_phase=following["phase"],
                                 next_launch_start_ns=api["start"] if api else None,
                                 next_launch_end_ns=api["end"] if api else None))
        queue = [max(0, row["start"] - row["_api"]["end"]) for row in rows if row["_api"]]
        devices.append(dict(device=device, kernel_activity=gpu, memory_activity=mem,
                            combined_kernel_memory_activity=combined,
                            kernel_free_gaps=durations(end - start for start, end in gaps),
                            combined_activity_free_gaps=durations(end - start for start, end in combined_gaps),
                            positive_queue_wait=durations(value for value in queue if value > 0),
                            nonpositive_queue_wait_count=sum(value == 0 for value in queue),
                            largest_kernel_free_gaps=top_gaps))
    launches = {row["_api"]["_rowid"]: row["_api"] for row in kernels if row["_api"]}
    by_thread = defaultdict(list)
    for api in launches.values():
        by_thread[api["globalTid"]].append(api)
    launch_spacing = []
    for tid, rows in by_thread.items():
        rows.sort(key=lambda row: row["start"])
        launch_spacing.append(dict(globalTid=tid, launch_count=len(rows),
                                   api_end_to_next_launch_start=durations(
                                       max(0, right["start"] - left["end"])
                                       for left, right in zip(rows, rows[1:]))))
    api_inside_steps = [api for api in apis if api["_step"]]
    api_summary = aggregate(api_inside_steps, lambda row: row["name"],
                            sum(row["end"] - row["start"] for row in api_inside_steps))
    for row in api_summary:
        row["fraction_of_api_duration_sum"] = row.pop("fraction_of_kernel_sum")
    checks = dict(kernel_count=len(kernels), nvtx_step_count=len(steps),
                  assigned_step_kernel_count=sum(row["kernel_count"] for row in per_step),
                  unassigned_step_kernel_count=sum(row["step_id"] is None for row in kernels),
                  unassigned_phase_kernel_count=sum(row["phase"] == "unassigned" for row in kernels),
                  category_sum_matches=(sum(row["sum_ns"] for row in categories) == total_ns),
                  phase_sum_matches=(sum(row["sum_ns"] for row in phases) == total_ns),
                  kernel_name_sum_matches=(sum(row["sum_ns"] for row in grouped_kernels) == total_ns),
                  step_count_partition_matches=(sum(row["kernel_count"] for row in per_step) +
                                                sum(row["step_id"] is None for row in kernels) == len(kernels)),
                  step_sum_partition_matches=(sum(row["gpu"]["sum_ns"] for row in per_step) +
                                              sum(row["end"] - row["start"] for row in kernels if row["step_id"] is None) == total_ns),
                  correlation=dict(correlation_status), step_assignment_methods=dict(step_methods),
                  phase_assignment_methods=dict(phase_methods), skipped_nvtx_ranges=dict(skipped_ranges),
                  prepare_fft_sequence_count=len(prepare_fft_sequences))
    if not all(checks[key] for key in ("category_sum_matches", "phase_sum_matches", "kernel_name_sum_matches", "step_count_partition_matches", "step_sum_partition_matches")):
        raise AssertionError(f"Partition checks failed: {path}")
    return dict(path=str(path), checks=checks, summed_kernel_ns=total_ns,
                kernels_per_step=len(kernels) / len(steps) if steps else None,
                median_projected_gpu_step_span_ns=statistics.median(row["gpu"]["span_ns"] for row in per_step) if steps else None,
                categories=categories, phases=phases, attributed_tags=tag_groups,
                devices=devices, steps=per_step, top_kernels=grouped_kernels[:top],
                launch_api_spacing=launch_spacing,
                cuda_apis_inside_step=api_summary,
                attribution_rules=dict(
                    correlation="correlationId and kernel.globalPid == runtime.globalTid & 0xFFFFFFFFFF000000; only unique matches accepted",
                    manual_ranges="Same-thread NVTX containment first; otherwise uniquely containing same-process manual range, explicitly marked temporal. CUDA API start AND end must be inside range.",
                    categories="Disjoint conservative full kernel-name classification: custom_spectral, FFT, roll_copy_pad, other. Unknown generic kernels remain other.",
                    preparation="Orthogonal direct same-thread prepare_training_kernel NVTX ancestry; kernel FFT backward additionally matches observed forward sequence IDs. Tagged groups overlap and cannot be added.",
                ),
                limitations=[
                    "Kernel sum adds all durations and can double-count concurrent time. Device union and span are separate; neither is complete-step wall latency.",
                    "NsightStep CPU range measures profiled host scope; asynchronous GPU work can extend beyond it. GPU step projection follows launches, not timestamp clipping.",
                    "Kernel-free gaps can contain memcpy/memset, dependency waits, untraced activity or profiling overhead. Launch spacing and queue waits alone do not establish a CPU bottleneck.",
                    "Cross-thread temporal manual attribution is not a structural NVTX parent relation. Unmatched or ambiguous correlations/ranges remain visible as unassigned.",
                    "CUDA API durations are CPU API intervals, may overlap GPU work, and must not be added to GPU kernel totals. Complete-step speed comes from the independent synchronized unprofiled benchmark.",
                ])


def markdown(reports):
    lines = ["# Nsight training summaries", "", "Kernel sums are accumulated GPU work, not complete-step wall time. GPU spans below are launch-projected profiled intervals. Use the independent synchronized benchmark for training speed.", ""]
    for report in reports:
        steps = report["checks"]["nvtx_step_count"]
        lines += [f"## {Path(report['path']).name}", "",
                  f"Steps: {steps}; kernels: {report['checks']['kernel_count']}; kernels/step: {report['kernels_per_step']}; correlation: {report['checks']['correlation']}.",
                  "",
                  "| Kernel class | Calls/step | Summed ms/step | Share of kernel sum |",
                  "| --- | --- | --- | --- |"]
        for row in report["categories"]:
            lines.append(f"| {row['name']} | {row['count'] / steps if steps else 0:.2f} | {row['sum_ns'] / 1e6 / steps if steps else 0:.3f} | {row['fraction_of_kernel_sum'] * 100:.2f}% |")
        lines += ["", "| Launch-attributed phase | Calls/step | Summed GPU ms/step |",
                  "| --- | --- | --- |"]
        for row in report["phases"]:
            lines.append(f"| {row['name']} | {row['count'] / steps if steps else 0:.2f} | {row['sum_ns'] / 1e6 / steps if steps else 0:.3f} |")
        lines += ["", "Orthogonal attribution (overlapping groups; do not add):", "",
                  "| Attribution | Calls/step | Summed GPU ms/step | Share of kernel sum |",
                  "| --- | --- | --- | --- |"]
        for row in report["attributed_tags"]:
            lines.append(f"| {row['name']} | {row['count'] / steps if steps else 0:.2f} | {row['sum_ns'] / 1e6 / steps if steps else 0:.3f} | {row['fraction_of_kernel_sum'] * 100:.2f}% |")
        lines += ["", "| Device | Kernel sum ms | Kernel union ms | Kernel span ms | Kernel-free gaps ms |",
                  "| --- | --- | --- | --- | --- |"]
        for row in report["devices"]:
            value = row["kernel_activity"]
            lines.append(f"| {row['device']} | {value['sum_ns'] / 1e6:.3f} | {value['union_ns'] / 1e6:.3f} | {value['span_ns'] / 1e6:.3f} | {value['kernel_free_inside_span_ns'] / 1e6:.3f} |")
        lines += ["", "Cross-thread phase attribution: " + str(report["checks"]["phase_assignment_methods"]),
                  "Kernel-free gaps and host launch spacing are diagnostic observations, not proof of a CPU bottleneck.", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("sqlite", nargs="+", help="Read-only SQLite paths or glob patterns")
    parser.add_argument("--output", type=Path, help="JSON output; defaults to stdout")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown tables")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    paths = []
    for pattern in args.sqlite:
        found = sorted(glob.glob(pattern))
        if not found:
            parser.error(f"No SQLite files match {pattern}")
        for path in found:
            path = Path(path).resolve()
            if path not in paths:
                paths.append(path)
    reports = [summarize(path, args.top) for path in paths]
    content = json.dumps(dict(reports=reports), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print("Saved", args.output)
    else:
        print(content)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(reports), encoding="utf-8")
        print("Saved", args.markdown)


if __name__ == "__main__":
    main()
