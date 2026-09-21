"""CPU-only preservation and official Perfetto-import audit of a compact trace.

Never edits the raw/compact trace, mapping report or skill. The installed
Perfetto package may fetch its SHA256-pinned CPU trace_processor binary on first
use. No PyTorch or GPU profiler is imported or invoked.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/Boyce/.codex/skills/torch-profiler-layer-track")
sys.dont_write_bytecode = True  # Importing the skill must not create its .pyc files.

SQL = {
    "kernel_count": "SELECT count(*) AS kernel_count FROM slice WHERE category = 'kernel';",
    "activity_lanes": """SELECT p.pid, th.tid, th.name, count(*) AS events,
       count(DISTINCT s.track_id) AS imported_tracks, max(s.depth) AS max_depth
FROM slice s
JOIN thread_track tt ON s.track_id = tt.id
JOIN thread th ON tt.utid = th.utid
JOIN process p ON th.upid = p.upid
WHERE s.category IN ('kernel', 'gpu_memcpy', 'gpu_memset')
GROUP BY p.pid, th.tid, th.name ORDER BY p.pid, th.tid;""",
    "activity_total": """SELECT count(*) AS activity_count,
       count(DISTINCT track_id) AS imported_tracks, max(depth) AS max_depth
FROM slice WHERE category IN ('kernel', 'gpu_memcpy', 'gpu_memset');""",
    "guide_total": """SELECT count(*) AS guide_count,
       count(DISTINCT track_id) AS imported_tracks, max(depth) AS max_depth
FROM slice WHERE category = 'layer_guide';""",
    "guides_by_name": """SELECT name, count(*) AS occurrences, max(depth) AS max_depth
FROM slice WHERE category = 'layer_guide'
GROUP BY name ORDER BY CAST(substr(name, 2) AS INT);""",
    "guide_lanes": """SELECT p.pid, th.tid, th.name, count(*) AS guides,
       count(DISTINCT s.track_id) AS imported_tracks, max(s.depth) AS max_depth
FROM slice s
JOIN thread_track tt ON s.track_id = tt.id
JOIN thread th ON tt.utid = th.utid
JOIN process p ON th.upid = p.upid
WHERE s.category = 'layer_guide'
GROUP BY p.pid, th.tid, th.name;""",
    "trace_bounds": "SELECT start_ts, end_ts FROM trace_bounds;",
    "flow_count": "SELECT count(*) AS imported_flow_count FROM flow;",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_list(document):
    return document if isinstance(document, list) else document["traceEvents"]


def top_metadata(document):
    return None if isinstance(document, list) else {key: value for key, value in document.items() if key != "traceEvents"}


def reverse_compaction(document, report):
    events = event_list(document)
    assert len(events) == report["output_event_count"]
    count = report["new_metadata_count"]
    assert count > 0 and all(event.get("ph") == "M" for event in events[-count:])
    restored = list(events[:-count])
    for retired in sorted(report["retired_track_metadata"], key=lambda item: item["source_event_index"]):
        restored.insert(retired["source_event_index"], retired["event"])
    relocations = 0
    for index, event in enumerate(restored):
        provenance = event.get("args", {}).get("_compact_gpu_track")
        if provenance is None:
            continue
        assert provenance["source_event_index"] == index
        event = {**event, "pid": provenance["pid"], "tid": provenance["tid"]}
        args = dict(event["args"])
        del args["_compact_gpu_track"]
        if provenance["had_args"]:
            event["args"] = args
        else:
            assert not args
            del event["args"]
        restored[index] = event
        relocations += 1
    assert relocations == report["relocated_events"]
    assert len(restored) == report["original_event_count"]
    output = restored if isinstance(document, list) else {**document, "traceEvents": restored}
    return output, relocations


def verify_solver_mapping(events, mapping, metadata):
    anchors = sorted((event for event in events if event.get("ph") == "X"
                      and "kernel" in event.get("cat", "").split(",")
                      and re.search(mapping["anchor_regex"], event.get("name", ""))
                      and event.get("pid") == mapping["pid"]), key=lambda item: item["ts"])
    scopes = [event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation"
              and event.get("name", "").startswith("Solver/L")]
    forwards = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and event.get("name") == "converse2d::forward":
            forwards[event.get("args", {}).get("External id")].append(event)
    call_map = metadata["call_map"]
    assert len(anchors) == mapping["matching_anchor_count"] == 80
    assert len(call_map) == len(scopes) == 80
    rows = []
    for index, anchor in enumerate(anchors):
        linked = forwards[anchor["args"]["External id"]]
        assert len(linked) == 1
        op = linked[0]
        parents = [scope for scope in scopes
                   if scope["pid"] == op["pid"] and scope["tid"] == op["tid"]
                   and scope["ts"] <= op["ts"]
                   and op["ts"] + op["dur"] <= scope["ts"] + scope["dur"]]
        assert len(parents) == 1 and parents[0]["name"] == call_map[index]["label"]
        assert call_map[index]["step"] == index // 40 and call_map[index]["index"] == index % 40
        rows.append(dict(pass_index=index // 40, solver=index % 40,
                         cpu_scope=parents[0]["name"], external_id=anchor["args"]["External id"]))
    return dict(verified_anchor_count=80, passes=2, solvers_per_forward=40,
                all_anchors_uniquely_linked_to_cpu_solver_scopes=True,
                interpretation="L0..L39 are zero-based Converse solver calls, not all neural-network layers; L(8*i) is DataNet and next seven are prior blocks",
                links=rows)


def verify_json(args):
    sys.path.insert(0, str(args.skill / "scripts"))
    from add_layer_track import annotate
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    assert mapping["source_sha256"] == sha256(args.raw)
    assert mapping["source_file"] == args.raw.name
    assert mapping["pid"] == 0 and mapping["device"] == 0
    assert mapping["num_layers"] == 40 and mapping["passes"] == 2 and mapping["anchor_offset"] == 0
    assert mapping["phase"] == "USRNet-forward-solver"
    recreated, _ = annotate(raw, **{key: mapping[key] for key in
                                    ("anchor_regex", "num_layers", "anchor_offset", "passes", "phase",
                                     "evidence", "pid", "device", "first_layer", "end_anchor_regex")})
    restored, relocated = reverse_compaction(compact, mapping["compaction"])
    assert restored == recreated, "Reverse compaction did not exactly reconstruct annotated intermediate"
    raw_events, recreated_events = event_list(raw), event_list(recreated)
    assert recreated_events[:len(raw_events)] == raw_events
    assert len(recreated_events) - len(raw_events) == mapping["added_event_count"] == 82
    assert top_metadata(raw) == top_metadata(restored) == top_metadata(compact)
    kernel_count = sum(event.get("ph") == "X" and "kernel" in event.get("cat", "").split(",") for event in raw_events)
    guide_count = sum(event.get("cat") == "layer_guide" for event in event_list(compact))
    assert kernel_count == 18282 and guide_count == 80
    source_matches = {name: dict(expected=value, actual=sha256(ROOT / name),
                                 matches=sha256(ROOT / name) == value)
                      for name, value in metadata["source_sha256"].items() if (ROOT / name).is_file()}
    return dict(passed=True, source_trace_sha256=mapping["source_sha256"],
                raw_event_count=len(raw_events), annotated_event_count=len(recreated_events),
                compact_event_count=len(event_list(compact)), restored_relocations=relocated,
                retired_metadata_restored=len(mapping["compaction"]["retired_track_metadata"]),
                restored_flow_relocations=mapping["compaction"]["relocated_flows"],
                exact_annotated_intermediate_reconstructed=True, exact_raw_event_prefix_reconstructed=True,
                non_event_metadata_unchanged=True, raw_kernel_count=kernel_count, guide_count=guide_count,
                raw_category_counts=dict(Counter(event.get("cat", "<metadata>") for event in raw_events)),
                compaction_groups=mapping["compaction"]["groups"],
                captured_source_files=source_matches,
                solver_mapping=verify_solver_mapping(raw_events, mapping, metadata))


def verify_perfetto(args):
    sys.path.insert(0, str(args.tools))
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    from perfetto.trace_processor.platform import PlatformDelegate
    binary = PlatformDelegate().get_shell_path(str(args.trace_processor) if args.trace_processor else None)
    version = subprocess.check_output([binary, "--version"], text=True, encoding="utf-8", errors="replace", timeout=20).strip()
    config = TraceProcessorConfig(bin_path=binary, load_timeout=30)
    results = {}
    with TraceProcessor(trace=str(args.compact), config=config) as processor:
        for name, sql in SQL.items():
            results[name] = [vars(row) for row in processor.query(sql)]
    assert results["kernel_count"][0]["kernel_count"] == 18282
    activity = results["activity_total"][0]
    assert activity["activity_count"] == 19074 and activity["imported_tracks"] == 1 and activity["max_depth"] == 0
    assert len(results["activity_lanes"]) == 1
    assert all(row["imported_tracks"] == 1 and row["max_depth"] == 0 for row in results["activity_lanes"])
    guides = results["guide_total"][0]
    assert guides["guide_count"] == 80 and guides["imported_tracks"] == 1 and guides["max_depth"] == 0
    assert len(results["guide_lanes"]) == 1
    assert results["guides_by_name"] == [dict(name=f"L{index}", occurrences=2, max_depth=0) for index in range(40)]
    return dict(passed=True, engine="Official Perfetto TraceProcessor Python API and pinned native CPU parser",
                binary=binary, binary_sha256=sha256(binary), version=version,
                sql=SQL, results=results,
                interpretation="One imported synthetic GPU activity lane, not a reduction of runtime CUDA streams; layer guides are anchor intervals and have no exclusive kernel ownership")


def main():
    parser = argparse.ArgumentParser(__doc__)
    folder = ROOT / "artifacts/training_research/torch_current"
    parser.add_argument("--raw", type=Path, default=folder / "full.trace.json")
    parser.add_argument("--compact", type=Path, default=folder / "full.layers.trace.json")
    parser.add_argument("--mapping", type=Path, default=folder / "full.layers.trace.json.layers.json")
    parser.add_argument("--metadata", type=Path, default=folder / "metadata.json")
    parser.add_argument("--skill", type=Path, default=SKILL)
    parser.add_argument("--tools", type=Path, default=ROOT / "artifacts/training_research/tools")
    parser.add_argument("--trace-processor", type=Path, help="Optional already verified local CPU parser binary")
    parser.add_argument("--output", type=Path, default=folder / "verification.json")
    args = parser.parse_args()
    protected = [args.raw, args.compact, args.mapping, args.metadata,
                 args.skill / "SKILL.md", args.skill / "scripts/add_layer_track.py", args.skill / "scripts/compact_gpu_tracks.py"]
    if args.output.resolve() in [path.resolve() for path in protected]:
        parser.error("Output must not overwrite trace, metadata or skill source")
    before = {str(path.resolve()): sha256(path) for path in protected}
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status="running", verifier_sha256=sha256(Path(__file__)), before_sha256=before)
    code = 0
    try:
        report["json_preservation"] = verify_json(args)
        report["perfetto_import"] = verify_perfetto(args)
        report["status"] = "passed"
    except BaseException as error:
        code = 1
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    report["after_sha256"] = {str(path.resolve()): sha256(path) for path in protected}
    report["all_protected_files_unchanged"] = before == report["after_sha256"]
    if not report["all_protected_files_unchanged"]:
        report["status"], code = "failed", 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".sql").write_text("\n\n".join(f"-- {name}\n{sql}" for name, sql in SQL.items()) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], protected_files_unchanged=report["all_protected_files_unchanged"],
                          json_preservation=report.get("json_preservation", {}).get("passed"),
                          perfetto_import=report.get("perfetto_import", {}).get("passed"),
                          error=report.get("error", {}).get("message"), output=str(args.output))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
