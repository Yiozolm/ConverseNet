"""CPU/std-library-only summary of the complete declared16-case/mode matrix.

Subset diagnostics stay in their raw reports and cannot masquerade as this
full-matrix summary.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics


NAMES = ("old_statistics", "production_full_layernorm")
KINDS = ("eager_warm", "disabled_runner_warm", "graph_runner_hit")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--affinity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        parser.error("Choose fresh summary outputs")
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    affinity = json.loads(args.affinity.read_text(encoding="utf-8-sig"))
    assert report["status"] == "complete" and affinity["status"] == "complete" and affinity["task_exit_code"] == 0
    assert len(report["cases"]) == 16 and len({row["case"]["name"] for row in report["cases"].values()}) == 8
    assert {row["mode"] for row in report["cases"].values()} == {"no_grad", "inference_mode"}
    assert all(report[name] for name in ("model_state_unchanged", "tensor_versions_unchanged", "bindings_restored",
        "source_integrity_passed", "checked_binary_integrity_passed", "all_round_affinities_match"))
    rows, original_graph_differences = [], []
    for name, case in report["cases"].items():
        assert case["status"] == "complete" and case["strict_numerical_lifecycle_gates_passed"]
        assert case["eager_outputs"][NAMES[0]]["output"]["sha256"] == case["eager_outputs"][NAMES[1]]["output"]["sha256"]
        assert case["first_capture"][NAMES[0]]["output"]["sha256"] == case["first_capture"][NAMES[1]]["output"]["sha256"]
        for route in NAMES:
            assert case["first_capture"][route]["matched_control_equal"] and case["lru_miss"][route]["matched_control_equal"]
            assert case["disabled_runner"][route]["byte_equal"] and case["disabled_runner"][route]["captures"] == 0
            life = case["lifecycle"][route]
            assert all(life[key] for key in ("outputs_independent", "changed_input_and_kernel_match_control",
                "same_shape_layout_reused_graph", "clear_recaptured_once"))
            if not case["first_capture"][route]["original_caller_eager_equal"]:
                original_graph_differences.append(dict(case=name, route=route))
        for layout, probe in case["layernorm_layout_probes"].items():
            assert probe["routes"][NAMES[0]]["sha256"] == probe["routes"][NAMES[1]]["sha256"]
            assert probe["routes"][NAMES[1]]["fused_dispatch_count"] == (1 if layout == "contiguous" else 0)
        row = dict(case=name, spec=case["case"], seed=case["seed"], declared=case["declared"],
            original_graph_eager_equal={route: case["first_capture"][route]["original_caller_eager_equal"] for route in NAMES},
            actual_full_model_layernorm_dispatches=case["actual_operator_shapes"]["production_layernorm_dispatch_count"],
            warm={}, setup_observations={})
        for kind in KINDS:
            row["warm"][kind] = dict(routes={route:case["formal_timings"][kind][route]["median"] for route in NAMES},
                paired_ratios={metric:dict(median=statistics.median(values), minimum=min(values), maximum=max(values), rounds=values)
                               for metric,values in case["paired_ratios"][kind].items()})
        for kind, source in (("empty_eager_cache", "eager_outputs"), ("first_capture", "first_capture"), ("lru_miss", "lru_miss"),
                             ("clear_recapture", "lifecycle")):
            field = "empty_eager_cache_timing" if kind == "empty_eager_cache" else "clear_recapture_timing" if kind == "clear_recapture" else "timing"
            row["setup_observations"][kind] = {route:case[source][route][field] for route in NAMES}
        rows.append(row)
    ranges = {kind:{metric:[min(row["warm"][kind]["paired_ratios"][metric]["median"] for row in rows),
                           max(row["warm"][kind]["paired_ratios"][metric]["median"] for row in rows)]
                    for metric in ("wall_ms", "cuda_event_ms")} for kind in KINDS}
    summary = dict(kind="measured_existing_layernorm_deployment_summary", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        report=str(args.report.resolve()), report_sha256=sha(args.report), affinity_metadata=str(args.affinity.resolve()),
        affinity_metadata_sha256=sha(args.affinity), summarizer_sha256=sha(Path(__file__)),
        status="complete", case_mode_count=len(rows), wrapper_elapsed_wall_s=affinity["wrapper_elapsed_wall_s"],
        rows=rows, observed_paired_median_ratio_ranges=ranges, original_graph_eager_differences=original_graph_differences,
        scope="Representative shapes chosen from existing model/evaluator support, not observed production traffic. Existing production LayerNorm versus historical statistics only; no new dispatch/fusion support or training claim.",
        graph_scope="All matched capture-context controls passed. Original-caller eager differences are preserved; no_grad Graph must not be described as a byte-identical replacement for original no_grad eager.",
        setup_scope="Single initialized-process observations, not process-cold CUDA or library-plan startup; do not combine with warm medians.")
    lines = ["## 完整 GPU 测量结果", "", f"本次独占 GPU 进程正常退出，wrapper wall {affinity['wrapper_elapsed_wall_s']:.3f} 秒；{len(rows)}/16 个 case/mode 完整完成。",
        "模型值/version/绑定、source/binary、每轮亲和性检查通过。所有配对 eager 输出、capture-context controls、布局 fallback 与 Graph 生命周期门槛通过。",
        "", "每个完整 forward 均实际记录70次既有生产 LayerNorm dispatch、35次padding后的prior和5次DataNet。",
        "原始 caller 比较保留了8个no_grad case在两个route中的Graph/eager字节差异（16项）；8个inference_mode case则在两个route中均相等。",
        "因此不能把Graph表述成原no_grad eager的逐字节替代；这里的严格Graph gate使用相同inference tensor、contiguous clone及缓存资格。",
        "", "下表时间均为毫秒，`旧→生产`是各route的7轮wall中位数；倍数是逐paired round旧/生产比值的中位数，所以不必等于两列中位数直接相除。", "",
        "| Case / mode | 完整 eager 旧→生产 | disabled runner 旧→生产 | Graph hit 旧→生产 | paired eager / Graph 倍数 |",
        "|---|---:|---:|---:|---:|"]
    def times(values):
        return f"{values[NAMES[0]]['wall_ms']:.3f}→{values[NAMES[1]]['wall_ms']:.3f}"
    for row in rows:
        warm = row["warm"]
        lines.append("| " + row["case"] + " | " + " | ".join(times(warm[kind]["routes"]) for kind in KINDS)
            + f" | {warm['eager_warm']['paired_ratios']['wall_ms']['median']:.3f} / {warm['graph_runner_hit']['paired_ratios']['wall_ms']['median']:.3f} |")
    lines += ["", "这些逐case的paired wall中位数在完整eager为 "
        f"{ranges['eager_warm']['wall_ms'][0]:.3f}–{ranges['eager_warm']['wall_ms'][1]:.3f}×，Graph hit为 "
        f"{ranges['graph_runner_hit']['wall_ms'][0]:.3f}–{ranges['graph_runner_hit']['wall_ms'][1]:.3f}×。",
        "这是所测shape的观测范围，没有线上流量权重，不计算或声称整体部署收益；完整逐轮比值、CUDA event span和allocated peak增量保留在JSON。",
        "", "### 初始化进程中的 setup 观测", "", "下列每项是一次完整调用wall观测，单位毫秒，旧→生产。cuFFT/cuDNN/CUDA进程已初始化；不称为冷进程启动时间，也不与warm中位数合并。", "",
        "| Case / mode | 空 eager spectrum cache | 首次 runner capture | LRU miss | clear/recapture |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + row["case"] + " | " + " | ".join(times(row["setup_observations"][kind])
            for kind in ("empty_eager_cache", "first_capture", "lru_miss", "clear_recapture")) + " |")
    lines += ["", "首次runner时间包含warmup2、capture、copy、replay及output clone；它与已命中的runner是不同口径。显存数据是live state之上的allocated peak增量，不能当作总VRAM或跨shape容量上限。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with markdown.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(dict(status="complete", cases=len(rows), ranges=ranges,
        original_graph_eager_difference_count=len(original_graph_differences), output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
