"""CPU-only report from immutable B4 profile, gate and complete-layer timings."""
import json
from pathlib import Path
import statistics

import wgrad_common as common


def main():
    root = common.ROOT
    directory = root / 'artifacts/training_followup'
    names = dict(dataset='wgrad_profile_dataset_b4_a.json', synthetic='wgrad_profile_b4_b.json',
                 gate='wgrad_gate_b4_a.json', performance='wgrad_perf_b4_a.json')
    records = {name: json.loads((directory / filename).read_text()) for name, filename in names.items()}
    for name, record in records.items():
        if record['status'] != 'complete':
            raise RuntimeError('Incomplete evidence: ' + name)
    dataset, synthetic, gate, perf = (records[name] for name in ('dataset', 'synthetic', 'gate', 'performance'))
    for source in (dataset, synthetic):
        attribution = source['attribution']
        assert not attribution['unmatched_wgrad_kernel_ids']
        assert attribution['mutually_exclusive_name_categories']['cuDNN_wgrad']['count'] == 147
        assert sum(row['calls'] for row in attribution['exact_modules'].values() if row['type'] == 'Conv2d') == 147
        assert sum(row['calls'] for row in attribution['exact_modules'].values() if row['type'] == 'LayerNorm') == 70
        assert abs(sum(row['wgrad_gpu_us'] for row in attribution['calls']) - attribution['mutually_exclusive_name_categories']['cuDNN_wgrad']['gpu_us']) < 1e-6
        assert common.sha(Path(source['trace_path'])) == source['trace_sha256']
        assert common.sha(Path(source['fixtures_path'])) == source['fixtures_sha256']
    assert common.sha(directory / names['gate']) == perf['gate_sha256']
    assert common.sha(Path(perf['diagnostic_profile']['path'])) == perf['diagnostic_profile']['sha256']
    tensor_count = sum(len(row['snapshot']) for row in gate['cases'].values())
    byte_equal = all(value['sha256'] == row['python_fp32'][name]['sha256'] for row in gate['cases'].values()
                     for name, value in row['snapshot'].items())
    assert gate['all_gpu_cases_passed'] and byte_equal and tensor_count == 54
    assert common.sha(directory / 'wgrad_profile_b4_b.source.py') == synthetic['metadata']['tool_sha256']['wgrad_profile.py']
    old = {}
    for filename in ('pointwise_numeric.json', 'pointwise_split_numeric.json'):
        path = root / 'artifacts/fp32_roadmap' / filename
        record = json.loads(path.read_text())
        old[filename] = dict(sha256=common.sha(path), all_gpu_cases_passed=record['all_gpu_cases_passed'], failures=record['failures'])
    summaries = {}
    for name, source in (('dataset', dataset), ('synthetic', synthetic)):
        a = source['attribution']
        target = a['exact_modules']['p.m_body.0.conv1.5']['wgrad_gpu_us']
        summaries[name] = {key: a[key] for key in ('kernel_count', 'kernel_sum_us', 'kernel_union_us', 'kernel_span_us',
            'streams', 'mutually_exclusive_name_categories', 'exact_modules', 'convolution_shapes', 'calls')}
        summaries[name]['layernorm'] = {key: value for key, value in a['layernorm'].items() if not key.endswith('_ids')}
        summaries[name]['single_module_wgrad_fraction'] = target / a['kernel_sum_us']
        summaries[name]['ideal_kernel_sum_speedup_if_single_module_wgrad_removed'] = 1 / (1 - target / a['kernel_sum_us'])
    raw_paths = [path for path in directory.glob('wgrad*') if path.is_file() and path.name != 'wgrad_summary.json']
    summary = dict(kind='B4_wgrad_followup_summary', decision='Reject candidate on complete-layer performance; no model replacement, broader gate or training run.',
        summary_tool_sha256=common.sha(__file__), raw_sha256={path.name: common.sha(path) for path in raw_paths},
        dataset_protocol=dataset['dataset_protocol'], frozen_worker_step_record=dataset['frozen_worker_step_record'],
        production_metadata=dataset['metadata'], profiles=summaries,
        numerical=dict(cases=len(gate['cases']), active_cases=gate['active_cases'], tensors=tensor_count,
                       all_fp64_zero_margin_gates_passed=gate['all_gpu_cases_passed'], all_native_fp32_tensor_sha_equal=byte_equal,
                       higher_order=gate['higher_order'], failures=gate['failures']),
        performance=perf['cases'], diagnostic_profile=perf['diagnostic_profile'], retained_historical_failures=old,
        scope_limits=['Dataset profile uses the frozen worker and DatasetProtocol batch3 after batches0/1/2, seed17; only a diagnostic step, not quality/convergence.',
                      'Candidate gate/performance uses current-model activations with synthetic LR/target, one module across five invocations; no all-module or dataset candidate admission.',
                      'Historical B1 baseline and current B4 scope differ. Their percentages are not before/after speed comparisons.',
                      'LayerNorm module ownership is orthogonal to kernel-name categories and must not be added to them.',
                      'Profiler kernel timings diagnose work and cannot establish unprofiled kernel speedup. The complete-layer AB/BA timings determine rejection.'])
    common.write_report(directory / 'wgrad_summary.json', summary)
    a = dataset['attribution']
    total = a['kernel_sum_us']
    wgrad = a['mutually_exclusive_name_categories']['cuDNN_wgrad']['gpu_us']
    ln = a['layernorm']
    speedups = [row['median_speedup']['wall_ms'] for row in perf['cases'].values()]
    lines = [
        '# B4 实际训练的 1×1 wgrad 归因与受限候选', '',
        '本轮没有修改生产源码。受限 channels-last weight VJP 的 54 个输出/梯度张量均与原生 FP32 逐字节相同，也全部通过同输入 FP64 的零裕量双指标门槛；但完整单层前后向约慢 18–20%，因此停止，不做整网替换或扩大候选。', '',
        '## 实际数据协议下的主要证据', '',
        '当前 checked 模型与预训练权重，B4、LR32²、s3、HR96²，prior 实际 FFT100²。直接复用未改动的 `DatasetProtocol` 与冻结 `train_usrnet_dataset.train_step`：seed17 的前四个不同 batch，前三步预热，第四步采集。沿用 F.mse_loss、microbatch4 的权重乘法、Adam 1e-5、H2D、全部 backward、有限性/梯度范数检查；数据生成、哈希及 CPU 验证在 capture 外。不包含验证集评估或 checkpoint 保存。', '',
        'TF32/AMP 关闭，cuDNN benchmark 关闭，cuDNN deterministic 与 deterministic_algorithms 均为 true，CPU affinity 为 `[0,1,10,11,12,13,22,23]`。旧 B1 profile 源码也明确开启确定性算法，不能用一个并不存在的 lane 差异解释历史训练耗时。', '',
        f'仅统计 Chrome trace 中 `cat=kernel` 的 {a["kernel_count"]} 个实际 GPU kernel；总和与并集约 {total/1000:.6f} ms，首尾跨度 {a["kernel_span_us"]/1000:.6f} ms。147 次 1×1 wgrad 共 {wgrad/1000:.6f} ms，占 kernel 总和 **{100*wgrad/total:.4f}%**，全部映射到 module path 和具体调用，无未归属项。CPU/NVTX/user annotation 未加入分母。', '',
        '| 输入 → 输出通道 / 空间 | 调用数 | wgrad GPU ms | 总 kernel 占比 |',
        '|---|---:|---:|---:|',
    ]
    for label, value in a['convolution_shapes'].items():
        lines.append(f'| `{label}` | {value["calls"]} | {value["wgrad_gpu_us"]/1000:.6f} | {100*value["wgrad_gpu_us"]/total:.4f}% |')
    lines += ['', '64→128 的 70 次调用比 128→64 更重；不能把旧 B1 中两组接近的占比直接移植到当前 B4。以下每行是实际模块，prior 模块的五次调用来自五次迭代；逐调用 shape、sequence、kernel ID 与 CPU 祖先保留在原始 JSON。', '',
              '| 模块路径 | 次数 | wgrad GPU ms |', '|---|---:|---:|']
    for name, row in sorted(a['exact_modules'].items(), key=lambda pair: pair[1]['wgrad_gpu_us'], reverse=True):
        if row['type'] == 'Conv2d':
            lines.append(f'| `{name}` | {row["calls"]} | {row["wgrad_gpu_us"]/1000:.6f} |')
    lines += ['', '## LayerNorm 的实际范围', '',
        f'70 次 LayerNorm 的前向归属为 {ln["forward_gpu_us"]/1000:.6f} ms，后向归属为 {ln["backward_gpu_us"]/1000:.6f} ms，集合并集 {ln["union_gpu_us"]/1000:.6f} ms，占 **{100*ln["union_gpu_us"]/total:.4f}%**。前后向集合无重叠。', '',
        '归因方法：前向 module scope；后向从该次输出的 grad_fn 向上遍历，遇原输入 grad_fn 或 AccumulateGrad 即停止，用内部节点 sequence 对应真实 backward CPU 区间，再经 External id 映射 GPU kernel。它包含对应 evaluate_function 区间内的工作，不包含独立参数 AccumulateGrad。该视图与 kernel 名称分类正交，不可相加。本轮没有实现训练 LayerNorm 融合。', '',
        '## 受限候选、数值与停止原因', '',
        '候选只研究 NCHW FP32 `(4,128,96,96)`、权重 `(64,128,1,1)` 的一阶 weight VJP。前向保持 F.conv2d；输入和 bias 梯度保持原 NCHW ATen convolution_backward；仅 weight-only cuDNN 调用在 backward 内把输入与 grad_output 转成 channels_last，并把 weight 梯度恢复 contiguous。高阶完全回原生 ATen。不改变求解公式、FFT、频谱或缓存；不使用 AMP、TF32、fast-math。cuDNN 算法/累加次序可能变化，所以必须独立验收。', '',
        '这是 cuDNN 布局/算法候选，与此前 direct GEMM 和 split-K 的自定义矩阵归约路径不同。原失败各 5/6 个 dweight 项仍保留，不能重标成功。', '',
        '数值矩阵：当前模型 `p.m_body.0.conv1.5` 五次真实调用的 activation/grad_output（固定合成 LR/target 驱动）、3 个独立合成 seed、梯度子集和非连续布局，共15案例，12个候选实际启用，54个逐张量门槛全过且SHA相同。高阶 ddx/ddweight 的 max_abs 都为0。后补的 dataset profile 不用于追认该候选的 dataset 数值或全模块覆盖。', '',
        '| 该模块调用 | native 全层 wall ms | candidate 全层 wall ms | native/candidate | candidate 额外峰值 bytes |',
        '|---|---:|---:|---:|---:|']
    for name, row in perf['cases'].items():
        base, candidate = (row['timings'][key]['median'] for key in ('native', 'candidate'))
        lines.append(f'| `{name}` | {base["wall_ms"]:.6f} | {candidate["wall_ms"]:.6f} | {row["median_speedup"]["wall_ms"]:.4f}× | {candidate["peak_extra_allocated_bytes"]} |')
    target_fraction = summaries['dataset']['single_module_wgrad_fraction']
    lines += ['', f'以上为5预热、9轮×10次、轮换 AB/BA，完整 forward + dx + dweight + dbias，含所有布局转换、自定义 autograd/Python 开销和同步。五次调用的 wall 加速比仅 **{min(speedups):.4f}–{max(speedups):.4f}×**；CUDA event 同方向。原生额外峰值28,344,832 bytes，候选56,656,896 bytes。全部原始 rounds 保留。', '',
        '独立诊断 trace 确认 weight kernel 从 `wgrad_alg1_engine` 变为 `wgrad_alg1_engine_NHWC`；该次 kernel 时间386.938→263.548 µs，但增加了两个布局 copy kernel（共205.917 µs）和初始化等工作，kernel 数7→11。该单次 profile 只解释完整算子负收益，不作为局部性能排名。数值完全一致不代表 cuDNN 没有改变布局路径。', '',
        f'只替换已覆盖单模块时，其 dataset wgrad 占总 GPU kernel 时间 {100*target_fraction:.4f}%，即使完全删除这部分工作的理想 kernel-sum 上限也仅 {1/(1-target_fraction):.6f}×，且不是完整流程收益承诺。替换全部70个匹配调用需另补各模块真实数值 gate；本次因局部负收益没有进行。', '',
        '## 复现与证据', '',
        '工具位于 `tools/training_followup/wgrad_{profile,channels_last,gate,perf,report}.py` 和 `wgrad_common.py`。运行 GPU 工具需先设置 `CONVERSE_MSVC_VERSION=14.44`、`TORCH_CUDA_ARCH_LIST=12.0`，经原 `tools/run_affinity.ps1 -Mask 0xC03C03` 启动；checked loader 强制复用已验证二进制，不自动构建。', '',
        '主要文件：`artifacts/training_followup/wgrad_profile_dataset_b4_a.json`（实际数据归因）、`wgrad_profile_b4_b.json`（先前合成输入诊断）、`wgrad_gate_b4_a.json`（逐张量FP64/SHA）、`wgrad_perf_b4_a.json`（原始轮次及单层trace）。各自 trace/fixture/log/affinity 元数据保留；完整 SHA、生产与工具链 manifest、具体调用和失败记录索引见 `wgrad_summary.json`。', '',
        '最初 `wgrad_profile_b4_a.log` 记录 checked loader 因未指定匹配工具链环境而拒绝加载；当时没有进入模型。纠正进程环境后使用新文件名，未改 manifest 或重编译。增加 dataset 分支前的已测 profile 源已保存在 `wgrad_profile_b4_b.source.py`，其哈希与原报告一致。', '',
        'CPU 重新汇总：`.venv/Scripts/python.exe -B tools/training_followup/wgrad_report.py`。本报告不声称长期训练提速、质量改善或收敛证明。', '']
    (root / 'docs/training_followup_wgrad.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote wgrad_summary.json and docs/training_followup_wgrad.md')


if __name__ == '__main__':
    main()
