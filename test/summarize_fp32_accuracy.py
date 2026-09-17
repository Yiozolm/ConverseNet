"""Summarize the paired accuracy audit without discarding failed comparisons."""
import collections
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/accuracy_fp32_20260917'


def main():
    data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    groups = collections.defaultdict(list)
    for row in data['rows']:
        group = row['group']
        if group in ('forward','training_forward') or group.startswith('gradient_'):
            group += '/stress' if row['case']['eps'] < 1e-5 else '/nominal'
        groups[group,row['version']].append(row)
    summary=[]
    for (group,version),rows in groups.items():
        finite=[r for r in rows if r['finite']]
        worst=max(finite,key=lambda r:r['max_abs']) if finite else None
        summary.append(dict(group=group,version=version,count=len(rows),nonfinite=len(rows)-len(finite),
            max_abs=max(r['max_abs'] for r in finite) if finite else None,
            max_relative_l2=max(r['relative_l2'] for r in finite) if finite else None,
            median_relative_l2=statistics.median(r['relative_l2'] for r in finite) if finite else None,
            max_abs_signed_mean=max(abs(r['signed_mean']) for r in finite) if finite else None,
            cases_exceeding_diagnostic_tolerance=sum(r['violations_3e5'] > 0 for r in finite),
            worst_absolute_error_case=worst['case'] if worst else None))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# FP32 版本精度统一复测（2026-09-17）','',
        f"实测 {len(data['rows'])} 条数值记录；异常 {len(data['errors'])} 条。GPU：{data['manifest']['gpu']}，PyTorch {data['manifest']['torch']}。",'',
        '所有被测输出及训练输入均为 FP32；仅参考计算使用 FP64。TF32 关闭。各版本共享已经量化为 FP32 的输入、参数和反向 upstream；FP64 参考由这些相同值转换得到。', '',
        '常规组使用 eps=1e-3 或 1e-5、bias=0；压力组使用 eps=1e-7、bias=-12。压力组不是默认配置。覆盖 3 个种子、10 个尺寸、倍率 1–4、独立/nearest prior、正归一化/带符号/均值核及 B/C 广播；核型和 eps 使用轮换设计，不是所有因素的完整笛卡尔积。', '',
        '最大绝对误差和 relative L2 均相对 FP64；两列各自取最坏值，可能来自不同样例。3e-5 + 3e-5×|参考| 是统一的诊断门槛，不代替各训练/模型测试的专用容差。没有通过调整门槛删除失败。', '',
        '梯度参考接近零时，relative L2 会失去实用意义：本轮 bias 梯度最大相对误差对应参考仅约 1e-17 至 1e-13，须同时查看绝对误差和 results.json 中的 reference_max。', '',
        '## 版本身份','',
        '- checkout_v2–v7：当前 Git 工作区重新构建，v3–v6 在当前源码中实际共用同一条实现路径，不代表六份历史实现。',
        '- pre_spectral_io_v7：Git 19c1bfc 的源码。',
        '- pre_batch_fft：artifacts/batch_fft/before.* 冻结快照。',
        '- nearest_before_c2r：artifacts/b_direct_integration/before.* 快照，已有 nearest 融合和 batch tiling。',
        '- snapshot_before_training：保存的 FP32 训练优化前快照，包含 C2R direct 等后续改进。',
        '- snapshot_after_training：验证快照原始哈希后，逐个严格匹配上下文应用保存的 changes.patch，在隔离命名空间构建。换行标准化后的源码哈希在 results.json。',
        '- nearest_cpp / nearest_spectral：当前 experiments 下两个独立实验。',
        '- warp 四版本：独立 scale=2 频谱实验，只比较其支持范围内的图像输出，不能与完整公开算子混排行。', '',
        '当前生产源码与旧 .build/cuda 二进制不匹配。本次使用 .build/accuracy_fp32_20260917，未覆盖旧产物；实验模块经其自身构建入口重建。', '']
    for group in sorted({r['group'] for r in summary}):
        lines += ['## '+group,'','| 版本 | 条数 | 最大绝对误差 | 最大相对 L2 | 中位相对 L2 | 超过诊断门槛的样例 |','|---|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['group']==group:
                lines.append(f"| {r['version']} | {r['count']} | {r['max_abs']:.3e} | {r['max_relative_l2']:.3e} | {r['median_relative_l2']:.3e} | {r['cases_exceeding_diagnostic_tolerance']} |")
        lines.append('')
    lines += ['## 边界与复现','',
        '模型加载本地 DnCNN、SRResNet、USRNet 预训练权重，使用 2 个种子 × 2 个输入尺寸。各模型参考是相同 FP32 参数转为 FP64 的 PyTorch 整网输出；对照还记录同一整网的 Python FP32 输出。为比较尚未接入包装层的实验版本，仅在测试进程中临时重定向 Converse 算子调用；模型其余部分保持相同。', '',
        '这是算子、梯度和整网数值一致性实验，不是有真值数据集上的 PSNR/SSIM、训练收敛或质量退化评估。未测 FP16/BF16。', '',
        '```powershell',
        "& ./experiments/warp_spectral/run.ps1 -TaskArgs @('test/accuracy_fp32_versions.py')",
        "& ./experiments/warp_spectral/run.ps1 -TaskArgs @('test/summarize_fp32_accuracy.py')",
        '```','',
        '原始数据：results.json；汇总：summary.json；图：accuracy.png。所有异常保存在 results.json 的 errors 字段。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(1,3,figsize=(19,6),constrained_layout=True)
    for ax,group,title in zip(axes,('forward/nominal','forward/stress','model'),('Operator / nominal regularizer','Operator / small regularizer stress','Pretrained models / synthetic inputs')):
        chosen=[r for r in summary if r['group']==group]
        names=[r['version'] for r in chosen]
        vals=[r['max_relative_l2'] for r in chosen]
        ax.scatter(vals,names,color=['#197a85' if 'checkout' in n else '#5965a3' for n in names],s=35)
        ax.set_xscale('log'); ax.invert_yaxis(); ax.set_title(title)
        ax.set_xlim((1e-6,1e-3) if group=='forward/stress' else (1e-7,1e-5))
        ax.set_xlabel('Worst relative L2 error vs FP64 (lower is better)')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('FP32 numerical accuracy audit | same inputs and weights | TF32 off')
    fig.savefig(OUT/'accuracy.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
