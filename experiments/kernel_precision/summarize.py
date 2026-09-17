"""Publish the clean-load rerun, retaining the earlier run for comparison."""
import collections
import json
import math
from pathlib import Path
import statistics

from extension import ROOT, NAMES

OUT=ROOT/'artifacts/kernel_precision'


def main():
    data=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    repeat=json.loads((OUT/'timing_repeat.json').read_text(encoding='utf-8'))
    accuracy=data['accuracy']
    summaries={}
    for name in NAMES:
        vals=[r['variants'][name] for r in accuracy]
        summaries[name]=dict(count=len(vals),max_abs=max(r['max_abs'] for r in vals),
            max_relative_l2=max(r['relative_l2'] for r in vals),median_relative_l2=statistics.median(r['relative_l2'] for r in vals),
            diagnostic_failures=sum(r['violations_3e5']>0 for r in vals),nonfinite=sum(not r['finite'] for r in vals))
    helpful=[r for r in accuracy if r['variants']['fp32']['relative_l2']>1e-6 and r['variants']['kernel_fp64']['relative_l2']<r['variants']['fp32']['relative_l2']/2]
    screens=[]
    for threshold in (100,300,500,1000,3000,10000,100000):
        selected=[r for r in accuracy if r['risk']>=threshold]
        values=[r['variants']['kernel_fp64' if r['risk']>=threshold else 'fp32'] for r in accuracy]
        heldout=[r for r in helpful if r['case']['seed']==260917]
        screens.append(dict(threshold=threshold,upgraded=len(selected),helpful=len(helpful),
            missed_helpful=sum(r['risk']<threshold for r in helpful),heldout_helpful=len(heldout),
            missed_heldout=sum(r['risk']<threshold for r in heldout),max_relative_l2=max(v['relative_l2'] for v in values)))
    benches=collections.defaultdict(dict)
    for row in data['benchmarks']:
        key=tuple(row['case']['shape']),row['case']['scale'],row['cache_mode']
        benches[key][row['variant']]=row
    graph=collections.defaultdict(dict)
    for row in repeat['rows']:
        if row['mode']=='hot_graph':graph[tuple(row['case']['shape']),row['case']['scale']][row['variant']]=row
    report=['# 核频谱高精度预计算与缓存研究','',
        '**建议优先采用：固定核在缓存未命中时用 FP64 生成频谱，转成 complex64 后缓存；先不要默认启用逐次同步的敏感核检测。**', '',
        '本报告使用用户关闭其他负载后的完整重跑：1,104 组精度样例、18 组训练梯度、3 个预训练模型，以及 11 轮 eager / 15 轮 CUDA Graph 性能测试。此前受干扰数据单独保存在 `with_background_load/`，未混入本轮统计。重跑前 GPU 利用率约 8%、显存约 1192 MiB；桌面 WDDM、未锁频。', '',
        '## 原型与范围','',
        '- fp32：当前源码的隔离基线。',
        '- kernel_fp64：只将小核转 FP64、补零/移位并做 FP64 RFFT，立即转回 complex64；功率、x/prior FFT、频谱求解、输出均保持 FP32。缓存仍存 complex64 频谱和 FP32 功率。',
        '- adaptive：先用 FP32 计算核频谱；在缓存未命中时计算敏感性指标并读回一个标量，超过预设阈值才重算 FP64。缓存命中时跳过检测。训练和未命中缓存的 graph capture 采用高精度准备。', '',
        '选择指标为 `max_filter(||K||_1² / (min_frequency(alias_mean(|FFT(K)|²)) + eps))`，阈值预先设为 1e4。它是启发式筛选，不是误差上界。eps 是 λ=sigmoid(bias−9)+eps 的下界，因此 bias 改变不会使决策依据失效；adaptive 的缓存键额外包含 eps。', '',
        '实验只在私有命名空间运行，生产 dispatch 未修改；以当前 v7 实现为底座，未将此次改动接入历史 FP32 融合训练快照或 nearest 实验。', '',
        '## 精度','',
        '| 方案 | 最大绝对误差 | 最大相对 L2 | 超过诊断门槛的样例 |',
        '|---|---:|---:|---:|']
    for name,r in summaries.items():report.append(f"| {name} | {r['max_abs']:.3e} | {r['max_relative_l2']:.3e} | {r['diagnostic_failures']}/{r['count']} |")
    report += ['',
        '基准为相同 FP32 输入值转为 FP64 后的独立 full-FFT 参考。诊断门槛保持 `3e-5 + 3e-5*abs(reference)`。没有 NaN/Inf。高精度方案仍有 32 组逐点超限，不能声称完全解决所有 FP32 误差；这些压力样例输出可达数千，其归一化误差已降到约 1e-7。', '',
        '主网格由上一轮 240 组及新正交网格 864 组构成。新网格包含三个种子、四个形状、scale=1/2/3、归一化随机核/带符号核/均值核/7×7 Gaussian，以及三组 eps/bias 和两种 prior。没有枚举全部核大小和权重范围。', '',
        f"最大相对 L2 从 {summaries['fp32']['max_relative_l2']:.3e} 降到 {summaries['kernel_fp64']['max_relative_l2']:.3e}，约降低 {summaries['fp32']['max_relative_l2']/summaries['kernel_fp64']['max_relative_l2']:.0f} 倍；这是两个方案各自最坏值的比值，不是同一样例的比值。", '',
        '## 敏感核筛选是否值得','',
        f"预设阈值 1e4 升级 {sum(r['selected_high'] for r in accuracy)}/{len(accuracy)} 组。把“FP32 相对 L2 大于 1e-6、且高精度至少改善两倍”作为事后受益标签，共 {len(helpful)} 组，其中漏选 {sum(not r['selected_high'] for r in helpful)} 组。", '',
        '| 阈值（事后模拟） | 升级样例 | 漏选受益样例 | 新种子 260917 漏选/受益 | 最大相对 L2 |',
        '|---:|---:|---:|---:|---:|']
    for r in screens:report.append(f"| {r['threshold']:g} | {r['upgraded']} | {r['missed_helpful']}/{r['helpful']} | {r['missed_heldout']}/{r['heldout_helpful']} | {r['max_relative_l2']:.3e} |")
    report += ['',
        '阈值 500 在这批数据上不漏选上述受益样例，但这是有限样本的事后分析，不能据此承诺全域安全。阈值 1000 恰在 Gaussian 核约 1000 的评分边界附近，浮点舍入即可改变分类，也说明不宜围绕某个样例硬调阈值。', '',
        '当前检测需要先做 FP32 FFT、功率归约并 `.item()` 同步，再决定是否做 FP64 FFT。固定核直接高精度预计算更简单；小图和需要升级的动态核上，同步检测的开销很容易超过节省的计算。较大且判为不敏感的动态核能比统一 FP64 更快，但仍比原 FP32 慢。', '',
        '## 性能：关闭其他负载后的重跑','',
        '以下为 eager CUDA Event 中位数，单位 ms。11 轮轮换顺序，缓存命中每轮 60 次，未命中/动态每轮 20 次；同时保存 wall time 和全部轮次。冷缓存指核频谱缓存未命中，FFT plan 已预热，不含编译和进程首次 FFT 初始化。动态模式使用不允许缓存的 inference tensor。', '',
        '| B×C×H×W / scale | 模式 | FP32 | 高精度核 | 选择策略 |',
        '|---|---|---:|---:|---:|']
    for (shape,s,mode),rows in benches.items():
        report.append(f"| {'×'.join(map(str,shape))} / {s} | {mode} | {rows['fp32']['event_ms']:.4f} | {rows['kernel_fp64']['event_ms']:.4f} | {rows['adaptive']['event_ms']:.4f} |")
    report += ['',
        'CUDA Graph 专门核对缓存命中后的纯设备执行。每图含 16 次调用，每轮 replay 20 次，共 15 轮轮换顺序；图捕获前为各版本建立独立、可持有的缓存。下列单位为 μs/次：','',
        '| 形状 / scale | FP32 | 高精度核 | 高精度变化 |',
        '|---|---:|---:|---:|']
    changes=[]
    for (shape,s),r in graph.items():
        baseline=r['fp32']['event_ms'];high=r['kernel_fp64']['event_ms'];change=(high/baseline-1)*100;changes.append(change)
        report.append(f"| {'×'.join(map(str,shape))} / {s} | {baseline*1000:.3f} | {high*1000:.3f} | {change:+.2f}% |")
    report += ['',f"高精度预计算后的稳态变化范围为 {min(changes):+.2f}% 至 {max(changes):+.2f}%。不能把这些小波动当成稳定加速。精度准备不会在缓存命中时重跑，测试计数器验证了这一点。",'',
        '例如 B1 C64 128²、scale=2，eager 冷缓存约 0.844→2.075 ms，额外约 1.23 ms；稳态约 0.508 ms。若一个缓存条目使用 100 次，单次平均额外成本约 0.0123 ms，即稳态的 2.4%；使用约 243 次后，此项摊销低于稳态的 1%。这只摊销准备成本，不含缓存容量不足造成的驱逐。', '',
        '## 缓存与训练检查','',
        '重复命中、权重原地修改、张量身份、storage 替换、非连续输入/不同 stream、eps 切换、graph 所有权及缓存驱逐、inference tensor 绕过缓存均通过。新增检查确认高精度频谱不会以 complex128 留在缓存。', '',
        '六种性能形状的三个方案缓存 payload 字节数逐一相同。例如 B1 C64 128²、scale=2 为 19,040,512 bytes（含 source 引用计数对应的数据量、complex64 频谱和 FP32 功率）。缓存未命中时仍需临时 FP64 张量；不能将缓存大小相同解读为峰值临时显存始终相同。', '',
        '18 组当前 ATen 训练路径的一阶梯度对照未见明显退化；weight 梯度最大绝对误差 8.414e-5→7.842e-5、最大相对 L2 5.497e-7→4.522e-7。prior 梯度接近零时，相对比值约 0.009，但绝对误差仅约 2e-8。尚未覆盖高阶梯度、完整训练收敛以及历史融合训练后端。', '',
        '三个预训练模型的本轮 FP64 对照：','',
        '| 模型 | 方案 | 最大绝对误差 | 相对 L2 |','|---|---|---:|---:|']
    for r in data['models']:report.append(f"| {r['model']} | {r['variant']} | {r['max_abs']:.3e} | {r['relative_l2']:.3e} |")
    report += ['',
        '整网包含卷积、归一化等其他 FP32 误差，高精度核 FFT 不保证每个模型的每项误差都降低。DnCNN 本轮略有变化，SRResNet/USRNet有所改善；这不是数据集 PSNR/SSIM。', '',
        '## 接入建议','',
        '1. 固定核推理：提供显式 `kernel_precision="fp64_precompute"` 策略，缓存未命中时高精度生成，缓存命中后走现有 FP32 算子。策略必须加入生产缓存键，不能混用已有低精度条目；本原型通过独立命名空间隔离。',
        '2. 训练/动态核：先保留显式选择，按用户精度需求和实测成本启用。训练每步改权重，无法直接享受跨步的固定核缓存收益。',
        '3. 自动敏感核判定：暂不默认落地当前同步阈值法。可继续研究离线标定、准备阶段一次性策略标签或不需主机同步的调度；需要扩大核大小、核分布、正则项和实际模型层的验证。', '',
        '生产源码本轮未修改。实验入口与复现命令见 `experiments/kernel_precision/README.md`。原始数据：`results.json`；稳态复测：`timing_repeat.json`；汇总：`summary.json`。','',
        '![误差与稳态成本](comparison.png)']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(dict(accuracy=summaries,threshold_sweep=screens,graph_percent_changes=changes),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    x=[max(r['variants']['fp32']['relative_l2'],1e-9) for r in accuracy]
    y=[max(r['variants']['kernel_fp64']['relative_l2'],1e-9) for r in accuracy]
    axes[0].scatter(x,y,s=10,alpha=.35,color='#167a87')
    axes[0].plot([1e-9,1e-3],[1e-9,1e-3],'--',color='gray',linewidth=1)
    axes[0].set(xscale='log',yscale='log',xlim=(1e-9,1e-3),ylim=(1e-9,1e-3),
        xlabel='FP32 kernel FFT: relative L2 vs FP64',ylabel='FP64 kernel FFT then c64: relative L2',title='1,104 paired FP32 output comparisons')
    labels=[];values=[]
    for (shape,s),r in graph.items():
        labels.append('x'.join(map(str,shape))+f' / s{s}')
        values.append(r['kernel_fp64']['event_ms']/r['fp32']['event_ms'])
    axes[1].scatter(values,labels,s=50,color='#5965a3')
    axes[1].axvline(1,color='gray',linestyle='--');axes[1].set_xlim(.9,1.1);axes[1].invert_yaxis()
    axes[1].set(title='Warm cache: CUDA Graph latency ratio',xlabel='High-precision preparation / FP32 baseline')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Kernel spectrum precompute | clean-load rerun | cache stays complex64')
    fig.savefig(OUT/'comparison.png',dpi=170)
    print(json.dumps(dict(accuracy=summaries,threshold_sweep=screens,graph_percent_changes=changes),indent=2))


if __name__=='__main__':main()
