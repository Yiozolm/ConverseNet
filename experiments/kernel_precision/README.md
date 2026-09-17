# FP32 核频谱高精度预计算与缓存

2026-09-17。独立研究原型，未修改生产算子。输入、输出、激活 FFT 和频谱求解保持 FP32，仅核 FFT 可用 FP64；随后立即转 complex64 并沿用缓存生命周期。

用户关闭其他负载后已完整重跑。**推荐先做固定核的高精度预计算缓存；暂不默认采用每次未命中都同步检测的敏感核策略。**

- 1,104 组输入：最大相对 L2 误差 `3.89e-4 → 4.60e-7`，约降低 847 倍（各自最坏值的比值）。
- CUDA Graph 缓存命中延迟变化 `−0.9%～+1.8%`；六种形状的缓存计费字节数逐一相同，保留 complex64/FP32。
- B1 C64 128²、scale=2：eager 缓存未命中 `0.844→2.075 ms`，缓存命中均约 `0.51 ms`。FFT plan 已预热，这不是进程首次调用时间。
- 预设选择阈值升级 160/1104 组，但漏掉 22/125 组按事后标准判定明显受益的样例。检测的 FP32 FFT、归约和设备同步有实测成本。
- 高精度方案仍有 32 组超过严格逐点诊断门槛；极端样例输出可达数千，相对 L2 已约 1e-7。不能声称解决全部 FP32 舍入误差。

详细结果：[研究报告](../../artifacts/kernel_precision/report.md) · [原始数据](../../artifacts/kernel_precision/results.json) · [缓存命中复测](../../artifacts/kernel_precision/timing_repeat.json)。受干扰的首轮数据保留在 `artifacts/kernel_precision/with_background_load/`。

## 原型

`extension.py` 读取当前生产 C++/CUDA 源码，生成三个私有命名空间，构建到 `.build/kernel_precision/`。保存原始和生成源码哈希，不安装全局包、不加载旧 `.build/cuda` 二进制。

| 模式 | 缓存未命中时的核频谱 | 缓存命中 |
|---|---|---|
| `fp32` | 原 FP32 | 复用 |
| `kernel_fp64` | 小核转 FP64 → PSF 准备 → FP64 RFFT → complex64 | 复用 |
| `adaptive` | FP32 预估敏感性，分数≥1e4 才重算 FP64 | 复用，不检测 |

分数为 `max_filter(||K||_1² / (min(alias_mean(|FFT(K)|²)) + eps))`，只是启发式。使用 eps 下界避免依赖可变 bias，eps 本身加入 adaptive 缓存键。每个模式使用独立缓存；将来接入统一 API 时还须把策略加入缓存键。

adaptive 的训练路径直接采用高精度准备。未命中缓存的 graph capture 避免主机判定，改走高精度准备；仍须按 CUDA Graph 常规要求预热所需 FFT plans。本轮测试的是有缓存所有权的暖捕获与驱逐后 replay，不是任意新形状的无预热捕获。

缓存验证涵盖权重修改、身份、storage、eps、bias、非连续输入、stream、inference tensor 以及 graph 所有权。训练验证针对当前 ATen 一阶反向，尚未覆盖历史融合训练后端、高阶导数或收敛。

## 复现

使用本机现有 CUDA 13.2 / MSVC / PyTorch 环境脚本：

```powershell
& ./experiments/warp_spectral/run.ps1 -TaskArgs @('experiments/kernel_precision/study.py','--iters','60','--rounds','11')
& ./experiments/warp_spectral/run.ps1 -TaskArgs @('experiments/kernel_precision/benchmark_graph.py')
& ./experiments/warp_spectral/run.ps1 -TaskArgs @('experiments/kernel_precision/summarize.py')
```

每次 `study.py` 会重新生成主结果文件；需要保留旧结果时先复制。完整精度网格还会读取上一轮 `artifacts/accuracy_fp32_20260917/results.json` 中的 240 组案例定义。

实验调用（已配置编译环境）：

```python
import torch
from experiments.kernel_precision.extension import load_all

ops, hashes = load_all()
# x, prior, weight, bias 均为 CUDA FP32；weight 在 inference_mode 外创建。
with torch.no_grad():
    output = ops['kernel_fp64'].forward(x, prior, weight, bias, scale, 1e-5, 'v7')
```

固定核要能复用正常版本计数。`inference_mode` 内新建的权重没有版本计数，本原型沿用生产行为，不会跨调用缓存它；动态权重和训练每步更新也不能假设只付一次准备成本。
