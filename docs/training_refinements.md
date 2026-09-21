# FP32 训练继续优化：核准备与 s=1 分派

2026-09-17，RTX 5060 Ti 16GB，Windows WDDM，PyTorch 2.11.0+cu130，CUDA Toolkit 13.2。
按用户最新要求，本轮只推进 FP32；FP16/BF16 草稿已从生产包装层撤下并单独归档。

本轮默认路径相对**上一轮已优化实现**，B8/B16、s3 的缩小 USRNet 完整训练步再加速约
**1.10×**，256²、s1 完整步再加速约 **1.06–1.07×**。可微核谱复用已实现，但完整模型
质量门槛仍有失败，因此它是显式候选，**默认关闭**。真实数据集评估入口已准备，未完成数据集收敛验证。

## 默认实现

1. 核准备分派同时考虑空间面积和实际核数量：`H*W >= 16384` 或
   `KB*KC*H*W >= 1048576`，且 `kh <= H/4` 时使用分离 FP64 FFT。
   小空间但每个样本/通道都有独立核的 DataNet 不再遗漏该路径。共享核的 KB 按实际的 1 计算，
   不把输入 batch 错当成重复准备数量。
2. FP64 谱转 complex64 时直接生成连续布局，避免频谱核心在前向、反向重复复制列优先数据。
3. `s == 1 && H*W >= 65536` 使用专用前向/VJP，合并逐频点操作并按梯度需求分配临时量。
   广播核仍确定性地按 batch/channel 归约，不使用浮点 atomic 或 fast-math。其余形状保留 generic。

激活/求解仍为 FP32；核准备、FFT/IFFT、pad/roll、λ 参数化保持自动求导，高阶仍回退 ATen。
未修改推理数学路径、模型参数名或 FP32 数值预算。

## 增量基线与测量

修改前冻结了生产 C++/CUDA/头文件、Python 模型、构建和基准入口，位于
`artifacts/training_refinements/source_before`，清单指纹为 `bc3926dc5ddc32e1`。
基准校验所有 SHA256 后从源码隔离构建，同时加载冻结 Python 模型，确保原 forward 不受候选模型改动影响。

**本报告 JSON 中兼容旧格式的 `dev` 字段表示该冻结的上一轮实现，不是原始 dev 分支。**
本轮结果不能与此前对原始 dev 的速度比相乘。

默认路径三个独立进程，各 6 轮 × 20 步，预热 5 步后重置为相同参数和零 momentum，
before/current 交替，TF32 关闭，eager。包括 zero_grad、forward、MSE、backward、SGD
(lr=1e-4,momentum=.9)；核准备及反向每步计时，不计编译、数据加载/H2D。每个工作负载
先检查四步变化输入的 loss/参数/梯度/momentum，计时后检查有限值。

时间为每进程轮次中位数再取跨进程中位数，速度比用表中延迟相除。
原始汇总还提供逐轮配对比率及跨进程范围；不同聚合不必完全相同。

| 完整训练步 | 上一轮 / ms | 当前 / ms | 上一轮÷当前 | peak allocated 上一轮→当前 / MiB |
|---|---:|---:|---:|---:|
| Operator B1 C32 256² s1 | 2.725 | 2.562 | **1.06×** | 104.77→100.42 |
| Operator B4 C32 256² s1 | 9.929 | 9.286 | **1.07×** | 376.89→376.89 |
| Operator B1 C32 256² s2 | 12.247 | 11.861 | 1.03× | 356.80→356.80 |
| Operator B1 C32 256² s3 | 27.245 | 26.497 | 1.03× | 757.11→757.11 |
| Operator B4 C32 256² s2 | 35.633 | 35.208 | 1.01× | 1329.64→1329.64 |
| Operator B4 C32 256² s3 | 74.311 | 73.579 | 1.01× | 2810.14→2810.14 |
| USRNet B8 RGB 16×20 s3 | 16.807 | 15.230 | **1.10×** | 452.82→435.42 |
| USRNet B16 RGB 16×20 s3 | 41.779 | 38.181 | **1.09×** | 862.98→819.73 |
| USRNet B1 RGB 16×20 s2 | 10.189 | 9.905 | 1.03× | 47.41→47.41 |
| ConverseBlock B16 C16 32×40 | 2.869 | 2.882 | 1.00× | 63.02→63.02 |

USRNet 性能表仍是两轮迭代、一个 block、64 隐藏通道，统一 alpha=.1 的缩小模型；
不能当作默认完整规模模型收敛或达到同等 PSNR 的训练总时间。Block 未显示稳定收益。

显存为单个变体驻留时的 PyTorch 总 peak allocated/reserved，包含模型、输入、优化器和
分配器内仍存活的工作区；不含分配器外的驱动/库内存，不是进程总显存。reserved 已含 allocated。
例如 B16/s3 USRNet reserved 从 1106→1066 MiB；B1/s3 算子 F+B 的 reserved
从 1044→1114 MiB，不能概括为所有配置都减少 reserved。

## s=1 独立消融

从当前源码只把 `use_scale1` 的选择条件改为 false，在独立命名空间重建，其他准备、模型、
编译精度和输入保持一致。两个独立进程，各 6 轮 × 20 步，区分专用核本身与核准备的收益。

| 完整训练步 | generic / ms | 专用核 / ms | 比率 |
|---|---:|---:|---:|
| B1 C32 256² s1 | 2.624 | 2.546 | **1.03×** |
| B4 C32 256² s1 | 9.918 | 9.311 | **1.07×** |
| B1 C32 256² s2，负对照 | 11.877 | 11.862 | 1.00× |
| B1 C32 256² s3，负对照 | 26.520 | 26.527 | 1.00× |

s1 的纯 forward+VJP 收益分别约 1.05×/1.10×。空间小图没有默认开启此候选；本轮没有扩大
到“只要 batch 大就使用 s1 专用核”的规则。

## Profiling 证据

单独采集 B16/s3 的完整训练步，不把 profiler 时间当基准延迟。最终默认路径中，归属核准备的
FFT 前向 GPU 时间约从 **2.03→1.16 ms/步**，按 autograd sequence 匹配的 FFT 反向约从
**3.94→1.46 ms/步**。每步仍准备 4 个核谱，FFT 调用数从 4 增为 6，但因先裁剪填充区，
实际工作减少。

当前源码使用 `converse2d::prepare_training_kernel` 标记；冻结代码按实际 pad→roll→FFT
序列识别，反向按 sequence_nr 匹配。只计 autograd engine 层，不重复计其内部 FftBackward
节点。各汇总组有包含关系，不能相加；per-use complex64 cast 在准备标记之外。

## 可微复用候选，默认关闭

```python
model = ConverseUSRNet(backend="cuda", reuse_training_spectra=True)
```

- 只在一次 forward 内复用重复固定权重；DataNet 每次生成的动态核不保留在复用列表。
- 保存 complex128 准备图，每次使用各自 cast 为连续 complex64，使共享 VJP 在 FP64 累加；
  不 detach，不跨 forward 或 optimizer.step 复用。
- key 包括 Tensor 身份、version、data pointer、grad_fn、requires_grad、空间尺寸、device 和 stream。
  grad_fn 检查覆盖 nonleaf 原位 detach 后版本/存储不变的情形。
- 嵌套作用域、线程/stream 隔离及异常 finally 清理均验证；缺少私有 API 的旧扩展透明回退。

候选在一轮独立缩小模型比较中，B8/s3、B16/s3 的总改动相对冻结基线分别约 1.14×/1.12×，
但 B1/s2 反而慢约 3%。这不是仅复用本身的独立速度比，不能与默认路径相乘。
更重要的是，其默认规模模型的短训练压力对照未过，故没有默认启用。

## 验证与未关闭的质量门槛

默认核心回归通过，包含新增 7 个 refinement 方法和 12 个作用域方法，以及之前的 FP32、
复数半谱、s3/大 batch、动态核、缓存、推理 Graph 和正确性套件。新增覆盖包含 s1 门槛两侧、
四种广播、15 种梯度需求、共享输入、高阶、conjugate/stride/非默认流、弱正则和总核谱量边界。
CPU-only 正确性 10 个方法通过、1 个 CUDA stream 方法跳过；预训练完整 USRNet 的
FP32/FP64 推理对照、生产 build_ext 和 sdist 通过。

严格完整模型审计**仍失败**，不能据核心回归宣称完整模型精度或收敛完成：

1. 固定 seed9214、默认 5 iterations/7 blocks、测试 alpha=.1，独立 FP64 门槛
   `atol=3e-5, rtol=3e-4` 下，冻结基线有 **16 个梯度张量、26 个逐点超限**。
   关闭复用的本轮实现与冻结基线该固定样例全部结果逐位相同；改用 FP64 图累加后的复用候选
   仍是相同 26 个失败位置。首次失败为第一层 1×1 卷积参数梯度，relative L2 约 `1.68e-6`，
   但逐点预算未满足。没有放宽容差、改 seed/λ/alpha，独立测试保持非零失败状态。
2. `--case usrnet-default-tiny --reuse-spectra` 的固定合成 SGD 压力配方也失败：四步检查中
   loss 已到约 `1.31e10`，两路径相对差 `3.893e-4` 超过 `3e-4` 门槛。
   没有修改学习率或残差门控来掩盖；未输出通过的完整模型性能或收敛结论。

这些是可复现的质量缺口，后续需拆分网络/频谱舍入贡献，并使用明确数据及训练配方验证
真实任务质量。FP16/BF16 按用户要求暂缓，相关草稿位于 `artifacts/training_refinements/deferred_amp`，
未纳入本轮默认实现或上述结果。

## 真实数据评估入口

`test/evaluate_usrnet_quality.py` 支持严格匹配 HR/LR 图像与共享/逐图 7×7 核，完整预训练
USRNet、scale2/3/4、RGB/Y PSNR/SSIM、裁边协议、current/pytorch 对照、分阶段时间和
allocated/reserved 峰值。不会自动 resize、生成退化数据或隐式归一化核。

仓库没有实际测试集或正式训练配方。当前仅完成 checkpoint、输入检查和明确标注的合成
I/O smoke，**未生成真实数据集质量结论**。仍需训练/验证集路径、退化方式、训练配置与多种子计划。

## 复现与原始记录

```powershell
& ./experiments/training_speed/run.ps1 test/test_training_refinements.py
& ./experiments/training_speed/run.ps1 test/test_training_scope.py
& ./experiments/training_speed/run.ps1 test/benchmark_training_refinements.py --iters 20 --rounds 6
& ./experiments/training_speed/run.ps1 test/training_s1_ablation.py --iters 20 --rounds 6
& ./experiments/training_speed/run.ps1 test/profile_training_refinements.py --case usrnet-b16-s3 --variant both
# 以下严格质量审计当前会报告失败：
& ./experiments/training_speed/run.ps1 test/test_full_usrnet_precision.py
# 有真实配对数据后：
& ./experiments/training_speed/run.ps1 test/evaluate_usrnet_quality.py --hr-dir DATA/HR --lr-dir DATA/LR --kernel DATA/kernel.npy --scale 3 --backend both
```

- [默认 FP32 三次汇总](../artifacts/training_refinements/fp32_summary.md)
- [s1 两次独立消融](../artifacts/training_refinements/s1_summary.md)
- [最终默认 profile](../artifacts/training_refinements/profile_final_default/usrnet-b16-s3.summary.json)
- [复用候选记录](../artifacts/training_refinements/reuse_candidate.json)
- [完整模型精度诊断](../artifacts/training_refinements/full_usrnet_c128_scope_diagnostic.json)
- [严格质量 gate 日志](../artifacts/training_refinements/full_precision_gate.log)
- [完整模型复用短训练失败](../artifacts/training_refinements/full_reuse_gate.log)

原始 artifacts 由 Git 忽略。复现增量基线需保留 `source_before` 和 manifest；旧二进制或单独
报告不能替代源码。未锁 GPU 频率，跨进程范围不是置信区间；不能外推未测设备和形状。
