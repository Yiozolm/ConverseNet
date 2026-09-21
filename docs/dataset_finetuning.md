# 1000 张真实图片上的 FP32 配对微调

2026-09-17。使用用户提供的 Open Images 图片，从仓库预训练权重微调完整 ConverseUSRNet，比较当前实现与上轮优化前冻结快照。本文的实验是多种子短程微调，不能代替完整收敛或外部测试集验证。

本文的 `before` 是冻结 C++/CUDA 实现。另有 [同 GPU Python/ATen 训练基线对照](python_training_comparison.md)，它使用独立交替计时，不与本文加速比相乘。

**六次正式训练全部完成，三组配对的质量与数据一致性门槛全部通过。** 每次 250 updates，合计 1500 updates。当前模型的三种子平均保留集图块 RGB PSNR 从 12.417 提升到 **27.674 dB**，Y PSNR 从 14.220 提升到 **30.084 dB**；优化前后 final PSNR 最大差仅 0.000026 dB。完整流程时间与显存数据见下文，不将这次短程质量改善称为已收敛。

![三种子质量和训练步时间](../artifacts/dataset_training/summary.png)

## 数据与固定配方

实际图像目录为 `dataset/.fiftyone/zoo/open-images-v6/validation/data`。1000 张均可解码，文件或规范 RGB 像素完全重复数为 0；最短边 296，全部可直接裁剪 96×96，无需 resize 或排除。986 张原生 RGB、13 张灰度、1 张 CMYK，统一 EXIF transpose 后转 RGB；两个 MPO 文件使用首帧，不另外应用 ICC 色彩转换。审计没有检测感知近重复。

按稳定 image ID 哈希划分 **900 train / 100 validation**，不按目录遍历顺序拆分，完全相同像素不能跨集合。该批图原属 Open Images 官方 validation 子集，900/100 是本地实验划分，不是官方独立测试集。现有 checkpoint 原始训练数据的重合情况未知。

| 项目 | 本轮固定设置 |
|---|---|
| 模型 | 完整 5 iterations、7 blocks、64 features，307,987 参数 |
| 初始化 | `model_zoo/converse_usrnet.pth` 严格加载；所有种子与后端完全相同初值 |
| 参数 | 全参数微调；初始化不改写 checkpoint 的 alpha/λ，它们随 Adam 正常更新，不人为重设门控 |
| 精度 | FP32，关闭 AMP/TF32；cuDNN deterministic=True、benchmark=False |
| 训练图块 | 随机裁剪 HR96×96，随机旋转/翻转；不 resize |
| 退化 | float32 RGB/[0,1] → 仓库五个 7×7 核之一的循环卷积 → 相位 0 每隔 3 取样 → AWGN σ=.01 |
| LR 输入 | 32×32；不对加噪输入裁剪或量化 |
| Loss | 未裁剪、未取整输出的 RGB MSE |
| Optimizer | Adam，lr=1e-5，betas=(.9,.999)，eps=1e-8，weight_decay=0；foreach=False、fused=False |
| Scheduler / gradient clipping | 均关闭 |
| Batch | effective batch=4、microbatch=4 |
| 种子与范围 | 17、29、43；每种子每后端 250 updates，1000 个训练图块，约 1.11 个训练集遍历 |
| 验证 | step 0/125/250，各评估全部 100 张固定中心图块；kernel/noise 独立于训练种子 |
| 指标 | 输出 clip/round 至 uint8；现有 RGB 和 MATLAB-style Y PSNR/SSIM；每侧裁 3 像素 |

种子只改变样本顺序、裁剪、增强、kernel 与噪声，不改变预训练初始化。每个 epoch 按固定种子排列训练图；跨 epoch 继续重新取样。五个 kernel 中包含非对称核，使用 `scipy.ndimage.convolve`，不能直接拿未翻核的 `torch.conv2d` 相关运算替代。

CPU 自检验证了非对称核的循环卷积及相位，与独立 torch FP64 circular-pad + flipped-kernel conv 的最大差为 **2.98e-8**；跨 epoch 的训练批次逐位可复现，100 张验证数据跨训练种子逐位相同，三个层次的 path/file/RGB 哈希均无跨集交叉。

这是根据仓库模型与退化定义明确制定的新微调配方，仓库没有提供作者的正式训练配置。本轮未更改旧 FP64、弱正则或合成 SGD 压力测试配方与失败状态。

## 对照与预先约定的门槛

- `before`：冻结快照 `bc3926dc5ddc32e1` 的 C++/CUDA 及 Python 模型，即上一轮 refinement 之前；不是最初 dev 提交。
- `current`：当前生产实现，`reuse_training_spectra=False`。
- 对照采用相同权重、拆分、每步训练 tensor hash、全部验证 tensor hash。每次只运行一个 GPU 工作进程，种子间交替后端先后顺序。
- 所有训练 loss/gradient 和评估时参数、输出须为有限值；非有限时停止，跳过该次 Adam 更新。
- 每个种子的 final RGB/Y PSNR 相对 before 下降不得超过 **0.05 dB**，RGB/Y SSIM 下降不得超过 **0.001**。门槛在正式运行前写入 [protocol.json](../artifacts/dataset_training/protocol.json)，不会根据结果放宽。

离线汇总验证了每组 **250 个逐步输入哈希**、共同的初始化张量/原 checkpoint/拆分/验证 payload，以及 worker/data/metric 源码身份；六次训练均完成，完整性错误为 0。每次 3 次评估均覆盖准确的 100 张保留图。所有 loss/gradient 有限，评估时参数和输出检查通过；保存模型的状态哈希发生变化。代表性 seed17 的 checkpoint 重新严格加载并完成推理，133 个参数张量均已变化，包含核生成网络和 `d.alpha`。

| seed | current RGB PSNR | current Y PSNR | current RGB SSIM | current Y SSIM | Y PSNR 差 current−before | 四项门槛 |
|---|---:|---:|---:|---:|---:|---|
| 17 | 27.671879 | 30.085773 | 0.739549 | 0.794997 | −0.000008 dB | 通过 |
| 29 | 27.563129 | 30.078734 | 0.740248 | 0.798153 | −0.000019 dB | 通过 |
| 43 | 27.786746 | 30.086320 | 0.745138 | 0.797616 | −0.000026 dB | 通过 |
| 三种子均值 | 27.673918 | 30.083609 | 0.741645 | 0.796922 | −0.000017 dB | 全部通过 |

RGB PSNR 最大差为 0.0000032 dB，RGB/Y SSIM 最大差约 0.00000086，均远小于事前固定门槛。数据集微调质量门槛是额外证据，不替代独立 FP64 逐点梯度门槛，也不自动解除同 forward 复用候选的默认关闭状态。

## 时间与显存口径

每个训练步包括同步的 H2D、forward/backward、有限值与梯度范数检查、Adam；数据解码、裁剪、退化及批次哈希单独计时。整个训练循环 wall time 另外包含三次评估、checkpoint 与日志 I/O。脚本记录各分项，但初始 checkpoint 同时属于 setup，不能把所有累计项无条件相加。

这是关闭 profiler 的配对微调流程，仍包含审计用同步检查。去除前 5 步后的训练步中位数与含数据的累计时间分开报告，不与 Nsight 的 kernel sum 或旧缩小模型 benchmark 混用。显存为单进程 PyTorch allocated/reserved 总高水位，reserved 包含 allocated；不能称为驱动、库和整个进程的 GPU 总峰值。

| seed | 预热后步中位 ms，before→current | 含数据和全部250步 s，before→current | 循环含评估/I/O s，before→current |
|---|---:|---:|---:|
| 17 | 526.413 → 501.645 | 138.978 → 134.126 | 149.046 → 144.702 |
| 29 | 527.740 → 501.427 | 139.144 → 132.642 | 149.129 → 142.648 |
| 43 | 524.676 → 500.993 | 135.711 → 132.516 | 148.050 → 142.553 |

三种子的 before/current 配对比率中位数：预热后步时间 **1.0494×**（范围 1.0473–1.0525×），含数据与全部训练步 **1.0362×**（1.0241–1.0490×），包含评估/I/O 的循环 **1.0386×**（1.0300–1.0454×）。这些范围不是置信区间，也不是最终达到相同收敛质量的时间证明。

必须保留一次计时波动：before/seed43 的 steps6–125 中位为 534.37 ms，126–250 为 478.96 ms，226–250 又回到 529.06 ms；current 同种子两段约 501 ms。其他四次运行基本平稳。波动集中在前后向测量区间，现有记录无法确定原因；未剔除较快窗口，全部纳入累计时间与中位数。不能据此宣称跨环境固定或完全平稳的加速。

六次运行的峰值大小一致地分为：before allocated **10.174 GB / 9.475 GiB**，current **9.736 GB / 9.067 GiB**，下降 **4.31%**；reserved 为 before 10.733 GB、current 10.373 GB。这里没有包含 GPU 进程总显存，Windows WDDM 下本轮仍未获得可靠的进程总峰值。

首次容量检查中 batch4/micro4 的当前模型峰值 allocated 约 9.736 GB、reserved 约 10.373 GB，三个训练步 loss/gradient 均有限，因此未降低 batch 或修改学习率。容量检查不纳入正式 A/B 结果。

运行环境：RTX 5060 Ti 16 GB、Windows WDDM，Python 3.12.13、PyTorch 2.11.0+cu130、NumPy 2.3.5、SciPy 1.18.1、Pillow 12.2.0。源码及 backend build/frozen manifest 写入各次运行记录。

## 可视化与尚未完成的验证

下图使用 manifest 中前四个验证 ID，没有按输出好坏选图。当前 s3 协议下原 checkpoint 有明显周期性伪影，250 步后减轻，但微调结果仍可见残留，不能称生产质量或完整收敛。图片仅按指标口径裁剪显示范围，没有对预测做后处理。

![固定保留图块的训练前后对比](../artifacts/dataset_training/qualitative_examples.png)

仍需更长训练、全图及独立数据集评估、达到相同质量的总时间与可靠进程总峰值显存。完整模型既有严格 FP64 逐点梯度差距和旧合成 SGD 压力失败继续保留。本轮没有启用复用候选或开展 FP16/BF16 优化。

## 复现与产物

```powershell
# 需要数据审计产出的固定 manifest；新 run-dir 不能已经存在。
& ./experiments/training_speed/run.ps1 test/train_usrnet_dataset.py --variant current --seed 17 --run-dir artifacts/dataset_training/repeat_current17
& ./experiments/training_speed/run.ps1 test/train_usrnet_dataset.py --variant before --seed 17 --run-dir artifacts/dataset_training/repeat_before17
& ./experiments/training_speed/run.ps1 test/summarize_usrnet_training.py
```

- [训练入口](../test/train_usrnet_dataset.py)、[数据协议](../test/usrnet_training_data.py)。
- [最终配对汇总](../artifacts/dataset_training/paired_summary.md)、[结构化记录](../artifacts/dataset_training/paired_summary.json)、[离线核验脚本](../test/summarize_usrnet_training.py)。
- [代表性 seed17 微调权重](../artifacts/dataset_training/current_seed17/final.pth)、[seed29 权重](../artifacts/dataset_training/current_seed29/final.pth)、[seed43 权重](../artifacts/dataset_training/current_seed43/final.pth)。
- [数据审计](../artifacts/dataset_training/dataset_audit.md)、[逐图清单](../artifacts/dataset_training/split_900_100.json)、[退化与复现自检](../artifacts/dataset_training/data_protocol_checks.json)。
- 每个 `artifacts/dataset_training/{variant}_seed{seed}/` 保存 `run.json`、`training.jsonl`、`evaluations.jsonl`、`initial.pth`、`best.pth`、`final.pth`。checkpoint 是含 `state_dict`、optimizer 状态、步数与配置的字典。
- `best.pth` 按固定保留集 Y PSNR 选择；保留集同时用于模型选择，不能当外部测试结果。
- 原图片未修改。`dataset/` 与 `artifacts/` 被 Git 忽略，复现时须另外保留数据、manifest、冻结快照和检查点。

用户已授权在需要更多数据时通过 Python 下载 1 万张训练图片。当前 1000 张足以完成本轮短程配对检查，未为这轮额外下载；扩大训练或声称收敛/泛化时应使用独立训练图片并保留验证集。
