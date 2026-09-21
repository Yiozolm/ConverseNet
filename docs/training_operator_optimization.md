# 生产训练算子优化

本页保存首轮生产接入的结果；后续核准备分派和 s=1 默认路径更新见
[FP32 训练继续优化](training_refinements.md)，不同轮次基线的比率不能相乘。

2026-09-17，分支 `codex/training-operator-optimization`，基于 `dev` 的
`b850e3885d566ad70cd90cd2213497215370980e`。结果来自 RTX 5060 Ti 16GB；尚未验证真实数据集收敛。

本次接通生产 FP32 半谱融合训练，减少重复分母计算，并优化可微核频谱准备。
三次独立进程测量中，缩小版 USRNet 完整训练步约加速 **1.23×**；两个 scale=2
算子完整训练步约加速 **1.60–1.74×**。存在明确取舍：256²、scale=1
约慢 **6%**，但总峰值 allocated 显存下降 **21%**，并满足本次固定 FP64 精度门槛。
没有将所有形状描述为加速，也没有把不同基线的加速比相乘。

## 实现

- 生产 `setup.py` 与测试 JIT 构建加入 `converse2d_training.cu`。
- CUDA、FP32、v7、GradMode 开启且任一输入需要梯度时，频谱核心使用融合前向和解析一阶反向。
  高阶导数仍重建可微 ATen 计算。CPU、FP64、低精度和 v2–v6 保留原训练路径。
- 核谱、激活 FFT、IFFT、pad/roll 和 λ 参数化均保持自动求导。
  每次调用重新准备核谱；不跨 optimizer.step 缓存，不用 detach 切断梯度。
- 训练核谱使用 FP64 准备后转 complex64，激活 FFT、输出和频谱求解仍为 FP32。
  当输出网格至少 16,384 像素、核高不超过网格高的四分之一时，先对核的 `kh` 行做
  横向 RFFT，再补零、居中并做纵向 FFT。反向会先裁剪填充行，再对仅 `kh` 行做横向
  FFT 反向；小图保留二维 FFT。此实现没有手写 FFT 反向或三角相位近似。
- JIT 记录源码/头文件、PyTorch/CUDA 及二进制指纹；`CONVERSE2D_SKIP_BUILD=1`
  拒绝旧二进制。同进程源文件变化要求重新启动。生产源码包包含训练头文件。

已有 s=1 专用 CUDA 实验仍保持独立，本次没有无条件启用。AMP 契约、推理策略和模型参数名未改动。

## 精度与回归

公开训练测试采用独立 full-FFT FP64 参考，输出 `atol=rtol=3e-5`，一阶梯度
`atol=rtol=5e-5`；方向二阶导数采用 `atol=2e-3, rtol=2e-4`。
弱正则使用 `eps=1e-8, bias=-40`，没有增大 λ 或放宽门槛。

新增 scale=1 高阶样例最初暴露了旧 dev 的核 FFT 舍入问题：一个核梯度的绝对误差
为 `2.118e-4`，整体相对 L2 为 `8.469e-7`，仍有一个元素超出逐点预算。
真实 dev 二进制与最初融合候选的对应高阶梯度逐位相同。保留原 seed、λ、容差，改进核准备后通过。
该诊断说明整体相对误差很小并不能替代逐点门槛。

验证通过：

| 测试入口 | 方法数 | 主要范围 |
|---|---:|---|
| `test_fp32_training.py` | 8 | FP64 输出/梯度、弱正则、全部 15 种梯度需求、共享输入、高阶、大图与矩形核、流、参数更新、USRNet SGD/累积 |
| `test_training_fusion.py` | 8 | 任意复半谱、四种广播、边界、gradcheck/gradgradcheck、共轭 view、版本检查 |
| `test_correctness.py --device cuda` | 11 | 独立空间求解、原接口、dtype、缓存和推理 |
| `test_batched_kernels.py` | 8 | 动态核、广播、DataNet、USRNet 训练 |
| `test_cache.py` | 3 | 缓存生命周期 |
| `test_cuda_graph.py` | 7 | 原推理 Graph 集成 |

合计 **45 个 CUDA 测试方法**，其中多项包含参数化子样例；不是 45 个输入样例。
CPU-only 构建运行原正确性套件，10 个方法通过、1 个 CUDA stream 方法跳过。
生产 `setup.py build_ext` 与 `sdist` 同时验证。

USRNet 数值回归包含四个变化输入的 momentum SGD 步及两个 microbatch 累积，对照 FP64
的 loss、所有已有梯度、参数和 momentum，并检查内部算子、数据分支及 KernelNet 有效更新。
这不是完整规模、多种子或达到同等 PSNR 的收敛验证。

## 完整训练步性能

Windows WDDM，PyTorch 2.11.0+cu130，CUDA Toolkit 13.2，MSVC 14.44，sm_120，TF32 关闭。
三个独立 Python 进程，各 6 轮 × 20 步，dev/current 交替；每个计时样本从相同参数和
零 momentum 状态开始，预热 5 步后重置。完整步包含 zero_grad、forward、MSE、backward、
SGD(lr=1e-4, momentum=0.9)。输入已在 GPU，不含数据加载/H2D，不用 CUDA Graph 或 AMP。
动态核准备及其梯度成本均在计时内，编译和检查在计时外。

下表时间为每个进程轮次中位数再取跨进程中位数，速度比用表中延迟相除。原始汇总另提供
逐轮配对比率和各进程范围；两种聚合不必完全相同。

| 完整训练步 | dev / ms | 当前 / ms | dev÷当前 | dev→当前 peak allocated / MiB |
|---|---:|---:|---:|---:|
| Operator B1 C32 64×80 s1 | 1.533 | 1.138 | 1.35× | 10.51→10.67 |
| Operator B1 C32 256² s1 | 2.549 | 2.690 | **0.95×** | 132.73→104.77 |
| Operator B1 C32 64×80 s2 | 2.336 | 1.341 | **1.74×** | 34.79→28.08 |
| Operator B1 C32 256² s2 | 19.623 | 12.287 | **1.60×** | 444.36→356.80 |
| ConverseBlock B1 C16 32×40 | 2.966 | 2.563 | 1.16× | 3.45→3.47 |
| 同上，两个 microbatch 累积 | 5.542 | 4.940 | 1.12× | 3.61→3.63 |
| USRNet B1 RGB 16×20 s2 | 11.666 | 9.451 | **1.23×** | 46.98→47.41 |

USRNet 使用真实模型的两轮迭代、一个 block、64 隐藏通道；各路径统一设置残差 alpha=0.1
以检查有效内部梯度。Operator 包含可训练 1×1 producer，确保算子输入也参与反向传播。

算子独立 forward+VJP 的延迟为：64×80 s1 `1.073→0.681 ms`，256² s1
`1.948→2.085 ms`，64×80 s2 `1.908→0.913 ms`，256² s2 `18.039→10.677 ms`。
这些时间不含 loss/optimizer，不能当成完整训练加速。

显存是每个样本仅驻留一个变体时的 **PyTorch 总 peak allocated/reserved**，包含模型、
优化器与输入，但不含驱动和库中 PyTorch 分配器以外的内存；不是整个进程显存。
reserved 已包含 allocated，不能相加。64×80 s2 完整步虽然 allocated 降低，reserved
仍从 48 MiB 升至 66 MiB；256² s1/s2 的 reserved 分别从 182→136、726→580 MiB。

全尺寸二维 FP64 核 FFT 的中间候选曾使 256² s1 完整步从 2.573 增至 3.575 ms。
因此没有直接保留它；分离 FFT 大幅降低了这一成本，但没有消除该形状相对旧 FP32 dev 的全部时间代价。
未锁定 GPU 频率，三次范围不是置信区间，也不能外推其他 GPU 或未测形状。

## 复现

```powershell
& ./experiments/training_speed/run.ps1 test/test_fp32_training.py
& ./experiments/training_speed/run.ps1 test/test_training_fusion.py
& ./experiments/training_speed/run.ps1 test/benchmark_fp32_training.py --iters 20 --rounds 6
```

基准从 Git 固定提取 dev 源码，在独立 namespace 构建，并检查当前生产训练图确实包含
`SpectralSolve`。不依赖旧安装包，JSON 记录源码指纹、每轮时间、短训练检查及显存。

- [三次测量汇总与波动](../artifacts/training_operator_optimization/summary_final.md)
- [第 1 次](../artifacts/training_operator_optimization/benchmark_final1.json)、[第 2 次](../artifacts/training_operator_optimization/benchmark_final2.json)、[第 3 次](../artifacts/training_operator_optimization/benchmark_final3.json)
- [最终公开训练验证日志](../artifacts/training_operator_optimization/fp32_training_separable.log)
- [旧 dev 精度问题诊断](../artifacts/fp32_higher_diagnostic.json)

原始 artifacts 由 Git 忽略；以上结果表和可复现脚本保留在版本管理中。

## s=3 和大 batch 补测

保持本轮生产源码不变的 s3、B4/B8/B16/B32 数值与性能扩展，见
[s=3 与大 batch 训练补测](training_scale3_batch.md)。该报告区分共享核算子与实际
USRNet 的完整训练步，不能把单算子的速度比直接用作整网收益。
