# s=2 训练频谱融合优化

2026-09-21。**已实现并验证可复用候选，默认未启用。** 候选把 s2 的四个 alias 交给同一线程，融合前向与部分反向，保留广播滤波器的独立归约。没有修改 s1/s3、核准备、FFT、缓存、低精度或 Converse 之外的层。

实施分支为 `codex/training-operator-optimization`，基线提交 `4c6314da1ee0147ad745a1a16df5e68602219226`。开始时工作区干净；冻结的默认生产依赖及模型文件在本轮前后逐份 SHA256 一致。[基线清单](../../HPC/artifacts/s2_train_20260921/before_manifest.json)。新文件 [training_scale2.cu](../Converse2D/torch_converse2d/training/training_scale2.cu) 不在默认构建清单，训练默认仍使用原分派。复现实验入口为 [study_training_scale2.py](../test/study_training_scale2.py)。

## 实现

1. 固定 `s=2`，消除倍率相关动态循环和计算；保留 const64 与 int32 两个消融版本。
2. 仅在 `s==2`、`p.numel()<=INT32_MAX-256`、`H/W<=INT32_MAX/2` 时使用 32 位索引；超出范围仍调用通用实现。维度、dtype、广播和梯度需求仍由原接口校验。
3. 前向在寄存器中保留四组滤波器和 prior，计算 q/d 后直接写出高分辨率半谱，减少一次 kernel 和重复全局读取。
4. 反向复用四组梯度／滤波器，合并 q 伴随和输入梯度；无核广播时同时生成核梯度。有 batch/channel 广播时保留按原 B/C 顺序的独立核梯度归约，无浮点原子操作。
5. 保留 q/d 保存量、高阶 ATen 回退、lambda 归约、显式范数舍入及 current stream。

| LR 半谱列 | 输出所有权 |
|---|---|
| `w=0` | 写四个直接 HR alias |
| 内部列 | 写四个 alias；镜像列使用共轭 correction |
| 偶数 W 的 `w=W/2` | 只写两个直接 alias，避免与镜像行争写 |

反向 Nyquist 列需要镜像行的 r/gd；其两个乘积已在本线程寄存器内，读取对应 q/d 后重算，避免跨线程同步。这里使用**寄存器复用**，没有新增手写 shared memory 流水线。

## 前向与全部 VJP

RTX 5060 Ti，Torch 2.11.0+cu130，CUDA 13.0，FP32，关闭 TF32/AMP。固定 CPU 随机输入和归一化 PSF，独立 prior；不是实测网络激活。H/W 均为 LR 尺寸，输出为 2H×2W。

包含核准备、FFT/IFFT、频谱求解和 x/prior/weight/bias 四类 VJP；不含 loss/optimizer、H2D、JIT 和 profiler。每路预热5次，六轮轮换，各30次。

| 场景 | 原实现 ms | 融合候选 ms | 配对加速中位数 |
|---|---:|---:|---:|
| B1/C3/3×5，共享 k3 | 0.630 | 0.645 | 1.005× |
| B1/C32/32×40，通道 k3 | 0.654 | 0.645 | 1.012× |
| B4/C32/64×80，batch 共享 k3 | 1.026 | 0.918 | 1.117× |
| B4/C32/64×80，动态 k3 | 1.910 | 1.841 | 1.040× |
| B8/C32/64×80，batch 共享 k3 | 2.259 | 2.125 | 1.064× |
| B1/C32/128²，通道 k3 | 1.680 | 1.631 | 1.034× |
| B1/C64/48²，动态 k7 | 0.783 | 0.754 | 1.045× |
| B4/C64/48²，动态 k7 | 1.764 | 1.691 | 1.046× |

加速列按逐轮比率取中位数，不等于耗时列简单相除。[工程化版本完整轮次](../../HPC/artifacts/s2_train_20260921/replay/operators_fused.json)。[初始 const64/int32 消融](../../HPC/artifacts/s2_train_20260921/operators.json) 与 [初始融合实验](../../HPC/artifacts/s2_train_20260921/operators_fused.json) 分别保留，不相乘或跨报告拼接。

## 算子完整训练步

单个 Converse 算子，固定输入，MSE、Adam1e-5，含 zero_grad、前向、loss、全部反向和 optimizer.step。每轮只有一路 fixture 驻留 GPU；预热5步后恢复相同参数并清零已分配的 Adam 状态，四轮 AB/BA 交替，各20步。仍不含数据搬运、验证和 profiler。

| 场景 | 原实现 ms/步 | 融合候选 ms/步 | 配对加速中位数 |
|---|---:|---:|---:|
| B4/C32/64×80，batch 共享 k3 | 1.309 | 1.203 | 1.087× |
| B4/C32/64×80，动态 k3 | 2.130 | 2.062 | 1.034× |
| B1/C64/48²，动态 k7 | 1.094 | 1.094 | 1.020× |
| B4/C64/48²，动态 k7 | 2.022 | 1.955 | 1.034× |

较大 batch 的样例有稳定方向的收益；B1/C64/48² 初测约回退2.4%，重放配对比约1.020×且耗时中位数基本相同，**不能据此宣称 B1 稳定加速**。候选尚未做经过验证的性能形状分派。[原始训练步](../../HPC/artifacts/s2_train_20260921/steps.json)、[工程化重放](../../HPC/artifacts/s2_train_20260921/replay/steps.json)。

## 正确性与精度门槛

- 16 项现有频谱／FP32训练测试通过，覆盖一阶／高阶、选择性梯度、弱正则、conjugate view、stream、参数更新与短程梯度累积。
- 额外80组 arbitrary-complex 边界／广播／dtype 案例，覆盖单维、奇偶尺寸、Nyquist 跨行依赖；400个前向／梯度张量与冻结基线逐字节相同，并与独立 ATen 数学参考对照通过。[边界结果](../../HPC/artifacts/s2_train_20260921/replay/boundaries.json)。
- 八个算子场景的40个输出／梯度张量全部与冻结基线逐字节相同。
- 四个算子场景各三步 Adam，共168个输出、梯度、参数和 optimizer 状态张量逐字节一致。
- 完整预训练 USRNet，B1/HR96/s2，固定合成输入，三步 Adam 的1,998个输出、参数、梯度及 optimizer 状态逐字节一致。[完整模型对照](../../HPC/artifacts/s2_train_20260921/replay/model.json)。这不是多种子真实图收敛或 time-to-quality 证明。

**Python FP32 非劣门槛仍失败。** 对同一 FP64 参考，逐张量同时比较 max_abs 与 relative_L2，融合候选只有 **15/40** 张量满足“不高于 Python FP32”；其余 **25/40** 与冻结生产的失败一致。张量顺序是 output、dx、dprior、dweight、dbias。没有放宽门槛或删除失败项。

[AGENTS.md](../AGENTS.md) 要求“精度和训练质量不劣于 Python FP32”。与当前生产逐位一致不能替代这个独立发布标准，因此 `production_eligible=false`，默认构建／分派保持不变。后续默认接入仍需解决既有数值非劣差异、明确性能分派范围并补相应训练质量证据。

## Nsight 复核

代表性 B4/C32/LR64×80/s2、batch 共享 k3，nearest prior，前向与 x/weight/bias VJP，捕获3次完整调用；该 scope 与上表独立 prior 的四类 VJP 略有不同。

- 原实现每次5个自定义谱 kernel，融合后3个；不广播的核可从4个降为2个。
- 捕获窗口内谱核累计时间 `0.7585→0.4088 ms`，约减少46.1%。它是 profiler 诊断，不能当成整算子正式加速比。原捕获中谱核占全部 GPU kernel 累计时间约24.3%；FFT及周边准备仍限制完整调用收益。
- NCU 广播滤波器梯度核：寄存器 `74→48 / thread`，DRAM throughput `67.79%→92.75%`。新的融合反向核44 registers/thread，DRAM throughput约85.69%。没有用更高 occupancy 直接替代速度证据，也没有把源码读取次数当作实测 DRAM 字节。

[NSYS 前](../../HPC/artifacts/s2_train_20260921/current_s2.nsys-rep)、[NSYS 后](../../HPC/artifacts/s2_train_20260921/fused_s2.nsys-rep)、[NCU 前](../../HPC/artifacts/s2_train_20260921/current_s2_filter.ncu-rep)、[NCU 后](../../HPC/artifacts/s2_train_20260921/fused_s2_kernels.ncu-rep)。工程化文件另经 [最终 NSYS 入口](../../HPC/artifacts/s2_train_20260921/replay/profile_nsys_fused/command.json) 重放。未运行 Compute Sanitizer，也未验证其他 GPU／操作系统。

## 复现

从 ConverseNet 根目录执行，`new_s2_study` 必须是新的实验目录。`build` 冻结当前默认实现及候选身份，不复用旧报告；失败构建可用 `--phase compile` 在原冻结目录续编。热装载核验派生文件和二进制 SHA256。

```powershell
& ./experiments/training_scale2/run.ps1 test/study_training_scale2.py --phase build --artifacts artifacts/new_s2_study
& ./experiments/training_scale2/run.ps1 test/study_training_scale2.py --phase contracts_fused --artifacts artifacts/new_s2_study
& ./experiments/training_scale2/run.ps1 test/study_training_scale2.py --phase fused --artifacts artifacts/new_s2_study
& ./experiments/training_scale2/run.ps1 test/study_training_scale2.py --phase training --artifacts artifacts/new_s2_study
& ./experiments/training_scale2/run.ps1 test/study_training_scale2.py --phase profile --route fused --tool nsys --artifacts artifacts/new_s2_study
```

`--phase operators` 比较 before/const64/int32；`--phase fused` 比较 before/int32/fused。`--tool ncu` 可采集同一路线的反向热点。每个 profile 输出目录拒绝覆盖，所有 GPU 任务串行。

最初生成脚本遗漏 int32 变体的失败、构建与 profiler 日志均保留，只有完整构建参与本文数据。执行参数记录在 `experiments/training_scale2/protocol.json`，它是测后整理的执行记录，不冒充预注册方案。
