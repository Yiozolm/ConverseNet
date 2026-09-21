# 从完整训练重新确定优化方向

2026-09-18，RTX 5060 Ti 16 GB / Windows WDDM / PyTorch 2.11.0+cu130。生产算子和模型未在本轮改动；本轮新增测量、轨迹、独立候选实验与调研记录。仅 FP32，TF32 关闭。

**首要优化对象应转向 1×1 卷积反向，以及激活 padding/crop 和 LayerNorm 的读写与反向。** 当前真实模型的核 FFT 准备只占本次 GPU kernel 累计时间约 4.8%，继续单独打磨它，难以带来大的整网收益。此前大图 s3 算子的瓶颈排序不适用于这个训练配方。

这里需要区分：**手写频谱核心约占 10.4%，完整 Converse2D 路径约占 37.4%**。后者包含 FFT、准备、padding/crop 与反向，仍是主要开销之一；不能把核心占比当整个算子的占比。

## 1. 重新建立的测量基线

仍用用户提供的 1000 图、固定 900/100 拆分、同一个预训练 checkpoint。完整 USRNet 为 5 iterations / 7 blocks / 64 features，HR96→LR32/s3，batch4/micro4，MSE、Adam1e-5，cuDNN deterministic=True、benchmark=False，谱复用关闭。

新增 [连续训练入口](../test/benchmark_sustained_training.py)：两轮 AB/BA，每路预热 5 步后恢复完全相同的模型与已分配但归零的 Adam 状态，各连续执行 100 次更新。一次只有一个 GPU fixture。

| 连续 100 步 | 当前 CUDA | 同 GPU Python/ATen | Python/current |
| --- | ---: | ---: | ---: |
| 第一轮：current→Python | 51.584 s | 79.022 s | 1.532× |
| 第二轮：Python→current | 51.627 s | 78.194 s | 1.515× |
| 两轮平均耗时/步 | 516.054 ms | 786.078 ms | 配对比中位数 1.523× |
| PyTorch peak allocated | 9.736 GB | 12.916 GB | 减少约 24.6% |

计时包含真实图像解码/退化/CPU 校验/批次哈希、H2D、forward/MSE/backward、全部 133 个参数梯度的有限值检查、Adam 和最终 GPU drain。不在计时中做验证、保存、打印、编译、预热、复位或 final hash；它是**持续训练吞吐**，仍不是完整收敛、全流程 wall time 或达到同等质量的总时间。

新循环每步只有一次 loss/finite 聚合 D2H 检查，更新前仍阻止非有限梯度；去掉逐阶段计时同步和仅用于诊断的梯度范数。计时前，current 和 Python 两路分别与旧 `train_step` 做两步对照，loss、全部梯度、参数和 Adam 状态均完全一致。初值、Adam 初值和全部 100 个批次哈希跨轮核对；原有三种子 250 步质量结果保留，未把本次吞吐实验冒充新质量门槛。

CPU 数据处理/校验/哈希每 100 步约 2.82–2.88 s，占 current 总时间约 5.5%；部分可与上一更新尾部重叠，因此不能把它直接从总时间减掉作为提速预测。20 步块记录的是未单独 drain 的 host interval / CUDA stream span，不能当作独立单步延迟。OS 文件缓存未控制；allocated/reserved 不代表进程总显存。

原始数据：[sustained_100.json](../artifacts/training_research/sustained_100.json)。这组结果与旧的预热 8 步约 1.516×相符，但绝对时间和统计口径不同，不能混用分母。相对上一轮冻结 CUDA 的约 1.04×全循环收益，仍是另一组比较。

## 2. 实际执行的模型

一次 forward 包含 147 次 1×1 Conv2d、70 次 LayerNorm、40 次频谱求解。7 个 prior block 的参数共享使用 5 次；并非 35 套独立参数。

| 求解调用 | 次数 | 实际形状/尺度 |
| --- | ---: | --- |
| 首次 DataNet | 1 | B4/C64，32²→96²，s3 |
| 后续 DataNet | 4 | B4/C64，96²，s1，动态 7×7 核 |
| Prior Converse2D | 35 | B4/C128，padding 后 100²，s1，共享 3×3 权重 |

因此本次训练 **39/40 次求解都是 s1**。当前 `H*W >= 65536` 的 s1 专用核门槛一次也未命中；40 次核准备均因实际核谱总量阈值走分离 FP64 FFT。已有实现和本次实际命中路径需要分开看。

## 3. 完整模型的新热点

新采集两个完整训练步，包含实际数据处理和旧的分阶段审计循环。Torch profiler 使用 `record_shapes=False`、`with_stack=False`，避免额外持有输入引用；用模块 hooks 标记 CPU 范围，并用 External id、forward/backward flow 和 Sequence 关联 GPU。profiling 只用于定位，正式速度使用上面的无 profiler 连续训练。

Torch trace 有 **18,282 个 kernel**，累计 **743.885 ms**。以下分母只包含原始 `cat=kernel`，不混入 GPU annotation、memcpy、memset 或 CPU runtime，也不把嵌套算子的 inclusive 时间重复相加。

| 互斥执行部件 | Kernel 累计占比 | 意义 |
| --- | ---: | --- |
| Conv2d 前后向及归属辅助核 | **42.11%** | 首要候选；其中两个 wgrad 核占全部 kernel 时间 31.28% |
| Solver 周围的 padding/crop、复制及其他支持操作 | **14.87%** | 需要逐项消除读写，不能全记成频谱求解 |
| LayerNorm 前后向 | **11.67%** | 70 次 channels-first 归一化值得融合 |
| 自定义频谱求解核心 | **10.41%** | 局部倍率需要按这个占比折算 |
| 激活 FFT 与归属支持操作 | **7.35%** | 保持自动求导，研究布局和往返次数 |
| 核准备及其自动反向 | **4.79%** | 不是本形状的第一瓶颈 |
| 其余 GELU、残差、循环及未归属项 | **8.80%** | 包括约 2.1% 未能可靠定位到模块的事件 |

这里“激活 FFT 部件”包含归属支持操作；另一种按 kernel 名字的分区中，**全部实际 FFT kernel 为 10.39%**。两套分类不能叠加。

两个 C64→128 的 prior 卷积路径，其权重反向合计占整体 kernel 时间 **23.93%**。已映射的 circular activation padding 反向 add/fill/copy 占 **6.26%**，solver 输出 crop 反向占 **2.06%**；它们是 solver 支持部件的子集，不要再加一次。额外未匹配 forward 序号的 CopySlices 保留为未归属。

归因完整覆盖所有 kernel，External id 均唯一，4,730 对 fwdbwd flow 的序号通过核对。388 个反向 kernel、约 8.75% 时间只能通过唯一 solver 序号区间推断归属，已在原始表单独标注，不能当作直接 flow 证据。

独立 Nsight Systems 同样捕获 **18,282 个 kernel**，累计 **795.645 ms**；两个 wgrad 核占 **31.94%**、实际 FFT 名称类占 **10.25%**、谱核心占 **10.20%**，与 Torch 的排序一致。两个 capture 的绝对时间不混算。该捕获的 kernel union/span≈83.5%，余下间隙可能含数据处理、提交、同步、复制、依赖或 profiler 成本，不能全部称作可消除的 CPU 空转。

来源：[逐 kernel 归因](../artifacts/training_research/torch_current/attribution.md)、[机器可读映射](../artifacts/training_research/torch_current/attribution.json)、[Systems 汇总](../artifacts/training_research/nsys_bootstrap_capture/summary.md)。`torch_summary.json` 是旧通用 helper 的诊断输出，其 GPU event 分母含 annotation；本报告不使用它计算占比。旧 Systems helper 不认识新增 Phase 名称，故其 phase 全标未归属；kernel 分类和 step/correlation 校验有效。

## 4. NCU 对卷积热点的核实

选择第一个完整 backward 中第 2 个 `wgrad2d_grouped_direct_kernel`（grid1024、block256），跳过输出层较小的第一次调用，采集 16-pass replay：

- DRAM SOL 8.26%，实测读写速率 36.95 GB/s；不是此前自定义谱核那种接近 DRAM 峰值的情况。
- SM throughput 80.95%，其中 ADU 同为 80.95%；FMA-heavy 77.49%、普通 FMA 12.22%、Tensor 0%。不能仅凭 SM 汇总指标断言 FP32 FLOPs 已经饱和。
- 实际 occupancy 63.54%，理论 66.67%，每 scheduler eligible warps 1.91；59 registers/thread。没有证据支持“先减寄存器就会更快”。

这支持优先比较卷积的算法与工作分配。NCU 没有固定时钟/清缓存，replay 还把备份放到系统内存；只使用硬件诊断，完全不将其时长或自动建议的 speedup 当正式收益。无可用 spill 指标，不声称无 spill。原始报告：[wgrad.ncu-rep](../artifacts/training_research/ncu_wgrad/wgrad.ncu-rep)、[摘要](../artifacts/training_research/ncu_wgrad/summary.md)、[完整计数器](../artifacts/training_research/ncu_wgrad/wgrad.raw.csv)。

## 5. 优化次序与可证伪实验

| 顺序 | 单变量候选 | 需要证明什么 |
| --- | --- | --- |
| 1 | 将满足 1×1/stride1/pad0/groups1 的卷积与等价 ATen batched matmul 对照；另做保留 deterministic 的 cuDNN autotune | 输出及 dx/dw/db 通过独立 FP64，再用真实层特征/VJP 和完整训练检查；bias、batch reduction、布局代价全部计时 |
| 2 | 激活 circular pad/crop 的 ATen 表达优化；LayerNorm 前后向融合 | 保留精确边界与 channels-first 归一化定义，使用自动求导；减掉具体重复 fill/copy，不能只减少 Python 调用数 |
| 3 | 对真实 B4/C128/100² 消融 force-generic / force-s1 | 只改变分派，其余准备/训练完全相同；按总频谱量、广播结构和实测决定新门槛 |
| 4 | 同 forward 可微核谱复用；核准备 pad/roll/cast | 先解决完整模型压力门槛；不跨步缓存、不 detach；现有失败状态保留 |
| 5 | 数据预取、聚合检查、Adam foreach/fused、局部 compile/Graph | 各自独立 A/B；优先减少 host 提交成本，但不以 kernel 数直接预测收益；完整步包含必需有限值检查 |

作为数量级约束：若整个 Conv2d 部件真能快 2 倍，当前 kernel-work 分母下仅约 **1.27×**；谱核心快 2 倍约 **1.055×**；核准备快 2 倍约 **1.025×**。这不是 wall-time 预测：真实收益还受新增开销、重叠与未优化部分影响。不同候选的倍率不能相乘。

Channels-last 必须按整个 block 测量，solver 入口需要 contiguous NCHW，局部 conv 快可能被布局往返抵消。Graph 不能消除本来就执行很久的 wgrad；CPU finite 分支也不能原样捕获。当前只有约 31 万参数，参数/Adam 状态很小，9.7 GB allocated 主要应继续从激活与 autograd 保存中分析；checkpoint 以重算换显存，不能默认当成速度优化。更大 batch 要固定有效 batch 和质量协议，单独比较 samples/s 与显存，不能偷换 microbatch 分母。

官方资料作为机制依据，实际版本和机器仍以本仓库实验为准：[PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)、[Profiler 的引用持有与额外开销](https://docs.pytorch.org/docs/2.11/profiler.html)、[CUDA/cuDNN 的算法与布局约束](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cudnn)、[Adam foreach/fused 选项](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)。详细候选、源码位置和证伪条件见 [独立调研记录](../artifacts/training_research/candidates.md)。

### 已执行的局部筛选：1×1 卷积与等价矩阵乘法

[独立探针](../test/probe_pointwise_training.py) 使用实际特征形状 B4/H96/W96、预训练权重、固定正常量级随机输入和归一化随机 VJP；它们不是采集自训练的真实激活。FP32、TF32 off、deterministic，四轮交替、每路预热5次后测20次；包含 eager forward、bias、dx/dw/db 和候选自身布局/广播归约。无 profiler、无 optimizer。

| 形状 | Conv2d 前向+全部VJP | 等价 matmul 前向+全部VJP | 四轮配对比中位数 |
| --- | ---: | ---: | ---: |
| C64→128 | 1.794 ms | 0.853 ms | **2.089×** |
| C128→64 | 0.729 ms | 0.564 ms | **1.294×** |
| 输出层 C64→3 | 0.222 ms | 0.527 ms | **0.424×，明显变慢** |

耗时为各路四轮中位数，配对比另取逐轮比值的中位数。两条方法、三种形状的输出/dx/dw/db，全部通过事前 FP64 `F.conv2d` 的 `atol=3e-5, rtol=3e-4` 逐点预算，0超限；没有放宽门槛。候选最大输出绝对误差1.20e-6，最大权重梯度绝对误差8.27e-7。

这支持**先对 prior 中 C64↔128 的卷积做整网候选，保留输出层现状**。同时，前两组 matmul 的局部 peak allocated 为152.11MB，对照123.80MB，增加约22.9%；完整训练的保存张量/显存收益仍未知。不能将局部2.089×写成整网加速，也不能无条件替换所有1×1卷积。下一步必须补真实层激活/VJP、full-model精度与质量、持续训练A/B后再决定接入。原始结果：[pointwise_probe.json](../artifacts/training_research/pointwise_probe.json)。

## 6. 层号导航与技能交付

使用用户指定的 [torch-profiler-layer-track](C:/Users/Boyce/.codex/skills/torch-profiler-layer-track/SKILL.md)。本模型的 L0–L39 表示一次 forward 中按执行顺序编号的 **40 次 solver 调用**：L0/8/16/24/32 是 DataNet；每个 DataNet 后的 7 个标签对应 prior block0–6。它们不是 Transformer 层号，也不是 40 组不同参数。

用 `solve_alias` 作起点；普通 guide 延续到下一个 `solve_alias`，每个 forward 的最后一个 guide 才由匹配的 `solve_output` 完成时刻闭合。两次完整 forward 各 40 个起点及谱核终点，通过 GPU External id→CPU solver scope→源码 call map 逐个核对，anchor offset=0。guide 只帮助导航：它会包含本次 IFFT、下次调用的部分前置工作，最后一个 guide 又不包含终点后的 IFFT，因此不能当完整模块耗时或独占归属。

- [原始 trace](../artifacts/training_research/torch_current/full.trace.json)
- [带 80 个标签的 compact view](../artifacts/training_research/torch_current/full.layers.trace.json)
- [层号与显示轨道映射](../artifacts/training_research/torch_current/full.layers.trace.json.layers.json)
- [反向恢复及 Perfetto SQL 验证](../artifacts/training_research/torch_current/verification.json)

原来就是 1 条实际 GPU activity stream，紧凑视图仍为 1 条 synthetic activity lane，另存 GPU annotation 轨道；不代表运行时减少了流。反向还原事件数组和 metadata 与原 trace 完全一致，原文件 SHA 未变。官方 Perfetto TraceProcessor 实际导入后，18,282 kernel、19,074 activity、80 guides 计数正确；activity 和 guide 分别单轨、depth=0。已在真实 Perfetto UI 中检查到 guide 与 GPU lane 相邻显示；本次没有制作截图交付。

## 7. 启动“报错”排查

`Error checking compiler version for cl` 来自 PyTorch 无参调用 `cl` 探测版本。`run.ps1` 设置的 `CL=/Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX` 使这次无源文件调用返回 2 / D8003。同一 cl19.44.35228，仅在探测子进程移除 CL 后返回 0 并给出版本；不是 CUDA 算子失败。不要关闭 ABI 检查或吞掉 ninja/算子异常来掩盖它。

前三次 Nsight 常规启动未生成报告，日志保留在 `nsys_current`、`nsys_current_retry`、`nsys_showout`。后来使用独立 bootstrap、先用现有 loader 核对 source/header/PyTorch/binary SHA 再加载已有构建，Systems/Compute 均成功。此处同时改变了启动形式和构建探测，尚未单独证明原 Nsight 失败的唯一原因。未改旧 loader 或全局环境；冷构建时旧 warning 仍可能出现。

已把成功方式封装为 [稳定采集入口](../test/profile_full_training_nsight.py)，并用新的 1 步完整模型采集验证 `returncode=0`、训练 metadata 与 `.nsys-rep` 均实际存在；记录见 [launcher.json](../artifacts/training_research/nsys_launcher_check/launcher.json)。入口要求已构建且 SHA 校验通过的库，陈旧构建会明确失败，不会跳过源码核验。

## 8. 下一轮验收标准

先固定原精度门槛，单独筛选候选，再做 full5/7 多种子真实训练。顺序为：独立 FP64 输出/各梯度与弱正则 → 真实层特征/VJP → 完整模型梯度审计 → 关闭 profiler 的持续 A/B → 三种子质量 → 更长训练与独立全图评估/同等质量时间。

不得放宽既有 `atol=3e-5, rtol=3e-4` 或通过改变 λ 让候选过关。冻结基线已有的完整模型 16 张量/26 逐点超限、复用候选的压力训练失败仍未解决；局部通过或 250 步 PSNR 接近不等于这些门槛通过。完整收敛、全图/独立集质量、time-to-quality 与进程总峰值显存仍是后续工作。1000 图足以做本轮性能定位，未额外下载 1 万图。

## 复现入口

```powershell
& ./experiments/training_speed/run.ps1 test/benchmark_sustained_training.py --steps 100 --rounds 2 --equivalence-backends both --output artifacts/training_research/new_sustained.json
& ./experiments/training_speed/run.ps1 test/profile_full_training.py --tool torch --output artifacts/training_research/new_torch
& ./experiments/training_speed/run.ps1 test/profile_full_training_nsight.py --tool nsys --output artifacts/training_research/new_nsys
& ./experiments/training_speed/run.ps1 test/profile_full_training_nsight.py --tool ncu --steps 1 --output artifacts/training_research/new_ncu
& ./experiments/training_speed/run.ps1 test/probe_pointwise_training.py --output artifacts/training_research/new_pointwise.json
# 每个输出目录必须为新路径；原始命令和生成的 bootstrap 随报告保存。
& 'F:\anaconda3\envs\vllm\python.exe' test/summarize_full_training_trace.py --help
& 'F:\anaconda3\envs\vllm\python.exe' test/verify_full_training_trace.py --help
```

所有正式性能数字来自无 profiler 的报告；raw traces、失败日志、候选结果与 source fingerprints 保存在 `artifacts/training_research/`（Git 忽略），不能只保留本 Markdown 就丢失复现依据。
