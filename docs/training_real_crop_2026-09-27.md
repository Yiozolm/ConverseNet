# real/crop 收尾融合：实现与单 epoch 验证（2026-09-27）

## 分支接入状态

本提交按用户要求将已验证的 real/crop V2 接入 `codex/nearest-phase-repair`，接在 pad commit `b80be04` 之后。main 保持不变。
接入前逐项核对 45 项 checked 源文件、5 项 Python 源、测试快照与二进制，确认与 V2 epoch 一致。提交检查仅移除 `real_crop_autograd.h` 末尾多余空行，其余源字节不变；对最终提交版重新 checked 构建，136/136 发布测试通过，FP64 记录与 V2 相同。源差异及本次构建/测试见归档中的 `landing_*` 记录；不重复训练。
旧单 epoch 负结果与后续固定权重诊断分开保留，历史触发因素未确定，不将后者改称 epoch 加速。
[版本化证据](real_crop_evidence/README.md)与[SHA256 索引](real_crop_evidence/index.json)提供下述本地 artifact 的归档副本。

## 原单 epoch 结论（保留当时记录）

**融合已实现，数值与 view/inplace 兼容性检查通过；完整训练性能未通过，本轮不建议据此合入。**

基线为 `b80be049b255f4ae0f5d9b55c67fd0aeb2714364`，已经包含重算 q 和 circular-pad 准备融合。
两组都关闭 NaN 填充，保持确定性算法、FP32/complex64、禁用 TF32/AMP/fast-math。
按要求各只运行一个 epoch：B4、seed17、900 张不同训练图、225 次 Adam 更新。

| 指标 | 基线 | real/crop 融合 | 本次结果 |
|---|---:|---:|---:|
| 完整 epoch 循环 | 110.128 秒 | 116.228 秒 | **耗时增加 5.54%，0.9475×** |
| 训练步耗时之和 | 103.700 秒 | 109.654 秒 | 耗时增加 5.74%，0.9457× |
| peak allocated | 9,803,483,648 bytes | 9,803,483,648 bytes | 无变化 |
| peak reserved | 10,399,776,768 bytes | 10,399,776,768 bytes | 无变化 |

这是单次固定顺序的跨进程对比，不能将其扩大为所有运行条件下的退化结论，也不能宣布训练加速。
后续仅做了算子级测量，没有追加训练 epoch，没有验证集或收敛声明。
该轮实验完成时，候选位于 `C:/Users/Boyce/.codex/worktrees/training-real-crop/ConverseNet`，当时尚未修改 nearest-phase-repair 或 main；本次接入状态见页首。

## 融合如何保留原有 view 语义

普通自定义 `autograd::Function` 返回 real/crop view 会标记 `IN_CUSTOM_FUNCTION`，导致原来合法的某些 inplace 操作被拒绝。
本候选采用另一种方式：**继续用原生 real/slice 创建 view，只替换其普通反向 Node**。

```text
原路径：ATen IFFT → real view → H slice view → W slice view
候选：完全相同的 forward views + RealCropBackward gradient edge
```

`RealCropBackward` 只保存完整 B/C/H/W 与 padding，不保存输入值；普通反向单 kernel 写出 contiguous complex64 梯度：

- crop 区域内：real 为对应上游 FP32 梯度，imag 为 `+0`；
- crop 区域外：real/imag 均为 `+0`；
- 只复制数值和写零，无浮点加法、归约或除法。

原生 `DifferentiableViewMeta`、跨 complex→real 的 ViewFunc、version counter、stride、storage offset 和 CreationMeta 均保留。
输出或 base 被 inplace 修改时，PyTorch 可以重建原生 view backward/CopySlices；因为自定义 Node 只实现同一个线性 view 的 VJP，这种回退仍然正确。
其机制依据当前 [PyTorch 2.11 的 variable.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/variable.cpp) 与本地 `functions/utils.h::set_history`，并由本次实际测试验证。

私有 schema 如实声明 alias：

```text
_training_real_crop(Tensor(a) spectrum, int padding) -> Tensor(a)
```

高阶反向维持原顺序：width slice_backward → height slice_backward → select_backward → view_as_complex。
CPU、no_grad/inference、冻结输入、非连续或 lazy-flag 输入、forward AD 输入使用原生路径；incoming forward-AD 梯度也不走原始 CUDA 指针写入。
CUDA kernel 支持一般正 stride、transpose、expand 上游梯度，并解析 lazy negative flag。

只在已经启用 pad 融合的 `_training_circular_s1` 收尾接入。其它 `spatial` 调用仍返回原来的 `real(ifft2(...))`。
所有 ATen FFT、1/N 归一化、逐调用可微 kernel FFT、共享频谱关系与求解器保持不变。

## 确认融合确实执行

在训练计时外，对真实 prior 尺寸 `[4,128,96,96]` 的完整 Converse2D 做一次前向/VJP 预检，不执行优化器更新：

| 观测 | 基线 | 候选 |
|---|---:|---:|
| 输出 grad_fn | SliceBackward0 | RealCropBackward |
| `RealCropBackward` 执行事件 | 0 | 1 |
| `aten::slice_backward` | 2 | 0 |
| `aten::select_backward` | 1 | 0 |
| `aten::fft_fft2` | 2 | 2 |

两组输出 stride 均为 `[2560000,20000,200,2]`，storage offset 均为 404。
前向仍包含原生 view 操作；它们没有对应的 GPU 数据拷贝，本次融合目标是反向的零嵌入。

## 数值与接口验证

- 最终 checked CUDA 构建成功，**136/136 测试通过，无跳过**：已有 123 项加 13 项 real/crop 验收。
- 覆盖 exact alias/layout、schema alias、grad/no_grad/inference、实际融合执行、CPU/非连续/conj/neg fallback、forward AD、signed zero/subnormal、strided/expand/negative 上游梯度、高阶与重复反向、output/base/alias/no_grad inplace、leaf 禁止原位修改、saved-view 版本检查、输入值不保存、共享祖先、retain_grad/hooks、dtype/padding 守卫及非默认 stream。
- 既有完整算子的 Python FP32 字节、独立 FP64 非劣及高阶检查全部通过，没有放宽数值门槛。
- 全 epoch 的 **225 个输入批次、loss 与梯度 L2 范数完全一致**；全部更新 finite 并执行 Adam。
- 终点 **666 项张量字节哈希完全一致**：133 个模型状态、133 个最终梯度、399 个 Adam 状态和最后一次输出。
- 独立重载 checkpoint 再核对 **532 项模型/Adam 张量**，全部逐字节一致；所有 Adam step=225，参数组及 CPU/CUDA RNG 一致。

V1 通过了最初的 136 项 eager 测试；复核后补齐 alias schema 并在已有测试中增加 schema 断言，形成 V2，再次 checked 构建和通过 136 项。
单 epoch 使用 V2。V1 的源码、二进制、构建和测试记录保留，未把较窄的初始验收等同于最终验收。

## 为什么不能只看融合 kernel 数量

完整 epoch 的候选进程中，未修改的 finite 检查和 Adam 阶段也一起变慢：

| 每步平均 wall ms | 基线 | 候选 |
|---|---:|---:|
| forward + backward | 443.094 | 467.663 |
| finite/梯度范数检查 | 7.178 | 8.046 |
| Adam | 9.533 | 10.485 |

这些观测说明不能把整轮差异全部归因于 real/crop kernel；运行状态影响尚未定位。
但它们也不足以推翻实测的完整 epoch 结果，本轮整步性能状态仍标为未通过。

为区分局部 VJP 与完整训练，另用同一个候选进程交替测量原生 real/crop 和融合 VJP。
这里只对固定 complex 输入求 VJP，不运行模型或优化器，不增加训练 epoch。
每种配置 6 轮，交替前后顺序，每组 30 次，预热 10 次；输出梯度字节哈希一致。
以下为 CUDA event 跨度/次的组均值中位数，包含可能的提交空隙，不是 profiler kernel 时间之和。

| B×C、完整 H×W、padding | 上游布局 | 原生 VJP ms | 融合 VJP ms | 配对加速比中位数 |
|---|---|---:|---:|---:|
| 1×128、100×100、2 | contiguous | 0.12218 | 0.04970 | 2.455× |
| 1×128、100×100、2 | transpose | 0.13325 | 0.05006 | 2.625× |
| 4×128、100×100、2 | contiguous | 0.78020 | 0.19393 | 4.028× |
| 4×128、100×100、2 | transpose | 0.83041 | 0.20707 | 4.008× |

这证明局部 VJP 的已测收益，**不构成模型完整步骤或 epoch 的加速证明**。
若继续研究，应使用同进程的完整步骤消融或新的诊断 trace 定位差异；不能直接将局部节省乘以调用数作为整体收益。

## 协议、代码与证据

环境沿用前两次实验：RTX 5060 Ti、PyTorch 2.11.0+cu130、CUDA Toolkit 13.0、MSVC 14.44、sm_120；进程 affinity `0xC03C03`，Torch intra-op=8、interop=24。
完整 5 次迭代、7 个 prior block，原 900/100 数据协议，B4/microbatch4、RGB MSE、Adam(lr=1e-5, foreach=False, fused=False)。
两组复用同一冻结 epoch helper，3 步预热后重载同一初始模型、清空梯度和 Adam 状态、重置种子，再从 epoch 0 第一个批次开始。
循环计时包含数据准备、输入哈希、原训练 worker 与进度记录；启动、审计、预检、预热、终点快照/checkpoint 不计入 epoch。没有计时中的 profiler。

实现新增 `real_crop.cuh` 与 `real_crop_autograd.h`，在 `production.cpp` 保留相同 IFFT 并接入新收尾，注册私有 alias-aware 接口。
该 Node 使用当前 PyTorch 的内部 autograd 接口，因此仍需依靠版本匹配的 checked 构建与 view 回归检查。

- [完整对比审计](../artifacts/training_real_crop/v2/comparison.json)
- [基线 epoch](../artifacts/training_real_crop/baseline_epoch/result.json) / [候选 epoch](../artifacts/training_real_crop/v2/candidate_epoch/result.json)
- [基线 checkpoint](../artifacts/training_real_crop/baseline_epoch/final.pth) / [候选 checkpoint](../artifacts/training_real_crop/v2/candidate_epoch/final.pth)
- [V2 发布测试](../artifacts/training_real_crop/v2/release_tests/result.json) / [日志](../artifacts/training_real_crop/v2/release_tests.log) / [构建日志](../artifacts/training_real_crop/v2/build.log)
- [V1 初始测试](../artifacts/training_real_crop/v1/release_initial/result.json)；初始源码和二进制在同层目录保留。
- [局部 VJP 交替测量](../artifacts/training_real_crop/v2/vjp_timing.json)
- [epoch 脚本](../tools/experiment_training_real_crop.py) / [算子级脚本](../tools/benchmark_real_crop_vjp.py)

```powershell
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
./tools/run.ps1 tools/experiment_training_real_crop.py --root C:/Users/Boyce/.codex/worktrees/training-pad-glue/ConverseNet --data-root H:/Python/ConverseNet --output artifacts/real_crop_repeat/baseline --arm baseline
./tools/run.ps1 tools/experiment_training_real_crop.py --root C:/Users/Boyce/.codex/worktrees/training-real-crop/ConverseNet --data-root H:/Python/ConverseNet --output artifacts/real_crop_repeat/candidate --arm real_crop
```

复现需要各自匹配的 checked 构建和本地数据；输出目录必须为新目录。
