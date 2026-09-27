# pad / crop / real 周边融合调研与单 epoch 实验（2026-09-27）

## 结论与本次范围

本次先实现 **s1 circular pad + FP32→complex64 准备融合**，并融合相应反向的取实部与循环边界归约。
中间的可微 ATen FFT、每次调用的 kernel FFT、重算 q 求解器、输出 IFFT、real 和 crop 均保持原路径。

两组均以 `codex/nearest-phase-repair` 的 `8082014829341d0f487a0fb07b2d9334b64148bc` 为基线，已包含重算 q，且均关闭 NaN 填充。
完整预训练 USRNet，B4、LR32²→HR96²、s3、seed17，各只运行 **一个 epoch：900 张不同训练图、225 次 Adam 更新**。

| 指标 | 重算 q 基线 | 再加准备融合 | 本次变化 |
|---|---:|---:|---:|
| epoch 循环（含数据准备） | 118.234 秒 | **107.845 秒** | **减少 8.79%，1.0963×** |
| 训练步耗时之和 | 111.680 秒 | **101.192 秒** | **减少 9.39%，1.1036×** |
| peak allocated | 9.809 GB | 9.803 GB | 基本不变，减少 5,144,576 bytes |
| peak reserved | 10.358 GB | 10.400 GB | 缓存池峰值增加 40 MiB |

GB 为十进制；allocated/reserved 是 PyTorch 指标，不是驱动总显存占用。
这是固定基线→候选顺序的单次 epoch 配对观测，没有多轮置信区间。没有将短程一致性称为收敛，也没有运行验证集评估。
本变更将已验证的 pad 准备融合纳入 `codex/nearest-phase-repair`，工作树为 `C:/Users/Boyce/.codex/worktrees/training-pad-glue/ConverseNet`；main 保持不变。

## 调研：原报告的开销口径

离线核查旧 `artifacts/training_followup/wgrad_profile_dataset_b4_a.trace.json.gz`，得到以下归因。
它来自 `ff7f8ba` 且 NaN 填充开启，**不是本次 8082014 + 关闭填充的现状 profile**。

| 原 trace 范围 | 总 kernel ms | NaN 填充 ms | 非填充 ms |
|---|---:|---:|---:|
| 循环 padding 前向 | 5.8152 | 1.4955 | 4.3198 |
| crop 的两个 SliceBackward | 9.5648 | 2.1804 | 7.3844 |
| 35 次 prior 的 real SelectBackward | 18.9793 | 3.5469 | 15.4323 |
| 循环 padding SliceBackward | 22.6526 | 4.5451 | 18.1075 |
| 循环 padding CopySlices | 17.4575 | 8.7166 | 8.7409 |
| 5 次数据项 real SelectBackward | 0.3250 | 0.1422 | 0.1828 |

非填充合计约 **54.168 ms**，占该 trace 的 536.568 ms kernel 时间和约 **10.10%**，占 618.356 ms GPU 首尾跨度约 **8.76%**。
原建议中的“约 7%”不能由这些相同分母直接推出；新 kernel 自身也需要时间，不能将待替代工作全额当作收益。
原“13.9% backward”还包含了 padding 前向；三个 backward owner 本身合计约 12.86%。旧数据原样保留，不以新结果重写旧结论。

原 trace 还确认：二维 `fft_fft2` 的实数输入先执行 `to(complex64)`，再执行 `_fft_c2c`。
81 次前向 FFT cast 的非填充时间约 6.964 ms，其中包含输入和 kernel PSF 的转换，不能全部归到 circular pad。
PSFPadRoll 已经融合，本次不把它计作新增工作。

## 为什么先做准备侧

`real + crop` 的前向本来是 view，代价主要在反向零嵌入。
它确实可以尝试用一个 kernel 构造完整 complex64 梯度，但如果简单使用自定义 autograd Function 返回原 view，会改变部分 view+inplace 限制。
因此本轮先保持收尾 view 的既有语义，独立验证准备侧。RealCrop 是后续单独候选，尚未实现，也没有被计入本次收益。

准备侧原路径为：

```text
FP32 x → ATen circular pad → FP32 padded x → cast complex64 → ATen fft2
```

候选为：

```text
FP32 x → CircularPadComplex（单 kernel）→ contiguous complex64 → 同一 ATen fft2
```

原反向依次取 complex 梯度实部，再走 circular pad 的 CopySlices/SliceBackward 链。
候选用一个 CUDA kernel，从 incoming complex64 的 real 分量按原顺序 gather 到 FP32 dx；每个输出元素唯一写入，没有 atomic。
原 pad 的前向复制顺序是中心、左、右、上、下，因此反向先垂直折叠下/上，再水平折叠右/左。
代码使用显式 `__fadd_rn`，并保留无贡献位置的 `+0`，避免改变负零和角点抵消结果；不能直接扁平求四项和。

高阶反向使用纯可微 ATen slice/pad/add 实现同样的有序折叠。
线性准备只保存形状与 padding，不保存输入值；测试确认不会引入输入值的假依赖或不必要的版本检查。

## 分派、FFT 与输出布局

快速路径仅用于普通整数 scale=1、variant=v7、正整数 circular padding、CUDA FP32 contiguous NCHW、无 lazy flags、GradMode 开启且任意输入/参数可微的调用。
padding 不超过输入 H/W。其它 scale/padding 模式、p0、非连续、negative view、CPU、Python backend、no_grad/inference 和全冻结输入保留原路径。
保留非法浮点 padding/scale 及运行时旧 variant 的拒绝行为，避免 `int()` 静默截断配置。

新私有入口为 `_training_pad_complex` 和 `_training_circular_s1`。
后者继续向求解器传入同一 `padded` 对象作为 x/prior，保留共享频谱和梯度累加关系，且每次调用仍重做可微 kernel FFT。
输出仍为原 ATen IFFT→real→两个 slice，未吸收 FFT 的 1/N，也未改成 contiguous 输出副本。

在 epoch 计时外，用实际 `Converse2D(128,128,3,padding=2)` 类执行一次前向预检，观测如下：

| dispatcher 事件 | 基线 | 候选 |
|---|---:|---:|
| `aten::_pad_circular` | 1 | 0 |
| `aten::_to_copy` | 2 | 1 |
| `aten::fft_fft2` | 2 | 2 |
| `aten::fft_ifft2` | 1 | 1 |
| `aten::real` | 1 | 1 |
| `aten::slice` | 14 | 2 |
| `converse2d::_training_circular_s1` | 0 | 1 |

两组 `[4,128,96,96]` 输出 stride 都是 `[2560000,20000,200,2]`，storage offset 都是 404。
测试还验证 `out.mul_().add_()` 后全部梯度及输出布局与原扩展/Python 参考一致。

## 验证结果

- 候选重新 checked CUDA 构建成功。
- 最终 **123/123 发布与融合验收测试全部通过，无跳过**：原有 108 项，加 15 项融合相关测试。
- 新增覆盖 signed zero/subnormal、padding 等于边长及角点重叠、复数上游梯度的非连续/expand/negative/conjugate 布局、NaN fill 两种设置、dtype/非法参数、所有梯度需求组合、请求子集、共享祖先/重复调用累加、二/三阶及重复 backward、输出 view/inplace、每调用 kernel FFT/参数更新、非默认 stream 和模块回退。
- 包含 padded 算子的独立 Python FP64 非劣检查，分别要求 max-absolute 与 relative-L2 不劣于 Python FP32，没有放宽门槛。
- **225 步输入、loss、梯度 L2 范数均完全一致**，均 finite 并实际执行 Adam。
- 终点 **666 项张量字节哈希全部一致**：模型、最终梯度、Adam 状态和最后一次输出。
- 独立重新加载 checkpoint，再比较 **532 项模型/Adam 张量**，全部逐字节一致；所有 Adam step=225，参数组和 CPU/CUDA RNG 状态一致。

首次 122 项测试已通过；只读复核发现缺少非法参数守卫后，补判断和一项测试，再运行最终 123 项。
保留两次测试记录，最终 epoch 的模型代码/checked manifest 与最后一次测试记录一致。

## 分支接入核验

接入时确认 43 个 checked 源文件、5 个 Python 源文件、验收测试快照和二进制均与最终 123 项测试及候选 epoch 的记录一致。本次只提交已验证代码，没有重新运行或扩大训练实验。旧 q-only 基线工作树保留在 8082014，继续可用于复现对照。

[版本化证据](training_pad_glue_evidence/README.md)及[SHA256 索引](training_pad_glue_evidence/index.json)保存下面本地 artifact 的无损压缩副本；完整 checkpoint 仍保留在本地，哈希在 epoch 报告中。

## 计时协议与复现

RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，MSVC 14.44，sm_120。
两组固定进程 affinity `0xC03C03`，Torch intra-op=8、interop=24；确定性算法与 cuDNN deterministic 开启，NaN fill/TF32/AMP/cuDNN benchmark 关闭。
完整 5 次迭代、7 个 prior block 的预训练 USRNet，原 900/100 数据协议，B4/microbatch4，RGB MSE，Adam(lr=1e-5, foreach=False, fused=False)。

两组复用完全相同的 `experiment_q_recompute.epoch()`，分别用独立 Python 进程加载 checked 扩展。
前向预检在计时外；每组预热 3 步后重新载入 checkpoint、清空梯度与 Adam 状态、重置种子，再正式处理 epoch 0 的全部 225 个批次。
循环包含逐批读图/解码、增强/退化、CPU 批次检查、输入哈希、训练 worker 和周期性进度记录。
训练步包含 H2D、前向/loss/反向、finite/梯度范数检查、Adam 与原有同步。
启动、审计、预检、预热、终点哈希、checkpoint 写盘不计入 epoch；没有计时中的 profiler，也没有新增验证集评估。

```powershell
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
./tools/run.ps1 tools/experiment_training_pad.py --root C:/Users/Boyce/.codex/worktrees/q-recompute/ConverseNet --data-root H:/Python/ConverseNet --output artifacts/training_pad_repeat/baseline --arm baseline
./tools/run.ps1 tools/experiment_training_pad.py --root C:/Users/Boyce/.codex/worktrees/training-pad-glue/ConverseNet --data-root H:/Python/ConverseNet --output artifacts/training_pad_repeat/candidate --arm circular_pad
```

复现需匹配的 checked 构建与本地数据。输出目录必须不存在，防止覆盖旧证据。

- [实验脚本](../tools/experiment_training_pad.py)
- [完整对比审计](../artifacts/training_pad_glue/v1/comparison.json)
- [基线 epoch](../artifacts/training_pad_glue/baseline_epoch/result.json) / [候选 epoch](../artifacts/training_pad_glue/v1/candidate_epoch/result.json)
- [基线 checkpoint](../artifacts/training_pad_glue/baseline_epoch/final.pth) / [候选 checkpoint](../artifacts/training_pad_glue/v1/candidate_epoch/final.pth)
- [最终 123 项测试](../artifacts/training_pad_glue/v1/release_guarded/result.json) / [日志](../artifacts/training_pad_glue/v1/release_guarded.log)
- [首轮 122 项测试](../artifacts/training_pad_glue/v1/release_tests/result.json) / [构建日志](../artifacts/training_pad_glue/v1/build.log)
- [候选 tracked patch](../artifacts/training_pad_glue/v1/candidate_epoch/candidate.patch)；完整新头文件、Python 模型和验收测试保存在同目录 `sources/` 下。

后续收尾融合应单独解决 RealCrop 的 view+inplace 兼容性，再过相同数值门槛；不能把两项理论收益直接相加。
