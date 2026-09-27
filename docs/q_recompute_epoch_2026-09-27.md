# s1 反向重算 q：单 epoch 配对实验（2026-09-27）

## 结论

在 B4、LR32²→HR96²、s3 的完整预训练 ConverseUSRNet 上，原版保存 q 和候选重算 q 各完成一个 epoch：
**900 张不同的训练图、225 次 Adam 更新、种子 17**。两组均关闭 NaN 填充，保持确定性算法开启。

| 指标 | 原版保存 q | 反向重算 q | 变化 |
|---|---:|---:|---:|
| epoch 循环耗时（含数据准备） | 120.268 秒 | 117.528 秒 | **减少 2.28%，1.0233×** |
| 训练步耗时之和 | 113.838 秒 | 111.194 秒 | 减少 2.32%，1.0238× |
| PyTorch peak allocated | 11.352 GB | 9.809 GB | **减少 1.544 GB / 13.60%** |
| PyTorch peak reserved | 11.901 GB | 10.358 GB | 减少 1.544 GB |

GB 按十进制 10⁹ bytes 计。精确 allocated 差值为 **1,543,503,872 bytes**；这是进程内 PyTorch 分配峰值，不是驱动总占用或其他应用的显存。

**当前更明确的价值是节省显存。** 本次单轮配对观察到约 2.3% 的速度收益，没有通过多轮重复证明该幅度稳定。
训练严格按要求各运行一个 epoch，未增加长期训练；没有验证集质量或收敛声明。
本变更将已验证实现纳入 `codex/nearest-phase-repair`；尚未合入 main，NaN 填充的生产默认配置未改。

## 实现范围

候选位于独立工作树 `C:/Users/Boyce/.codex/worktrees/q-recompute/ConverseNet`。
原版对照使用 `C:/Users/Boyce/.codex/worktrees/nan-fill-experiment/ConverseNet`。
二者从同一提交 `cb60572be18449a07032c70a8814a82220b47bb2` 出发，该提交已合入 main。
主工作区尚未提交的排版改动未参与候选修改。

只改 s1 的 q 保存方式：

1. 前向仍计算 `q=(y-k*p)/d` 与 `out=p+conj(k)*q`，但取消 q 的完整张量分配和写入。
2. `save_for_backward` 的原 q 槽位使用 undefined Tensor，继续保存已有 `y,p,k,lambda,d`。
3. 普通 s1 反向在需要核/正则梯度时，在 kernel 寄存器中重算 q。
4. B2/B4 融合反向的两个阶段各自重算 q，保留原有广播、四累加器归约和共享输入梯度顺序。
5. 使用原来的 `product`、`add` 及 complex64 复数除法表达式，没有代数简化或改成标量除法。高阶梯度仍走原 ATen 回退。

没有重跑输入/核 FFT，也没有跳过或跨调用复用可微核 FFT；重算使用本次调用原本已保存的频谱。
s2/s3、推理、公共 FP32/complex64 类型规则未改变。没有 AMP、TF32 或 fast-math。

源文件为 `training/full_spectrum/{scale1.cuh,batch_reduce.cuh,batch_reduce.h,full_fusion.cpp,full_fusion.cu}`，新增 `recompute_q.cuh`。
候选的完整生产源码快照、包括新增头文件的源码哈希和二进制身份，保存在每个候选结果目录。

## 数值与显存机制检查

- 候选重新 checked CUDA 构建；**107/107 发布测试通过，无跳过**，涵盖已有广播、梯度子集、共享输入、弱正则、布局和高阶梯度检查。
- **60 项独立 FP64 误差记录与原版完全一致**，没有放宽 max-absolute 或 relative-L2 门槛。
- 额外 8 个真实/代表尺寸配置（B1/B4、C128×100²、C64×96²、核广播、共享/独立输入）共 **36 项输出/VJP 字节哈希全部一致**。
- `saved_tensors_hooks` 检查 s1 `[4,3,5,7]` 的复数输入：除输入频谱以外额外保存的 complex64 张量由 **3,360 bytes 降至 0**，确认没有把 q 换一个名字继续保存。
- 全 epoch 的 **225 个输入批次哈希、225 个 loss 和 225 个梯度 L2 范数全部一致**；所有 Adam 更新均 finite 并实际执行。
- 终点 **666 项张量字节哈希全部一致**：133 个模型状态、133 个最终梯度、399 个 Adam 状态、最后一次前向输出。
- 独立从磁盘重载 checkpoint，再次逐字节比较 **532 项模型/Adam 状态**，全部通过；全部 133 个 Adam step 计数器均为 225，CPU/CUDA RNG 和 optimizer parameter groups 也一致。

上述结果只覆盖本次检查矩阵和一个 epoch，不能证明所有形状、平台或编译器都逐字节一致。

## 执行协议

环境沿用 NaN 填充实验：RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，MSVC 14.44，sm_120；
进程 CPU affinity 为 `0xC03C03`，Torch intra-op=8、interop=24。两组环境、模型和训练 worker 源码哈希一致。
`use_deterministic_algorithms(True)`、cuDNN deterministic 开启，`fill_uninitialized_memory=False`，cuDNN benchmark/TF32/AMP 关闭。

完整 5 次迭代、7 个 prior block 的模型，原 900/100 数据拆分与退化协议，B4/microbatch4，RGB MSE，Adam(lr=1e-5, foreach=False, fused=False)。
两组分别在新 Python 进程中加载各自 checked 扩展，按原版→候选的固定顺序运行。
每组预热 3 步后重新加载同一预训练 checkpoint、清空梯度和 Adam 状态、重置种子，再正式执行 epoch 0 的 225 个批次；预热不计入正式更新。

epoch 循环计时包含读图/解码、裁剪/增强/退化、CPU 批次检查、输入哈希、原训练 worker 和周期性进度记录。
训练步计时保持原 worker 的 H2D、zero_grad、前向/loss/反向、有限值与梯度范数检查、Adam 及同步。
启动、源码/数据审计、预热、最终哈希、checkpoint 写盘和验证集评估不在 epoch 计时内；本次没有跑验证集评估，没有使用 profiler。
峰值在预热及状态重置后清零，在终点快照/保存 checkpoint 之前读取。

首次候选构建因批量反向 kernel 的一个形参声明遗漏而失败，修正后重新 checked 构建成功。
首次构建日志和当时的 tracked patch 原样保留，未将它作为数值失败或运行成功计数。

## 分支接入与证据归档

接入时逐文件核对生产源码及 checked 二进制，确认与完成 epoch 的版本完全相同。新增 `test/test_q_recompute.py`，覆盖 B1/B2/B4 和共享/独立输入，要求 s1 只保留输入频谱与实数分母，防止重新保存完整 q。接入后的完整发布测试为 **108/108 通过，无跳过**；对应 `landing_release_tests` 记录已归档，checked 构建 manifest 与单 epoch 实验完全一致。

[版本化证据目录](q_recompute_evidence/README.md)与[SHA256 索引](q_recompute_evidence/index.json)包含下述原始结果的无损 gzip 副本。`artifacts/` 链接是当前工作区的原始本地证据，未安装数据集或复制实验产物的 checkout 可读取版本化副本。

## 复现与证据

脚本：[experiment_q_recompute.py](../tools/experiment_q_recompute.py)。输出目录必须不存在，脚本拒绝覆盖旧证据。

```powershell
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$baselineRoot = 'C:/Users/Boyce/.codex/worktrees/nan-fill-experiment/ConverseNet'
$candidateRoot = 'C:/Users/Boyce/.codex/worktrees/q-recompute/ConverseNet'

# 两个工作树需要各自的 checked 构建；候选修改保留在 candidateRoot。
./tools/run.ps1 tools/experiment_q_recompute.py --root $baselineRoot --data-root H:/Python/ConverseNet --output artifacts/q_recompute_repeat/baseline_gate --mode gate --arm baseline
./tools/run.ps1 tools/experiment_q_recompute.py --root $candidateRoot --data-root H:/Python/ConverseNet --output artifacts/q_recompute_repeat/candidate_gate --mode gate --arm recompute
./tools/run.ps1 tools/experiment_q_recompute.py --root $baselineRoot --data-root H:/Python/ConverseNet --output artifacts/q_recompute_repeat/baseline_epoch --mode epoch --arm baseline
./tools/run.ps1 tools/experiment_q_recompute.py --root $candidateRoot --data-root H:/Python/ConverseNet --output artifacts/q_recompute_repeat/candidate_epoch --mode epoch --arm recompute
```

- [对比汇总与审计结果](../artifacts/q_recompute/v1/comparison.json)
- [原版完整 epoch](../artifacts/q_recompute/baseline_epoch/result.json) / [候选完整 epoch](../artifacts/q_recompute/v1/recompute_epoch/result.json)
- [原版 checkpoint](../artifacts/q_recompute/baseline_epoch/final.pth) / [候选 checkpoint](../artifacts/q_recompute/v1/recompute_epoch/final.pth)
- [候选发布测试](../artifacts/q_recompute/v1/release_tests/result.json) / [日志](../artifacts/q_recompute/v1/release_tests.log) / [FP64 误差](../artifacts/q_recompute/v1/release_tests/fp64_errors.json)
- [原版算子探针](../artifacts/q_recompute/baseline_gate/result.json) / [候选算子探针](../artifacts/q_recompute/v1/operator_gate/result.json)
- [候选 tracked patch](../artifacts/q_recompute/v1/operator_gate/candidate.patch)；新增头文件及其他生产源码在同目录 `sources/` 中。
- [最终构建日志](../artifacts/q_recompute/v1/build_corrected.log) / [首次构建失败日志](../artifacts/q_recompute/v1/build.log)
