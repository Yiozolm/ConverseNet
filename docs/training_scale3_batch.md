# s=3 与大 batch 训练补测

本页保存补测时的源码结果；之后的默认 FP32 改进见
[FP32 训练继续优化](training_refinements.md)。

2026-09-17，RTX 5060 Ti 16GB，分支 `codex/training-operator-optimization`。
本轮只扩展测试和基准，生产算子源码指纹与
[上一轮优化](training_operator_optimization.md)一致；基线仍为 dev
`b850e3885d566ad70cd90cd2213497215370980e`。

三次独立进程复测中，s=3 算子完整训练步加速 **1.41–1.88×**，大 batch 的 s1/s2
算子加速 **1.24–1.86×**。整网收益较小：USRNet B8/s3 基本持平，B16/s3 约
**1.05×**，B16/s2 约 **1.07×**。s3 算子完整步的 peak allocated 降低约 **16–22%**。

## 数值验证

[test_training_scale3_batch.py](../test/test_training_scale3_batch.py) 的 5 个测试方法、
**44 个参数案例全部通过**，记录了 291 次输出或梯度比较。包括：

- s=3，B8/B32，四种 batch/channel 核广播组合，独立 prior 和 nearest prior；
- s=1 的共享输入，s=3 的全部 15 种梯度需求及方向二阶导数；
- 完整 B32/C32/64×80/s3 与 B4/C32/256²/s3：CUDA 一次执行真实整批，FP64 参考
  按样本求解并累加共享 kernel/bias 梯度，以限制参考路径的显存；
- B8/B32 弱正则，固定 `eps=1e-8, bias=-40`，含零核。

保持上一轮门槛：输出 `atol=rtol=3e-5`，梯度 `atol=rtol=5e-5`；
弱正则输出采用 `atol=1e-6, rtol=1e-5`。没有更改 λ、seed 或误差容差。
291 次记录中，最大绝对误差 `1.725e-4`，最大相对 L2 `3.64e-6`，最大逐点预算比
`|error|/(atol+rtol*|reference|)` 为 **0.320**，超限元素数为 **0**。
绝对误差的最大值来自梯度，并不意味着所有参考数值都接近 1；验收采用既定绝对加相对预算。
方向二阶导数另按原测试的固定门槛检查。

## 测量范围与协议

新增 11 组算子 forward+VJP 和 15 组完整训练步：

- 算子 C32：B1 的 64×80、256²、s3；B8/B32 的 64×80、s1/2/3；B4 的 256²、s1/2/3。
- ConverseBlock：B16/C16/32×40。
- 缩小版 USRNet：B8/B16 RGB 16×20、s3，以及 B16、s2。

三个独立进程，每个使用 6 轮 × 20 步，dev/current 顺序交替；每轮预热后重置为相同参数及
零 momentum，完整步包含 zero_grad、forward、MSE、backward、SGD(lr=1e-4,momentum=0.9)。
四步变化输入的 loss/参数/梯度/momentum 检查位于计时外；计时后另检查有限值。
生产分派检查覆盖 s1/2/3，确保测试实际走融合训练。

FP32 激活与求解，可微 FP64 核准备；每次准备及反向成本均计时。TF32 关闭，普通 eager，
不使用 AMP 或 CUDA Graph；数据已在 GPU，不计数据加载/H2D。
模型结构仍为两轮迭代、一个 block、64 隐藏通道，alpha=0.1，与前次小模型协议一致。
这些是短程合成输入测试，不是完整模型或真实数据集收敛验证。

显存记录单个变体驻留时的 PyTorch 总 peak allocated 与 reserved；后者包含前者，
不能相加，且两者均不包含分配器之外的驱动/库内存。

## 完整训练步结果

时间为每个进程轮次中位数再取跨进程中位数，单位 ms；速度比用表中延迟相除。
下表为同步 wall time，CUDA Event 结果、逐轮配对比率和进程间范围在原始汇总中。
不同聚合方式的速度比可能略有差别。图像尺寸均为输入尺寸，s3 输出的高宽各乘 3。

| 负载 | B / C / H×W | s | dev / ms | 当前 / ms | dev÷当前 | dev→当前 peak allocated / MiB |
|---|---|---:|---:|---:|---:|---:|
| Operator | 1 / 32 / 64×80 | 3 | 2.341 | 1.665 | 1.41× | 76.15→61.54 |
| Operator | 1 / 32 / 256² | 3 | 44.454 | 27.229 | 1.63× | 964.48→757.11 |
| Operator | 8 / 32 / 64×80 | 1 | 1.598 | 1.166 | 1.37× | 68.37→59.72 |
| Operator | 8 / 32 / 64×80 | 2 | 6.854 | 3.805 | 1.80× | 240.74→205.73 |
| Operator | 8 / 32 / 64×80 | 3 | 17.855 | 9.964 | 1.79× | 530.76→442.68 |
| Operator | 32 / 32 / 64×80 | 1 | 5.872 | 4.520 | 1.30× | 266.74→233.16 |
| Operator | 32 / 32 / 64×80 | 2 | 34.458 | 18.477 | 1.86× | 947.87→815.91 |
| Operator | 32 / 32 / 64×80 | 3 | 71.230 | 37.819 | **1.88×** | 2054.13→1719.11 |
| Operator | 4 / 32 / 256² | 1 | 12.338 | 9.930 | 1.24× | 446.42→376.89 |
| Operator | 4 / 32 / 256² | 2 | 63.140 | 35.615 | 1.77× | 1572.92→1329.64 |
| Operator | 4 / 32 / 256² | 3 | 131.467 | 74.469 | **1.77×** | 3413.23→2810.14 |
| ConverseBlock | 16 / 16 / 32×40 | 1 | 3.162 | 2.723 | 1.16× | 46.93→46.95 |
| USRNet | 8 / RGB / 16×20 | 3 | 16.927 | 16.783 | **1.01×** | 471.28→452.82 |
| USRNet | 16 / RGB / 16×20 | 3 | 44.170 | 41.917 | **1.05×** | 903.53→862.98 |
| USRNet | 16 / RGB / 16×20 | 2 | 15.070 | 14.086 | **1.07×** | 419.58→402.67 |

B4/256²/s1 已有速度收益，不能将上一轮 B1/256²/s1 的约 6% 时间代价直接外推到更大 batch。
USRNet B8/s3 的约 1% 差异只宜视为基本持平；不能把共享核 Operator 的近 2× 收益当作整网收益。

较大 s3 完整步的 reserved 峰值也下降：B32/64×80 从 2994→2468 MiB，B4/256²
从 4724→4254 MiB；USRNet B16/s3 从 1212→1106 MiB。ConverseBlock B16 的
allocated 基本不变，reserved 为 62→64 MiB，不能概括为所有场景都节省显存。

## 独立算子 forward+VJP

该测量请求 x/kernel/bias 梯度，不含 loss 和优化器；与完整训练步分别报告。

| B / C / H×W | s | dev / ms | 当前 / ms | dev÷当前 |
|---|---:|---:|---:|---:|
| 1 / 32 / 64×80 | 3 | 1.972 | 1.390 | 1.42× |
| 1 / 32 / 256² | 3 | 40.930 | 23.811 | 1.72× |
| 8 / 32 / 64×80 | 1 | 1.140 | 0.764 | 1.49× |
| 8 / 32 / 64×80 | 2 | 6.065 | 3.092 | 1.96× |
| 8 / 32 / 64×80 | 3 | 15.705 | 7.787 | 2.02× |
| 32 / 32 / 64×80 | 1 | 4.401 | 3.091 | 1.42× |
| 32 / 32 / 64×80 | 2 | 30.305 | 14.232 | 2.13× |
| 32 / 32 / 64×80 | 3 | 62.696 | 29.355 | 2.14× |
| 4 / 32 / 256² | 1 | 9.341 | 6.944 | 1.35× |
| 4 / 32 / 256² | 2 | 56.129 | 28.742 | 1.95× |
| 4 / 32 / 256² | 3 | 117.725 | 60.690 | 1.94× |

所有 26 组负载均完成三次进程复测；算子数值对照及完整训练步的短训练/有限值检查均通过。
未锁 GPU 频率，原始范围不是置信区间；这些结果不说明其他设备、未测形状或真实数据收敛的性能。

## 复现

```powershell
& ./experiments/training_speed/run.ps1 test/test_training_scale3_batch.py
& ./experiments/training_speed/run.ps1 test/benchmark_fp32_training.py --suite scale3-batch --iters 20 --rounds 6
```

默认输出独立于原测试组，不覆盖旧实验记录：

- [FP64 数值记录](../artifacts/training_scale3_batch_validation.json)
- [数值测试日志](../artifacts/training_operator_optimization/scale3_batch_validation.log)
- [三次延迟、显存与波动汇总](../artifacts/training_operator_optimization/summary_scale3_batch.md)
- [第 1 次](../artifacts/training_operator_optimization/scale3_batch_run1.json)、[第 2 次](../artifacts/training_operator_optimization/scale3_batch_run2.json)、[第 3 次](../artifacts/training_operator_optimization/scale3_batch_run3.json)

原始 artifacts 由 Git 忽略；报告中的结果表和测试/基准入口随源码提供。
