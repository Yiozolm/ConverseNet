# 当前训练实现对 Python/ATen 基线

2026-09-17。在同一 RTX 5060 Ti 上，以完整 ConverseUSRNet、FP32、真实图 HR96→LR32/s3、batch4/microbatch4 直接比较。Python 基线是仓库的 `backend="pytorch"` / `models.converse_core.converse2d_reference`，所有张量和 FFT 都在 CUDA 上运行，不是 CPU 版本。

四轮交替测量的完整训练步配对加速比中位数为 **1.516×**，allocated 峰值显存减少 **24.62%**。三种子的 250 步任务质量对照全部通过，final RGB/Y PSNR 最大差小于 **0.00002 dB**。上一轮报告的约 1.049× 比较的是优化前后的 C++/CUDA 实现，是不同基线，不能与本次结果相乘。

## 直接性能对照

| 指标 | Python/ATen | 当前 CUDA | Python/current 配对比 |
|---|---:|---:|---:|
| 完整训练步 wall | 654.996 ms | 431.998 ms | 1.516× |
| forward + loss + backward wall | 633.008 ms | 407.134 ms | 1.554× |
| PyTorch peak allocated | 12.916 GB | 9.736 GB | 减少 24.62% |
| PyTorch peak reserved | 13.453 GB | 10.412 GB | 分别记录，不与 allocated 相加 |

耗时列是四个轮次中位数的中位数；加速比列是四个逐轮配对比率的中位数，二者计算口径明确区分。完整步配对比率范围 **1.506–1.527×**，forward/backward 为 **1.554–1.556×**。范围不是置信区间。等价地，本次完整步耗时减少约 34%。

测试方法：

- 使用相同 full 5 iterations/7 blocks/64 features 模型、133 个参数张量、307,987 个参数，严格加载同一个原预训练 checkpoint；保持原门控初始化，关闭复用候选。
- 相同 seed17 真实图 CPU batches，记录每批 tensor hash。退化、MSE/Adam1e-5、TF32/AMP 关闭、cuDNN deterministic=True，与上一轮数据协议一致。
- 一次仅一个 GPU 模型；轮次顺序为 Python→current、current→Python，交替四轮，每路预热 5 步后测 8 步。
- 每轮预热后严格恢复同一模型初值，清除梯度，将已分配 Adam 的 step/exp_avg/exp_avg_sq 归零；模型与 Adam 初值均核对哈希。
- 完整步直接复用 `train_usrnet_dataset.train_step`，包含同步 H2D、forward/loss/backward、有限值/梯度范数检查及 Adam。图像解码/退化/哈希、编译、fixture 创建、预热/复位、数值快照和报告 I/O 在计时外。没有 profiler。
- 显存是 PyTorch allocated/reserved 的总高水位，包含模型、优化器和该训练步张量，不包含所有驱动/库分配，也不是 Windows WDDM 的进程总显存。

Python 路径在 batch4 单微批即可完成，不需要 OOM 后降低 batch。因此两路使用相同物理 batch，也没有拿梯度累积版本与单 batch 版本比较。

本次是预热并复位后的短段交替 A/B；上一轮约 501 ms 的 current 时间来自连续 250 步、穿插验证和 I/O 的另一组运行。不能把旧值直接除以本次 Python 时间，也不能据短段计时承诺完整收敛所需时间。

## 数学路径与数值诊断

Python/ATen 基线使用可微全谱 complex64 FFT 与自动反向，显式保留频谱、power、alias correction 等中间图。当前实现使用半谱路径、融合频谱核及解析一阶 VJP，核 FFT 仍以可微 FP64 准备后转 complex64。因此这是整个实现路径的对照，收益不能全部归因于 Python 解释器开销。

未计时的前两次 Adam 更新从相同模型/输入开始，检查了真实 autograd 图：Python 不含 `SpectralSolve`，current 包含。输出、参数、梯度和 Adam 状态均有限。

| 更新 | 输出 max abs | 输出 relative L2 | 全参数梯度 max abs | 全参数梯度 relative L2 |
|---|---:|---:|---:|---:|
| 1 | 4.262e-6 | 9.490e-7 | 2.289e-5 | 4.040e-7 |
| 2 | 7.421e-6 | 1.159e-6 | 5.555e-5 | 1.065e-6 |

这些是 current 对 FP32 Python 路径的差异统计，未另选容差来宣称独立 FP64 门槛已通过。既有完整模型 FP64 逐点梯度差距继续保留。

## 250 步任务质量对照

使用相同 900/100 图片拆分和种子 17/29/43，各自从相同原预训练权重运行 250 步。Python 新运行通过适配入口强制所有求解层使用 `pytorch`；每个完整 forward 必须恰好调用 Python reference 40 次（35 个 residual solver + 5 个 DataNet），首次训练图另验证无 `SpectralSolve`。

适配入口保留旧 worker 的 `config.variant=current` 参数形式；实际执行路径以新增的 `comparison_backend=pytorch`、独立 backend manifest 和 `backend_verification` 为准。每次完整运行记录 250 个训练 forward、300 个评估 forward、22,000 次 Python reference 调用，纯 Python 路径不构建或调用原生求解扩展。

current 质量结果复用上一轮保存的同源码、同 worker/data/metric、同物理 batch 三种子运行；逐批输入和初始化等哈希已逐项核对。这里只复用其训练质量证据，不直接用历史时间计算本次速度比。原训练 worker、数据协议和指标函数未改。

质量门槛沿用事前约定：每个种子的 current final RGB/Y PSNR 相对 Python 下降不超过 0.05 dB，RGB/Y SSIM 下降不超过 0.001，全部 loss/gradient/评估输出有限。

| seed | Python final Y PSNR | current final Y PSNR | Y 差 current−Python | RGB 差 current−Python | 四项质量门槛 |
|---|---:|---:|---:|---:|---|
| 17 | 30.085767 | 30.085773 | +0.000006 dB | +0.000006 dB | 通过 |
| 29 | 30.078747 | 30.078734 | −0.000013 dB | +0.000002 dB | 通过 |
| 43 | 30.086333 | 30.086320 | −0.000013 dB | +0.000002 dB | 通过 |

三个种子的四项门槛全部通过，最大 RGB/Y SSIM 差小于 0.000001。离线审计核对了每对 250 个输入 batch hash、同一原 checkpoint 和初始参数、拆分、验证 payload、共享源码及保存 checkpoint 的文件哈希；3/3 配对完整可比，完整性错误为 0，模型状态确实改变。Python 运行的分派计数均符合 550 次 forward / 22,000 次参考求解，梯度和评估输出有限。

独立 FP64 逐点审计及旧合成压力测试状态不变。这里验证的是固定 100 张保留图块上的短程微调一致性，不是完整收敛、全图恢复质量或达到最终相同质量的总时间证明。

## 复现与原始文件

```powershell
& ./experiments/training_speed/run.ps1 test/benchmark_python_training.py --batch-size 4 --microbatch-size 4 --rounds 4 --steps 8 --warmup 5 --output artifacts/python_training_comparison/repeat_timing.json
& ./experiments/training_speed/run.ps1 test/train_usrnet_python_comparison.py --backend pytorch --seed 17 --steps 250 --eval-every 125 --batch-size 4 --microbatch-size 4 --run-dir artifacts/python_training_comparison/repeat_pytorch17
& ./experiments/training_speed/run.ps1 test/summarize_python_training.py
```

- [交替性能基准](../test/benchmark_python_training.py)、[纯 PyTorch 训练适配入口](../test/train_usrnet_python_comparison.py)。
- [完整计时与数值 JSON](../artifacts/python_training_comparison/paired_timing_b4_m4.json)、[容量检查](../artifacts/python_training_comparison/capacity_b4_m4.json)。
- [计时/质量离线汇总](../artifacts/python_training_comparison/summary.md)、[结构化审计](../artifacts/python_training_comparison/summary.json)、[汇总脚本](../test/summarize_python_training.py)。
- [此前真实图协议与 current 三种子](dataset_finetuning.md)。

工具记录了当前源码、build manifest、Python/Torch/CUDA/GPU、原 checkpoint 与输入哈希。所有原始结果位于 Git 忽略的 `artifacts/python_training_comparison/`，不修改预训练权重或原图片。
