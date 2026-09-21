# 当前 FP32 训练算子的 Nsight 分析

2026-09-17，分支 `codex/training-operator-optimization`。本轮只增加采集、解析与分析材料，未修改生产算子。

主要结论：大图 s3 的 GPU kernel 时间约 **43% 在 FFT、13–21% 在自定义频谱核**；额外的复制、roll、pad 也很显著。自定义核大多已有较高 DRAM 吞吐，应优先减少全图中间读写。FP64 核准备 FFT 则呈现不同限制：主要的 768 点 double C2C 代表 launch，双精度执行管线活跃率约 81–83%。因此后续优化需要按核准备占比和 batch 分派，不能只调一个 CUDA 核的 occupancy。

![各工作负载 GPU kernel 时间占比](../artifacts/nsight_training/final/kernel_time_share.png)

## 采集范围与可信度

- RTX 5060 Ti 16 GB，Windows WDDM，driver 616.92；PyTorch 2.11.0+cu130。
- Nsight Systems 2025.3.2、Nsight Compute 2025.3.0；使用当前源码校验后的 JIT 扩展，确认 s1/s2/s3 均进入 `SpectralSolve`。
- FP32、TF32 关闭、eager；每个进程预热 5 次。核准备保持可微 FP64，随后转 complex64；频谱复用默认关闭。没有 FP16/BF16。
- Systems 每例捕获 3 次执行，以 `cudaProfilerStart/Stop` 排除编译、初始化和预热；记录 CUDA、手工 NVTX 阶段及 autograd NVTX。
- Compute 记录 4 组算子的自定义核、1 组 B4 s3 FFT，共 31 个 launch；每个选中 launch 进行 16 passes kernel replay。`cache-control none`、`clock-control none`，未更改 GPU 时钟策略或系统权限。
- 算子 shape 是 **LR 输入**：C32、256²、s3 对应 768² 输出。检查 x/kernel/bias VJP；s1 使用共享 x 作为 prior，s3 包含 nearest prior 及其反向。这些算子案例没有 loss/optimizer。
- USRNet 是 B16/RGB/16×20/s3、2 iterations/1 block 的缩小模型，计入完整 SGD 步；使用已有合成配方 alpha=.1、lr=1e-4、momentum=.9。不是完整规模模型或收敛验证。

当前主文件指纹：`converse2d.cpp` 为 `53adc684912b…`，`converse2d_training.cu` 为 `f94db6914275…`；完整源码、fixture、脚本哈希及准确命令见逐例 metadata/command JSON。

Systems 的最终报告在 `artifacts/nsight_training/final/`。父目录的首轮 Systems 采集曾跨迭代保留上一轮返回张量，现已修正并全部重采，本文不使用首轮数据。Compute 仅捕获一个 step，不受该生命周期问题影响；无 profiler 的计时也逐次丢弃返回值。

## Systems：时间主要花在哪里

以下是 **GPU kernel 累计时间/次**，百分比分母为所有 kernel duration 的和，不是端到端 wall time。分类互斥，不能与后文 NVTX 正交归因重复相加。

| 工作负载 | kernel 数/次 | 累计 ms/次 | FFT | copy/roll/pad | 自定义频谱核 | 其他 |
|---|---:|---:|---:|---:|---:|---:|
| B1 C32 256² s1 | 45 | 1.543 | 56.45% | 20.26% | 12.87% | 10.42% |
| B4 C32 256² s1 | 46 | 5.516 | 45.21% | 9.32% | 25.83% | 19.64% |
| B1 C32 256² s3 | 56 | 21.037 | 43.02% | 32.03% | 12.61% | 12.34% |
| B4 C32 256² s3 | 57 | 53.875 | 42.52% | 19.53% | 20.79% | 17.15% |
| B32 C32 64×80 s3 | 57 | 27.764 | 44.42% | 12.60% | 20.75% | 22.24% |
| 缩小 USRNet B16 s3 完整步 | 648 | 31.932 | 22.24% | 14.67% | 10.97% | 52.11% |

“其他”保守保留未分类通用核，包括逐点运算、归约、nearest 和网络卷积，不能统一解释为 launch 开销。USRNet 中较大的单项包括 vectorized add 2.662 ms、卷积 wgrad 2.146 ms、fill 1.137 ms、卷积 dgrad 1.063 ms。仅继续优化频谱 CUDA 核会受到其整步占比限制。

B4 256² s3 的阶段归因是 prior **1.077 ms**、forward **18.273 ms**、backward **34.525 ms**。五个自定义核分别为 `solve_alias` 1.665、`solve_output` 2.550、`adjoint_q` 1.762、`adjoint_inputs` 2.622、`adjoint_filter` 2.604 ms/次。共享核反向值得单独研究，但其约 4.8% 的全算子 kernel 时间也限定了这一优化的作用范围。

核准备的独立 NVTX 归因如下。前向列包含 preparation 内的所有 kernel；反向列仅含匹配 preparation FFT sequence 的 autograd node，**未包含全部 cast/pad/roll 反向**。这些操作已经包含在上表中。

| 工作负载 | 核准备前向 ms | 核 FFT 反向 ms |
|---|---:|---:|
| B1 256² s1 | 0.467 | 0.392 |
| B4 256² s1 | 0.443 | 0.392 |
| B1 256² s3 | 4.340 | 3.051 |
| B4 256² s3 | 4.285 | 3.008 |
| B32 64×80 s3 | 0.233 | 0.273 |
| 缩小 USRNet B16 s3 | 1.475 | 1.389 |

共享固定核的 preparation 成本随 B 增长较小：B1 大图优先优化它更有价值，而 B32 小图已经主要由激活变换、数据流量和反向占用时间。不能把某个形状的准备策略无条件推广到其他形状。

### wall time 与时间线间隔

同一 harness 在独立进程中不挂 profiler，预热后连续执行 20 次的同步 wall 均值如下，仅作本轮诊断参照，并非多轮配对速度结论：B1 s1 **2.010 ms**、B4 s1 **6.225 ms**、B1 s3 **22.440 ms**、B4 s3 **60.065 ms**、B32 s3 **28.133 ms**、USRNet B16 s3 **36.182 ms**。记录在父目录 `*.none.metadata.json`，与历史完整 SGD benchmark 的 scope 不同。

CPU NVTX 范围不能代表完整步延迟。例如 B4 s3 三次 host Step 仅 2.017/1.706/1.437 ms，GPU 投影跨度则约 55.6–55.9 ms。USRNet 三步的 kernel union 为 95.797 ms、跨度为 112.891 ms；计入 memcpy 后仍有 10.602 ms 未捕获到 CUDA 活动。这提示进一步检查 launch/sync，但不能直接判定 CPU 瓶颈，间隔也可能含 WDDM 调度、依赖等待或 profiler 开销。不同进程的 kernel sum 与无 profiler wall 不应相减解释为固定的 CPU 开销。

## Compute：硬件限制的证据

| case / kernel | registers/thread | 理论 / 实际 occupancy | DRAM SOL | 实测 GB/s | eligible warps/scheduler |
|---|---:|---:|---:|---:|---:|
| B4 256² s3 `solve_alias` | 46 | 83.33 / 73.09% | 94.63% | 423.59 | 0.25 |
| B4 256² s3 `adjoint_q` | 56 | 66.67 / 60.49% | 94.33% | 355.63 | 0.24 |
| B4 256² s3 `adjoint_filter` | 74 | 50.00 / 43.93% | 80.80% | 283.19 | 0.64 |
| B32 64×80 s3 `adjoint_filter` | 74 | 50.00 / 45.64% | 80.99% | 362.51 | 0.68 |
| B4 256² s1 `filter_scale1` | 72 | 50.00 / 46.92% | 96.13% | 430.26 | 0.10 |

DRAM SOL 使用 `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`，GB/s 来自独立的 `dram__bytes.sum.per_second`，不能混为同一指标。B4 s3 的 `solve_alias/adjoint_q` long-scoreboard 分别占 warp 指令间隔约 90%/85%，同时可发射 warp 很少，支持内存依赖限制的判断。这不是 kernel 时间百分比。

共享核 filter 的 74 个寄存器把 256-thread block 限制在 3 blocks/SM，即理论 50% occupancy。B32 filter 还存在 wait 23.34%、short-scoreboard 11.47% 的指令间隔占比，值得检查串行广播归约、索引和寄存器活跃范围。但 s1 filter 的 DRAM SOL 已达 96%，单纯提高 occupancy 未必改善速度。

FP64 FFT 的行为不同：

| 768 点 FFT | registers/thread | 实际 occupancy | FP64 pipe | DRAM SOL |
|---|---:|---:|---:|---:|
| double C2C，前向代表 launch | 104 | 31.02% | 83.41% | 36.01% |
| double C2C，反向代表 launch | 104 | 31.02% | 81.10% | 36.33% |
| float R2C | 40 | 85.92% | 0% | 85.30% |
| float C2C | 62 | 62.59% | 0% | 82.28% |
| float C2R | 39 | 85.99% | 0% | 86.68% |

FP64 pipe 为 `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed`。double C2C 更符合双精度执行及调度/同步限制；FP32 大 FFT 更符合带宽限制。这里没有将高精度准备改回 FP32，独立精度门槛继续保留。

NCU 的 replay duration 和规则 `Est. Speedup` 均不作为端到端时间或预期收益。local-spilling 派生指标为 `no data`，不能宣称没有溢出；本轮没有 PC sampling，因此聚合 stall 也不能精确定位到某条 SASS 指令。

## 后续优化顺序

1. **先减少 FFT 周围的大图读写。** 针对 copy/layout/cast、可微 pad/roll 和 nearest 相关数据通路设计独立候选；保留 FFT 自动求导，先看实际流量是否下降，再看完整步。B1 s3 的 copy/roll/pad 已占 32%，收益空间比单独调整谱核寄存器更明确。
2. **按形状继续优化核准备。** B1 大图核准备占比高，可以研究减少变换工作；B32 小图准备成本已较低，优先级应下降。同 forward 复用仍受完整模型质量门槛约束，继续默认关闭。
3. **单独优化 s3 共享核反向。** 检查 `adjoint_filter` 的广播归约组织、索引与寄存器生命周期。它有真实热点证据，但不应直接强制 max-registers 或把更高 occupancy 当作成功。改变归约顺序后需重新跑固定 FP64、弱正则、共享输入及高 阶回归。
4. **整网优化需覆盖其他算子。** 缩小 USRNet 的谱核仅占约 11%，还应分析卷积反向、add/fill 与时间线间隔；当前数据不支持把算子加速等同于整网加速。

完整模型已有的严格 FP64 梯度差距及真实数据收敛/质量验证仍未解决；本轮 Nsight 采集不改变这些状态。

## 复现与原始报告

```powershell
# 若源码发生变化，先运行 extension_loader.py 重建；否则允许校验后复用二进制。
$env:CONVERSE2D_SKIP_BUILD='1'
& ./experiments/training_speed/run.ps1 test/profile_nsight_training.py --tool nsys --case op-b4-256-s3 --output-dir artifacts/nsight_training/repeat
& ./experiments/training_speed/run.ps1 test/profile_nsight_training.py --tool ncu --case op-b4-256-s3 --steps 1 --output-dir artifacts/nsight_training/repeat
& ./experiments/training_speed/run.ps1 test/profile_nsight_training.py --tool ncu --case op-b4-256-s3 --steps 1 --launch-count 14 --kernel 'regex:.*regular_fft.*' --output-dir artifacts/nsight_training/repeat_fft
& ./experiments/training_speed/run.ps1 test/summarize_nsight_training.py 'artifacts/nsight_training/final/*.sqlite' --output artifacts/nsight_training/final/nsys_summary.json --markdown artifacts/nsight_training/final/nsys_summary.md
& ./experiments/training_speed/run.ps1 test/summarize_ncu_training.py artifacts/nsight_training/op-b4-256-s3.ncu.ncu-rep --export --output artifacts/nsight_training/repeat_ncu_summary.json
```

- [采集脚本](../test/profile_nsight_training.py)、[Systems 解析脚本](../test/summarize_nsight_training.py)、[Compute 解析脚本](../test/summarize_ncu_training.py)。
- [Systems 最终汇总](../artifacts/nsight_training/final/nsys_summary.md)、[结构化 JSON](../artifacts/nsight_training/final/nsys_summary.json)。
- [Compute 完整指标](../artifacts/nsight_training/ncu_summary.md)、[结构化 JSON](../artifacts/nsight_training/ncu_summary.json)。
- [B4 s3 Systems GUI 报告](../artifacts/nsight_training/final/op-b4-256-s3.nsys.nsys-rep)、[B4 s3 Compute 报告](../artifacts/nsight_training/op-b4-256-s3.ncu.ncu-rep)、[FFT Compute 报告](../artifacts/nsight_training/fft/op-b4-256-s3.ncu.ncu-rep)。

解析通过 `(process, correlationId)` 唯一匹配 launch 和 kernel；autograd worker 的手工阶段采用明确标记的同进程唯一时间包含归因，FFT preparation 反向另用 forward sequence 对应。六份报告 correlation 全部唯一、step/phase 零未分配，类别/核名/阶段/step 总和均通过一致性检查。B4 s3 与官方 `nsys stats cuda_gpu_kern_sum` 精确一致：171 个 kernel、161,623,515 ns。此次 kernel sum 等于 union，但解析器仍区分二者以支持并发时间线。

`artifacts/` 被 Git 忽略；保存报告与源指纹后才能在别处复核本轮数据。
