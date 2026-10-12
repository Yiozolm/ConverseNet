# 当前 Flash FFT：RTX 5060 Ti 16GB / SM120 瓶颈与流程加速上限

2026-10-12，HEAD `6dd4bbb`，partial-compensation 版本。

**主要结论：当前 fused forward/backward 的首要瓶颈是低驻留并行度下的全局访存延迟。** 每个 block 占约 80KB shared memory、125–128 registers/thread，两个资源都限制为 1 block/SM；实际 occupancy 约 33%。long-scoreboard 约占 stall 采样的 38–39%，DRAM 使用仅约 27–36%。FFT shared-memory 重放与同步是次要问题。

**对完整流程的意义：在本次 USRNet 训练研究接入中，两个 fused kernel 占 GPU kernel 总耗时 16.95%。将它们加速 2×，其余工作固定，整体 GPU 工作加速约 1.093×；即使完全消除，上限也只有 1.204×。** 卷积、kernel preparation、LayerNorm 和主机开销必须另算。

![耗时归因和 Amdahl 上限](analysis/bottleneck_amdahl.png)

## 测量口径与硬件

- 本机 RTX 5060 Ti 16GB，CC 12.0，36 SM；PyTorch 2.11.0+cu130，CUDA toolkit 13.0，Nsight Compute 2026.3.1，Windows WDDM。
- NCU 工作负载：B4 C128 96×96，circular pad 2，实际 FFT plane 100×100；当前 `static_s1_forward<100,100,...>` 与 `static_s1_backward<100,100>`。每个 grid 为 512 blocks，每 block 512 threads。
- 完整模型：预训练 USRNet，B4 LR32×32 → HR96×96，scale 3，5 iterations，35 次 C128 prior Converse2D、5 次 data-term、147 次 pointwise convolution、70 次 LayerNorm。输入和 target 为固定 seed 的合成数据，已经驻留 GPU；完整训练步包含 forward、MSE、backward、Adam。
- `model_train_002` 为原生产路径。`model_flash_002` 仅在研究 harness 内将 7 个 prior 模块的 35 次调用换为现有 `train_study.FlashS1`；data-term 与其余模型实现保留生产路径。生产代码没有接入该研究 kernel。
- FP32/complex64，TF32 与 AMP 关闭；模型 profiling 采用 deterministic 算法与 `0xC03C03` CPU affinity。
- kernel 编译使用 `-O3 -lineinfo -std=c++17`，原生 `sm_120` codegen；无 fast-math。研究源码 SHA256：`29288d898273b1f22c7580151c4dc902c9b6793d689343bd9498bc91be50c752`，二进制 SHA256：`357a391401e9e30b2d88245f1c5f5bbe445182629f25a230c30e973e01bc4f44`。

本次直接核验并加载 checked production binary，完整源文件/工具链/二进制指纹检查仍然启用。研究二进制先由 NCU worker 构建，后续模型及数值测量核对同一源码和二进制哈希。

## 1. kernel 内部瓶颈

| NCU 指标 | fused forward | fused backward | 含义 |
|---|---:|---:|---|
| registers/thread | 125 | 128 | backward 的 512 threads 使用全部 65536 个 SM registers |
| dynamic shared memory | 80000 B | 80000 B | 整个 complex64 plane 驻留片上 |
| static shared memory | 0 B | 128 B | NCU 另计每 block 1024 B reserved shared memory |
| register / shared residency limit | 1 / 1 block | 1 / 1 block | 降低单一资源不会自动获得双 block 驻留 |
| achieved occupancy | 33.18% | 33.35% | theoretical 16/48 warps = 33.33% |
| eligible warps/scheduler/cycle | 0.68 | 0.92 | 少于 1，难以隐藏依赖和访存延迟 |
| issue-active | 35.04% | 44.77% | 有大量发射空档 |
| SM throughput | 33.18% | 42.22% | 没有达到计算吞吐屋顶 |
| DRAM utilization | 26.85% | 36.41% | 不是当前 kernel 的带宽饱和瓶颈 |
| long-scoreboard / source samples | 38.43% | 39.11% | 主要等待 L1TEX/global-load 依赖 |
| wait / source samples | 13.86% | 12.94% | 指令依赖延迟亦明显 |
| local load/store sectors | 0 / 0 | 0 / 0 | 本 shape 没有 spill traffic |

NCU 的两条 SOL rule 均为 `Latency Issue`；warp-state rule 均为 `Long Scoreboard Stalls`。因此，不能仅因为 compensated FP32 指令多就将当前 SM120 kernel 定性为 FP32 算力饱和。

源码 stall 采样的热点：

- forward：[`fused.cu:326`](../../tools/v4_flash_fft/fused.cu#L326)，输入 pad/load 的 consumer 位置，9825 long-scoreboard samples；[`fused.cu:191`](../../tools/v4_flash_fft/fused.cu#L191)，solve 依赖，4804 samples。
- backward：[`fused.cu:607`](../../tools/v4_flash_fft/fused.cu#L607) 的 x load，4749 samples；[`fused.cu:617`](../../tools/v4_flash_fft/fused.cu#L617) 的 g load，6028 samples；[`fused.cu:629`](../../tools/v4_flash_fft/fused.cu#L629) 的 k/Y 依赖消费位置，3603 samples。
- FFT 阶段的 shared store、barrier 与 twiddle/operand 等待在 [`fused.cu:262`](../../tools/v4_flash_fft/fused.cu#L262)、[`fused.cu:267`](../../tools/v4_flash_fft/fused.cu#L267)、[`fused.cu:288`](../../tools/v4_flash_fft/fused.cu#L288) 也可见。

采样定位到的是等待数据的 consumer instruction，不一定就是发起 load 的 producer；应结合 SASS/source dependency 做下一轮改动。

NCU 还报告 forward/backward 的 global excessive sectors 分别占 23% / 18%；global load 每 32B sector 平均使用 17.6B / 18.8B。对应 global-access rule 的估计改善约 20.7% / 15.6%，只是诊断提示，不能与其它估计相加。shared load/store bank-conflict rule 分别只估计约 3.8%/4.7% 与 2.7%/3.5% 的改善，优先级低于全局访存延迟。

grid 已有 512/36 = 14.22 个单 block waves。假设等长 block，15 个离散 waves 相对理想连续调度的倍率仅 1.055；本机当前版本的首要问题不是 grid 太小或历史 A100 的 per-batch backward launch。

## 2. SM120 架构约束

以本机 device attributes 为准：每 SM **48 warps、65536 registers、102400 B shared memory**；单 block opt-in shared memory 为 **101376 B（99KiB）**。与 [NVIDIA CUDA Programming Guide 的 CC 12.x 资源表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) 一致。

这里的 16GB 是全局显存容量，不能增加 shared memory 或 block 驻留数。若仍使用 512 threads，要实现 2 blocks/SM，既需要 registers/thread 不超过约 64，也需要每 block 的 shared allocation 加 reserved memory 降到约 50KiB 以内。整张 100×100 complex64 plane 的 80KB 分配本身无法满足后一个条件。

- s2 64→128 的完整谱面为 128KiB，s3 48→144 为 162KiB，超过本卡 99KiB/block 上限；当前容量开关会落回 production。不能用训练半频谱绕过当前政策。
- 本机 `device__attribute_cluster_launch=1`，cluster 的 potential size 为 8。DSM/cluster 可以作为大谱面的研究方案，但增加跨 SM 通信与同步，尚无本工作负载上的实测收益。
- 当前严格 FP32 策略排除了 TF32/AMP。SM120 的 Tensor Core 宣传峰值不适合作为这些 FFT/adjoint 的可达 FP32 roofline；[官方 Tensor Core 输入类型表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) 也没有精确 FP32 输入 lane。
- 本卡理论 GDDR7 带宽 448GB/s，见 [NVIDIA RTX 5060 系列规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/)。本次 NCU 的 DRAM 百分比明显低于饱和，优先解决延迟隐藏与访问效率。

## 3. 单算子收益和完整流程收益是两个数

新跑的 B4 C128 HR96 pad2 单算子 forward+VJP，8 轮、每轮 15 次，包含每次可微 kernel preparation：

| 路径 | median | production / Flash |
|---|---:|---:|
| 当前 production | 3598.14 µs | — |
| 当前 Flash partial compensation | 1617.07 µs | **2.225×** |

这是局部算子速度，不能直接乘到模型吞吐上。原 production 模型的 prior 占 GPU kernel sum 39.09%；使用此局部倍率、假设其它工作固定，替换 prior 的 Amdahl 估计为 **1.274×**。

研究接入 Flash 后的新单步 trace，GPU kernel sum/union 为 **337.693ms**，kernel 首尾跨度为 **400.591ms**。sum 和 union 基本相同，没有重叠 GPU kernels 导致重复计时；CPU/user annotation 未加入分母。

| Flash 接入后的工作 | GPU kernel 时间 | 占比 |
|---|---:|---:|
| 所有 1×1 convolutions：forward + backward | 168.705ms | **49.96%** |
| LayerNorm：forward + backward | 52.975ms | 15.69% |
| Flash fused forward | 16.108ms | 4.77% |
| Flash fused backward | 41.126ms | 12.18% |
| prior 的 kernel FFT / pad / 参数梯度 / batch reductions 等其余工作 | 19.952ms | 5.91% |
| data-term reverse operator | 9.701ms | 2.87% |
| GELU、Linear、optimizer、residual 等其它工作 | 29.126ms | 8.63% |

最大单项为 cuDNN `wgrad2d_grouped_direct_kernel`：**105.661ms，31.29%**。它已超过两个 Flash fused kernels 的总和 **57.234ms，16.95%**。此 wgrad 使用当前模型调用的实际梯度形状；不能从另一个 shape 的 GEMM 成功记录直接推断收益。

## 4. 理论加速上限

采用 Amdahl 定律，`p` 为可改善工作当前占比，`r` 为该工作自身倍率，其余工作固定：

\[
S(r)=\frac{1}{1-p+p/r},\qquad S_{\max}=\frac{1}{1-p}.
\]

| 优化范围 | 当前 p | 自身 2× | 自身 3× | 自身 4× | 自身耗时→0 的上限 |
|---|---:|---:|---:|---:|---:|
| 仅当前 fused backward | 12.18% | 1.065× | 1.088× | 1.101× | **1.139×** |
| 当前 fused forward + backward | 16.95% | **1.093×** | 1.127× | 1.146× | **1.204×** |
| 整个 Flash prior 算子，含 preparation 和参数 VJP | 22.86% | 1.129× | 1.180× | 1.207× | **1.296×** |
| prior + data-term 的所有 reverse 工作 | 25.73% | 1.148× | 1.207× | 1.239× | **1.346×** |

这些是本 workload 上**GPU 工作的理想上限**，不是已测得的端到端速度承诺。CPU 调度、数据读取/生成、H2D、评估、保存 checkpoint 不会随 kernel 加速而自动消失。若把当前 profiled kernel 首尾跨度作为分母，两个 fused kernels 的 p 进一步降为约 14.29%，2× 对应约 1.077×，极限约 1.167×；该跨度含 profiler 影响，也不能冒充无 profiler 的 wall time。

分别启动的无 profiler 全步骤计时中，production median wall 为 437.213ms，Flash 为 368.230ms。Flash 的 5 次测量从 415ms 漂移到 330ms，且两组不是模型级配对实验，**不将二者相除宣称实测模型收益**。本报告的模型结论来自分项归因和明确条件下的 Amdahl 计算。

## 5. 下一步优先级

1. **若目标是训练流程吞吐，优先 pointwise wgrad，再评估 LayerNorm。** 两者分项占比约 50% 和 16%；全删 Flash core 也只剩约 20% 的理论倍率收益。wgrad 改动需真实 module activation/upstream gradient 的 frozen-FP32/FP64 gate，包含 layout copies 的完整 VJP 配对测量，并保持高阶回退。
2. **若继续改 Flash，先减少 global-load 等待和 excessive sectors。** 从 x/g pad-load、k/Y scratch 依赖以及 twiddle access 的实际 source/SASS 热点入手；可对 channel/batch grid 顺序、分段预取、独立指令交错进行独立 ablation。当前每线程寄存器已接近/达到 128，不能盲目通过更多 live registers 做预取。
3. **双 block 驻留需要同时重设计谱面存储和 register live ranges。** 分块 FFT、两阶段方案或有容量约束的异步 shared staging 都需实测，额外流量可能抵消 occupancy 收益。当前 SM120 无法只调 launch bounds 或只减寄存器就获得 2 blocks/SM。
4. **shared-layout swizzle、减少 FFT stage/barrier 是后续项。** bank-conflict rule 的估计只有几个百分点；不宜依据 shared-plane 设计就将它先验定为首要瓶颈。所有 arithmetic 改动仍遵守 FP32 数值预算，不放宽 fast-math、TF32 或训练谱复用。

## 验证与证据

- profiled s1 shape，8 seeds × output/dx/dweight/dbias：Flash 与当前 production 都通过 frozen Python FP32 baseline / independent Python FP64 gate；原始 64 个 comparison verdict 保存在 `analysis/operator_gate_timing.json`。这是数值局部 gate，不是完整模型或收敛证明。
- relevant release lanes 共 49 项：首轮 48 通过，唯一报错为 `TemporaryDirectory` ACL 导致数值报告无法写入。校验临时路径在本工作区后，在相同未改测试上重跑该项，1/1 通过。原失败与新成功日志均保留，未改任何测试或放宽 gate。
- checked-loader 的最初环境检查拒绝加载，因 PATH 中未发现已记录的 MSVC、architecture 环境与 manifest 不同；恢复真实工具链环境后通过全部指纹检查。最初尝试及其 traceback 保留在 `reports/model_train.log`。
- full NCU 初次受限环境未能启动，第二次采集完成两个 kernel 的 41 passes，并导出可完整读取的 `.ncu-repz`。CLI 在导出后仍占用 CPU 超过 9 分钟，故只停止该 collector。采集数据、全部 metric values、rule results 已成功解析；collector 非零退出与停止原因保存在 `analysis/full_collector_stop.json`。source NCU 正常退出，分别完成 5/6 passes。
- 没有覆盖所有任意大小、独立 prior、broadcast 或更高阶训练调用；两次模型 trace 是单步诊断。研究 wrapper 的 35 次 prior 接入用于本次性能归因，没有验证完整模型的数值/训练收敛，也未作为 release 功能。
- 本次全部新输出位于本目录，历史成功/失败记录保持原状。

主要文件：

- `reports/full.ncu-repz`、`reports/source.ncu-repz`：本机完整 NCU 数据。
- `analysis/ncu_summary.json`、`analysis/metrics_all_*.json`、`analysis/rules.json`：通过 `ncu_report` 提取。
- `reports/model_train_002/trace.json.gz`、`reports/model_flash_002/trace.json.gz`：GPU trace；相邻 `summary.json` 含 source/checkpoint SHA、调用归属和原始 timing。
- `analysis/amdahl.json`、`analysis/bottleneck_amdahl.svg`：可复算的数据与导出图。
- `reports/release_tests.log`、`reports/release_test_permission_retry.log`：两轮结果。
- `harness/`：standalone Python worker、模型与数值 harness、研究 C++/CUDA 源码快照。生产环境通过现有 checked loader 验证。

复算已有数据：

```powershell
& 'H:/Python/ConverseNet/.venv/Scripts/python.exe' profile/flash-bottleneck-20261012/analysis/analyze_ncu.py
& 'H:/Python/ConverseNet/.venv/Scripts/python.exe' profile/flash-bottleneck-20261012/analysis/render_summary.py
```

重新采集应创建新的 `profile/<run>/`，复制 harness 并更新 run-local 路径；不要复用本次 reports 或覆写旧失败日志。
