# SM120 bound 重新评估：新 LayerNorm + 当前 Flash FFT

2026-10-12，HEAD `1c2faf9`。本机 RTX 5060 Ti 16GB，CC 12.0，36 SM。

**新瓶颈排序是 pointwise wgrad → Flash backward / prior 周边 → GELU / LayerNorm 周边。**
当前 grouped wgrad 单项占 GPU 工作 **37.08%**，所有 pointwise convolution 占 **59.50%**。
Flash core 占 **12.95%**，LayerNorm 调用及其 autograd 工作占 **5.26%**。
旧报告的“Flash 首先受全局访存延迟限制”和“LayerNorm 占 16%”已经不适用于当前实现。

三个热点属于不同的 bound：

- **新 fused LayerNorm：带宽 bound。** NCU DRAM 利用率 forward **81.34%**、backward **90.98%**。
- **Flash FFT：shared-memory / 同步 / 指令依赖造成的 latency bound。** global-load 等待大幅下降，
  FP32 算术屋顶和 DRAM 屋顶均未饱和；不能继续用旧版 long-scoreboard 归因。
- **grouped wgrad：地址处理与指令管线 bound，伴随缓存访问依赖。** ADU **77.72%**、
  Shared FMA Heavy **71.70%**，DRAM 只有 **11.61%**；其逻辑 FP32 FLOPs 仅达到约 **2%** 屋顶。
  “compute 更忙”不等于“FP32 FFMA 吞吐已经接近峰值”。

![当前耗时归因及条件加速上限](analysis/bound_refresh.png)

## 口径和版本

完整模型沿用历史 workload：预训练 USRNet，B4 LR32×32 → HR96×96，scale 3，5 iterations，
35 次 C128 prior reverse、5 次 data reverse、147 次 pointwise convolution、70 次 LayerNorm。
每步包含 `zero_grad(set_to_none=True) + forward + MSE + backward + Adam`，数据预先驻留 GPU。
Adam 保持 `foreach=False, fused=False`；未计 DataLoader、H2D、评估、checkpoint 保存。

研究接入仅替换 7 个 prior 模块的 forward 为现有 `train_study.FlashS1`，保持每调用可微 kernel FFT、
参数 VJP、batch reductions；data term 保留生产实现。当前 Flash 为 forward **768 fused** /
backward **512 fused**。新 LayerNorm 为 release 默认实现。没有修改生产源码、dispatch 或算术。

所有 GPU 测量使用 FP32/complex64，关闭 AMP、TF32，开启 deterministic 算法与 cuDNN deterministic，
CPU affinity `0xC03C03`。PyTorch 2.11.0+cu130，CUDA toolkit 13.0，MSVC 14.44，NCU 2026.3.1，Windows WDDM。
研究 flags 为 `-O3 -lineinfo -std=c++17`，原生 sm_120，无 fast-math。

Checked production binary SHA256：
`e6b425c32293965d00b74bb5103566f9ef293fb16c64c2d73cff28ee31b7ceb6`。
当前 Flash source SHA256：
`a2ddd155b6367943ac377176e0eec2567bb6ef2e57d7693f0269450b4cfe7b0a`；binary：
`ccfbbbe2bed55d35f686830281d625424713a3542281a94712a3f98ea9d4a69c`。
本次 checked build 正常执行，Ninja 确认无须重编译，最终完整 manifest 与采集前副本一致。

NCU 使用 `--set full --section PmSampling` 和 `--set source --section SourceCounters`，
分别采集当前 Flash forward/backward、LayerNorm forward/backward 和 workload 匹配的
1×1 convolution wgrad。解析均使用 `ncu_report` Python API，包括 rules、source counters、SASS 和 PM samples。
未锁 GPU clocks，未 flush caches；后台桌面负载约 9%。短 kernel 的 NCU 时间不作为模型热态单调用时间。

## 1. 完整模型：重新采集的耗时分母

两次独立 Flash trace 的 GPU kernel sum 为 **282.195ms / 301.596ms**。
下表采用第二次 `model_flash_03`；第一次的占比同时给出，避免把单步波动当成精确常数。
两次均为 5714 kernels，sum 与 union 一致；CPU annotation 没有计入 GPU 分母。
第二次 kernel span 为 339.221ms，不能将 span 或 profiler 时间当成无 profiler 的 wall time。

| 当前 Flash + 新 LN 的工作 | GPU 时间（第二次） | 占比（第一次 / 第二次） |
|---|---:|---:|
| 全部 pointwise convolution，forward + backward | 179.456ms | 59.39% / **59.50%** |
| 其中 grouped direct wgrad（77 launches） | 111.833ms | 36.98% / **37.08%** |
| Flash forward（35 launches） | 10.685ms | 3.80% / **3.54%** |
| Flash backward（35 launches） | 28.374ms | 9.52% / **9.41%** |
| prior 的可微 kernel preparation / 参数 VJP / reductions 等 | 23.781ms | 7.84% / **7.89%** |
| LayerNorm 调用及其 autograd 工作 | 15.853ms | 4.72% / **5.26%** |
| data reverse | 11.070ms | 3.62% / **3.67%** |
| GELU | 18.956ms | 6.90% / **6.29%** |
| 其它 + Linear | 13.423ms | 4.22% / **4.45%** |

`grouped wgrad` 是 pointwise 行的子集，不能重复加到总和。prior 总计 **62.839ms / 20.84%**；
全部 reverse 为 **73.909ms / 24.51%**。Flash core 与 prior 总体是两个不同的优化范围。

### LayerNorm 仍有 5 次 fallback

70 次 LN 中，**65 次**实际使用新 fused forward/backward。其它 5 次全来自
`p.m_body.0.conv1.0`，输入 shape 相同，但 stride 为
`[1179648, 18432, 192, 2]`，非 contiguous，正常落回 ATen。
这些调用及其归属 autograd 工作共 **4.514ms / 1.50%**。
这是 layout boundary，不能归因为新 fused kernel 没有生效。

两个 fused LN kernel 合计仅 **7.135ms / 2.37%**；LN 组的其余时间包括上述 fallback、
参数 reductions、gradient accumulation、fill 等。全删两个 fused kernel 的理想模型 GPU 倍率仅 **1.024×**。
若处理这 5 次 fallback，应评估 stride-aware 路径或上游 layout，包含必要 copy 的完整 VJP 计时和数值 gate。
本次未变更该边界。

## 2. 硬件 counters：新的 bound

| NCU 指标 | Flash fwd | Flash bwd | LN fwd | LN bwd | grouped wgrad |
|---|---:|---:|---:|---:|---:|
| kernel time，µs（单次 replay） | 251.04 | 571.49 | 42.05 | 83.71 | 1788.45 |
| threads/block | 768 | 512 | 128 | 128 | 256 |
| grid blocks | 512 | 512 | 1152 | 288 | 1024 |
| registers/thread | 67 | 119 | 38 | 40 | 59 |
| shared allocation，B，含 reserved | 82560 | 82688 | 2304 | 3328 | 1024 |
| achieved active-warps % | 49.30 | 33.32 | 84.67 | 86.30 | 63.42 |
| eligible warps/scheduler/cycle | 1.43 | 1.74 | 0.56 | 0.24 | 1.93 |
| issue-active % | 54.34 | 62.99 | 28.05 | 18.05 | 58.28 |
| SM throughput % | 53.11 | 59.62 | 25.26 | 32.41 | 77.72 |
| DRAM throughput % | **43.52** | **46.93** | **81.34** | **90.98** | **11.61** |
| long-scoreboard，cycles/issued instruction | 1.43 | 0.39 | 33.50 | 34.39 | 3.78 |
| barrier，同上 | 1.81 | 0.39 | 2.71 | 1.31 | 0.00 |
| mio-throttle，同上 | 1.24 | 0.59 | 0.09 | 1.13 | 1.09 |
| wait，同上 | 1.68 | 0.96 | 2.10 | 1.36 | 1.79 |
| math-pipe-throttle，同上 | 0.21 | 0.06 | 0.22 | 0.05 | 1.98 |

DRAM 百分比来自 SM120 实际 metric
`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`；本机不存在旧名
`dram__throughput.avg.pct_of_peak_sustained_elapsed`，不存在的 metric 未补成零。
warp-state ratio 是 cycles per issued instruction，与 source 采样百分比不同。

### Flash：访存优化已经有效，后续是片上流程和指令成本

对照旧 `flash-bottleneck-20261012`：forward long scoreboard **4.41 → 1.43**，
backward **3.52 → 0.39**；eligible warps **0.68 → 1.43 / 0.92 → 1.74**。
新 forward occupancy 约 50%，旧版约 33%；new core time 251/571µs，旧版 413/865µs。
这是独立诊断 replay 的历史对照，未作为新配对 wall-time 倍率。

NCU 的 SOL rule 仍均为 `Latency Issue`，但当前 forward 最大等待项已是 barrier / wait，
backward 的 long scoreboard 小于 wait、short scoreboard 与 mio 等待。
forward/backward 分别执行 **53.33M / 136.55M** warp instructions，shared wavefronts 为
**9.59M / 14.56M**。FP32 roofline rule 仅约 **14% / 22%**；这不是 FP32 算力饱和。
当前 shared load/store conflict rule 估计改善分别为 **8.9%/7.9%** 与 **5.9%/5.4%**，
优先级已高于旧版；rule estimates 不能相加或直接作为收益承诺。

硬件运行时仍为 65536 registers/SM、102400B shared/SM、101376B opt-in/block、48 warps/SM。
当前两个 Flash kernels 的 register/shared residency limit 都是 **1 block/SM**。
80KB 全谱面仍限制双 block；仅降低寄存器不能解除这条约束。
每个 grid 有 14.22 waves/SM，等长 block 的离散末波倍率约 15/14.22 = **1.055**，
并非小 grid 导致的主要欠填充。本次不以未提供的 per-SM counters 宣称已证明完全负载均衡。
PM samples 已保存，Tensor Core activity 为零；不借 Tensor Core 宣传峰值评价严格 FP32 FFT。

### LayerNorm：低 issue rate 伴随 DRAM 饱和，不能仅看 long scoreboard 定成延迟问题

LN forward/backward 的长等待确实很高，source hotspots 在
`layernorm.cu:31/122` 和 `layernorm.cu:159/184` 的数据消费位置。
但 DRAM 已为 **81%/91%**，NCU SOL rule 为 `High Throughput`，明确指向 DRAM。
进一步调 block/occupancy 的潜力有限；更有意义的是减少输入重读、跨算子 materialization 或 layout 开销。
若不减少流量，只把同等流量推到 100% sustained，隔离 kernel 的带宽尺度空间约
**1.23× / 1.10×**，不是经过证明的算法速度上限，也不能乘到 LN 整组或全模型。
热态模型 trace 中，LN 在 L2 内的时间与本次 NCU standalone replay 不同，不能用 42/84µs × 70 推模型耗时。

### wgrad：最大的收益来源，且存在明显的通用卷积指令开销

NCU wgrad 使用真实模型的一个主流 geometry：NCHW `[4,64,96,96]`，weight `[128,64,1,1]`，
upstream `[4,128,96,96]`，deterministic cuDNN，随机 seed 固定。
这不是捕获的真实 activation/gradient fixture，不能替代未来 wgrad 算术改动的真实数据数值 gate。
对应 kernel 与模型的主要热点名称一致。

NCU 的 `High Compute Throughput` 与 `High Utilization` 指向 ADU **77.72%** 和 Shared FMA Heavy **71.70%**。
SASS 热点包含大量 `IMAD/IMAD.HI/SHF/SEL` 等索引/地址指令；Tensor Core activity 为 **0**。
该 geometry 的逻辑 wgrad FLOPs 为
`2 × (4×96×96) × 64 × 128 = 603,979,776`，以 1.788ms 计仅约 **0.338 TFLOP/s**，
NCU FP32 roofline 约 **2%**。这说明通用 direct convolution 的高 SM utilization
不代表有效 FP32 矩阵乘已经达到算术屋顶。

因此下一条值得验证的路线是明确的 FP32 SIMT 1×1 wgrad/GEMM 或已有 backend 的更合适 layout，
减少通用 indexing/循环和重读；必须使用真实 activation/upstream gradient，计入转置/copy/reductions，
保持 broadcast、共享梯度顺序和高阶 ATen fallback。不得借 TF32、AMP、fast-math 绕过 release 约束。
本次未做任何 wgrad 实现替换，也没有宣称某个 GEMM 形状的既有成功即可迁移。

## 3. 当前剩余加速上限

条件：只加速表中选定工作，其余 GPU 工作固定。`S(r)=1/(1-p+p/r)`，
`S(infinity)=1/(1-p)`。分母为第二次 trace 的 GPU kernel sum，而非 wall time。

| 优化范围 | 当前 p | 自身 2× | 自身 3× | 自身耗时→0 |
|---|---:|---:|---:|---:|
| grouped direct wgrad | 37.08% | **1.228×** | 1.328× | **1.589×** |
| 全部 pointwise convolution | 59.50% | **1.424×** | 1.657× | **2.469×** |
| Flash backward | 9.41% | 1.049× | 1.067× | 1.104× |
| Flash core，forward + backward | 12.95% | **1.069×** | 1.094× | **1.149×** |
| 整个 Flash prior，含 preparation / 参数 VJP | 20.84% | 1.116× | 1.161× | **1.263×** |
| 全部 reverse，含 data term | 24.51% | 1.140× | 1.195× | 1.325× |
| LN 调用整组，含 fallback / autograd | 5.26% | **1.027×** | 1.036× | **1.055×** |
| 仅两个 fused LN kernels | 2.37% | 1.012× | 1.016× | 1.024× |
| Flash core + LN 整组 | 18.21% | 1.100× | 1.138× | 1.223× |

两次 trace 的主要份额很接近：grouped wgrad 36.98–37.08%，Flash core 12.95–13.32%，
LN 整组 4.72–5.26%；因此排序稳定。旧报告 Flash core 的全删上限 **1.204×** 更新为约 **1.149×**。
只继续优化两个 Flash kernels，无法在此 workload 上带来几十个百分点的完整流程收益。

### 无 profiler 的配对模型计时

同一模型、checkpoint、驻留 inputs/target、Adam；每个 timed block 前恢复 checkpoint、
清空 optimizer state，并运行一次相同 warm step。每组 6 轮 ABBA，交替 `ABBA/BAAB`，
每位置计时 2 steps，路径各 24 steps。排除恢复和 warm step，包含完整公共模型步骤。

| 控制与当前路径 | 控制 median wall | 当前 median wall | 实测倍率 |
|---|---:|---:|---:|
| 当前 production prior + 新 LN → 当前 Flash prior + 新 LN | 436.523ms | 314.043ms | **1.390×** |
| 当前 Flash prior + ATen LN → 当前 Flash prior + 新 LN | 358.200ms | 317.834ms | **1.127×** |

两行是不同的配对组，不相乘来宣称“旧模型→全部新优化”的测量结果。
完整 raw rounds、CUDA events、loss 和源码/二进制指纹位于 `analysis/paired_model.json`。
这是本机短测量窗口的吞吐，不能证明完整 Flash 模型数值等价或训练收敛。
另测 B4 C128 HR96 pad2 的完整单算子 forward+VJP（含逐调用 preparation）：
production **4271.94µs**、Flash **1436.20µs**，约 **2.974×**；这是局部倍率。

## 4. 下一步优先级与大谱面范围

1. **优先 grouped wgrad。** 它占 37%，FP32 有效算术利用率低且通用 indexing 成本高。
   用真实训练 fixture 做完整 VJP 配对测量和 frozen-FP32 / FP64 gate，再选择 layout 或 SIMT 路线。
2. **其次 prior 周边及 Flash backward。** preparation / 参数 VJP / reductions 尚占 7.9%，
   Flash backward 占 9.4%。逐调用可微 FFT不能取消；可研究 materialization、固定序 reductions、
   shared store layout 和 inverse-load 融合，在当前数值预算内逐项验证。
3. **LayerNorm 留意 5 次 strided fallback及周边融合。** fused 本体已接近带宽屋顶，
   继续只微调其 block 配置的整步回报低。GELU 另占 6.3%，可另行评价有精度门控的周边融合。

s2/s3 cluster 不在本次 USRNet prior 的 NCU/替换范围内；data term 仍为生产路径。
当前已保留的 v4b 证据（`profile/flash-cluster-v4-20261012`）显示 s2 64→128 cluster
的主 stall 已是 mio-throttle **6.00/2.99**、barrier **2.01/1.13**，long-scoreboard **3.01/1.48**；
同样应从片上 FFT/DSM 流程继续诊断，而非沿用 v4 的 global latency 排序。
该历史报告的算子倍率 s2 **2.33×**、s3 **1.07×**没有在本次重新测量，未混入新模型收益。
SM120 99KiB/block 的容量限制仍存在，训练继续保持 full spectrum；cluster 是容量/通信的取舍。

## 验证、失败与证据保存

- Checked build正常执行，完整 source/toolchain/binary manifest通过；采集前后 production manifest一致。
- 当前 profiled Flash shape：8 seeds × output/dx/dweight/dbias，candidate 和 production 均针对
  frozen Python FP32 baseline / independent Python FP64 reference 通过全部 **64** comparison verdict。
  candidate 最大 rel-L2 ratio **1.00383**、max-abs ratio **1.02515**，使用原数值预算及 floors。
- 完整未修改 release suite **225/225 通过**，包括 LayerNorm 几何/masks/seeded 参数梯度、
  dtype/dispatch、broadcast/shared-input、高阶fallback及 pretrained regression。
  第一轮选定72项为71通过、1项临时目录ACL错误；错误来自 numerical-policy报告测试，
  并非数值误差。随后以相同工作区TEMP运行完整套件正常通过，原失败日志保留。
- NCU full首次无法启动目标，第二次 harness 使用了不支持的 LayerNorm constructor argument；
  修正 harness 后第三次 full 与第一次 source 均正常退出，并分别收集5个kernel。
  所有旧失败日志保留，未改生产代码或改写历史成功/失败。
- Flash 本次 gate是局部算子gate；production release pretrained测试不是完整研究Flash模型的
  FP64/收敛验证。本次模型接入继续仅用于性能研究。

主要证据：

- `reports/full_03.ncu-repz`、`reports/source_01.ncu-repz`。
- `analysis/ncu_summary.json`、`metrics_full_*.json`、`metrics_source_*.json`、`sass_*.txt`。
- `reports/model_flash_02/`、`reports/model_flash_03/`、`reports/model_production_01/`：trace及module attribution。
- `analysis/amdahl.json`、`analysis/paired_model.json`、`analysis/operator_gate_timing.json`。
- `reports/checked_build.log`、`reports/release_tests.log`、`reports/release_tests_02_all.log`。
- `harness/fused_snapshot.cu`、`harness/bind_snapshot.cpp`、`analysis/production_manifest.json`及本次harness。

复算：

```powershell
& H:/Python/ConverseNet/.venv/Scripts/python.exe profile/sm120-bound-refresh-20261012/analysis/analyze_ncu.py
& H:/Python/ConverseNet/.venv/Scripts/python.exe profile/sm120-bound-refresh-20261012/analysis/summarize.py
```

重新采集请创建新的run目录，保留本次原始报告。
