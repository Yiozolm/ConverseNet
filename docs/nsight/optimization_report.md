# Nsight 优化分析

后续更新：[CUDA Graph 实现与验证](cuda_graph_implementation.md) ·
[计数器开放后的 Nsight Compute 分析](compute_analysis.md)。下文保留原始采样时的状态与结论。

实现进展：[v7 频谱读写优化与精度验证](spectral_io_optimization.md)。

归档日期：2026-09-11。采样代码为 main 提交 `5be56ef7d9dd526b7d1b5be905e835636ab0e5f9`；本报告保存于 dev 分支，数据不代表 dev 当前代码的性能。源码链接固定到采样提交，文件指纹见 [manifest.json](manifest.json)。

当前最值得做的是：固定形状推理的 CUDA Graph、恢复块里的 LayerNorm/逐元素融合，然后再优化动态频谱准备和 FFT 搬运。只继续打磨 DataNet 内核，对整网收益会受到恢复块占比的限制。

## 采样与限制

RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，Nsight Systems 2026.1.2。采集了 12 份有效 Systems trace，覆盖静态核、动态核、非规则 FFT 尺寸、恢复块、USRNet 推理和反传。

Nsight Compute 已连接目标进程，但驱动返回 `ERR_NVGPUCTRPERM`。因此没有 SM occupancy、DRAM 利用率、缓存命中率或 stall counter 结论；没有修改系统计数器权限。

统计来自 SQLite 中 `capture` NVTX 区间，排除了 cudaProfilerStart 的初始化耗时。GPU 活动占比是 kernel/copy/memset 时间区间的并集覆盖率，**不是 SM 占用率**。Profiler 有额外开销，带详细 ATen NVTX 的采样只用于定位操作；实际速度收益用无 profiler 的 A/B 实验验证。

本次分析未改生产算法；采样工作区只修复了 profiler 工具在 Windows 把 `.BAT` 误当 `.exe` 的问题。本次归档包含报告与四份 JSON 数据，不包含该工具修复或生产代码变更。

## 时间线概览

| 场景 | 每次 capture 耗时 ms | GPU kernel 累计 ms | kernel 数/次 | GPU 活动覆盖率 |
|---|---:|---:|---:|---:|
| 静态 C64 256×256 s1 | 0.435 | 0.335 | 9 | 90.6% |
| 静态 C64 128×128 s2 | 0.661 | 0.436 | 12 | 77.2% |
| 静态 C32 128×128 s3 | 0.744 | 0.529 | 12 | 84.4% |
| 静态 C32 127×129 s3 | 1.663 | 1.391 | 23 | 88.9% |
| 动态核 DataNet C64 128×128 s2 | 1.341 | 0.953 | 27 | 74.4% |
| DataNet 前向+反传 64×80 s2 | 3.231 | 1.035 | 129 | 32.4% |
| 单个恢复块 64×80 | 1.462 | 0.228 | 51 | 15.8% |
| USRNet 32×40 s2 推理 | 42.530 | 8.542 | 1908 | 20.4% |
| USRNet 64×80 s2 推理 | 47.026 | 31.552 | 2048 | 68.9% |
| USRNet 32×40 s2 前向+反传 | 210.973 | 44.320 | 8601 | 21.6% |

详细 NVTX 会明显放大主机开销：例如同一 DataNet 训练样例，light trace 为约 3.23 ms，detail trace 为约 6.89 ms，因此没有拿后者当正常耗时。

## P0：固定形状 CUDA Graph，已做概念验证

小尺寸 USRNet 每次前向约 1908 个 kernel，GPU 时间线上存在大量空隙。其主要开销包含 Python/ATen/驱动提交。CUDA Graph 可以把重复提交改为图重放；固定地址与图内存生命周期要求见 [PyTorch 2.11 CUDA Graph 文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。

使用同一原始 checkpoint、FP32、关闭 TF32。下面是无 Nsight 的 5 轮×20 次实测，图模式包含 GPU 输入复制和输出 clone：

| 输入 / scale | Eager ms | Graph+GPU I/O ms | 加速比 | 改变输入/核后的最大差 |
|---|---:|---:|---:|---:|
| 1×3×32×40 / 2 | 30.154 | 10.801 | 2.79× | 0.0e+00 |
| 1×3×64×80 / 2 | 44.591 | 39.936 | 1.12× | 0.0e+00 |

结论：小尺寸推理的 Graph 收益已验证；较大尺寸更多时间在 GPU 运算上，收益明显较小。本次没有验证训练 Graph，不能把推理倍数套用到训练。

落地时需要按 shape/batch/dtype/device/scale 管理图，保持输入/输出缓冲和模型参数存储有效，并在权重/配置变化时失效重建。当前全局频谱 LRU 不能成为图中悬空指针的来源。

原型采用冷频谱缓存捕获，使频谱计算与分配进入图私有池；捕获后清空外部缓存，再改变输入及核验证两次，结果一致。这会把部分常量频谱重算也放入每次 replay，后续可以研究图拥有的固定频谱缓冲。

图会使用私有内存池。JSON 中 reserved_growth_mib 仅是捕获前后的净快照差，不代表图的全部显存开销；生产环境需要按图数量和尺寸另外预算。

## P1：恢复块的 LayerNorm 与逐元素融合

USRNet 的 prior_stack 占累计 GPU kernel 时间约 96%–97%，DataNet 仅约 2.5%–3.4%。因此整网优先级应放在恢复块。

一个 64×80 的预训练恢复块，共 51 个 kernel；两次 LayerNorm 就占 **20 个启动（39%）**，约 **20% GPU kernel 时间**。当前通道优先 LayerNorm 分开执行 mean、sub、square、mean、sqrt、div、affine。

建议先实现保持 NCHW 布局的融合 LayerNorm，并评估 residual/scale/add 等相邻逐元素融合。避免盲目转 channels_last 后在 FFT 入口再转回。先验证前向，再补齐梯度及二阶梯度；不能靠 detach 或漏算 backward 提速。

代码：[LayerNorm](https://github.com/Yiozolm/ConverseNet/blob/5be56ef7d9dd526b7d1b5be905e835636ab0e5f9/models/util_converse.py#L46)、[恢复块](https://github.com/Yiozolm/ConverseNet/blob/5be56ef7d9dd526b7d1b5be905e835636ab0e5f9/models/util_converse.py#L299)。

## P2：动态核的频谱准备

静态 s2 算子每次约 12 个 kernel，动态核 DataNet 为 27 个。详细 NVTX 定位到：PSF pad/roll、real²+imag²、半谱 flip/roll/cat、mean 和 nearest upsample。

可做两组融合：

- PSF 的补零和中心移位合并为一个 OTF 写入 kernel，减少全尺寸 roll 临时张量。
- 动态推理时，在现有 alias_correction 读取 FB 的同一次循环中累计 |FB|²，直接形成分母，避免先还原完整功率谱再归约。

训练目前走 ATen 路径，每次 DataNet 前向+反传约 129 个 kernel，布局搬运、逐元素操作和归约占明显份额。训练侧融合需要完整 backward/gradgrad 支持，属于下一阶段验证。

代码：[频谱准备与功率归约](https://github.com/Yiozolm/ConverseNet/blob/5be56ef7d9dd526b7d1b5be905e835636ab0e5f9/Converse2D/torch_converse2d/converse2d.cpp#L80)、[训练半谱还原](https://github.com/Yiozolm/ConverseNet/blob/5be56ef7d9dd526b7d1b5be905e835636ab0e5f9/Converse2D/torch_converse2d/converse2d.cpp#L148)。

## P2：FFT 输入复制与归一化

在 C64、HR=256×256 的样例里，每次都有约 **16.12 MiB** 的 D2D 复制。详细范围明确是 `fft_irfft2 → _fft_c2r → clone → copy_`，而不是外部 H2D 输入传输。静态样例中这次复制约占 capture 时间的 11%–14%。

优先可验证把 IFFT 的归一化乘法并入已有 correction kernel，然后选择匹配的 FFT norm，省掉一次完整输出遍历。数值等价但舍入路径不同，需做回归。

取消 C2R 保护复制更复杂：cuFFT 的 out-of-place C2R 会覆盖输入，需要推理专用、可销毁的输入缓冲与可靠的 plan/workspace 生命周期，不能直接删 clone。见 [cuFFT 13.0 数据布局说明](https://docs.nvidia.com/cuda/archive/13.0.0/cufft/index.html#data-layout)。

代码：[逆 FFT 出口](https://github.com/Yiozolm/ConverseNet/blob/5be56ef7d9dd526b7d1b5be905e835636ab0e5f9/Converse2D/torch_converse2d/converse2d.cpp#L160)、[融合频谱更新](https://github.com/Yiozolm/ConverseNet/blob/5be56ef7d9dd526b7d1b5be905e835636ab0e5f9/Converse2D/torch_converse2d/converse2d_kernels.cu)。

## P3：FFT 尺寸与索引微优化

C32、scale=3 下，128×128 与127×129 的 HR 像素数几乎相同，但后者走 prime_fft_factor<127>/<43>，FFT kernel 累计时间约 **0.296 → 1.118 ms**，约3.8倍；kernel 总数12→23。

若业务允许选择训练 patch/request bucket，可优先考虑小质因数尺寸；cuFFT 对2/3/5/7因子的优化见 [官方性能建议](https://docs.nvidia.com/cuda/archive/13.0.0/cufft/index.html#accuracy-and-performance)。改变FFT尺寸会改变这里的周期边界，任何 padding/crop 策略都必须显式验证输出与图像指标。

scale=1 的 correction_scale_one 仍使用64位索引。可以针对可安全索引的尺寸试验32位版本，但在没有 Compute 计数器前，不能断言当前瓶颈是整数除法或寄存器压力；这是低于整网调度/LayerNorm融合的候选。

## 建议实施顺序

1. 固定尺寸推理 Graph + 缓冲/缓存生命周期管理（已有无损概念验证）。
2. 恢复块 LayerNorm/逐元素融合。
3. 动态频谱准备融合与推理 IFFT 归一化融合。
4. 推理专用 FFT 输入/工作区复用，以及业务允许的尺寸分桶。
5. 获得 Compute 权限后，再按寄存器、带宽、stall 数据调整 block size、向量化与索引。

除 Graph A/B 外，其余项目是根据时间线提出的优化候选，尚未实现，因此没有把可触达开销当作承诺加速比。

## 复现与文件

以下命令记录原始采样工作区的运行方式，依赖其中的 `.build/run.ps1` 和 `artifacts/nsight_optimization/` 分析脚本；这些本机脚本未包含在本次报告归档中，不能直接在 dev 分支运行。

```powershell
& ./.build/run.ps1 artifacts/nsight_optimization/run_profiles.py
python artifacts/nsight_optimization/analyze.py
& ./.build/run.ps1 artifacts/nsight_optimization/probe.py --workload usrnet --H 32 --W 40 --scale 2 --graph-experiment --output artifacts/nsight_optimization/graph_small.json
```

[汇总 JSON](summary.json) · [小图 Graph A/B](graph_small.json) · [大图 Graph A/B](graph_large.json) · [源码指纹](manifest.json)

上述四份 JSON 与报告一起纳入版本控制。原始 `.nsys-rep`、SQLite、分析脚本、逐操作摘要与 Compute 错误日志保留在本机 `D:/Python/ConverseNet/artifacts/nsight_optimization/`（Git 忽略目录），未纳入本次提交。
