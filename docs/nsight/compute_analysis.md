# 计数器开放后的 Nsight Compute 分析

2026-09-11 已确认计数器可用，未再出现 `ERR_NVGPUCTRPERM`。环境为 RTX 5060 Ti
（SM 12.0）、驱动 616.92、Nsight Compute 2026.1.0。采样对象是当前 dev 工作区的
v7 FP32 推理内核，包含此前 CUDA Graph 缓存修改；源码 SHA-256 见
[计数器汇总](compute_summary.json)。本次没有修改生产内核的算法或 launch 配置。

## 四组详细采样

使用 `--set full --replay-mode kernel --cache-control all --clock-control none`，
每个 kernel 40 个采集 pass。先预热算子，再在 `cudaProfilerStart/Stop` 区间内
采集一次调用。下表是 NCU 单 kernel 结果，不能代替整网或普通 CUDA event 耗时。

| 场景（B=1） | kernel | 耗时 μs | DRAM 吞吐 % | 实际 occupancy % | 寄存器/线程 | 溢出指令 |
|---|---|---:|---:|---:|---:|---:|
| C64，256×256，s1 | correction_scale_one | 140.800 | 92.53 | 74.94 | 30 | 0 |
| C64，128×128，s2 | alias_correction | 97.728 | 95.75 | 89.78 | 39 | 0 |
| 同上 | apply_correction | 111.648 | 91.67 | 84.74 | 30 | 0 |
| C32，128×128，s3 | alias_correction | 102.016 | 94.83 | 89.77 | 40 | 0 |
| 同上 | apply_correction | 129.920 | 90.22 | 80.07 | 30 | 0 |
| C32，127×129，s3 | alias_correction | 105.248 | 94.64 | 91.63 | 40 | 0 |
| 同上 | apply_correction | 123.072 | 91.90 | 78.53 | 30 | 0 |

所有样例的理论 occupancy 均为 100%，local load/store sector 与寄存器溢出
指令计数均为 0。由这些数据判断，当前大尺寸 correction kernel 更受内存访问
限制，优先降低寄存器数量或调整 block size 的依据不足。

Warp State Statistics 中，long scoreboard 占相邻指令发射间隔的约
68%–96%；s2/s3 的 alias_correction 分别约 90%/96%。这是等待 L1TEX 数据依赖
的 warp 停顿指标，不是“可直接消除的总耗时百分比”。

s2/s3 的 alias_correction 分别有约 8.90%/7.07% 的额外 global sectors；奇数
尺寸为 10.78%。apply_correction 约 1.75%–3.14%。访存合并值得改善，但没有
证据支持大规模转置或增加完整频谱副本。奇数尺寸的两个 correction kernel
合计耗时与偶数尺寸接近，也与历史 Systems 报告中“奇数尺寸主要慢在 FFT”
的发现一致。

## 缓存状态与数据质量

NCU 默认每次 replay 前清缓存，这会提高 DRAM 访问量。本次另做一组 s2 的
`--set basic --replay-mode application --cache-control none` 对照，保留前序算子
形成的缓存状态；两个采样条件没有混用来计算优化加速比。

保留缓存对照共 9 次 application replay：

| s2 kernel | 耗时 μs | DRAM 吞吐 % | SM 吞吐 % | 实际 occupancy % |
|---|---:|---:|---:|---:|
| alias_correction | 104.064 | 94.39 | 16.30 | 90.46 |
| apply_correction | 114.304 | 91.79 | 33.29 | 86.41 |

这组数据仍显示 DRAM 接近饱和，支持该大尺寸样例的访存限制判断。两组使用不同
replay/section 设置且未锁频，耗时差不能解读成“缓存开关导致的性能变化”。
本结论限定于所测静态核和尺寸，动态核、小尺寸或整网图推理仍需对应采样。

详细采样中，s2 apply_correction 的 L2 命中率出现 192.21%，属于无效读数。
汇总将该值标为 null，同时保留原始异常值，未用于分析。NCU 自动建议中的
“Est. Speedup”也没有作为实际收益引用。多 pass 的比率可能受不同 pass 的
执行差异影响；缓存控制、时钟控制、采样与正常执行的耗时差异见
[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility)。

## 下一步优化顺序

1. **整网：恢复块 LayerNorm/相邻逐元素融合。** 依据此前 Systems 对整网的
   时间归因，恢复块仍是主要 GPU 开销；本次 Compute 只测 correction kernel，
   不把其占用率或带宽结果外推到 LayerNorm。
2. **算子：减少频谱及 FFT 的读写。** 验证将 IFFT 归一化并入已有频谱更新，
   以及动态核在读取 FB 时直接累计功率分母，减少中间张量。
3. **进一步改善 alias_correction 的数据复用和访存合并。** 先用分块复用、
   减少 q 的中间读写等原型验证收益；避免增加大范围拷贝抵消优化。
4. **保留 32 位索引和 block size 微调为较低优先级。** s1 虽仍使用 64 位索引，
   但当前计数器未显示寄存器溢出或算力饱和，不能据此承诺明显收益。

这些是优化候选，尚未实施。每次实现仍需独立空间参考、奇偶尺寸、批量核、
前向/梯度回归，并用无 profiler 的算子与整网 A/B 证明收益。

## 复现

在配置好 CUDA 编译环境的仓库根目录运行：

```sh
python test/profile_nsight.py --kind compute --set full --scale 1 --C 64 --H 256 --W 256 --iters 1 --output artifacts/compute_analysis/s1
python test/profile_nsight.py --kind compute --set full --scale 2 --C 64 --H 128 --W 128 --iters 1 --output artifacts/compute_analysis/s2
python test/profile_nsight.py --kind compute --set full --scale 3 --C 32 --H 128 --W 128 --iters 1 --output artifacts/compute_analysis/s3
python test/profile_nsight.py --kind compute --set full --scale 3 --C 32 --H 127 --W 129 --iters 1 --output artifacts/compute_analysis/s3_odd
python test/profile_nsight.py --kind compute --set basic --scale 2 --C 64 --H 128 --W 128 --iters 1 --cache-control none --replay-mode application --output artifacts/compute_analysis/s2_warm
```

本机可用 `& ./.build/run.ps1` 代替上述 `python`，以加载 Visual Studio/CUDA 环境。
原始 `.ncu-rep`、完整 CSV 和工具文本在本机 `artifacts/compute_analysis/`，
继续由 Git 忽略；精简计数器汇总与本报告一同保存到 `docs/nsight/`。
