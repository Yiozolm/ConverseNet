# LeetCUDA 对当前 Converse2D 全谱训练优化的启发

2026-09-22。结论：**优先删除不被消费的梯度缓冲、重复布局拷贝，再研究不跨归约边界的融合；向量化和线程配置随后评估。TMA、warp specialization 和 Tensor Core 不宜作为当前第一步。**

本次使用 [leetcuda-cpp-kernel skill](C:/Users/Boyce/.codex/skills/leetcuda-cpp-kernel/SKILL.md)，按章节目录读取相关段落，并交叉检查当前生产源码和已有 Nsight 记录。没有修改算子、编译扩展或启动新的 GPU 实验。以下“候选”均为待验证假设，不是新增性能结果。

## 依据与当前状态

LeetCUDA 本地未找到，已浅克隆并稀疏检出到 `artifacts/leetcuda_reference_20260922`，固定提交 `39c5ca21914f3359df6dd0f924c5f60577ed37ce`。主要读取 ch00/01 的 profiling、Roofline，ch02 的归约，ch03 的向量化，ch07 的布局，ch14 的 SM120 流水线与编译取证。书的源码和本次身份核验见 [调研记录](../artifacts/leetcuda_research_20260922/audit.json)。

当前 CUDA FP32 v7 训练已经使用 complex64 全谱，并已接入 s1 专用逐点融合。核准备/FFT/IFFT 为逐调用 FP32 autograd；推理仍为半谱。同 forward 核 FFT 复用关闭。调研期间工作区又接入了 `scale2.cuh` 的受限 s2 分派（W>1且HR复数张量字节数不超过INT32_MAX），本次不审定这项并行变更的验收状态。**旧半谱的面积门槛、FP64 核准备和9月21日半谱 pad/crop 加速数字均不用于判断当前性能。**

调研开始时，已有 [生产完整训练步 Nsight 汇总](../../HPC/artifacts/full_spectrum_s1_20260922/nsys_production_model/summary.json) 的元数据与36份生产构建依赖逐份 SHA256 相符。收尾时检测到 `full_fusion.cpp/.cu` 因接入 s2 而变化，新增 `scale2.cuh`；`scale1.cuh` 保持相同。**下表绑定原 s1 版本，不冒充包含新 s2 代码的二进制性能。** 在这一份捕获中：

| 当前自定义核 | 次数 | GPU 累计时间 |
|---|---:|---:|
| `scale1_adjoint` | 39 | 41.638 ms |
| `scale1_forward` | 39 | 17.221 ms |
| `adj_kernel` | 40 | 3.906 ms |
| 全部 Converse 全谱自定义核 | — | 63.656 ms |

上述自定义核时间不包含 FFT、ATen 归约、准备和边界操作，也不等于完整 Converse 耗时。全模型 kernel 累计为488.948 ms，不能把它当完整训练步墙钟时间。`scale1_adjoint` 是当前已记录的 Converse 自定义核首要热点；此前旧 `adj_prediction` 的88.95% DRAM吞吐不能直接套给这个新核。后续应对当前 `scale1_adjoint` 单独采集 NCU。

## 最直接的三个源码机会

### 1. 共享输入时，不再写出无用的 gy

[scale1.cuh](../Converse2D/torch_converse2d/training/full_spectrum/scale1.cuh:24) 的反向无条件创建并写出 `gy/gp/direct/prediction/gd` 五张复数张量。可是 [FullSolve::backward](../Converse2D/torch_converse2d/training/full_spectrum/full_fusion.cpp) 在 `shared` 时最终返回合并后的 `gp`，另一输入槽为空，单独的 `gy` 缓冲没有消费者。

这对应 ch00/01 的“先识别实际读写，再减少中间量”方法。建议仅消除 `gy` 的分配和全局写回，仍在寄存器计算 `gyi`，保持共享梯度的 `(G+gy)+gprediction` 顺序。各个 ATen 归约、复数除法、FMA 和高阶回退不变。

以 B4/C128/100×100 为例，一张 complex64 `gy` 为：

`4 × 128 × 100 × 100 × 8 = 40,960,000 B = 39.0625 MiB`

35次 prior 调用对应约 **1.4336 GB 的逻辑写入量/训练步**。这不是实测 DRAM 流量，更不是常驻显存或峰值显存可以减少1.43 GB；临时量不会全部同时存活，cache与写回事务也会影响实际流量。需要用 NCU 和完整前后向计时确认收益。

同一类改动还包括把 `needs_input_grad` 下沉到分配/计算阶段。当前一阶反向主要在结果返回时筛选梯度，此前已经算完核梯度与 λ 梯度。冻结 kernel 或 λ 时，可避免对应分支；不能删除仍被其他梯度依赖的中间值。完整模型通常需要参数梯度，梯度子集收益必须另列，不能拿冻结参数测量冒充常规训练。

### 2. 共享谱只整理一次布局，布局转换尽量由生产者完成

[scale1_forward 的 host 包装](../Converse2D/torch_converse2d/training/full_spectrum/scale1.cuh:44) 分别执行 `plain(y0)` 和 `plain(p0)`，而 `plain` 包含 `resolve_conj/resolve_neg/contiguous`。在 `y0.is_same(p0)` 且确实需要整理布局时，可以考虑让两个只读入口使用同一个整理结果，避免重复拷贝。这是本次调用的临时输入共享，不是核 FFT 图复用，更不是跨 optimizer 步缓存。

更进一步可研究让谱输出直接写成 IFFT 所需布局，消除 [production.cpp](../Converse2D/torch_converse2d/training/full_spectrum/production.cpp:23) 的 `empty_like + copy_`。ch07 关于转置与缓存层级的分析适用于这里，但必须同时满足：

- 保留传给 cuFFT 的 shape/strides，不能直接删掉布局恢复。既有 [layout_probe](../../HPC/artifacts/full_spectrum_default_20260922/layout_probe.json) 已记录布局变化带来的数值非劣失败。
- 分别考虑输出 `out` 与保存用于反向的 `q/d`，不能为了输出写入方便而改变后续归约输入布局。
- 验证是否减少总访存，避免省去一次 copy 却使核心读写变为严重跨步访问。

外层 circular pad/crop 也仍有重新评估的价值，当前 [模型入口](../models/util_converse.py:185) 仍使用原 pad/双切片。可以重放已有 cat/view 思路，但必须以当前全谱默认重新验精度和计时，旧半谱结果不能继承。

### 3. 无广播的 s1，直接完成核梯度合成

当 `KB==B && KC==C` 时，direct 与 prediction 的核形状归约不需要跨元素累加。可以研究把 [adj_kernel](../Converse2D/torch_converse2d/training/full_spectrum/full_fusion.cu:57) 的最终逐点合成并入 `scale1_adjoint`，减少 direct/prediction 的写回、再读取和一次 launch。

适用对象包含 batch1 且通道不广播的 prior，以及 batch4 的逐样本逐通道动态核。batch4 的共享 prior 核不满足这个条件，应保留原路径。

融合时仍按现有 nested-add/FMA 顺序计算，λ 的归约仍保留；`gd` 的 complex 存储及其实部 stride2 暂不改变。没有跨元素归约不代表可以任意重排浮点表达式。该候选的寄存器需求可能增加，须检查 spill，而非只看少了几个 kernel。

## 书中方法如何分层应用

| 方法/章节 | 对 Converse 的具体用途 | 顺序与关键约束 |
|---|---|---|
| ch00/01：profile、Roofline、寄存器 | 对当前 adjoint 测 DRAM/L2 流量、store 指令、stall、寄存器与 spill，判断删除 scratch 的效果 | 最先做；不把旧核的 NCU 结论当新核事实 |
| ch03：向量化与合并访问 | 比较每线程1/2个 complex64、block128/256/512，检查编译后是否产生预期宽访存 | 删除无用工作后再测；两复数打包需要16B对齐、完整尾部和广播边界守卫 |
| ch00 + 当前源码：指令开销 | 当前使用 int64 索引与除余运算；可在所有尺寸、乘积和最大偏移可证明安全时研究 int32 版本 | 不改变浮点计算；超界回退；收益需要 SASS/NCU 验证 |
| ch02：warp/block 归约 | 用于分析 B/C 广播梯度和 s2/s3 alias 的组织方式 | 优先保留 ATen 归约边界；确定顺序也不保证与 Python 同精度 |
| ch07：布局与转置 | 同输入谱的重复 contiguous、IFFT 前布局恢复、pad/crop 的总读写 | 保留 FFT 输入布局与伴随；区分 cache 内工作集与大 batch |
| ch14：TMA/warp specialization、编译取证 | 未来有可重复使用 tile、足够计算/搬运重叠时，再研究流水线；用 SASS 核实实际生成的指令 | 当前逐点 s1 先不引入；不能直接复制书的 SM120a 构建/寄存器参数 |

另一个可测试方向是每线程处理少量 batch 上相同频点，在寄存器复用共享 K/D，同时保持每个元素原计算顺序。收益取决于已有 L2 命中和并行度：如果 K/D 已被缓存命中，新增循环和寄存器可能抵消节省。没有证据前，不增加 shared memory、barrier 或 persistent CTA。

## 哪些内容不能直接照搬

**向量化不等于减少四分之三的数据量。** ch03 明确区分访存指令数和 sector 流量。我们的目标应是改善指令开销或访问效率，不能先承诺 float4 带来4×加速。书也指出 relu/dot 教学示例有尾部守卫不足的问题；可参考 `base.cuh` 的 `elementwise_add_vec4` 完整尾部范式，不能直接搬教学代码。

**warp reduce 不自动满足精度要求。** 不能把 `sum(direct)+sum(prediction)` 合成 `sum(direct+prediction)`，不能把 `real(gd)` 的 stride2 静默改成连续 FP32，再假设 ATen 会选择同样的归约顺序。跨 block 浮点 atomic 累加还可能引入运行间差异。NVIDIA 的[浮点说明](https://docs.nvidia.com/cuda/floating-point/index.html#operations-and-accuracy)也明确指出，运算顺序和 FMA 边界会改变结果。

**Tensor Core/TF32/FP16/FP8 不属于本轮路线。** 书的主要高峰值结果来自 GEMM/attention，与当前复数逐点求解、广播归约、cuFFT 的结构不同。不能迁移其 TFLOPS 或对 cuDNN 的比率，不能用降低精度满足速度目标。LayerNorm和普通卷积不在用户当前优化范围内。

**TMA/双缓冲需先证明复用和重叠机会。** 逐点核引入额外 gmem→smem→register、barrier、producer warp 未必合算。只有分析显示访存延迟或共享数据复用是瓶颈，才值得独立试验。架构支持、工具链和 SASS 必须在本机核对，书中其他卡的寄存器预算、L2容量和成绩不能当5060 Ti事实。

**书中 min-of-N 不是我们的验收协议。** 保留事前固定轮数、交替 A/B、预热与冷启动分开、profiler与正式计时分开的口径。occupancy 只作解释指标；[NVIDIA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)也不把更高 occupancy 等同于更快。

## 建议下一轮的最小实验

1. 冻结当前全谱默认源码/二进制，在 B1/B4/B8、C128/100² 与 C64/96² 上采集当前 s1 前后向，重点针对 `scale1_adjoint` 做 NCU。
2. **只删除 shared 情况的 gy 分配/写回**，保留寄存器计算、归约及所有舍入顺序。先单独验这一项，不混入 vectorization、int32、pad/crop。
3. 通过原 Python FP32 相对同一 FP64 的 max_abs/relative L2 非劣检查后，再测算子完整 forward+VJP和包含 optimizer 的整步；补共享/独立输入、KB/KC四组合、梯度子集、布局、弱正则、高阶与stream。
4. 再独立比较共享输入布局去重、无广播末端融合。每次记录实际流量、分配量、kernel次数和墙钟时间；只有数值和性能证据同时成立才考虑默认接入。

这轮最有把握的启发是寻找“算完却不用、拷贝后重复读取”的工作。它能给出明确、较易审查的代码变更，并且有机会延续当前全谱路径的精度保证；具体可节省多少时间，仍需下一轮实测。

## 定位到书的章节

- [ch00 profiling](https://github.com/xlite-dev/LeetCUDA/blob/39c5ca21914f3359df6dd0f924c5f60577ed37ce/kernels/interview/book/chapters/ch00-profiling.tex#L52)：工具分工、L1/L2/DRAM、stall与spill。
- [ch01 Roofline](https://github.com/xlite-dev/LeetCUDA/blob/39c5ca21914f3359df6dd0f924c5f60577ed37ce/kernels/interview/book/chapters/ch01-arch-roofline.tex#L194)：从计算量与数据移动判断策略。
- [ch02 reduction](https://github.com/xlite-dev/LeetCUDA/blob/39c5ca21914f3359df6dd0f924c5f60577ed37ce/kernels/interview/book/chapters/ch02-reduce-dot.tex#L247)：两级归约与浮点累加次序。
- [ch03 vectorization](https://github.com/xlite-dev/LeetCUDA/blob/39c5ca21914f3359df6dd0f924c5f60577ed37ce/kernels/interview/book/chapters/ch03-vectorize-atomic.tex#L147)：宽访存、对齐、尾部以及“指令数≠事务数”。
- [ch07 layout](https://github.com/xlite-dev/LeetCUDA/blob/39c5ca21914f3359df6dd0f924c5f60577ed37ce/kernels/interview/book/chapters/ch07-rope-transpose.tex#L199)：布局搬运与不同工作集的收益差别。
- [ch14 SM120](https://github.com/xlite-dev/LeetCUDA/blob/39c5ca21914f3359df6dd0f924c5f60577ed37ce/kernels/interview/book/chapters/ch14-sm120-tma-ws.tex#L35)：流水线适用条件与编译后取证。


## 实施跟进（2026-09-22）

原调研与其历史源码/性能依据保留。后续按上述顺序分别制作了共享输入 `gy` 删除、共享谱布局去重、无广播 s1 核梯度末端融合三个递进候选；实现差异、逐项验证与是否接入默认路径以 [HPC 实施报告](../../HPC/docs/conversenet_full_spectrum_leet.md) 及其原始记录为准。该链接不把上文研究假设或旧版本性能数字改写为新版本通过结论。
