# FP32 全清单优化活动

活动进行中。开始 `2026-09-25 04:02:31 UTC`，八小时上限 `12:02:31 UTC`；基线 `a33e3af`。发布分支为 `codex/fp32-p0-optimization`：首批实现 `ca03a22`、负视图元数据修复 `06a9673`、完整并行 LayerNorm 推理融合 `ff7f8ba`。生产源码已冻结于 `ff7f8ba`，长期质量验证正在执行。隔离研究分支 `codex/fp32-roadmap-research` 已保存到 `cc244e3`，结果和原始证据分别见该分支的 `research/RESULTS.md`、`research/evidence/index.json`。

发布契约保持 FP32/complex64、可微输入全谱训练、逐调用可微核 FFT、原广播归约与共享输入梯度顺序、高阶 ATen 回退。没有启用 AMP、TF32、fast-math 或训练谱缓存。用户批准的半谱训练、谱复用和无 FFT 训练研究仅存在于隔离分支。

## 全清单状态

| 方向 | 当前结果与边界 |
|---|---|
| 冻结输入分派、同频谱物化、缓存布局 | `d37e963` 已有实现，整合回归通过。同地址/版本/布局的负视图标志变化导致缓存误命中的旧版 CPU/CUDA 反例已保存；`06a9673` 将 negative/conjugate 标志纳入缓存身份，最终 checked 构建及回归通过。 |
| 梯度需求 mask | `d37e963` 已有实现，保留高阶与 Graph 回退，本轮整合回归通过。 |
| 可微 PSF pad/roll | `a33e3af` 已有实现；训练仍逐次生成可微核 FFT。`06a9673` 在推理 PSF 读取指针前显式解析 lazy negative/conjugate view，CPU/CUDA 回归通过。 |
| s2 无广播核梯度 | `a33e3af` 已有实现，整合回归通过。 |
| 当前全谱 s3 融合 | `ca03a22` 保留九 alias 的四累加器树、两个不同 mean 因子、complex gd 的 stride-2 实部和广播归约。W=1/大索引回退；逐字节及逐张量 FP64 门槛通过。 |
| s1 共享核批量归约 | `ca03a22` 仅融合 B2/B4、KB=1、KC=C、W>1、32 位安全索引且同时需要核和输入 VJP 的情形。保留原归约树，复用所需梯度缓冲；其他几何、核单梯度及高阶回原路径。 |
| alpha 残差、LayerNorm 外围 | `ca03a22` 的残差/独立 affine 自动融合仍限制在无 GradMode、至少 `2**21` 元素的推理张量。`ff7f8ba` 另将完整并行 LayerNorm 融合用于 no_grad/inference_mode、NCHW 连续 FP32、C64/128、HW>1、32 位安全索引且无 lazy 标志的输入；六种 mean/variance 归约组合、offset1/2、尾部、FP64、流/Graph 门槛通过。训练、负视图、CPU、指定 pytorch 后端及不支持输入保留原 ATen 表达式。 |
| Graph runner | `ca03a22` 合并模块遍历，并补全共享注册绑定、全局 forward hook 检查。`06a9673` 补全 negative/conjugate 签名，CPU/CUDA 生命周期回归通过。runner 自身的同模型消融仍约 1.00×；LayerNorm 组件的 Graph 收益另行记录。 |
| k2/s2 nearest 无 FFT | 显式 nearest 的 CUDA 残差式候选在 180 案例中失败 17 个；另一系数式失败 14 个。未替换模型，也没有质量或速度通过结论。 |
| s1 传递函数缓存 | uncached 共享传递式失败 23/32 个张量；缓存半谱传递式失败 209/240 个案例，并且未通过 negative-view 缓存身份契约。原候选和反例均保留，未替换模型或进入生产缓存路径。 |
| nearest 去 HR prior FFT | 当前频谱构造候选失败 45/96 个输出/VJP 张量，未合入。 |
| 准备量复用、重构归约 | 无梯度缓存与精确 B2/B4 归约已有实现；可微谱缓存未进发布。同一 forward 内谱复用研究失败 33/264 个张量，全部为核梯度。 |
| 半谱训练 | 隔离候选失败 114/216 个张量，未合入。 |
| 混合空间求解 | 隔离候选失败 119/216 个张量，未合入。 |
| 直接小核 DFT | 纯 FP32 候选失败 140/216 个张量，未作为生产后端。 |
| cuFFT LTO callbacks | CUDA 13 primitive 探针可运行；32×40 相对 L2 门槛失败，256²/260² 的已测 primitive 通过。260² 热调用有局部收益但冷启动增加；没有完整 Converse/Graph/VJP 集成。 |
| cuFFTDx | NVRTC 1D 36/44/64/100 探针通过。两遍 2D+完整 s1 在 32×40、36×44、100² 的 126 案例中分别失败 53、41、67 个，未合入。平台可运行不等于完整算子通过。 |
| 新 profile 发现的 1×1 wgrad | 直接 FP32 GEMM 在 22 案例的 80 个输出/VJP 张量中失败 5 个；独立 split-K256 FP32 bmm+sum 在 25 案例的 92 个张量中失败 6 个，包含非整除尾部和真实模型样本。两种候选均未进入模型替换或性能准入。 |

失败仅对应这些实现和输入矩阵，不代表该方向不可能优化。案例与张量是不同计数单位，不能混用百分比或用多数通过替代每张量门槛。

## 数值与性能验收

路线 A 优先逐字节比较，并保留非连续/共轭频谱、共享祖先、梯度子集、弱正则、大 mean 计数、流与 Graph 生命周期。路线 B 使用同一量化输入的独立 Python FP64 参考，分别计算 Python FP32 和候选误差；每个输出/VJP 的 max-absolute 与 relative-L2 都不得劣化，没有额外裕量。

`ca03a22` 的 93 项 CUDA 测试全部通过；当时 CPU-only 独立构建为 12 项通过、81 项按策略跳过，随后新增的 2 项 CPU 自动分派边界测试通过。最终 `ff7f8ba` 的 checked CUDA 构建为 **107 项通过**；独立 CPU-only 构建为 **18 项通过、89 项 CUDA 测试按策略跳过**。完整 LayerNorm 新增的 8 个测试方法全部通过；训练仍沿用原 ATen 表达式。对应日志为 `full_ln_production_cuda_tests.log` 和 `full_ln_production_cpu_tests.log`。

负视图旧版证据保留两版探针。V1 的 8 个 s3/s4 inference_mode 差异受到缓存资格不同引起的既有舍入路径影响，不能归因于错误符号；V2 在相同 context 内物化参考，复现独立的冷 PSF 与缓存元数据错误，并确认旧源码、manifest 和二进制均未变化。修复后的发布测试使用相同 context 的比较路线；没有将 V1 原始失败改写成修复成功。

四组 B2/B4 比较的张量哈希、准备状态与 FP64 门槛全部一致。加长的两组完整算子（pad/crop、准备、FFT、全部指定 VJP）中，B2 all 约为 1.04–1.07×，B4 all 约为 1.08–1.10×；完整 B4 USRNet Adam 步约为 1.025–1.042×。B4 算子额外 allocated 峰值减少 115,589,120 bytes，模型峰值改善远小于这个局部数值。

未固定 CPU 亲和性的 B1 eager、首次 Graph 和训练步存在明显跨进程波动，全部正反结果保留，不挑最快的一组宣传。独立 same-model Graph runner 对照约为 1.00×，不支持 runner 本身加速。残差/独立 affine 的交错测量仅支持较大推理张量的自动融合；完整 LayerNorm 使用后续独立研究结果和自己的受限分派条件。

后续采用明确的新协议：Windows 实测 P 核逻辑 CPU 为 `[0,1,10,11,12,13,22,23]`，mask 为 `0xC03C03`。`tools/run_affinity.ps1` 只设置本任务进程及其子进程，验证所选 Python 的继承结果；不修改其他进程、优先级、功耗策略或 OMP/MKL。每次有外层 manifest。旧数据仍标记为未固定亲和性，尚不能证实旧波动就是 P/E 核迁移。

在该亲和性协议下，并行完整 LayerNorm 的两轮**研究组件消融**在同进程交错运行同一模型的两个分支：B1 LR32/s3 eager 为 1.281–1.295×、Graph hit 为 1.130–1.136×；B4 eager 为 1.113–1.123×、Graph hit 为 1.081–1.083×。范围包括完整 runner 的输入/输出复制、检查和 clone。它证明该组件在这些模型输入上的收益，不能当作生产整包相对于 `a33e3af` 的同幅收益；原始报告为 `full_ln_parallel_model_a.json`、`full_ln_parallel_model_b.json`。

最终生产整合的固定亲和性**单组跨进程**比较中，所有已记录张量哈希及逐张量 FP64 门槛一致；B1 LR32/s3 eager 约 1.42×、Graph hit 约 1.00×、完整 Adam 训练步约 0.98×（`compare_production_integration.json`）。这些是不同进程、完整改动包的观测值，与上面的同模型组件消融不具备相同解释，不能混用或据此宣布训练加速。生产路径专用的同模型消融留待长跑后执行。

[Nsight 审计](fp32_roadmap_profile.md) 使用当前 FP32 基线重新采集：1×1 权重梯度占真实 GPU kernel 总和约 42.7451%。CPU/NVTX 与 GPU 活动不重复相加，NCU 16-pass replay 时间不作基准。[研究门槛审计](research_validation_audit.md) 分别列出各候选的范围和覆盖缺口。

## 真实数据长期验证

保留既有 900/100 拆分与退化协议、预训练初始化、Adam、学习率 `1e-5`、unclipped RGB MSE、HR96/s3，以及真实 batch=4/microbatch=4。种子为 17/29/43，每条 before/current 最多 4000 更新，每 250 更新完整评估 100 张固定验证图。每进程预算 3000 秒，并受活动总截止时间限制。

稳定条件预先确定：至少 1000 更新，最近五次完整评估的 RGB/Y PSNR 跨度均小于 0.02 dB、SSIM 跨度均小于 0.0005。每个种子分别检验质量门槛（PSNR 不降超过 0.05 dB、SSIM 不降超过 0.001），不平均掩盖失败。达到步数或时间上限但没有满足窗口，仍是未完成稳定性验证；窗口满足也不证明长期收敛或外部数据泛化。

真实 CUDA pause/resume 已通过：连续 0→10 与 0→5→resume→10 的 133 个模型张量、完整 Adam、CPU/CUDA/Python/NumPy RNG、数据哈希、loss、梯度范数及指标轨迹全部一致。worker 保存每步身份、源/二进制身份、完整 checkpoint 和分阶段时间；汇总器拒绝缺步、缺评估、身份不符或无效续跑链。

种子 17 的 before 已完成 4000 更新，进程总耗时 2661.63 秒；固定 100 张验证集的末次 RGB PSNR/SSIM 为 30.4638 dB / 0.820822，Y PSNR/SSIM 为 32.2884 dB / 0.848760。状态为 `max_steps_reached`，停止原因为 `max_steps_without_stability`，五次窗口未稳定。这是一条基线轨迹，不能独自判断候选质量、训练加速或长期收敛。

其余五条由 supervisor 于 `2026-09-25 07:58:40 UTC` 启动串行队列，顺序为 current17、current29、before29、before43、current43；每条仍使用同一稳定性门槛、4000 步上限和 3000 秒预算，worker 截止为 `11:50 UTC`。实时进度写入 `quality_remaining_supervisor.jsonl`。尚未完成三种子的全部配对审计，没有稳定性或达到同等质量总时间的最终结论。

## 复现入口

- `tools/benchmark_fp32_roadmap.py`：完整算子/VJP、B1 USRNet Adam、eager、Graph first/miss/hit。
- `tools/benchmark_batch_training.py`：B2/B4 共享 prior 算子和 B4 完整训练步，每轮恢复共同 Python FP32 预备状态。
- `tools/benchmark_peripheral_inference.py`：交错外围推理与同模型 Graph runner 对照。
- `tools/profile_fp32_roadmap.py`：当前版本 Torch/Nsight 采样。
- `tools/roadmap_quality/`：数据协议、可续跑长期微调及三种子配对审计。
- `tools/roadmap_quality/summarize_long_training.py`：CPU-only 合并续跑父链，核对逐步数据哈希、计划评估、源/配方身份、四项质量门槛及各 session 时间；不把终点张量一致当作品质通过的替代证据。
- `tools/roadmap_quality/plot_long_training.py`：由审计 JSON 生成 RGB/Y PSNR/SSIM 对更新数的 PNG/SVG 及终点总耗时/实测峰值内存图；端点不同不标等工作加速，不虚构逐评估累计 wall time。
- `tools/summarize_fp32_roadmap.py`：CPU-only 证据索引，保留失败、计时范围、计数单位及来源哈希。

原始数据位于 `artifacts/fp32_roadmap/`。历史成功/失败仍保存在 `1b579ea` 与 `codex/training-operator-optimization`，没有改名为本轮成功。

## 下一次更新

本节保留给活动结束前的实际结果：补齐三种子 before/current 的续跑链与共同数据前缀审计、各自四项质量门槛、稳定窗口状态、完整进程时间及实测峰值内存；生成带审计身份的最终图表。另列生产路径的同模型推理消融。若截止时仍有未完成任务，记录已完成步数、预算停止和缺失证据，保持未完成状态，不将局部性能或单条轨迹改写为训练质量/收敛结论。
