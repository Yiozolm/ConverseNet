# 当前 FP32 profile 与全清单审计

本页记录截至 **2026-09-25 06:07 UTC** 已存在的证据，不替代活动结束时的发布验收。八小时活动从 04:02:31 UTC 开始，截止 12:02:31 UTC；此时约剩 6 小时。基线为 `a33e3af`，当前发布候选尚需最终统一比较和提交。生产契约仍为 FP32/complex64、可微输入全谱训练、逐调用可微核 FFT、高阶 ATen 回退；研究 worktree 的例外不能自动成为发布授权。

## 当前版本的热点

完整可审计数据在 [profile_summary.json](../artifacts/fp32_roadmap/profile_summary.json)，解释与互斥分类在 [profile_summary.md](../artifacts/fp32_roadmap/profile_summary.md)。原始 Systems capture、NCU report/raw CSV、SHA、完整构建身份、SQL 和 kernel rowid 均保留。该 profile 测量当前 FP32 **基线**，没有沿用历史 FP64 准备路径的占比。

USRNet，LR `B1×3×32×32`、s3、HR96²、Adam，预热 3 步后捕获 1 个完整训练步。9273 个实际 GPU kernel 的时间和并集均为 **105.279335 ms**；单 stream 首尾跨度 **200.247093 ms**。CPU/NVTX annotation 和 memcpy 没有加入 kernel 分母。94.967758 ms 的间隙含提交、依赖、同步、复制及 profiler 影响，不能全部算成可消除的 CPU 开销。

| 互斥 kernel 类别 | 时间 ms | kernel 时间占比 |
|---|---:|---:|
| cuDNN 1×1 权重梯度 | 45.001739 | 42.7451% |
| Converse 频域求解 | 9.658161 | 9.1738% |
| cuFFT GPU 核 | 8.136249 | 7.7282% |
| cuDNN 前向及索引 | 4.370010 | 4.1509% |
| cuDNN 输入梯度 | 4.167997 | 3.9590% |
| cuDNN scalePacked 辅助 | 1.123567 | 1.0672% |
| PSF pad/roll 前后向 | 0.597691 | 0.5677% |
| 其他 kernel（含小线性核） | 32.223921 | 30.6080% |

147 次权重梯度的 NVTX 形状均直接验证为 1×1；70 次 `128→64` 和 70 次 `64→128` 的 HR96² 卷积合计占 **42.7112%**。因此 1×1 wgrad 是这个训练负载的真实热点。39 次 s1 前向和 39 次 s1 adjoint 共 **9.573652 ms**，唯一 s3 的 6 个通用频域核仅 **0.084509 ms（0.0803%）**。后者不含它的 FFT、准备和外围工作，不能当作整个上采样调用的总成本。

独立重建的 LayerNorm 前向+含 autograd 累加的后向为 **8.857077 ms（8.4129%）**，alpha residual 为 **1.871527 ms（1.7777%）**。这些来自精确 ATen 模式及 forward/backward seq 映射，属于上表“其他”的子集，不能再加到总数上。

NCU 使用相同冻结模型，跳过首个匹配 wgrad 后采样第二个：`input=[1,128,96,96]`，`weight=[64,128,1,1]`。该核每线程 59 寄存器、按 64 分配，限制 4 blocks/SM；理论 occupancy 为 66.6667%，实测 62.9492%。SM composite 的 82.3139% 与 ADU 指标同值，不能称作 FP32 FLOP 利用率。DRAM 仅 4.6842%（20.665866 GB/s），L2 hit 为 99.9725%；此 replay 没有接近 DRAM 带宽上限。平均 warp issue interval 的 not-selected/math-throttle/long-scoreboard/wait 分别为 22.6840%/19.8344%/17.4487%/15.4132%，这些不是可直接消除的 wall-time 损失。local spilling 为 no data。**16-pass replay 的 346.208 µs 仅作诊断，不能作为无 profiler benchmark。**

如果只对当前 kernel 总和做理想估算，wgrad 快 2 倍对应约 1.272×，完全消除对应约 1.747×；这不是完整步加速预测，也不构成降低 FP32 门槛的理由。后续预算优先看 wgrad、39 次 s1 及外围 block，但本 B1 capture 不能证明 B2/B4 归约候选或推理 Graph runner 的收益。

## 对照原清单的进度

“已合入”只用于本轮开始前已有提交；本轮工作树中的候选不能冒充最终发布结果。所有失败均保持原始记录，算法候选失败不等于证明整条方向不可行。

| 原计划方向 | 当前证据与处理 | 尚缺什么 |
|---|---|---|
| 冻结输入缓存、共享频谱物化、布局键；梯度 mask | 已在 `d37e963` 合入，进入整体回归；未重启先前失败的 frozen→fused 求解路由 | 最终统一发布记录 |
| 可微 PSF pad/roll、s2 无广播核梯度 | 已在 `a33e3af` 合入；当前 profile 中 PSF 仅占 0.5677% | 不再用旧占比预估新增收益 |
| 全谱 s3 专用融合 | 本轮发布候选，inference-policy 最终日志 84 项 CUDA 测试通过，保留 mask/广播/高阶边界 | 与最终源码一致的完整模型双向配对 |
| LayerNorm/外围逐点 | 标量和向量训练候选的完整步未稳定受益，默认训练恢复 ATen；向量推理仅在 GradMode 关闭且输入 `numel>=2**21` 时自动启用。高阶逐字节探针失败保留，既有高阶容差检查通过 | 阈值分派的最终模型结果；不能把早期 all-grad 或无阈值计时归到最后版本 |
| Graph runner | 每 call 的结构/config/hooks/version/布局检查保留并补绑定/global-hook 缺口；CPU 签名 336.1945→277.463 µs，保留 metadata 增约 32 KB | 完整 runner 没有可重复加速证据，详见下文 |
| s1 同调用归约/临时空间复用 | B2/B4、KB1/KC=C 候选研究阶段 66 项测试通过；四组算子及完整步哈希相同、逐张量非劣全过，已迁移至发布工作树 | 迁移后 checked 重构建及统一测试进行中，尚未作最终发布结论；B1 profile 不覆盖此分派 |
| k2/s2 nearest 无 FFT | 原 ATen 特例 15/72 张量失败；checked CUDA 扩展 180 case 中 17 case 失败，模型阶段被阻断 | 达到逐张量 FP64 双指标非劣后才有实际模型速度/质量资格 |
| s1 传递函数缓存 | 纯 FP32 transfer 原型 23/32 张量失败 | 算法数值资格未过，尚无可发布的完整缓存生命周期后端 |
| nearest 去 HR prior FFT | 首个频谱原型 45/96 张量失败；新的 coefficient 代数原型 180 case 中 14 case 失败 | 不能将另一公式的数学等价作为 FP32 通过 |
| 训练准备量复用/同调用谱复用 | 同调用复用 33/264 张量失败，均为 dweight；失败记录保留 | 生产仍逐调用可微核 FFT；无梯度几何缓存的单独收益未证明 |
| 半谱训练 | 研究原型 114/216 张量失败，且发布 AGENTS 禁止训练半谱 | 仅隔离研究，不得以速度为由合入 |
| 混合空间求解 | 研究原型 119/216 张量失败 | 数值资格和实际训练质量均未过 |
| 直接小核 DFT | 研究原型 140/216 张量失败 | 动态核/miss/完整算子未获速度资格 |
| cuFFT LTO callbacks | 当前平台 GPU primitive 可执行，不能称平台不支持；某些尺寸有局部速度，但存在逐指标非劣失败 | 未整合 Converse 准备、求解、VJP、Graph 或模型 |
| cuFFTDx | NVRTC 的 1D 64/36/44/100 primitive 已运行；2D 32×40、36×44、100×100 各 126 case 分别失败 53/41/67 | 1D 通过不授予 2D/算子通过，2D 计时与默认集成被阻断 |

研究数字分别来自 `research_numeric.json`、`fftfree_numeric.json`、`nearest_coefficient_numeric.json`、`cufftdx2d_*.json` 和隔离 worktree 的 `research/fft_backends/*_gpu_*.json`。这里明确区分“张量数”和“case 数”，没有用多数张量更准确替代每张量门槛。

## 分层性能结论

完整 runner 的独立配对使用同一个模型对象和 checkpoint，仅替换旧/新 runner，并包含锁、检查、签名、输入复制、replay、输出 clone 和事件。在 `peripheral_inference_a/b.json` 中，旧/新输出相同；no_grad 的逐轮配对 wall 中位比为 **0.99972/0.99792×**，inference_mode 为 **0.99866/1.00059×**。不能将 CPU 签名收益转称完整 Graph 加速。两 runner 的 Graph 输出均不与 eager 逐字节相同，旧/新之间相同；此既有区别原样保留。

B2/B4 候选的四个独立配对 `compare_batch_research_a/b/c/d.json` 均无 hash 差异、无独立 FP64 相对 Python FP32 非劣失败。B4 完整 USRNet+Adam wall 比为 **1.01898/1.08066/1.04203/1.02501×**，跨度明显；不得仅报最优一组。后两组更长的 c/d 计时中，完整算子 all-VJP 的 B2 为 **1.04456/1.07480×**，B4 为 **1.08266/1.10340×**。这为发布迁移提供了依据，仍需迁移后的统一检查。该比较包含完整真实 batch4 训练，不是数据集质量或收敛证明。算子 input-only/kernel-only 分支的上下波动也保留，未被无关分支获得的快慢掩盖。

早期 scalar/all-grad、vector/all-grad 失败或收益不稳定的计时仍保存在 `before_roadmap_a/b.json`、`after_roadmap_a.json`、`after_vector_b.json`。最终 inference-only 策略修改发生在其后，必须以最终源匹配的新比较为准。外围独立计时显示大小形状间收益不一致，不能将 B4 局部最优外推为 B1 完整模型普遍收益。

## 质量与长期验收边界

900/100 数据协议已恢复，CPU 五项检查包含全部 1000 个 RGB 文件 hash。质量 worker 保持默认 250 步短程语义；长期模式显式启用，至少 1000 次更新、每 250 步评估，最近 5 次 RGB/Y PSNR 跨度均小于 0.02 dB 且 SSIM 跨度均小于 0.0005 才可标稳定。到最大步数或预算安全停不等于稳定，更不等于收敛证明。

真实 CUDA 的 5→10 步续跑已与 0→10 步连续控制核对，结果见 [resume_cuda_check.json](../artifacts/fp32_roadmap/resume_cuda_check.json)：133 个模型张量、optimizer state、Torch CPU/CUDA/Python/NumPy 四类 RNG、步骤 6–10 的数据 hash/loss/grad norm、完整 metric_history、源码/build 与计数器逐字节一致。时间字段不参与这个等价门槛。容量/续跑试跑只证明执行与恢复，不能替代 17/29/43 三种子的成对长期轨迹。

正式质量比较仍须按每 seed 的 RGB/Y PSNR 0.05 dB、SSIM 0.001 非劣门槛逐项判断，单列稳定状态、每个 session 的训练/评估/总耗时、续跑间隙与共同步骤数据身份。长跑不稳定时不得称作达到收敛或同质量更快。固定验证拆分之外的独立测试集、全图质量与达到同质量的总时间仍是额外缺口，不能从短轨迹、hash 保持或 profile 推导。

## 06:18 UTC 补充：CPU 亲和性协议

后续跨进程 B1 完整步出现较大 wall 波动，因此新增独立固定亲和性协议，不改旧记录。Windows 实读的 P 核为逻辑 CPU **0、1、10、11、12、13、22、23**，group0 mask **0xC03C03**；不是连续 0–7。`tools/run_affinity.ps1` 显式接收 mask，只改变当前 PowerShell 并由新任务子进程继承，验证可用 mask 和独立 Python probe，不改系统、其他进程、OMP/MKL。真实 CPU 子进程已核对继承值。

同一亲和性下的 before/current 和正式质量应在外层 campaign manifest 标记新协议。上文未固定亲和性的既有计时不能与新协议直接混为一组。当前 3 秒采样中 agent 客户端合计约 0.334% 全机 CPU，没有观察到饱和；这不证明历史波动由 P/E 迁移引起，也不能排除历史 CPU 竞争。完整证据和限定见 [cpu_affinity_notes.md](../artifacts/fp32_roadmap/cpu_affinity_notes.md)、`cpu_topology.json`、`cpu_activity_snapshot.json` 和 `affinity_wrapper_probe.json`。
