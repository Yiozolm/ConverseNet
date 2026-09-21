# Converse2D 优化约定与进度

更新：2026-09-18。以下依据当前源码及保存的实验记录；实验验证不代表已接入主线或完成真实训练收敛验证。

## 目标

实现 torch.nn.ConvTranspose2d 类似的训练速度。不管与Converse kernel之外的如Layernorm等kernel

## 精度与速度 trade-off

- 当前仅关注 FP32 精度的训练与推理优化。BF16/FP16 仅作为实验选项，待 FP32 训练与推理优化完成后，再单独开展低精度优化。
- 用户最新指令（2026-09-18）：**以原 Python 算子的 FP32 精度作为数值基准**，随后明确选择“**精度和训练质量不劣于 Python FP32**”，不要求逐点贴合其舍入结果。使用此前训练对照的 `backend=pytorch` 全谱 FP32/ATen 实现作基线；对相同高精度参考比较两者输出/梯度的绝对误差、相对误差与弱正则表现，另验证真实训练PSNR/SSIM及到同等质量的时间。旧逐点容差比较保留为诊断，不能把“与Python FP32有舍入差异”直接当精度退化；也不能通过增大λ或看过结果后调整误差/质量门槛掩盖退化。独立FP64用于同条件误差度量及解析导数检查，**不再要求候选通过先前对FP64的固定逐点预算**。这条用户指令覆盖下文历史实验中的旧验收约定，旧失败报告不得删除或改成通过。
- 主执行以 FP32 为基线。固定核推理优先评估 FP64 核 FFT 预计算后转 complex64 缓存；训练/动态核需单独计入重复准备成本。AMP、原生低精度 FFT 暂作独立候选。
- 推理检查 PSNR/SSIM、端到端延迟与显存；训练比较完整训练步和达到同等质量的总时间。区分冷/热缓存、算子/整网、eager/Graph，不能相乘不同基线的加速比。

## 训练与推理实现

- 保留统一接口和数学定义，按梯度模式与输入梯度需求选择后端，不能只依据 `model.eval()`。
- 全网使用 autograd；频谱求解核心采用手写解析一阶反向及融合 CUDA，FFT/IFFT、核准备、λ 参数化等保留自动求导。高阶梯度回退可微 ATen 实现，保留参考路径验证。
- 训练可在同一次 forward 内复用可微核频谱；不得复用参数更新前的缓存或用 detach 缓存切断梯度。重点检查复数共轭、半谱边界、广播归约及共享输入的梯度累加。

## 性能分析

- 优先用 Nsight Systems（nsys）分析代表性 FP32 推理和完整训练步，定位 CPU 提交间隙、同步、数据搬运及 FFT/前向/反向的时间占比，再决定优化对象。
- 对已定位的热点 CUDA 核，用 Nsight Compute（ncu）检查内存流量、带宽、寄存器、occupancy 和 stall 原因，以实测支持融合、分块或归约方案，不能仅凭 occupancy 判断性能。
- Profiling 与正式计时分开；收益用关闭 profiler、预热后的同条件 A/B 验证。报告记录源码版本、设备、形状、缓存状态及采集命令，原始报告保存在 `artifacts/`；旧版本的瓶颈结论需在当前实现上复核。

## 已完成

- 当前分支：稳定残差公式、实数 FFT/半谱推理、CUDA 频谱融合及固定核缓存；已接入 CUDA FP32 v7 训练融合及解析一阶反向，高阶仍回退可微 ATen。CPU、FP64、低精度及 v2–v6 训练保留原 ATen 路径。
- 训练核准备：为满足独立 FP64 逐点梯度门槛，核 FFT 使用可微 FP64 准备后转连续 complex64，激活和求解维持 FP32。空间面积 ≥16,384 或实际核谱总量 ≥1,048,576，且核高 ≤输出高/4 时采用分离 FFT；其余保留二维 FFT。默认每次调用重新准备，成本计入完整训练步，无跨步缓存。详见 [本轮继续优化](docs/training_refinements.md)。
- s=1 专用核已按空间面积 ≥65,536 接入；本轮消融中 256² 完整步相对相同准备路径的 generic 核再加速约 1.03×（B1）/1.07×（B4）。FP32 默认改动相对上一轮使 B8/B16 s3 缩小 USRNet 完整步再加速约 1.10×，不能与旧 dev 比率相乘。
- 同 forward 可微核谱复用已实现为 `reuse_training_spectra=True` 显式候选，默认关闭；仅复用重复固定权重，保存 complex128 准备图、逐调用转 complex64，包含版本/图身份/流失效及异常清理。功能回归通过，但完整模型短训练压力门槛未过，不能当作已验证的默认训练能力。
- 已提供真实 HR/LR 配对与 7×7 核的 USRNet PSNR/SSIM 评估入口。用户提供的 1000 张真实图已按固定 900/100 拆分用于 s3 合成退化微调；原作者正式训练配方仍未提供。完整模型严格 FP64 审计发现冻结基线已有的 16 个梯度张量/26 个逐点超限，原门槛和失败状态均保留。
- 真实图微调：完整预训练 USRNet、FP32、HR96/batch4、Adam1e-5/MSE，3 种子各做 before/current 250 步（共 1500 updates），全部逐批哈希/有限值及事前质量门槛通过。current 三种子平均 Y PSNR 从 14.220 到 30.084 dB，final 与 before 最大差 0.000026 dB；本次预热步配对比约 1.049×、allocated 下降 4.31%，保留 before/seed43 的时间波动。只是固定保留图块的短程微调，不代表全图或完整收敛；见 [1000 图微调](docs/dataset_finetuning.md)。
- Python 训练基线补测：同 GPU 的 `backend=pytorch` 全谱 FP32/ATen，与当前实现保持 full USRNet、HR96/s3、batch4/micro4。独立四轮交替预热计时完整步 655→432 ms，配对比约 1.516×、allocated 12.916→9.736 GB（−24.62%）；三种子 Python 各250步与既有 current 逐批/质量核对通过，final PSNR 最大差 <0.00002 dB。此基线不同于上条冻结 C++/CUDA，不乘加速比；详见 [Python 训练对照](docs/python_training_comparison.md)。
- 训练补测：s=3 与 B8/B32、最大形状 B4/C32/256²/s3 的 44 个独立 FP64 数值案例通过，包含共享核梯度累加、高阶与弱正则。性能按算子/完整步/整网分别记录，见 [s=3 与大 batch 补测](docs/training_scale3_batch.md)，不能直接将算子收益外推到大 batch USRNet。
- Nsight 分析：当前 FP32 路径完成 6 组 Systems 与 5 组 Compute 采集，覆盖 s1/s3、B1/B4/B32 和缩小 USRNet。大图 s3 的 GPU kernel 累计时间约 43% 在 FFT；自定义核多呈带宽限制，FP64 核准备 FFT 呈双精度执行限制。建议优先减少 FFT 周边读写，再按形状评估准备与共享核反向；这些是诊断结论，不是新增加速结果。见 [Nsight 分析](docs/nsight_training_analysis.md)。
- 训练实验：融合前后向、高阶回退及 s=1 专用核；564 项检查和短程 SGD/梯度累积验证通过。本机 FP32 测量中，已有融合在缩小版 USRNet 上约 1.30×；新增 s=1 核在已测 256² 算子完整训练步上再加速 1.06–1.07×，已测小图/缩小网络无稳定额外收益。
- 精度实验：1,104 组中，高精度核预计算使最大相对 L2 从 3.89e-4 降至 4.60e-7，热缓存延迟基本不变；冷缓存成本增加，仍有 32 组逐点诊断超限，未接主线。
- 推理实验：nearest 频域先验融合、warp 组织候选已有独立验证；不能直接视为主线能力或训练收益。

## 待完成（按优先级）

当前用户目标（2026-09-18）：继续优化 **Converse2D完整前后向**，尽量接近 `torch.nn.ConvTranspose2d`，已明确是原生转置卷积。默认groups=1为主要速度标杆、depthwise另列；数学不等价，不能替代FP64正确性参考。新原生基线、隔离s1候选及边界诊断见 [原生deconv目标实验](docs/converse_deconv_target.md)。优先推进这一算子目标，下面整网热点排序保留为背景。

最新继续进展见[共享s1训练优化](docs/shared_s1_training.md)：精确共享传递系数已实现为隔离解析CUDA核心，45项内部梯度/高阶/stream/版本/重复性检查通过。以用户确认的Python FP32精度非劣标准（对同一FP64参考，逐张量max_abs与relativeL2均不大于Python）筛选，原native s1_module fixture中共享CUDA+cat/view为3.639ms，原生dense3.528ms、Python FP32 11.225ms、生产FFT4.974ms；FP32核准备候选被精度筛选排除，仍用可微FP64准备。当前共享CUDA在完整模型136项中仍有8张量某项误差指标高于Python，未接默认、未完成新候选真实图质量验证，目标保持进行中。旧FP64逐点失败仅诊断，依照顶部用户新指令，不再作为旧固定阈值发布阻碍。

后续最终mixed方案已进一步推进：FP64仅用于小KernelNet/5个核投影及核FFT准备，5次DataNet激活/求解/IFFT保持FP32，35次prior用共享CUDA。固定完整模型136张量的max_abs和relativeL2均不高于Python FP32；三种子候选/控制各250步、共1500 updates，对Python及新current六对质量全部通过，完整性错误0，新current/候选完整循环配对中位数1.093×（非完整收敛证明）。新的models/converse_training.py抽取通过CPU24/GPU33逐位等价检查；主库新shared_s1.cpp/.cu仅完成等价迁入，尚未加入构建/默认调用。真实40-op重放中，仅共享prior路线局部100张量非劣筛选有36项未过，未计时，不能借单层结果声称整体对齐native；原报告及完整mixed模型通过结果按各自范围保留。接下来需处理实际调用接入、剩余局部精度/整体速度证据及更长训练验证，目标仍active。

2026-09-18 目标实验结果（默认未接入）：s1专用分派+pad/crop组合，在真实prior模块上4.381ms，对原生dense配对耗时比1.142；完整USRNet三种子各250步的模型参数/Adam状态/指标完全相同，实际循环配对加速中位数1.107×。另有保持Converse数学的nearest k3/s3精确非重叠空间解，经Inductor融合并加自动高阶回退封装，在B32/C32/64×80同fixture/确定性/编译设置下4.519ms，同轮生产FFT33.325ms、native eager7.908ms/native compiled8.119ms，FP64、弱正则及所测高阶契约通过。编译前与全过程必须关闭functorch donated_buffer，冷编译另计；封装首次执行复用了已编译图，不能声称8.8ms冷编译。这个k3/s3特例不命中本次fullUSRNet，不能外推整网收益。预训练局部fixture的1点dw失败和完整模型固定压力fixture（非预训练）的16张量/26点失败均保留，禁止据短训练通过绕过独立精度门槛。完整命令、范围与失败记录见上面报告。

2026-09-18 补充调研：[完整训练重新定位](docs/training_research.md)。完整 HR96/B4/s3 模型每 forward 是 1 个 s3 + 39 个 s1，现有大空间 s1 专用门槛未命中；新 Torch/NSYS/NCU 指向普通 1×1 卷积反向。唯一 GPU kernel 分母下，Conv2d 相关约42.1%、完整 Converse2D 路径约37.4%（其中手写谱核心10.4%）、LayerNorm11.7%、核准备4.8%（属于 Converse2D 子集，不能重复相加）。两轮各100连续步，含实际数据处理与有限值检查的 current/Python 配对比1.515–1.532×；仍不含验证保存或收敛时间。生产默认未在此次调研中改变，后续优先对照1×1卷积算法、激活pad/crop与LayerNorm，再评估真实形状s1分派/核准备。

1. 按完整训练的新profile优先验证1×1卷积算法、激活pad/crop和LayerNorm；局部等价matmul在B4/H96的C64→128与C128→64前向+全部VJP分别约2.09×/1.29×，独立FP64局部门槛通过，但C64→3明显退化且局部显存增加，尚未接入整网或证明质量/整网收益。随后继续按真实形状评估核准备和s=1分派，避免无条件替换；大图专用路径已接入。复用候选须先通过完整模型质量门槛再考虑默认启用，生产构建/JIT分派及独立算子FP64回归已接通。
2. 在已完成的 1000 图多种子短程微调基础上，补齐全图/独立数据集 PSNR/SSIM、完整模型多种子收敛、达到相同质量的总时间和进程总峰值显存；定位现有完整模型严格 FP64 逐点梯度差距。需要更多数据时，用户已授权用 Python 下载 1 万张训练图。AMP 的输入/参数 dtype 契约随 BF16/FP16 优化，在 FP32 训练与推理优化完成后再统一；相关草稿已隔离，未纳入本轮默认实现或性能结论。
3. 推理评估接入高精度核缓存与 nearest 融合；训练研究可微核 pad/roll 融合、同一次 forward 内核频谱复用，再按 profiling 决定后续优化。

依据：[训练实验](experiments/training_speed/RESULTS.md)、[核精度实验](experiments/kernel_precision/README.md)、[nearest 实验](experiments/nearest_spectral/README.md)、[warp 实验](experiments/warp_spectral/README.md)。原始记录在 `artifacts/`（Git 忽略）；旧报告和 `.build` 二进制不能代替当前源码状态。
