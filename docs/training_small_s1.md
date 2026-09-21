# 重构后的小图 s1 与 pad/crop 优化

2026-09-21。本轮已完成可运行的隔离候选、算子消融、真实图完整训练步对照和 Nsight Systems 复核。**性能筛选通过，默认分派未改变**：候选与冻结生产的已测输出/梯度逐位一致，但仍继承部分 Python FP32 非劣检查失败，不能仅凭逐位兼容宣布达到该精度目标。

仅处理 Converse 路径，没有修改 LayerNorm、普通 Conv2d、shared_s1/mixed、精度、核准备或缓存策略。生产源码与模型文件在本轮前后保持原始字节一致，见 [汇总](../artifacts/small_s1_20260921/summary.json)。

## 候选和身份

- 按新的 `build_config.py` 冻结实际依赖和模型文件，使用 `legacy_sources()` 导出的自包含文本构建独立命名空间。基线保留 `s==1 && H*W>=65536`；隔离候选强制 s1 专用核，核体、计算次序和高阶回退不变。
- 真实模型的 39 个 s1 都命中候选：35 个 prior 实际为 100×100，4 个 DataNet 为 96×96。唯一 s3 保留原 generic。
- 边界仅替换 CUDA FP32、GradMode 开启且有梯度需求、v7、C128/96×96/s1/circular-pad2 的 prior。使用 `models/converse_training.py` 已有的两次 `cat` 环形填充和保留 strides/offset 的 `crop_view`；仍由 ATen 求导。no_grad 和不匹配形状走冻结原模型入口。
- 本轮快照始于 `2dcdbfc` 加用户未提交重构；期间用户将重构提交为 `b51c91b`。实验以 [源码字节清单](../artifacts/small_s1_20260921/before_manifest.json) 为身份，不以 HEAD 推断代码内容。兼容研究头另存 [补充清单](../artifacts/small_s1_20260921/adapter_manifest.json)。
- 独立二进制及其 SHA、派生源码文本 SHA、实际 flags 均在结果中记录。研究 loader 的文本哈希与磁盘 CRLF 字节哈希有不同语义，热装载分别核验文本与二进制，不能混用。

事前 [实验协议](../experiments/training_speed/small_s1_protocol.json) 保留不变。实际 GPU 为 RTX 5060 Ti、Torch 2.11.0+cu130；FP32，TF32/AMP/Graph 关闭，无训练谱复用，所有 GPU 作业串行。

## 算子前后向

固定预训练 prior 权重、seed9214 的 FP32 输入/上游梯度、C128/96×96、k3、pad2、eps1e-5。包含填充、核准备、FFT、求解、裁剪和 x/weight/bias 全部 VJP；不含 optimizer。预热5次，四轮轮换，每轮每路20次。

| Batch | 冻结生产 ms | 仅 s1 分派 ms | 仅边界 ms | 组合 ms | 生产/组合配对中位数 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.856 | 1.788 | 1.259 | 1.364 | 1.383× |
| 4 | 4.781 | 4.403 | 4.196 | 3.986 | 1.204× |
| 8 | 12.409 | 12.130 | 10.745 | 9.471 | 1.288× |
| 32 | 46.397 | 43.914 | 38.043 | 35.638 | 1.302× |

耗时列是各路轮次中位数，加速列是逐轮比率的中位数，两种统计不必相等。B1 的仅边界与组合存在波动，不据此宣称二者严格排序。全部四路、四个 batch 的输出与三类 VJP 均与冻结生产逐位一致。[原始四路记录](../artifacts/small_s1_20260921/operators.json)。

另有不含外层边界的 [s1 分派矩阵](../artifacts/small_s1_20260921/dispatch_sweep.json)：10组常规/弱正则数值及实际 kernel 分派检查通过。B4/C128/100² 完整前后向 3.451→3.175 ms，配对1.089×；B4/C64/96² 动态 k7 为2.122→1.986 ms，配对1.069×。它使用另一套固定输入，不能与上表相乘。

## 完整训练步

预训练完整 USRNet，5次迭代、7个 prior block、C64、HR96/s3，来自既定900/100拆分的真实图训练批次，Adam1e-5、MSE。GPU常驻输入；计时含 zero_grad、forward、loss、backward、Adam，不含 H2D、数据处理、验证、保存及 profiler。

每路每轮新建一个模型，预热5步后恢复相同初始参数并清零 Adam 状态；四轮 AB/BA 交替，各计6步，只有一个 GPU 模型驻留。

| Batch | 生产中位 ms/步 | 组合中位 ms/步 | 配对加速中位数 | 四轮配对比率 |
|---|---:|---:|---:|---|
| 1 | 217.294 | 199.438 | 1.130× | 0.967、1.058、1.202、1.229 |
| 4 | 454.079 | 396.086 | 1.146× | 1.154、1.176、1.139、1.092 |

B4四轮均改善；B1波动明显且第一轮回退3.4%，保留所有轮次。B4 peak allocated 两路均9,738,292,736 B；B1为2,646,151,680→2,651,759,104 B，增加约5.6 MB。没有宣称显存优化。

B1/B4 各自三步训练的3个输出、133组参数、133组梯度和399组 Adam 状态，共 **668/668 项逐位相同**。最终入口另做 B4 有限值与逐位复核。该验证不是新的多种子 PSNR/SSIM、全图收敛或 time-to-quality 证明。

本次B4未复现重构记录的约9秒/步异常；不据此给之前异常补写“显存分页”等未经证明的归因。[B1](../artifacts/small_s1_20260921/model_b1.json)、[B4](../artifacts/small_s1_20260921/model_b4_retry.json)、[有限值复核](../artifacts/small_s1_20260921/model_finite_b4.json)。

## Nsight 与回归

Nsight Systems 关闭采样，预热后仅采集2个完整训练步，独立于上表正式计时。

| 两步合计 | 冻结生产 | 组合 |
|---|---:|---:|
| 全部 kernel 启动数 | 16,908 | 15,282 |
| generic forward `solve_alias` | 80 | 2 |
| `forward_scale1` / `backward_scale1` | 0 / 0 | 78 / 78 |
| CUDA 谱核心累计 ms | 79.964 | 57.039 |
| 全部 kernel 累计 ms | 766.361 | 753.251 |

GPU kernel 首尾跨度880.448→832.733 ms，kernel busy区间并集766.361→753.251 ms。启动次数、copy/fill工作和谱核心开销减少，但全网仍含其他计算；profiler中的时间与正式计时不能混作同一加速比。这里没有修改 kernel 的 block 配置或 warp 组织，也没有新增 NCU 调优结论。[基线报告](../artifacts/small_s1_20260921/nsys_before_verified/launcher.json)、[候选报告](../artifacts/small_s1_20260921/nsys_combined_verified/launcher.json)。

额外回归：[15项候选 CUDA 契约](../artifacts/small_s1_20260921/contracts.json)通过，覆盖任意复数、广播、选择性梯度、高阶、共享输入、stream/布局、版本检查和弱正则；复用辅助函数 [CPU24项](../artifacts/small_s1_20260921/helpers_cpu.json)和 [CUDA33项](../artifacts/small_s1_20260921/helpers_cuda.json)通过。

## 精度限制与接入决定

与同一独立 FP64 参考比较，16个“batch×输出/梯度”张量中，11个通过 Python FP32 的 max_abs 和 relative L2 零余量非劣检查；以下5个的 max_abs 较高，relative L2 均较低。**原生产与候选误差逐位相同**，仍按原门槛记录失败：

| Batch / 张量 | 生产及候选 max_abs | Python FP32 max_abs |
|---|---:|---:|
| B4 / db | 1.306756e-6 | 1.230919e-6 |
| B8 / output | 7.272466e-5 | 6.737279e-5 |
| B8 / dw | 6.808452e-4 | 6.324694e-4 |
| B32 / dw | 4.797877e-4 | 4.690384e-4 |
| B32 / db | 1.412830e-6 | 1.155718e-6 |

因此本轮交付为隔离实验入口，**没有放宽门槛、没有修改生产默认分派或模型默认 pad/crop**。下一步需单独定位这些已有误差来源，再决定是否接入；不能把旧 mixed 路线的精度通过结果挪作此路线的证据。本轮也没有重新对照原生 ConvTranspose2d，不宣称已经达到原生性能。

## 重放

每次使用新输出路径。以下入口读取本轮已冻结的重构源码，不改写历史报告：

```powershell
& ./experiments/training_speed/run.ps1 test/study_training_small_s1.py --phase build --output artifacts/small_s1_20260921/new_build.json
& ./experiments/training_speed/run.ps1 test/study_training_small_s1.py --phase operators --warm-builds artifacts/small_s1_20260921/new_build.json --output artifacts/small_s1_20260921/new_operators.json
& ./experiments/training_speed/run.ps1 test/study_training_small_s1.py --phase model --batch 4 --iters 6 --warm-builds artifacts/small_s1_20260921/new_build.json --output artifacts/small_s1_20260921/new_model_b4.json
& ./experiments/training_speed/run.ps1 test/profile_small_s1_nsight.py --route combined --warm-builds artifacts/small_s1_20260921/new_build.json --output artifacts/small_s1_20260921/new_nsys
& ./experiments/training_speed/run.ps1 test/check_small_s1_contract.py --warm-builds artifacts/small_s1_20260921/new_build.json --output artifacts/small_s1_20260921/new_contracts.json
```

初次快照缺少仅供研究使用的兼容头、Adam标量的字节视图检查错误，以及Nsight首次无报告/Windows文本哈希换行不匹配均已修复；初次日志和失败JSON保留，没有用于性能统计。成功的 Nsight 入口逐份核验冻结源码、派生源码文本和库的二进制 SHA，直接热装载，不进行编译器探测。
