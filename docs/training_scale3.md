# s3 训练优化：九 alias 融合候选

2026-09-22。**候选已实现、验证并加入当前 codex 分支，默认未启用。** 只优化 Converse s3 训练频谱前后向；s1/s2、核 FFT 准备、缓存、LayerNorm 和普通 Conv2d 均不改变。已有未提交的 s2 文件逐份 SHA256 核验未改变。

基线为 `codex/training-operator-optimization` 的 `4c6314da1ee0147ad745a1a16df5e68602219226`，实际默认依赖和模型源码已冻结。[基线身份](../../HPC/artifacts/s3_train_20260922/before_manifest.json)。代码：[training_scale3.cu](../Converse2D/torch_converse2d/training/training_scale3.cu)；入口：[study_training_scale3.py](../test/study_training_scale3.py)。默认构建清单及分派仍使用原实现。

## 实现与 s2 的区别

- 固定倍率3，并在 `s==3 && p.numel()<=INT32_MAX-256 && H<=INT32_MAX/3 && W<=INT32_MAX/3` 时使用32位索引；其他情况回退通用实现。
- 每个 LR 半谱频点在寄存器中缓存9个 HR alias 的数据，融合 q/d 求解与输出写回。
- 反向融合 q 伴随和输入梯度；无核广播时同时写核梯度，有 batch/channel 广播时仍用独立的确定顺序归约核。保留保存量、高阶 ATen 回退和 lambda 归约。
- LR 内部列拥有9个输出；DC／偶数 Nyquist 列只写未镜像的6个输出，避免镜像行争写。
- **s3 的 DC 列也有跨行反向依赖。** 例如 HR 列 W 是内部频点，会同时接收当前 LR 行和镜像 LR 行的伴随。边界处重算镜像行的 q 伴随，使用实际 q/d，不假定任意复数半谱满足额外共轭约束。此处不能照搬 s2 的处理。

未引入浮点原子操作、shared memory 分工、低精度或跨训练步缓存。

## 前向与全部反向

RTX 5060 Ti，Torch 2.11.0+cu130，CUDA13.0；FP32，TF32/AMP关闭。固定 CPU 随机输入与归一化 PSF，独立 prior。H/W 均为 LR 尺寸，输出3H×3W。C64/32² 对应 HR96/s3 的 DataNet 尺寸，但这些不是捕获的整网激活。

包含核准备、FFT/IFFT、频谱求解及 x/prior/weight/bias 四类 VJP，不含 optimizer、H2D、编译或 profiler。每路预热5次，6轮轮换，各30次。

| 场景 | 原实现 ms | 融合候选 ms | 配对加速中位数 |
|---|---:|---:|---:|
| B1/C3/3×5，共享 k3 | 0.708 | 0.735 | 0.958× |
| B1/C32/32×40，通道 k3 | 0.723 | 0.715 | 1.011× |
| B4/C32/64×80，batch 共享 k3 | 3.232 | 3.206 | 1.009× |
| B4/C32/64×80，逐样本 k3 | 6.163 | 5.982 | 1.027× |
| B8/C32/64×80，batch 共享 k3 | 7.021 | 6.627 | 1.060× |
| B1/C32/128²，通道 k3 | 4.685 | 4.474 | 1.045× |
| B1/C64/32²，逐样本 k7 | 0.892 | 0.858 | 1.043× |
| B4/C64/32²，逐样本 k7 | 1.922 | 1.880 | 1.021× |

已测较大场景约1.01–1.06×；tiny 样例回退约4.2%，不支持无条件替换全部形状。加速列由逐轮比值取中位数，不能直接用耗时列相除。[完整轮次](../../HPC/artifacts/s3_train_20260922/operators_fused.json)。

固定倍率本身没有稳定收益，int32 地址路径的收益依形状而异。[const64/int32 消融](../../HPC/artifacts/s3_train_20260922/operators.json)。另尝试去掉长期保存的索引数组、重算地址并简化有界取模，所有数值检查通过，但没有稳定胜过简单融合，故未选用。[该候选及原融合的同轮对照](../../HPC/artifacts/s3_train_20260922/compact/operators.json)。这个负结果不被删除，也不据理论寄存器压力推断它更快。

## 算子完整 Adam 步

GPU常驻输入、MSE、Adam1e-5，包含 zero_grad、forward、loss、全部 backward 和 optimizer.step。每轮仅一路 fixture 驻留GPU；预热5步后恢复参数并清零已分配的 Adam 状态，四轮 AB/BA，各20步。

| 场景 | 原实现 ms/步 | 融合候选 ms/步 | 配对加速中位数 |
|---|---:|---:|---:|
| B4/C32/64×80，batch 共享 k3 | 4.203 | 4.156 | 1.011× |
| B4/C32/64×80，逐样本 k3 | 7.284 | 7.047 | 1.033× |
| B1/C64/32²，逐样本 k7 | 1.234 | 1.220 | 1.017× |
| B4/C64/32²，逐样本 k7 | 2.228 | 2.203 | 1.011× |

最终入口测得约1%–3%的训练步收益，幅度有限，不能外推为整网或所有分辨率收益。[最终四轮记录](../../HPC/artifacts/s3_train_20260922/steps.json)。首次探索的训练步汇总保留在 [training.log](../../HPC/artifacts/s3_train_20260922/training.log)；工程化重放覆盖了首次同名逐轮 JSON，首次逐轮数据未完整保留，未计入主表。随后入口增加防覆盖检查，已有结果目录拒绝再次写入。

## 验证与发布门槛

- 21项现有频谱／FP32训练／s3大batch回归通过。其中 s3专项保留44组独立FP64比较记录。[21项日志](../../HPC/artifacts/s3_train_20260922/final_contracts.log)、[专项记录](../../HPC/artifacts/s3_train_20260922/large_batch_contracts.json)。
- 80组复数半谱边界、广播及 complex64/128 案例，覆盖单维、奇偶尺寸、DC/Nyquist跨行关系；400个前向／梯度张量与基线逐字节一致。[边界记录](../../HPC/artifacts/s3_train_20260922/boundaries.json)。
- 八个算子场景的40个输出／梯度张量与基线逐字节一致；按 output、dx、dprior、dweight、dbias 排列。
- 四个算子场景各三步 Adam，共168个输出、梯度、参数及优化器状态张量有限且逐字节一致。
- 完整预训练 USRNet，B1/HR96/s3，固定合成输入，三步 Adam 的1,998个输出、参数、梯度及 Adam 状态张量有限且逐字节一致。[完整模型记录](../../HPC/artifacts/s3_train_20260922/model.json)。这是短轨迹兼容性验证，不是新的真实图 PSNR/SSIM、完整收敛或 time-to-quality 证明。

**Python FP32 非劣门槛仍未通过。** 对同一FP64参考，逐张量要求 max_abs 与 relative_L2 均不高于 Python FP32，候选通过 **10/40**，另 **30/40** 失败，与冻结基线一致。[逐张量指标](../../HPC/artifacts/s3_train_20260922/operators_fused.json)。没有放宽容差或改动 lambda。

[AGENTS.md](../AGENTS.md) 明确要求“精度和训练质量不劣于 Python FP32”。因此 `production_eligible=false`；默认接入仍需处理继承的数值非劣差异，并确定有可靠收益的形状范围。

## Profile 证据

代表场景为 B4/C32/LR64×80/s3、batch共享k3、nearest prior；捕获三次前向及 x/weight/bias VJP，scope 与独立 prior 的正式四类 VJP计时不同。

- 原实现每次5个自定义谱核，融合后3个；无核广播时可从4个降为2个。
- 三次调用的谱核累计时间 `1.711942→1.446573 ms`，约下降15.5%。原谱核占所有GPU kernel累计时间约18.7%，FFT和准备仍占较大部分。以上是 profiler 诊断，不当作正式整算子加速比。
- NCU 广播滤波器梯度核：74→47 registers/thread，DRAM throughput 71.99%→92.52%。融合反向核86 registers/thread、DRAM throughput约68.59%，9-alias 融合仍有资源开销。

[NSYS基线](../../HPC/artifacts/s3_train_20260922/baseline.nsys-rep)、[NSYS融合](../../HPC/artifacts/s3_train_20260922/profile_nsys_fused/capture.nsys-rep)、[NCU基线](../../HPC/artifacts/s3_train_20260922/baseline_filter.ncu-rep)、[NCU融合](../../HPC/artifacts/s3_train_20260922/profile_ncu_fused/capture.ncu-rep)。未用 occupancy 单一指标判定优化成功；未运行 Compute Sanitizer，也未验证其他GPU/操作系统。

## 复现

从 ConverseNet 根目录执行，使用新实验目录；每个阶段结果拒绝覆盖。`compile` 只用于尚无测量结果的冻结构建续编。热装载检查派生源码与二进制 SHA256。

```powershell
& ./experiments/training_scale3/run.ps1 test/study_training_scale3.py --phase build --artifacts artifacts/new_s3_study
& ./experiments/training_scale3/run.ps1 test/study_training_scale3.py --phase contracts_fused --artifacts artifacts/new_s3_study
& ./experiments/training_scale3/run.ps1 test/study_training_scale3.py --phase fused --artifacts artifacts/new_s3_study
& ./experiments/training_scale3/run.ps1 test/study_training_scale3.py --phase training --artifacts artifacts/new_s3_study
& ./experiments/training_scale3/run.ps1 test/study_training_scale3.py --phase profile --route fused --tool nsys --artifacts artifacts/new_s3_study
```

`--phase operators` 比较 before/const64/int32，`--tool ncu` 采集反向热点。全部GPU工作串行，profiling与计时分开。`protocol.json` 是测后整理的执行记录，不冒充预注册方案。
