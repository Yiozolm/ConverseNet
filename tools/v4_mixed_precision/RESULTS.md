# FP16 / BF16 第一轮实验结果

本轮**不作生产放行**，不修改预先声明的误差门槛。生产 API 继续只接受 FP32，频谱保持 complex64。Level 1A 的部分输入量化方案通过了本矩阵，但完整数值矩阵仍失败，且已通过分区的完整算子计时全部慢于 FP32。

[results.json](results.json) 保存原始报告路径与 SHA256、逐报告实测源码哈希、公共 checked build 身份、全部 89 个失败行的分类/指标/门槛、模型分布、短训练漂移和性能配对轮次。原始报告没有改写；预期范围拒绝没有改记为数值通过。

**数值矩阵：972 项完成。** 960 个常规覆盖案例中 879 通过、81 失败；另有 12 个显式范围探针，其中 8 个按预期拒绝，4 个数值通过。整个 gate 的 `passed=false`、`gpu_admission=false` 保持不变。

| 低精度存储 | 消融 | 常规案例 | Level 1A：FP32 输出通过 | Level 1B：低精度输出通过 |
|---|---|---:|---:|---:|
| FP16 | 仅 activation | 240 | 240 | 205 |
| FP16 | activation + weight | 240 | 234 | 204 |
| BF16 | 仅 activation | 240 | 240 | 235 |
| BF16 | activation + weight | 240 | 240 | 235 |

每组包含 192 个推理案例和 48 个训练案例。这些计数只证明该矩阵内相对于量化参考的实现边界，不能升级为模型质量或生产承诺。

81 个常规失败保留为三类：

- **4 个 FP16 低精度 weight 案例违反独立 FP32 core 的 max-abs 预算**，都属于弱正则推理。量化后输入不能掩盖原 FP32 求解器相对于 RQ64 的额外误差。
- **2 个 FP16 低精度 weight 的 dweight 表示溢出**：`base134`（s3、7×1、transpose、零核）和 `base234`（s4、17×19、strided、零核），均为 bias=-40、eps=1e-8 的训练案例。
- **75 个额外 Level 1B 失败**全部是输出 max-abs 的 kernel-extra 门槛超限。全部 966 个实际执行的 Level 1B 输出均满足 `C_B == cast(C_A)`；这与微小 FP32 差异跨越低精度输出舍入边界后放大的解释一致，不是 adapter 漏做或错误执行输出 cast。门槛未因此放宽。

| FP16 activation+weight core 失败 | baseline max-abs | candidate max-abs | 允许上限 |
|---|---:|---:|---:|
| `base003` | 4.04226348e-5 | 6.12388307e-5 | 6.06339523e-5 |
| `base012` | 1.36444974e-4 | 2.38594195e-4 | 2.04667462e-4 |
| `base197` | 1.70052055e-4 | 2.92122368e-4 | 2.55078083e-4 |
| `base207` | 1.70052056e-4 | 2.92122368e-4 | 2.55078084e-4 |

原规则仍是 normal total≤1.25×EQ、weak≤1.50×EQ、kernel-extra≤0.25×EQ，分别检查 rel-L2 与 max-abs，并保留预先声明的 1e-7/1e-6 下限。Level 1B 必须同时通过 Level 1A；继承的 Level 1A 失败与新的输出 cast 失败在 JSON 中分开记录。

**预训练模型：固定 6 个 held-out 96×96 中心 crop。** 两份模型报告的全部 264 条记录均有限，参数及原始 FP32 输出恢复成功。输入使用明确的合成退化：DnCNN 灰度 AWGN 25/255，SRResNet 仓库 bicubic ×4，USRNet 已知 7×7 circular blur / 下采样 / AWGN .01。PSNR/SSIM 对预测 clip+round 到 uint8 后计算，原始输出差异另外保留。质量阈值尚未设置。

| 模型/协议 | 原 FP32 平均 PSNR | 原 FP32 平均 SSIM |
|---|---:|---:|
| DnCNN | 33.038680 dB | 0.887123 |
| SRResNet x4 | 34.501038 dB | 0.882245 |
| USRNet s3 | 10.677122 dB | 0.198078 |
| USRNet s1 | 39.139540 dB | 0.952055 |

USRNet s3 的原始 FP32 基线只有 10.677 dB。该报告独立保留，不能用其中很小或略正的 PSNR 差值证明低精度质量合格。另一次 s1 去模糊实验使用同一组 held-out crop 身份，但输入、倍率和指标裁边按 s1 协议重新生成；s1/s3 不合并平均，也不替换彼此。

下表是**全部边界 input + weight + output cast**相对于各自 FP32 基线的结果。这仍是量化后还原 FP32 的敏感性模拟，不是完整 AMP 模型、实际低精度计算或内存收益测量。

| 模型/协议 | dtype | 平均 rel-L2 | 最大 max-abs | 平均 ΔPSNR | 最差 ΔPSNR | 平均 ΔSSIM |
|---|---|---:|---:|---:|---:|---:|
| DnCNN | FP16 | 3.18099e-05 | 0.000154972 | -0.000079 dB | -0.000608 dB | -6.18135e-06 |
| DnCNN | BF16 | 0.000227584 | 0.000920266 | +0.000536 dB | -0.006348 dB | -2.13567e-07 |
| SRResNet x4 | FP16 | 0.000146387 | 0.00189465 | +0.000541 dB | -0.004109 dB | -2.95008e-05 |
| SRResNet x4 | BF16 | 0.0010545 | 0.0117006 | -0.023082 dB | -0.049322 dB | -0.000276266 |
| USRNet s3 | FP16 | 0.00157941 | 0.00658266 | +0.000056 dB | -0.001488 dB | +0.000124792 |
| USRNet s3 | BF16 | 0.0124487 | 0.0782609 | +0.003032 dB | -0.004783 dB | +0.00211754 |
| USRNet s1 | FP16 | 0.00045869 | 0.001872 | -0.004189 dB | -0.022658 dB | -2.15837e-06 |
| USRNet s1 | BF16 | 0.00363322 | 0.0144057 | -0.284925 dB | -0.952616 dB | -0.00212665 |

**USRNet s1 的消融值得继续用于质量阈值校准。** FP16 全部 cast 的平均 ΔPSNR 为 −0.004189 dB、最差 −0.022658 dB；BF16 对应 −0.284925 dB、最差 −0.952616 dB，不能视作无损。

| USRNet s1 消融 | FP16 平均 ΔPSNR | BF16 平均 ΔPSNR |
|---|---:|---:|
| activation_only | -0.000009 dB | -0.124961 dB |
| activation_weight | -0.003534 dB | -0.145952 dB |
| output_only | -0.000033 dB | -0.164663 dB |
| activation_output | -0.001398 dB | -0.251869 dB |
| activation_weight_output | -0.004189 dB | -0.284925 dB |

所有五种消融、两种 dtype、各模型协议的完整分布和最差样本 ID 均在 JSON 中。USRNet 的 `d` 五次 data-step 同时纳入：输入 x 按 activation 开关量化，逐次生成的 kernel 按 weight 开关量化，alpha 始终 FP32。逐层范围/量化误差可以定位敏感边界，但不是隔离单层改动后的因果贡献排名。

**训练诊断：3 seeds × 3 steps × 3 lanes × 2 dtypes，共 54 条 lane-step，全部有限、执行更新且 master/Adam 状态保持 FP32。** FP16 两个 mixed lane 使用相同初始 loss scale=128；BF16 不用 scaler；没有启用 autocast。

| dtype | RQ 相对原 FP32 的最大 max-abs | 最大 rel-L2 | candidate 相对 RQ |
|---|---:|---:|---|
| FP16 | 5.24744391e-5（output） | 3.28461912e-4（dx） | 所有已记录指标均为 0 |
| BF16 | 3.44242901e-4（output） | 2.78330791e-3（dx） | 所有已记录指标均为 0 |

这是 1×2×5×7、s3 的单算子短轨迹，只验证该诊断范围内的有限性、更新和状态 dtype。它没有建立完整模型训练支持或收敛，也不把不同轨迹状态下的差异冒充同输入 kernel-extra gate。

**性能：只测数值合格的三个 Level 1A 推理分区。** 4 种形状 × 3 分区 × warm/cold =24 行，每行两次 9 轮 AB/BA。48 组 wall/CUDA 中位速度比全部小于 1，均慢于 FP32；没有性能优化放行。低精度输出及 FP16 低精度 weight 的失败分区未计时。

主要计时采用 `mixed_perf_002.json`，其全部源码快照与当前文件匹配。`mixed_perf_001.json` 是较早完成的相同成本研究，原 SHA 和全部 24 行中位速度比保留在 JSON：仅未参与 perf 执行的 `profile_ncu.py` / `profile_worker.py` 快照在随后修正，实际计时脚本、adapter、gate、policy、生产构建和计时协议未改变。这是辅助文件身份补齐后的重复测量，不是算法失败，也没有覆盖旧结果。

速度比定义为 FP32 时间 / candidate 时间；小于 1 表示变慢。B4/C128/100×100/s1 的 warm-attempted 结果：

| 分区 | wall 两次速度比 | CUDA 两次速度比 |
|---|---:|---:|
| float16 activation-only | 0.893310x / 0.882202x | 0.855833x / 0.883634x |
| bfloat16 activation-only | 0.919032x / 0.890503x | 0.914812x / 0.879119x |
| bfloat16 activation+weight | 0.722434x / 0.763814x | 0.721951x / 0.762637x |

该大形状的 activation cast 将 allocator 峰值增量从 83,886,592 B（约 80 MiB）提高到 104,858,112 B（约 100 MiB）。较小形状的低精度 weight warm-attempted wall 速度比约为 0.36–0.44×：每调用 upcast 的 FP32 weight 身份变化使 warm 频谱复用不能被假定，相关准备和分配成本已计入。

JSON 保留主要 `_002` 的全部 432 个配对轮次的 wall/CUDA/峰值增量、顺序和调用次数，以及 cast-only 测量与不重复计数的 profiler 分组。较早 `_001` 的另 432 个配对轮次仍在其原始报告中，汇总保留每行两次中位速度比及报告 SHA；两次研究没有混合统计。逻辑 storage 字节和 allocator 峰值都不是实测 DRAM 流量。

**NCU 实际 DRAM：以修正后的 `_002` 三份 capture 为主要证据。** 工作负载为 B4/C128/100×100/s1、shared prior、activation-only、FP32 weight/bias/output；每种模式预热 5 次后捕获 1 次完整调用，采用 application replay、cache-control=none、clock-control=none。

| 外部 activation storage | kernel 数 | DRAM read | DRAM write | DRAM 总字节 | 相对 FP32 |
|---|---:|---:|---:|---:|---:|
| FP32 | 9 | 49,140,480 B | 63,024,384 B | 112,164,864 B | 基线 |
| FP16 | 10 | 52,449,536 B | 71,440,384 B | 123,889,920 B | +10.4534% |
| BF16 | 10 | 48,199,936 B | 76,126,720 B | 124,326,656 B | +10.8428% |

修正后的三个 capture 均没有 `FillFunctor`。它们仍执行相同的 `correction_scale_one<float>`，低精度模式增加了 low→FP32 copy kernel。本次捕获没有显示完整调用 DRAM 流量下降，不能从外部 storage 减半推导谱核带宽减半或 2× 加速。每模式仅一份 capture，没有重复统计或置信区间；profiled kernel duration sum 不作为独立计时或整网性能结论。

初始 `_001` 三份 capture 也完整成功，但因 worker 开启全局 `torch.use_deterministic_algorithms(True)`，引入 perf.py 中不存在的 `FillFunctor`，所以**保留为测量协议不匹配，排除主要对比**。它们不是工具失败，原始文件没有覆盖：

| 初始 capture | kernel 数 | FillFunctor 数 | 原 DRAM 总字节（排除） |
|---|---:|---:|---:|
| FP32 `_001` | 13 | 4 | 256,544,768 B |
| FP16 `_001` | 15 | 5 | 238,974,208 B |
| BF16 `_001` | 15 | 5 | 273,473,024 B |

旧 worker 已精确归档为 `history/profile_worker_deterministic_fill.py`，SHA256 `1a70941e5eafe30dad5343956d8c5d92c2408ee8f632016883e25ee423f7418c`，与三份旧 launcher 记录完全匹配。修正 worker 使用全局 deterministic=False、cuDNN deterministic=True，并及时释放每次 warmup 输出。JSON 的 `ncu_dram_evidence` 保存六份 capture 的 launcher/summary/worker/capture/CSV/log 哈希、源码/build 身份和排除原因；`_002` 的输出 SHA 与对应 `_001` 一致。

**回归与重现。** `.build/mixed-release-002.log` 记录 192 项通过，SHA256 为 `4bbb45968b94a75186fb84a4105b8fba2c3064d0edcb311ce8c36d755c73bfc9`。这些 release/boundary 测试通过不撤销上面的低精度矩阵失败。五份主要报告及保留的较早性能报告使用同一 checked binary：`7ef164bb346eb787219b0cc27f3c9f0039430f93644616b9f4c4f799c83b88ed`。

原始报告与完整 SHA256 索引：

| 证据 | 原始路径 | SHA256 |
|---|---|---|
| operator_gate | `artifacts/v4_campaign/mixed_gate_001.json` | `3123a0bbbf4b43e1c5cc734d9f2f8da522e4c4183db1c3fb2dbc8eaeabb7ae54` |
| models_dn_sr_usr_s3 | `artifacts/v4_campaign/mixed_models_001.json` | `4d0f763542996f25352d26b6f7bdb733aad1b244891a6315327352237d2f7b60` |
| models_usr_s1 | `artifacts/v4_campaign/mixed_models_usr_s1_001.json` | `6455aa5634dc831ebada68bb3a8d28ee0783e88a6cf42f98899db9e5ea554c7f` |
| operator_training | `artifacts/v4_campaign/mixed_training_001.json` | `aa29f9042669e8071e644c59c6c53ec19c9df6b893999fd38a6fee6ff4cb4991` |
| operator_performance（主要） | `artifacts/v4_campaign/mixed_perf_002.json` | `5a831225782a16c742e3a0308d21143acabd974949317aa62dc1d843e74f2e4b` |
| operator_performance_earlier（保留） | `artifacts/v4_campaign/mixed_perf_001.json` | `84439b83c029693e738df95a2d2bab033f185232afa73f1d29f6c334ad030482` |

在匹配的 checked build、项目 Python 环境和独占 GPU 窗口下，用新的输出路径重跑：

```text
python tools/v4_mixed_precision/gate.py --device cuda --output NEW_GATE.json
python tools/v4_mixed_precision/model_study.py --samples 6 --output NEW_MODELS_S3.json
python tools/v4_mixed_precision/model_study.py --samples 6 --models usrnet --usr-scale 1 --output NEW_USR_S1.json
python tools/v4_mixed_precision/training.py --device cuda --backend cuda --output NEW_TRAINING.json
python tools/v4_mixed_precision/perf.py --gate NEW_GATE.json --output NEW_PERF.json
python tools/v4_mixed_precision/profile_ncu.py --dtype fp32 --output NEW_NCU_FP32
python tools/v4_mixed_precision/profile_ncu.py --dtype fp16 --output NEW_NCU_FP16 --identity-from NEW_NCU_FP32/launcher.json
python tools/v4_mixed_precision/profile_ncu.py --dtype bf16 --output NEW_NCU_BF16 --identity-from NEW_NCU_FP32/launcher.json
```

模型使用既有 `artifacts/v4_campaign/dataset_absolute.json` 的固定 holdout；数据、checkpoint、脚本与 build 哈希均须匹配。gate 退出码 2 与本轮保留的失败一致；perf 根据报告中的合格推理分区选择工作负载，不将整体失败改成通过。
