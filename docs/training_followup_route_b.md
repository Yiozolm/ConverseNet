# 路线 B 最小反例：nearest prior 的频谱构造

本次只分析旧 `nearest_spectral` 候选。原实现、公式、FFT、归约和缓存均未修改，生产代码未改动。CPU 与 GPU 都复现了输出门槛失败；这是一项失败定位结果，不能视为算法准入。

旧来源为 `codex/fp32-roadmap-research` 的 `cc244e3` 中 `research/algorithms.py`，原始研究 `artifacts/fp32_roadmap/research_numeric.json` 的 **45/96 个输出/VJP 张量失败**保持原判。本次选中其中第 0 个案例：seed **41191**，B2/C3、LR5×7、s2/k3、共享 KB=KC=1、nearest prior、eps=1e-5；只追踪这个案例的 forward output，未重判其 VJP 或其他案例。

## 原失败与忠实性核验

[复现脚本](../tools/training_followup/route_b_nearest.py) 包含冻结候选和独立 Python 参考的自包含副本，仅依赖 PyTorch；`--verify-legacy` 会将所用函数的 AST 与旧研究树逐一比较。插入阶段记录后的 FP32 候选、Python FP32 与 Python FP64 输出，均分别与未插桩原函数逐字节相同；调用旧研究树原函数的结果也相同。CPU/GPU 的输入包字节完全相同，研究源码运行前后哈希保持不变。

GPU 为 RTX 5060 Ti，PyTorch 2.11.0+cu130；未启用 AMP、TF32 或 fast-math，无 extension 构建。GPU 原案例的四个误差数字与旧研究记录**完全一致**：

| 第 0 案例输出 vs 同一独立 FP64 | Python FP32 | 冻结候选 |
|---|---:|---:|
| max absolute | 1.7303519825873082e-6 | 2.3393899724055700e-6 |
| relative L2 | 1.3000409394295847e-7 | 1.3402357062694780e-7 |

两项均劣化。CPU 原案例仅 relative L2 劣化，仍失败；CPU 与 CUDA 的 FFT/归约实现不同，没有宣称两平台逐字节相同。

证据为 [CPU 报告](../artifacts/training_followup/route_b_cpu.json)、[CUDA 报告](../artifacts/training_followup/route_b_cuda.json) 和 [完整性审计](../artifacts/training_followup/route_b_integrity.json)。最后一份审计用 stdlib 核对测量脚本身份、旧算法哈希、输入包一致性、插桩/单变量诊断及旧 GPU 误差，errors 为 0；`passed` 只代表失败忠实复现。

## 去随机化的最小合法输入

在相同 **k3/s2 固定契约**内，把数据变成单点输入和单位核，去除随机数、广播和病态正则的干扰：

```text
B=C=1, LR H=W=2, HR H=W=4, scale=2, eps=1e-5
x = [[1, 0], [0, 0]]
weight = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
bias = 0
prior = nearest(x) = [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
```

所有公开输入均为连续 FP32，kernel 有一个非零系数，正则仍为原 `sigmoid(bias-9)+eps`。HR 必须装下 3×3 核，故 LR 的每个轴至少为 ceil(3/2)=2，B/C 至少为 1；这证明固定 k3/s2 下的形状最小性。非零 x 与非零 kernel 各只占一个元素。这个输入是原失败机制的确定性简化构造，不声称每个中间裁剪都保留失败，也不声称对其他核尺寸、公式或浮点数值幅度的全局最小性。原案例 seed41191 保留；最小案例不使用 RNG。

单位核使 K=1、核能量=1，理论 nearest prior 的降采样恰等于 x，所以精确 residual 为 0，精确输出为 prior。本例 Python FP32 与独立 FP64 都恰好输出该 prior；冻结候选在 CUDA 的 `[0,0,1,2]` 位置产生 **2.9802322387695312e-8 = 2^-25**，另有约 1e-16 级残差。CPU 最大残差位于 `[0,0,2,1]`；两平台最终误差指标相同，但输出位置/字节不必相同。

| 最小案例输出 vs 独立 FP64 | Python FP32 | 冻结候选，CPU/CUDA |
|---|---:|---:|
| max absolute | 0 | 2.9802322387695312e-8 |
| relative L2 | 0 | 1.4901161193847663e-8 |

候选在两项逐输出零裕量门槛上均失败，不使用多数张量通过、容差或相对误差的平均来覆盖失败。

## 首次偏离和误差传播

两条 FP32 路径在 prior、PSF、K、power、Y、lambda、alias_power、denominator 上逐字节相同。沿共同计算图观察，首次不同的张量是 **P（HR prior 频谱）**：

```text
Python32: P = FFT2(nearest(x))
candidate: P = repeat(FFT2(x), 2, 2) * row[:, None] * col[None, :]
row[f] = col[f] = sum(a=0..1, exp(-i * 2*pi*a*f/4))
```

这里的精确单轴频谱可直接写成 `[2, 1-i, 0, 1+i]`，不需要 FFT 或三角函数参考。旧 FP32 相位实际为：

```text
[2+0i,
 0.9999999403953552-1i,
 0+8.742277657347586e-8i,
 1+1i]
```

相对实数/FP64，相位子图最先在 `angle1` 的 FP32 角度量化出现误差：f=2 对应的 `-pi` 被表示为 `-3.1415927410125732`。它与 FP64 角度正确舍入后的 FP32 位模式相同；随后对**这个已舍入的角度**计算 polar，得到非零的 sin 分量。因此不能把这个现象单独归为 CUDA 三角函数实现错误。f=1 的实部相对 1 为一个 FP32 可表示数间隔；f=2 本应为实数零的 Nyquist 相位留下约 8.74e-8 虚部，也破坏了精确 Hermitian 对称。两轴乘法将偏差带入 P，后续原求解器传播这一偏差。

以下为最小案例的 CUDA 阶段数据。max absolute 是候选与原 Python32 的复数模/实数绝对差；ULP 为实部/虚部分别计算的 FP32 有序可表示数距离，合并正负零。

| 阶段 | max absolute 差 | 最大分量 ULP 距离 | 不同实数分量数 |
|---|---:|---:|---:|
| P | 1.7484555315e-7 | 876330286 | 17 |
| K × P | 1.7484555315e-7 | 876330286 | 17 |
| alias mean prediction | 8.7422776573e-8 | 867941678 | 3 |
| Y − alias prediction | 8.7422776573e-8 | 867941678 | 3 |
| q（复数除法后） | 8.7411116567e-8 | 867940037 | 3 |
| P + conj(K) × tiled(q) | 1.6858739404e-7 | 872415232 | 19 |
| complex IFFT output | 7.0036442423e-8 | 865494769 | 25 |
| 最终 real output | 2.9802322388e-8 | 855638016 | 9 |

零附近的可表示数极密，以上很大的 ULP 距离**不代表很大的物理误差或许多次舍入操作**，必须和绝对误差一起读。对理论值全零的 numerator/q/correction，relative L2 没有可解释的分母；原始诊断保留统一 1e-300 分母下限的数值，完整性摘要明确标记 zero-reference，本文使用绝对/L2 误差而不把这些巨大归一化比值当作门槛证据。最终非零输出的 relative L2 分母则正常。

两个只改变现有 solver `P` 参数的诊断都通过：用原 HR FFT 的 P 代入，恢复 Python32 输出字节；把候选 P 代入同一个 solver，重现候选输出字节。它们只是只读定位实验，没有构造新优化候选。初始差异位于频谱生成，而非 kernel FFT、能量/广播归约、正则、缓存身份或新 VJP；相同下游计算会放大、抵消并重新分布该差异。

## 输入字节和复现

[输入包](training_followup_route_b_inputs.json) 将原案例及最小案例的 x/weight/bias 保存为 little-endian FP32 Base64 字节、形状和 SHA256；它与原 `artifacts/training_followup/route_b_cpu_inputs.json` 逐字节相同并随分支保存。prior 由相同量化后的 x 做 nearest；FP64 只提升这些同一字节的值，绝不重新随机采样。原 seed 流中用于独立 prior 的随机数虽不参与 nearest 输入，仍按原顺序消耗，保证原 k/b 不变。

| 最小输入 | FP32 字节 SHA256 |
|---|---|
| x，16 bytes | `ccaf6f183579497e8bfcd71045c04286fd3c2e60f3641e3eea164b761a4494b7` |
| weight，36 bytes | `1d72f46d9b27195d60220f6cf0da5b44f27f0210969d69d82e5e4a757d95ba9f` |
| bias，4 bytes | `df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119` |

在仓库内用新输出名复现；现有报告不会被覆盖：

```powershell
.venv/Scripts/python.exe -B tools/training_followup/route_b_nearest.py --device cpu --inputs docs/training_followup_route_b_inputs.json --output artifacts/training_followup/route_b_replay_cpu.json --verify-legacy
.venv/Scripts/python.exe -B tools/training_followup/route_b_nearest.py --device cuda --inputs docs/training_followup_route_b_inputs.json --output artifacts/training_followup/route_b_replay_cuda.json --verify-legacy
```

脚本与输入包可独立复制到安装了 PyTorch 的环境，省略 `--verify-legacy` 即不依赖任何仓库模块。[独立 CPU 回放](../artifacts/training_followup/route_b_standalone_cpu.json) 已通过这种方式复现两个失败。正常退出码 0 表示 `counterexample_reproduced`；所有报告始终写 `candidate_admitted=false`。GPU 命令应在获分配的独占时段执行；本次 GPU 计算约 0.56 秒，已释放设备。

本轮仅诊断一个既有算法的两个 forward 输入，没有改相位公式或提高候选内部精度，没有增加训练/VJP/Graph/性能/质量准入声明。历史研究矩阵保持原判。本结果给出了相位生成这一明确的后续研究边界，尚未提供修复方案。
