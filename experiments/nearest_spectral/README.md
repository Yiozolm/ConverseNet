# nearest 先验的频域融合实验

本实验直接复用低分辨率 `FFT(x)`，在 CUDA 频谱校正中按需计算 nearest 先验
的频率值，省去高分辨率 `x0 = interpolate(x)` 和 `FFT(x0)`。
正式算子、模型和上一个 `nearest_cpp` 实验均未修改。

## 计算方法

对于整数倍率 `s`，nearest 复制满足
`x0[s*i+a, s*j+b] = x[i,j]`（`0 <= a,b < s`），因此：

\[
P[k,l] = Y[k\bmod H,l\bmod W]\,\Phi_{sH}(k)\,\Phi_{sW}(l),
\qquad
\Phi_N(k)=\sum_{a=0}^{s-1}e^{-2\pi i k a/N}.
\]

`Y=FFT(x)`，`P=FFT(x0)`。本式采用 PyTorch 默认 FFT 归一化；无需额外除以
`s²`。原求解器的 alias 平均仍保留自己的 `s²` 除数。

CUDA 实现包含一个小的 phase 生成 kernel 和两个频谱校正 kernel：

1. 生成两个方向的一维 phase，共 `sH+sW` 个复数，每次调用都计入成本。
2. 计算 alias 残差时，用 `Y` 和 phase 按需构造 `P`。
3. 写最终高分辨率频谱时，再按需构造 `P` 并加入校正，随后调用原方式的 IFFT。

不分配完整的 `x0` 或它的频谱。卷积核频谱、缓存、lambda、IFFT 和输出精度
沿用原实现。半谱读取同时反射两个坐标；相位使用 double 三角函数并精确处理
DC、alias 零点及 Nyquist。未使用 fast math。

## 范围与构建

仅支持 CUDA 推理，须在 `no_grad()` 或 `inference_mode()` 中调用；没有实现
融合反向。FP16/BF16 输入先升到 FP32 计算并返回原 dtype；FP64 保留双精度。
`scale=1` 调用原实现，作为对照。

独立注册名：`torch.ops.converse2d_nearest_spectral_experiment.forward_nearest`。
构建时将生产 C++ 复制到 `.build/nearest_spectral/baseline.cpp` 并修改注册
namespace；实验 CUDA 包装文件包含生产 CUDA，只在 include 前取消 Windows
SDK 的 `small` 宏。没有安装或替换生产扩展。

本机运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/nearest_spectral/run.ps1 --iters 50 --rounds 7 --quiet-build
```

其他已配置编译环境：

```sh
python experiments/nearest_spectral/study.py --iters 50 --rounds 7
```

`run.ps1` 可用 `-Python/-Msvc/-Sdk/-SdkVersion/-Cuda` 覆盖本机路径，环境设置
仅作用于该进程。脚本支持 `--validation-only`、`--benchmark-only`、
`--profile-only` 和 `--output`。结果保存于 `artifacts/nearest_spectral/`，
包含源码 SHA256、设备/编译环境、误差及逐轮原始计时。

## 验证与性能测量

- 与独立 full-FFT FP64 参考对照，涵盖四种 dtype、倍率 1–5、单例/奇偶/矩形
  尺寸、广播卷积核、非连续输入、权重修改及不缓存的动态核。
- 极小输入及正则项单独做压力测试，分别记录 baseline 与融合误差。
- 检查 CUDA Graph capture/replay 和错误输入，明确拒绝训练调用。
- 使用 profiler 检查预热固定核路径中的 nearest 调用与输入/先验 RFFT 次数。
- 以当前生产 v7 加显式插值为基线，FP32 推理，预热后交替测量，记录 CUDA
  Event、同步 wall time 与峰值额外分配显存。计时包含每次 phase 生成。

CUDA Event 计时可能包含主机提交间隙；结果是算子整体推理时间，不是纯 kernel
计算时间，也不代表完整模型或训练收益。

## 2026-09-16 实测结果

环境：RTX 5060 Ti（驱动 616.92）、PyTorch 2.11.0+cu130、CUDA Toolkit 13.2、
MSVC 14.44。首次测量 7 轮 × 50 次；独立复测 11 轮 × 100 次。
下表使用复测的 CUDA Event 中位数，显存为预热后的额外峰值分配：

| 输入 B×C×H×W | s | 核缓存 | 原路径 ms | 融合 ms | 耗时降低 | 原/融合显存 MiB |
|---|---:|---|---:|---:|---:|---:|
| 1×32×32×40 | 2 | 预热 | 0.15348 | 0.12097 | 21.18% | 3.34 / 2.07 |
| 1×64×128×128 | 2 | 预热 | 0.54567 | 0.33110 | 39.32% | 84.44 / 52.32 |
| 1×32×128×128 | 3 | 预热 | 0.67802 | 0.40647 | 40.05% | 92.31 / 56.23 |
| 8×32×128×128 | 2 | 预热 | 3.88179 | 2.20577 | 43.18% | 337.75 / 209.25 |
| 1×64×128×128 | 1 | 预热，对照 | 0.10621 | 0.10923 | −2.84% | 16.19 / 16.19 |
| 1×32×127×129 | 3 | 预热 | 1.57156 | 0.88064 | 43.96% | 128.20 / 92.16 |
| 1×32×96×128 | 4 | 预热 | 1.05212 | 0.67395 | 35.94% | 121.80 / 73.72 |
| 1×64×128×128 | 2 | 不缓存 | 0.69652 | 0.40996 | 41.14% | 100.56 / 68.44 |

典型 s=2、1×64×128×128 的额外显存减少 32.12 MiB（38.04%）。所有 s>1
样例的额外显存降低 28.11%–39.48%。不缓存一行使用 inference tensor 卷积核，
每次重新准备核频谱；不包含上游网络生成动态卷积核的时间。

首次测量的 s>1 耗时降低为 16.89%–44.49%；复测为 21.18%–43.96%。典型
s=2 两次分别为 39.12% 和 39.32%，大 batch 分别为 43.35% 和 43.18%。
s=1 没有融合，其变化从首次 +7.06% 到复测 −2.84%，属于调用开销/计时波动，
不能作为算法收益。同步 wall time 的复测结果与 CUDA Event 结论一致。

### 省掉的操作

预热固定核、s=2 的 profiler CPU 算子事件计数：

| 操作 | 原路径 | 融合 |
|---|---:|---:|
| `aten::upsample_nearest2d` | 1 | 0 |
| `aten::fft_rfft2` | 2 | 1 |
| `aten::fft_irfft2` | 1 | 1 |

这与代码及显存下降一致：显式 `x0` 与其 FFT 已被消除。phase 每次重新生成，
其 launch 和计算成本已经包含在上述时间中。

### 数值验证和限制

480 组常规对照、28 组缓存/权重修改对照、3 次修改输入后的 CUDA Graph replay
及 15 项输入/训练模式拒绝检查通过。常规与缓存测试相对独立 FP64 full-FFT
参考的最大误差如下；低精度行含输出量化误差：

| 输出 dtype | 最大绝对误差 | 最大相对 L2 误差 |
|---|---:|---:|
| FP64 | 9.59e-14 | 4.93e-15 |
| FP32 | 1.99e-5 | 3.00e-6 |
| FP16 | 7.81e-3 | 2.58e-4 |
| BF16 | 8.83e-2 | 2.22e-3 |

常规容限为 FP64 `rtol=atol=1e-10`、FP32 `rtol=atol=1e-4`，FP16/BF16
为 `rtol=2*dtype.eps, atol=1e-4`；结果 JSON 另存独立参考量化回原 dtype
时的误差。八个性能样例的融合输出与原路径最大绝对差不超过 2.87e-6。

**极小数压力测试并非全部通过。** 21 项中 15 项满足严格相对 L2 门槛，
6 项未通过，且这些失败的 baseline 与融合误差相同：

- 极小核与 `eps=1e-42`、s=2：两种缓存模式的相对 L2 误差均约 1.12e-4，
  略超 FP32 的 1e-4 门槛。
- 极小核与 `eps=1e-45`、s=2/3：两种缓存模式的相对 L2 误差约 27.0%–27.5%。
  这些系数处在 FP32 subnormal 极限，本实验没有解决原路径的这一精度限制。
- 单独的微小输入测试（FP32 1e-38/1e-40、FP64 1e-310，s=2/3/5）全部通过。
  FP32 最差相对 L2 为 6.53e-6，FP64 为 2.23e-14。

因此 JSON 中 `stress_passed=false`、`overall_accuracy_passed=false` 被明确保留；
`passed=true` 表示常规验证、操作计数及性能测试完成，不表示全部压力测试通过。
目前结论仅适用于已测的 CUDA 推理范围；没有实现融合训练反向，也没有做整网
质量或训练收敛验证，不将此实验直接接入正式模型。

原始记录：[完整验证与首次测量](../../artifacts/nearest_spectral/results.json) ·
[性能复测](../../artifacts/nearest_spectral/repeat.json)。复测命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/nearest_spectral/run.ps1 --benchmark-only --iters 100 --rounds 11 --quiet-build --output artifacts/nearest_spectral/repeat.json
```
