# 数值简化与 v2–v7 优化整合

整合入口为当前 main。普通 Converse2D 已有稳定残差公式与融合/rFFT 路径；本次补齐简化分支的 ConvReverseDataNet 路径，使 USRNet 的动态核也能使用同一套实现。

## 如何组合

所有路径使用同一残差解：

```text
Q  = (FFT(x) - M(FB * FFT(x0))) / (M(|FB|²) + lambda)
FX = FFT(x0) + conj(FB) * E(Q)
```

M 是跨频谱块对应位置的均值，E 是频谱周期平铺。rFFT 路径使用正确的二维共轭索引完成等价操作，除法始终留在频域。

| 优化来源 | 整合方式 |
|---|---|
| 数值简化分支 | 稳定残差公式用于所有后端，保留独立 x0 的语义 |
| v2 缓存 | 保留权重身份、版本、存储、尺寸、CUDA 流与推理状态检查 |
| v3 归约 | 修正混叠分组后，将归约与修正量计算融合到推理内核 |
| v4 零插值 | 残差公式已消除此步骤，无需再运行上采样内核 |
| v5 批次 FFT | 保留批次并行，不额外保留仅做 reshape 的包装 |
| v6 复数运算 | 推理融合共轭乘法与残差更新；训练保留完整 ATen 梯度 |
| v7 实数 FFT | 使用半谱计算并正确恢复共轭频率，减少 FFT 数据量 |

运行时 v3–v6 仍为共享实现的兼容标签，不是六个按顺序执行的阶段。

## 本次新增的整合

- 扩展 Python/C++/CUDA 对核形状的支持：`(1|B, 1|C, kh, kw)`。支持批次共享、通道共享，以及每样本每通道的独立核。
- CUDA 内核根据当前样本和通道选择正确频谱，避免把第一张图的核用于整个 batch。
- ConvReverseDataNet 统一调用 `converse2d_reference` 或 `torch.ops.converse2d.forward`，不再维护另一套闭式计算。
- ConverseUSRNet 新增 `backend` 和 `variant` 参数，控制数据模块和恢复模块；state_dict 的参数名保持不变。
- KernelNet 按线性层权重的 dtype 处理核输入，支持完整模型的 float64 验证。
- 扩展版本更新为 0.3.0，需要重新编译安装以启用批次核支持。

训练继续使用可自动求导的 ATen 路径，包含传回 KernelNet 的梯度；融合 CUDA 路径用于 no_grad/inference-mode 推理。动态生成的 inference tensor 不跨次缓存，避免错误地复用其他样本的频谱。

## 验证

环境：RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，Windows/MSVC 14.44。

- 原有 11 项算子回归测试通过（252 组前向、18 组梯度对照等）。
- 新增整合测试：CPU/GPU 各 8 项通过，包含各 192 组前向、36 组四类梯度对照。
- 覆盖核批次/通道广播、非叶子动态核、不同通道的正则系数、稠密空间域参考、gradcheck/gradgradcheck、DataNet 低精度、padding 和端到端训练。
- 无自定义 CUDA 内核的 CPU-only 构建验证通过。
- 原 USRNet checkpoint 严格加载成功；B=1/2、scale=1/2、float32/float64 共 8 个整网对照全部通过。
- 实际 `setup.py build_ext --inplace` 生成的扩展再次通过相同 USRNet checkpoint 对照。

GPU 算子对全 FFT Python 参考的最大绝对差：float64 约 2.49e-14，float32 约 1.43e-5。完整预训练 USRNet 对照的最大差：float32 约 3.46e-6，float64 约 5.11e-15。整网严格对照关闭 TF32，库不修改应用全局精度设置。

## 动态核性能

以下测的是 ConvReverseDataNet，C=64、kernel=7、eps=1e-3、float32。每次调用都克隆核，模拟 KernelNet 每次生成新核，不假设可以复用核缓存。预热后测 5 批，每批 10 次，取 CUDA event 均值的中位数。

“简化 Python”来自 `dcc9896` 的 DataNet；它与简化分支的 C++ 算子是不同测量对象。不要把本表与之前固定 3×3 核的算子表直接混合比较。

| B×H×W / scale | 旧公式 Python ms | 简化 Python ms | 结合后 v2 ms | 结合后 v7 ms |
|---|---:|---:|---:|---:|
| 1×64×80 / 1 | 0.8109 | 0.5763 | 0.3980 | 0.2447 |
| 1×64×80 / 2 | 1.6786 | 1.0251 | 0.6072 | 0.4012 |
| 1×128×128 / 3 | 20.4248 | 12.3527 | 9.7914 | 4.1911 |
| 2×64×80 / 2 | 3.7427 | 2.3461 | 1.6862 | 0.6647 |

本批推理样例中，结合后的 v7 比简化 DataNet Python 路径快约 **2.4–3.5 倍**。收益同时来自移除重复 Python 计算、避免 CPU 临时频谱初始化、C++ 调度及融合/半谱计算，不全部归因于单个 CUDA 内核。

前向加 x/kernel/alpha 梯度的训练耗时：

| B×H×W / scale | 简化 Python ms | 结合后 v2 ms | 结合后 v7 ms |
|---|---:|---:|---:|
| 1×64×80 / 1 | 1.6361 | 1.3878 | 1.3783 |
| 1×64×80 / 2 | 2.6110 | 2.1862 | 2.5127 |
| 1×128×128 / 3 | 30.0116 | 29.8625 | 22.1972 |
| 2×64×80 / 2 | 5.6866 | 5.4986 | 4.0313 |

训练提升较小，部分小尺寸下 v2 更快。应按工作负载选择；这些是本机样例结果，不是所有模型的速度保证，也不是完整 USRNet 的端到端性能测量。

## 使用

在配置好 PyTorch/CUDA 的环境中重新安装当前 main 的扩展：

```sh
python -m pip install ./Converse2D --no-build-isolation
```

```python
import torch
import torch_converse2d
from models.converse_usrnet import ConverseUSRNet

model = ConverseUSRNet(backend="cuda", variant="v7").cuda().eval()
# model.load_state_dict(...)  # 原 checkpoint 可直接严格加载
with torch.inference_mode():
    out = model(x, k, sf=2)
```

用 `backend="pytorch"` 做稳定全 FFT 参考；用 `backend="cuda", variant="v2"` 选择 C++ 全 FFT 路径。直接调用普通 Converse2D 的方式保持兼容。

复现：

```powershell
& ./.build/run.ps1 test/test_combined.py
& ./.build/run.ps1 test/test_usrnet_combined.py
& ./.build/run.ps1 test/benchmark_combined.py
```

数据：[GPU 整合测试](combined_correctness_cuda.json)、[CPU 整合测试](combined_correctness_cpu.json)、[USRNet checkpoint](combined_usrnet_checkpoint.json)、[动态核测速](combined_benchmark.json)。

代码入口：[ConvReverseDataNet/ConverseUSRNet](../models/converse_usrnet.py)、[C++ 公共实现](../Converse2D/torch_converse2d/converse2d.cpp)、[CUDA 融合内核](../Converse2D/torch_converse2d/converse2d_kernels.cu)。
