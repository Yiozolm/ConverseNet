# CUDA Graph 推理实现与验证

在 dev 的 `600b15f` 基础上实现优化报告的 P0：固定形状 USRNet 图推理。
入口为 [`USRNetCUDAGraph`](../../models/cuda_graph.py)，checkpoint 格式和普通
`model(...)` 的训练接口保持兼容。本次尚未实现 LayerNorm 或 FFT 算术融合。

## 实现

- 按输入/核 shape、batch、dtype、device、scale 缓存图，默认只保留一张，LRU
  淘汰前等待未完成的 replay。每次复制输入，输出为独立张量。
- 每次调用检查模型参数地址、版本、模块配置和后端选项；正常权重更新或加载
  checkpoint 会重新捕获。`.data` 原地写入不更新版本计数，仍需手动清缓存。
- 固定权重频谱在独立的预热缓存中计算，由图入口持有至图释放，避免每次 replay
  重算。动态核频谱在图内计算；两者均不依赖全局可淘汰的 eager 频谱缓存。
- 直接使用 `torch.cuda.graph` 捕获算子时绕过全局频谱缓存，在图内计算频谱。
- 顺序调用可跨 CUDA stream，通过事件保护共享缓冲；捕获失败会退出频谱缓存
  作用域。旧版扩展会被检测并提示重新编译。
- KernelNet 使用 `reshape` 展平模糊核，支持转置、切片产生的非连续输入。

## 实测

RTX 5060 Ti，PyTorch 2.11.0+cu130；仓库预训练 USRNet，batch=1，scale=2，
FP32，关闭 TF32。无 profiler，五轮交替顺序，每轮 20 次，取 wall time 中位数。
图入口计时包含参数/配置检查、GPU 输入复制、replay 和输出 clone。

| LR 输入 | 普通推理 ms | 图入口 ms | 加速比 | 两组改变输入/核后的最大绝对误差 |
|---|---:|---:|---:|---:|
| 32×40 | 19.796 | 8.271 | 2.39× | 0 |
| 64×80 | 35.287 | 29.658 | 1.19× | 0 |

首调用（模型已预热，但包含图入口自身的预热和捕获）分别为 95.77/157.19 ms，
不计入稳态加速。捕获前后 allocated 显存净增约 39.55/121.76 MiB；这些是本次
运行快照，包含缓存差异，不能等同于图私有池大小或生产环境显存上限。
多尺寸请求需权衡图数量、显存与重新捕获成本，训练未使用图优化。

完整轮次、CUDA event 时间、显存快照和生产源码 SHA-256 见
[测量数据](graph_runner_benchmark.json)。与历史报告采用各自同轮 eager 基线比较，
不直接用两次测量的绝对耗时推算速度变化。

## 使用与验证

先重新编译扩展，再加载权重并将模型移到 CUDA、设为 eval。
模型参数应在 `inference_mode()` 外创建，输入支持 FP32/FP64，关闭 autocast。

```python
import torch
from models.cuda_graph import USRNetCUDAGraph

runner = USRNetCUDAGraph(model, max_graphs=1)
with torch.inference_mode():
    output = runner(image, kernel, scale=2)
runner.clear()
```

```sh
python test/test_cuda_graph.py
python test/test_error.py --device cuda
python test/test_error.py --device cpu
python test/test_batched_kernels.py
python test/test_usrnet.py
python test/benchmark_cuda_graph.py
```

图回归覆盖更换输入与核、输出独立性、频谱缓存清空、权重原地更新和替换、
checkpoint 加载、配置切换、LRU、batch/倍率/FP64、跨 stream、捕获异常恢复，
以及图推理后恢复训练。完整约束和 API 用法见
[CUDA Graph 使用说明](../../Converse2D/README.md#cuda-graph-inference)。

原有算子回归：CUDA 11 项通过；CPU 10 项通过、1 项 CUDA stream 测试按预期跳过。
批量动态核与训练回归 8 项通过，包含梯度检查。预训练 USRNet 的 FP32/FP64、
batch 1/2、scale 1/2 共 8 组 Python/CUDA 对照通过，最大绝对误差分别为
`3.4571e-6` 和 `5.1071e-15`。图入口独立回归 7 项通过。
