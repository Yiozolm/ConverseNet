# nearest 插值移入 C++ 的隔离实验

本实验比较两个入口：

```python
# 原路径
x0 = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
y = op.forward(x, x0, weight, bias, scale, eps, "v7")

# 实验路径：相同的插值在 C++ 入口内执行
y = op.forward_nearest(x, weight, bias, scale, eps, "v7")
```

`op` 是私有 dispatcher namespace `torch.ops.converse2d_nearest_experiment`。
构建时将当前生产 `converse2d.cpp` 复制到 `.build/nearest_cpp/baseline.cpp`，
只重命名注册 namespace；CUDA 包装文件 include 原源码，仅先取消 Windows SDK
的 `small` 宏以避免变量名冲突。没有修改、安装或替换正式算子，
也没有修改模型调用路径。生成文件及结果分别放在 `.build/nearest_cpp/` 和
`artifacts/nearest_cpp/`。

## 实验边界

这次测的是将插值纳入 C++ 算子接口的效果。内部仍然执行
`at::upsample_nearest2d`，分配高分辨率 `x0`，再计算 `FFT(x0)`；并未合并 GPU
kernel 或消除这两个步骤。因此不能将它称为已经完成频域融合。

保留原 dtype 插值的顺序，避免改变 FP16/BF16 反向归约。
`scale=1` 直接令 `x0=x`，保留原算子的 FFT 复用。
沿用 `CompositeImplicitAutograd`，原 v2-v7 路径和高阶梯度均可使用。
实验入口会先验证输入，再调用原实现；原实现会再次验证参数。

真正减少 GPU 工作的后续方向是利用整数倍 nearest 的频域关系：

\[
\widehat{x_0}[k,l]=\widehat{x}[k\bmod H,l\bmod W]
\left(\sum_{a=0}^{s-1}e^{-2\pi i ka/(Hs)}\right)
\left(\sum_{b=0}^{s-1}e^{-2\pi i lb/(Ws)}\right).
\]

将这个表达式用于频谱校正 kernel 才能省去显式 `x0` 和它的高分辨率 FFT；
这不属于本次实验的实现范围。

## 复现

在已配置 PyTorch、Ninja、C++ 编译器和 CUDA 的环境中，从仓库根目录执行：

```sh
python experiments/nearest_cpp/study.py --iters 50 --rounds 7
```

本机 Windows 路径可直接使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/nearest_cpp/run.ps1 --iters 50 --rounds 7 --quiet-build
```

`run.ps1` 的 `-Python`、`-Msvc`、`-Sdk`、`-SdkVersion`、`-Cuda` 参数可覆盖本机路径；
环境变量仅作用于该进程。`--validation-only` 跳过性能测试，`--cpu` 仅作 CPU 验证。
本机使用 CUDA 13.2 编译器和 MSVC 14.44；Windows CUDA 编译显式开启
`/Zc:preprocessor`。首次尝试 CUDA 13.0 时 `cudafe++` 崩溃，故切换到已安装的 13.2。

## 验证与测量方法

- 111 组输出、一阶梯度和 no-grad 路径对照：FP64/FP32/FP16/BF16，
  倍率 1/2/3，连续/非连续输入及 batch/channel 广播核；另外覆盖 v2-v6。
- 小尺寸 FP64 `gradcheck`、`gradgradcheck`，以及 13 项非法输入检查。
- 五种 FP32 v7 输入规模，预热固定核缓存，每轮 50 次、交替执行顺序，取 7 轮中位数。
- 同一调用循环记录 CUDA Event 时间、包含末尾同步的 wall time 和增量峰值分配显存。
  CUDA Event 测量也可能包含 GPU 等待主机提交的空隙，不等于纯 kernel 计算时间。
- `results.json` 保存环境、源码 SHA256、全部校验记录及逐轮原始测量。

测量为预热后的算子推理，不包含首次编译、初始化、模型前后处理或完整训练步。

## 本机结果（2026-09-16）

环境：RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.2。
首次运行及复测均通过全部 111 组对照、`gradcheck`、`gradgradcheck` 和 13 项
非法输入检查。111 组的前向、no-grad 输出和 x/weight/bias 一阶梯度最大绝对差
均为 **0**。

首次为 7 轮 × 50 次。倍率 3 出现较大时间差异，故增加到 11 轮 × 100 次复测，
保留两份结果。下表时间来自复测的 CUDA Event 中位数：

| 输入 B×C×H×W | scale | Python 插值（ms） | C++ 插值（ms） | 两者峰值额外显存（MiB） |
|---|---:|---:|---:|---:|
| 1×32×32×40 | 2 | 0.16690 | 0.16353 | 3.34 |
| 1×64×128×128 | 2 | 0.47803 | 0.47772 | 84.44 |
| 1×32×128×128 | 3 | 0.59767 | 0.59118 | 92.31 |
| 8×32×128×128 | 2 | 3.53494 | 3.54631 | 337.75 |
| 1×64×128×128 | 1（对照） | 0.14204 | 0.14647 | 16.19 |

两个运行中测得的耗时降低比例（正值更快、负值更慢）：

| 输入 / scale | 首次 | 复测 |
|---|---:|---:|
| 1×32×32×40 / 2 | +0.32% | +2.02% |
| 1×64×128×128 / 2 | −0.69% | +0.07% |
| 1×32×128×128 / 3 | −22.73% | +1.09% |
| 8×32×128×128 / 2 | −5.15% | −0.32% |
| 1×64×128×128 / 1 | −1.04% | −3.12% |

复测没有重现倍率 3 的大幅变慢，原始逐轮计时也存在明显波动。本实验支持
**功能上可以移入 C++，但未观察到稳定、明显的性能收益，也没有显存收益**。
不能据此声称 GPU kernel 已融合或速度提高；后续性能实验应针对消除 `x0` 和
它的 FFT。

原始记录：[首次](../../artifacts/nearest_cpp/results.json) ·
[复测](../../artifacts/nearest_cpp/repeat.json)。复测命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments/nearest_cpp/run.ps1 --iters 100 --rounds 11 --quiet-build --output artifacts/nearest_cpp/repeat.json
```
