# 训练加速实验（eager-only）

本实验按用户要求不接入 CUDA Graph。实验代码位于本目录，生产算子与模型文件不变。

注意：以上实验结论对应接入前的源码。生产 FP32 融合训练接入后，`checkout`
仍表示当前工作区源码，因此不再是原来的 ATen 基线；重现接入前后的比较请使用
`test/benchmark_fp32_training.py`，它固定从 `b850e38` 构建 dev 基线。

生产 `training.cu` 进一步接入大图 s1 分派后，本目录 `fused` 也会继承该路径，
不能再把当前 `fused`/`s1` 当作历史报告中的独立消融。现在应使用
`test/training_s1_ablation.py`：它从当前源码隔离构建，仅禁用 s1 selector，并记录补丁与指纹。

实测结果见 [RESULTS.md](RESULTS.md)：564 项数值检查通过。三次独立 eager 复测中，新 s1 内核将 256² 完整训练步进一步加速约 1.06–1.07×；小图及 USRNet 没有稳定额外收益。

比较三条 FP32 训练路径：

- `checkout`：从当前生产源码隔离构建的训练实现；历史报告中它是 ATen，接入后会包含生产融合及高精度核准备。
- `fused`：已有半谱融合前向与解析 VJP，由实验绑定接入。
- `s1`：在同一融合后端上使用新增 scale=1 专用 CUDA 内核，合并逐频点前向与反向。其他 scale 直接回退相同的 fused 内核，作为负对照。

`fused` 和 `s1` 共用 FFT、保存张量与高阶梯度回退。没有引入 fast-math、低精度 FFT 或浮点原子累加。空间接口仅支持 FP32 CUDA，内部谱接口支持 complex128，用于独立导数验证。

## 复现

从项目根目录运行完整套件：

```powershell
& ./experiments/training_speed/run.ps1 experiments/training_speed/run_all.py --iters 20 --rounds 6
```

依次执行数值验证、短训练检查和三个独立进程的 eager 基准。每个进程各做 6 轮 × 20 次，路径顺序交替；编译/预热不计入。启动器的 Python、CUDA、MSVC 路径可通过命名参数覆盖，仅影响子进程环境。

也可以单独运行：

```powershell
& ./experiments/training_speed/run.ps1 experiments/training_speed/validate.py
& ./experiments/training_speed/run.ps1 experiments/training_speed/study.py --mode operators
& ./experiments/training_speed/run.ps1 experiments/training_speed/study.py --mode training
```

在已配置 CUDA 编译环境的其他平台可直接执行 Python。构建位于 `.build/training_speed/`，不加载 `.build/cuda` 旧扩展，不依赖历史 artifacts 快照。源 SHA256 记录于结果，并参与构建失效检查。

## 文件与测量范围

| 文件 | 作用 |
|---|---|
| [scale1.cu](scale1.cu) | 新增 s1 训练前向与反向融合 |
| [bindings.cpp](bindings.cpp)、[extension.py](extension.py) | 隔离加载 current/fused/s1，生产源码不变 |
| [validate.py](validate.py) | FP64 空间参考、任意复数半谱、广播/选择性梯度/二阶导数验证 |
| [runtime.py](runtime.py) | 普通 eager MSE、backward、momentum SGD 与梯度累积 |
| [workloads.py](workloads.py) | 单算子生产层、真实 ConverseBlock、缩小版 USRNet 的实例级路由 |
| [study.py](study.py) | 前向＋VJP 与完整训练步的配对基准 |

算子基准测前向和 x/weight/bias 梯度；不含优化器。完整训练步包含 `zero_grad(set_to_none=True)`、MSE、backward 和 SGD(lr=1e-4, momentum=0.9)。输入已在 GPU 上，直接传入普通 eager 模型，不引入固定缓冲区或额外 D2D 拷贝；不计数据加载/H2D。

各路径从相同参数和优化器状态开始，预热后及每个计时区段前重置。四个连续变化输入/目标的训练步比较 loss、所有梯度、参数和 momentum，并检查实际参数更新；另测两个 microbatch 累积后才更新。计时结束后在区间外检查有限值。

USRNet 使用两次迭代、一个 block、64 隐藏通道；各路径统一把 alpha1/alpha2 初始化为 0.1，使短实验确实产生内部算子梯度。它不是默认完整规模网络或数据集收敛验证。完整训练步结果不能直接外推成达到相同 PSNR 的训练总时间。

结果保存在 `artifacts/training_speed/eager_results_1.json` 至 `eager_results_3.json`，数值记录为 `validation.json`。此前产生的含 Graph 探索数据仅留在 artifacts 中，当前入口不会执行这些路径。
