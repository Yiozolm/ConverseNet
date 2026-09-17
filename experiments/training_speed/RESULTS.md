# 不使用 CUDA Graph 的训练实验

日期：2026-09-17。最终实验入口为 **eager-only**，没有 Graph 捕获、重放或静态输入缓冲。生产算子和模型未修改，全部实现位于本目录。

**结论：已有半谱融合在完整训练步中有明确收益；新 s1 内核的可靠增益集中在较大尺寸。256×256 的算子前向＋反向再加速 1.07–1.10×，包含优化器的训练步再加速 1.06–1.07×。小图及缩小版 USRNet 没有显示新内核的稳定额外收益，不建议将其无条件替换所有 eager 训练配置。**

## 实现与对照

| 路径 | 内容 |
|---|---|
| checkout | 从当前生产源码隔离构建的 ATen 训练路径 |
| fused | 仓库已有半谱融合 forward/VJP，通过实验绑定接入 FP32 |
| s1 | 相同 fused 后端，新增 scale=1 CUDA 专用前后向 |

[scale1.cu](scale1.cu) 将原来的两个前向 kernel 合并；反向逐频点 adjoint 也合并。非广播核梯度直接写出，广播核保持固定 b→c 累加顺序；lambda 归约保持原实现。不使用 fast-math 或浮点 atomic。s>1 直接调用原 fused 函数，作为无优化负对照。

两个融合变体使用相同 FFT、保存张量方式及二阶 ATen 回退，因此 fused vs s1 隔离了本次 kernel 改动。构建不依赖历史 artifacts 快照，也不加载旧 .build/cuda 二进制。源码指纹保存在原始 JSON。

## 验证

[validate.py](validate.py) **564 / 564 项通过**，两后端各 282 项。包括独立 full-FFT FP64 输出与一阶梯度、任意 complex128 半谱、s1/2/3/4/5、奇偶/单维尺寸、四种核广播、same/nearest/独立 prior、15 种梯度需求、非连续/共轭 view、gradcheck/gradgradcheck、非默认 stream、连续参数修改与保存输入版本检查。

空间输出/梯度分别采用 atol=rtol=3e-5 / 5e-5；complex128 前向/梯度分别为 1e-12 / 1e-11。完整误差及各项门槛见 [validation.json](../../artifacts/training_speed/validation.json)。这不是对任意弱正则输入的精度上界。

完整训练每个工作负载连续执行四个不同输入/target 的 SGD 步，对照 loss、所有梯度、参数及 momentum。验证内部算子、USRNet 的 KernelNet 和数据分支参数确实更新；另有两次 microbatch 梯度累积后执行一次参数更新的测试。每个正式计时区段结束后另行检查有限值。

三次运行的所有训练检查通过。新 s1 相对 fused 最大参数差约 3.64e-12、梯度差 2.98e-8、momentum 差 5.96e-8。融合变体相对 checkout 最大参数差 1.49e-8，梯度差 5.72e-5、momentum 差 4.58e-5，均通过绝对＋相对容差。没有宣称逐位相同或数据集收敛已经验证。

## 测量协议

RTX 5060 Ti 16GB，Windows WDDM，PyTorch 2.11.0+cu130，CUDA Toolkit 13.2 / MSVC 14.44 / sm_120。FP32，TF32 关闭。

三个独立 Python 进程，逐个运行，种子和输入相同；每个进程每路径 6 轮 × 20 次，交替顺序。记录 CUDA Event 和同步 wall time，编译与预热在计时外。每轮完整训练从相同参数、零 momentum 状态重新开始。

完整训练使用普通 eager 的 zero_grad(set_to_none=True)、MSE、backward、SGD(lr=1e-4,momentum=0.9)。输入已在 GPU，直接传入，没有额外 D2D 拷贝。数据加载/H2D 不计入。profile 单独运行，不混入速度统计。

以下表格为各进程中位数再取中位数，单位 ms；额外列出每次运行的新内核收益，避免把进程间波动隐藏在单个数字中。未锁 GPU 频率，未建立跨日期置信区间。

## 算子前向＋一阶梯度

此处请求 x/weight/bias 梯度，不含 loss、优化器。

| B×C×H×W / scale | fused | s1 | 三次 fused/s1 加速比 |
|---|---:|---:|---|
| 1×32×64×80 / 1 | 0.5903 | 0.5916 | 1.075 / 1.046 / 0.993× |
| 8×32×64×64 / 1 | 0.7117 | 0.7075 | 1.029 / 1.006 / 1.012× |
| 1×32×256×256 / 1 | 1.0746 | 0.9897 | **1.086 / 1.068 / 1.096×** |
| 1×32×64×80 / 2，负对照 | 0.8072 | 0.8095 | 1.020 / 0.966 / 0.979× |

256² 三次都有收益。其余小图收益很小、甚至反转；s2 两路径实际上调用同一套 kernel，仍有几个百分点差异，显示本机 eager 小样例的噪声不可忽略。

## 完整训练步

operator 是 1×1 producer 加 Converse2D，确保算子输入需要反向传播。Block 为实际 ConverseBlock；C16 输入在内部扩为 C32，padding 后 FFT 尺寸为 36×44。USRNet 为实际网络的两轮迭代、一个 block、64 隐藏通道，alpha1/alpha2 在各路径统一设为 0.1，以使短实验产生有效内部梯度。

| 工作负载 | checkout | fused | s1 | 三次 s1 相对 fused 加速比 |
|---|---:|---:|---:|---|
| operator B1 C32 64×80 s1 | 1.7843 | 1.2090 | 1.1978 | 1.050 / 1.018 / 1.009× |
| operator B4 C32 64² s1 | 1.7714 | 1.1860 | 1.1287 | 1.021 / 1.051 / 0.982× |
| operator B1 C32 256² s1 | 2.4748 | 1.5275 | 1.4337 | **1.065 / 1.063 / 1.072×** |
| operator B1 C32 64×80 s2 | 2.6647 | 1.3252 | 1.3370 | 0.950 / 1.001 / 0.991× |
| Block B1 C16 32×40 | 3.3169 | 2.8857 | 2.8721 | 1.017 / 1.019 / 0.973× |
| 同上，梯度累积两次 | 6.4357 | 5.4852 | 5.4443 | 1.012 / 1.021 / 0.903× |
| USRNet B1 RGB 16×20 s2 | 11.9413 | 9.1999 | 9.2520 | 0.982 / 0.994 / 1.010× |

按上述跨进程中位数，已有 fused 相对 checkout 的完整步加速为约 **1.15–2.01×**，USRNet 为约 **1.30×**。新 s1 进一步改善了大图 operator 完整训练步，但没有稳定改善 Block 和 USRNet。这里的 1.30× 来自接入已有半谱融合，不是本次新 s1 kernel 的收益。

## 下一步

保留 s1 作为大图训练候选，不默认用于全部 eager 小图。小图下一项值得实验的是可微 PSF pad/roll 融合，或同一次模型 forward 内复用共享层的可微核频谱，以同时减少准备操作和 autograd 调度。不能把推理频谱缓存直接跨 optimizer.step 复用。

本次没有 AMP、真实数据集、默认完整规模 USRNet 的收敛/PSNR/SSIM 验证，也未测完整进程显存上限。JSON 中增量 allocated 峰值不是总训练显存。

## 复现与原始记录

    & ./experiments/training_speed/run.ps1 experiments/training_speed/run_all.py --iters 20 --rounds 6

- [第 1 次 eager 运行](../../artifacts/training_speed/eager_results_1.json)
- [第 2 次 eager 运行](../../artifacts/training_speed/eager_results_2.json)
- [第 3 次 eager 运行](../../artifacts/training_speed/eager_results_3.json)
- [数值验证](../../artifacts/training_speed/validation.json)
- [实验入口与说明](README.md)

此前含 Graph 的探索数据留在 artifacts 作为历史记录；本报告仅使用以上 eager_results_1–3，当前实验代码没有 Graph 执行路径。
