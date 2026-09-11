# Converse2D 修复与 GPU 验证报告

日期：2026-09-11。修复基于 main 的 `c7eb880`，代码保留在当前工作区，未创建提交或推送。

## 实现

- Python 与 C++ 统一使用等价残差公式，消除末尾大数相减后除以小 lambda 的写法。
- 修正频率混叠分组与周期平铺；v7 在频域求解，补全半谱时同时反射行列并取共轭。
- v2 是全频谱 ATen 基准；v3–v6 合并为共享的全频谱融合推理实现；v7 是默认半频谱实现。历史标签保留，但不声称它们仍是六套独立内核。
- 融合 CUDA 仅用于无梯度推理；训练统一走 ATen，保留 x/x0/weight/bias 的一阶、二阶导数。
- 缓存记录张量身份、存储地址、版本、尺度、CUDA 流及 inference-mode 状态；保留源张量身份以防地址复用，训练和无版本号的 inference tensor 绕过缓存。上限 64 项、计入张量大小 256 MiB。
- float16/bfloat16 用 float32 计算后转回，支持非 2 的幂尺寸；scale=1 也正确支持独立 x0。
- Windows/Linux 使用各自编译参数；CUDA 源文件采用独立 basename，修复 Windows setuptools 的 .obj 冲突。

## 环境与验证

GPU：NVIDIA GeForce RTX 5060 Ti；驱动 616.92；PyTorch 2.11.0+cu130；CUDA Toolkit 13.0；Python 3.12；MSVC 14.44；Nsight Systems 2026.1.2 / Compute 2026.1.0。

GPU 回归：11 项测试通过，包含 252 组前向子用例和 18 组四类梯度对照。CPU-only 编译后运行同一套测试通过，CUDA 流测试跳过。

验证包含独立空间域稠密线性求解、gradcheck/gradgradcheck、连续训练反传、缓存更新、半精度提升、奇偶/单维尺寸、scale=1/2/3/4、非连续输入、矩形核及 padding。安装入口 `setup.py build_ext --inplace` 已构建，并用生成的包运行预训练模型集成检查。

跨版本前向对照中的最大绝对误差：

| 精度 | v3–v6 对 Python 全谱参考 | v7 对 Python 全谱参考 |
|---|---:|---:|
| float64 | 3.553e-15 | 1.599e-14 |
| float32 | 1.907e-06 | 5.722e-06 |

这些是回归样例的误差，不是任意权重/尺寸的全局上界，也不承诺 float32 逐位相同。

## 推理性能

全部 B=1、kernel=3、float32、bias=0、eps=1e-5。新旧实现使用同一输入、权重、先验与 padding，均为 warm-cache/no_grad；每项预热后取 5 批、每批 20 次 CUDA event 均值的中位数。legacy 为实际编译的 `c7eb880` C++，不是 Python 近似。

| C×H×W / scale | legacy ms | v2 ms | v6 ms | v7 ms | legacy/v7 |
|---|---:|---:|---:|---:|---:|
| 32×128×128 / 1 | 0.3586 | 0.2041 | 0.1447 | 0.1707 | 2.10× |
| 32×128×128 / 2 | 1.3003 | 0.8096 | 0.5461 | 0.2522 | 5.16× |
| 32×128×128 / 3 | 4.8985 | 2.9541 | 1.9637 | 0.6638 | 7.38× |
| 64×256×256 / 1 | 4.3438 | 2.9134 | 1.4792 | 0.3894 | 11.16× |
| 64×256×256 / 2 | 18.2884 | 11.7151 | 7.7592 | 3.8497 | 4.75× |

v7 在这五个样例中比旧 main 快约 2.1–11.2 倍。小尺寸 scale=1 样例中 v6 比 v7 更快，实际选型仍应按工作负载测速。

C=64、256×256、scale=2 的增量峰值 allocated 显存由约 1184.0 MiB 降至 272.9 MiB。此项不等于整进程总显存或 CUDA allocator 的 reserved 显存。

同一组大尺寸测试对 float64 参考的最大绝对误差：

| C×H×W / scale | legacy | v7 |
|---|---:|---:|
| 32×128×128 / 1 | 1.185e-03 | 3.767e-05 |
| 32×128×128 / 2 | 9.294e-04 | 4.768e-06 |
| 32×128×128 / 3 | 1.516e-03 | 3.815e-06 |
| 64×256×256 / 1 | 1.563e-03 | 1.059e-04 |
| 64×256×256 / 2 | 1.384e-03 | 7.153e-06 |

scale=1 的大图仍可有约 1e-4 的 float32 绝对误差，说明稳定公式改善了误差，但不会消除 FFT 和病态频率位置的全部舍入影响。

## 训练性能

计时包括一次前向和 x/x0/weight/bias 四类梯度，x0 为独立叶子张量。每项取 5 批、每批 10 次的中位数。

| C×H×W / scale | legacy ms | v7 ms | legacy/v7 |
|---|---:|---:|---:|
| 32×128×128 / 1 | 1.8861 | 1.4790 | 1.28× |
| 32×128×128 / 2 | 5.7987 | 2.7916 | 2.08× |
| 32×128×128 / 3 | 19.2494 | 8.0223 | 2.40× |
| 64×256×256 / 1 | 17.7722 | 6.6422 | 2.68× |
| 64×256×256 / 2 | 73.6505 | 41.7464 | 1.76× |

训练提升约 1.3–2.7 倍；训练路径没有使用缺少 backward 的裸 CUDA 算子。

## Nsight 定位与优化

Nsight Systems 对 1×32×128×128、scale=2 的预热后 10 次调用采集了有效 CUDA/NVTX 时间线。旧实现包含大量频谱复制、逐元素复数运算和归约；新实现用两个 CUDA 内核计算混叠修正与半谱更新，FFT 数据量也缩小。

第一轮时间线显示两个自定义内核合计约占 GPU kernel 时间的 38%。据此加入可验证范围内的 32 位索引（大张量保留 64 位回退）及 scale=2/3 编译期展开。

| 内核 | 优化前平均 μs | 优化后平均 μs |
|---|---:|---:|
| alias_correction | 31.36 | 26.47 |
| apply_correction | 28.80 | 22.05 |

合计由约 60.16 μs 降至 48.52 μs，减少约 19%。这是具体样例的 trace 结果；端到端测量会受到 WDDM 调度和 GPU 时钟影响。未把 profiler 启动开销计作算子耗时。

Nsight Compute 已连接进程，但驱动返回 `ERR_NVGPUCTRPERM`。本次没有取得硬件性能计数器，不能据此声称已测得 occupancy、带宽利用率或 cache hit rate；也没有修改系统的计数器权限。

## 预训练模型与 TF32

使用仓库自带的 DnCNN、SRResNet 权重，严格加载 state_dict，在固定随机 24×32 输入上比较完整模型。默认允许 TF32 时，SRResNet 的差异达到约 8.05e-4；控制实验关闭外围 cuDNN/matmul TF32 后下降至约 1.31e-6，float64 下约 3.28e-15。该现象是外围低精度卷积放大了 FFT 舍入差异，不是混叠索引仍有错误。

正式模型回归测试关闭 TF32，并采用比最初更严格的 float32 容差 1e-5、float64 容差 1e-11：

| 模型 | dtype | 最大绝对差 |
|---|---|---:|
| converse_dncnn | torch.float32 | 2.980e-07 |
| converse_dncnn | torch.float64 | 4.441e-16 |
| converse_srresnet | torch.float32 | 1.311e-06 |
| converse_srresnet | torch.float64 | 3.275e-15 |

库不修改应用全局 TF32 开关。需要严格整网对照时，由调用方设置 `torch.backends.cudnn.allow_tf32=False` 和 `torch.backends.cuda.matmul.allow_tf32=False`。以上是集成验证，不是图像数据集 PSNR 测试。

## 复现与文件

运行方法见 [test/README.md](../test/README.md)；API 与变体含义见 [Converse2D/README.md](../Converse2D/README.md)。

本机使用 `D:/anaconda3/envs/vllm/python.exe`。当前工作区保留了 `.build/run.ps1`，它仅为子进程配置已安装的 MSVC 14.44、Windows SDK 和 CUDA 13.0；例如 `& ./.build/run.ps1 test/test_error.py --device cuda` 可复跑 GPU 测试。该机器专用脚本和编译产物在 Git 忽略目录中，通用脚本位于 test/。

- [GPU 正确性](correctness_cuda.json)、[CPU-only 正确性](correctness_cpu.json)
- [推理测量](gpu_benchmark.json)、[训练测量](gpu_training_benchmark.json)
- [模型验证](pretrained_smoke.json)、[TF32 控制实验](pretrained_precision_diagnosis.json)
- [旧实现 Nsight 摘要](nsight_legacy_summary.txt)、[最终 v7 摘要](nsight_v7_summary.txt)
- [索引优化前摘要](nsight_v7_before_index_optimization.txt)
- 本机原始 trace：`analysis/profiles/systems_v7_s2.nsys-rep`、`analysis/profiles/nsys_legacy.nsys-rep`（二进制报告未加入 Git）。

构建与测试使用了本机已有的 Python/CUDA 环境；没有安装到全局环境、修改 GPU 驱动设置或推送代码。通过 `.data`/裸指针绕过版本计数的原地写入仍需调用 `clear_cache()`，这是明确的缓存使用约束。
