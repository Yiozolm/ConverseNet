# v7 频谱读写优化与精度验证

本次在 `19c1bfc` 的 v7 上保留两项 CUDA 推理优化：

1. PSF 补零与中心移位合并为一次输出写入，替代完整补零张量再 roll。
2. 未缓存的动态核在 correction kernel 读取 FB 时同时计算功率分母，省去
   real/imag 平方、求和、半谱还原和额外功率归约中间张量。固定核仍预计算并
   缓存分母，训练和 CPU 路径保持 ATen 自动微分实现。

功率计算显式保留分别平方再相加的舍入步骤，没有开启 fast-math、TF32 或降低
计算 dtype。普通 `no_grad()` 中符合缓存条件的 leaf 核继续走缓存分母路径；
动态核加速表针对 `inference_mode()` 下的未缓存核。s3 等倍率的归约顺序仍可能产生少量浮点差异，因此不能宣称所有
输出逐位一致。融合 PSF 的频谱已验证与旧路径逐位一致。

## 保留原有 IFFT 归一化

曾验证将归一化提前并入频谱更新，但它在接近下溢的输入上损失精度：
FP32 identity PSF、输入幅度 `1e-40` 时，按输入幅度归一化后的最大误差从
`1.4013e-5` 增至 `1.8357e-3`。因此该方案已撤销，最终仍在 IFFT 后归一化，
没有删除 cuFFT C2R 的保护性输入复制。

最终版本在 FP32 幅度 `1/1e-30/1e-38/1e-40` 和 FP64 幅度
`1/1e-280/1e-310` 的 identity 回归中，与基线误差相同。

## 同进程性能对比

RTX 5060 Ti，PyTorch 2.11.0+cu130，FP32、关闭 TF32，使用 `inference_mode()`。通过 Git 固定基线源码，
以独立算子命名空间加载，交替顺序测试相同输入。下表为预热后 7 轮、每轮
100 次的 wall time 中位数，包含动态核 clone；未使用 profiler 计时。

| 动态核输入 B×C×H×W / scale | 基线 ms | 优化后 ms | 加速比 |
|---|---:|---:|---:|
| 1×64×128×128 / 1 | 0.2023 | 0.1384 | 1.46× |
| 1×64×128×128 / 2 | 0.9910 | 0.6364 | 1.56× |
| 1×32×128×128 / 3 | 1.2999 | 0.8291 | 1.57× |
| 1×32×127×129 / 3 | 2.6709 | 2.1646 | 1.23× |
| 2×64×32×40 / 2 | 0.2636 | 0.1618 | 1.63× |

对应静态缓存路径的比值在 `0.997×–1.005×`，基本不变。动态核的增量峰值
allocated 显存减少约 0.27–2.03 MiB；这只是该算子测量的峰值差，不代表全部
中间读写流量或整网显存收益。

预训练 USRNet 的 32×40/64×80、scale=2 对比也已执行（5 轮×20 次），包括普通
推理和 CUDA Graph。输出最大差为 0，但整网耗时差很小且有运行波动，不据此
宣称显著整网加速。主要收益集中在未缓存动态核的频谱准备。

所有轮次、显存、误差及源码指纹见 [测量数据](spectral_io_results.json)。

## 精度与回归

- 新增谱 I/O 回归：矩形/奇偶/单维 PSF，batch/channel 广播，倍率 1–5，
  `eps=1e-8`、FP32/FP64，缓存与未缓存两条路径均对照独立 FP64 参考。
- 性能样例中，动态 s3 相对旧 v7 最大差 `3.8147e-6`；其他算子样例逐位一致。
  对 FP64 参考的误差满足原有容差，没有放宽旧测试。
- 下溢压力测试、PSF 频谱逐位一致检查，以及原有前向、梯度、二阶梯度、
  CUDA Graph 缓存生命周期和预训练模型检查均用于验证最终实现。

## Nsight 验证

同进程动态 s2（B1/C64/128×128）时间线，每条路径 10 次：kernel 启动数从
**26 降到 15**。可见 pad/fill、roll、flip、cat、独立平方/功率求和与归约被移除，
新增一次 `prepare_psf` 写入，功率归约合并到 `alias_correction`。

GPU kernel 累计时间约 `835 → 550 μs/次`，仅用于时间线归因；性能结论采用
上面的无 profiler A/B。Memcpy 仍为 2 次/调用、16,920,832 字节，C2R 保护复制
被保留；减少的主要是由 kernel 执行的中间张量读写。

GPU 事件通过 runtime correlation ID 归属到 NVTX 的 baseline/optimized 区间，
避免用异步 GPU 时间戳直接裁剪 CPU 区间而漏计。原始文件为本机
`artifacts/spectral_io/dynamic_s2.nsys-rep`，汇总一并收入测量 JSON。

最终共 39 项回归通过，CPU 的 1 项 CUDA stream 检查按预期跳过；另有 USRNet
8 组、DnCNN/SRResNet 4 组预训练模型对照通过。

## 复现

```sh
python test/test_spectral_io.py
python test/test_error.py --device cuda
python test/test_batched_kernels.py
python test/test_cuda_graph.py
python test/test_usrnet.py
python test/test_pretrained_smoke.py
python test/benchmark_spectral_io.py
python test/benchmark_spectral_io.py --operators-only --iters 100 --rounds 7 --output artifacts/spectral_io_operators_repeat.json
```

本机可用 `& ./.build/run.ps1` 代替 `python`。冻结的对照源码/编译产物位于
`.build/spectral_baseline/`；运行日志和原始 profiler 文件位于 `artifacts/`。
运行 `test/benchmark_spectral_io.py --profile` 会在 CUDA profiler 区间内输出
动态 s2 的 `baseline`/`optimized` NVTX 区间。
