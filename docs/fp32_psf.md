# 可微 PSF 与 s2 核梯度融合（2026-09-24）

在 `codex/fp32-p0-optimization` 上继续推进，基线为上一批 P0 提交 `d37e963`。本轮保留全谱 FP32 训练，完成两项融合；没有改变推理求解路径。

## 实现

**可微 PSF pad/roll。** 前向按原 pad/roll 索引把核直接写入补零并移位后的数组；一阶反向直接采集原核位置的梯度，返回紧凑布局。两者仅搬运数值，没有归约或 atomic。支持非连续、channels-last、零 stride、负视图和零 padding；大索引保留 int64 回退。开启高阶求导时，反向使用原 ATen 逆 roll 与负 padding/clone，保持图连接和梯度布局。

**s2 无广播核梯度。** 仅在既有 s2 融合条件成立、需要核梯度且核 batch/channel 均不广播时启用。将最终核梯度写回并入 VJP，省去 direct/prediction 等中间结果和独立合成阶段。保持复数除 4、实数功率项除 4、共轭、乘 2 和三次加法的原有舍入边界。广播、W=1 等其余形状仍走原路径。

每次调用仍计算可微核 FFT；没有训练谱复用、半谱训练、AMP、TF32 或 fast-math。构建仍为原来的 13 个翻译单元，新增头文件纳入递归构建指纹。

## 验收与数值边界

| 检查 | 结果 |
|---|---|
| checked CUDA 构建与当前完整测试集 | 57/57 通过，无跳过；本轮新增 9 项 |
| 独立 CPU-only 构建 | 6 项适用测试通过，51 项 CUDA 专用测试按设计跳过 |
| PSF 边界与布局 | k1、奇偶矩形核、核等于目标尺寸、单维/零 padding、转置、channels-last、切片、expand、负视图的空间输出及 VJP 逐字节通过 |
| 共享梯度顺序 | 同一可微 base 同时生成输入、prior、核、bias，并连接额外 loss；其梯度逐字节通过 |
| 高阶与生命周期 | 非线性 loss 的二/三阶对照、side stream、现有和新增 Graph 捕获/更新后重放通过；实际设备为单 GPU |
| s2 融合 | 无广播梯度子集、零核/弱正则、转置、非连续上游梯度逐字节通过；内部任意 complex/懒共轭测试保留既有容差 |
| 严格基准 | 41 个完整算子场景，每次 110 个输出/VJP 张量逐一对同一 FP64 比较，max_abs 与 relative-L2 均不劣于 Python FP32，无附加裕量 |
| 整理前后保持 | 两轮确定性对照，每轮 379 个张量 SHA256 全同：110 个算子输出/VJP、269 个完整 USRNet 输出、loss、参数梯度及一步 Adam 后参数；跨复测也一致 |

没有据这些结果宣称新的数据集 PSNR/SSIM、长期收敛或 time-to-quality。

### 默认 replicate-padding 的已有非确定性

最初默认配置的**未修改基线**有两项 SRResNet 第二级上采样输入梯度没有通过 FP64 非劣门槛。进一步在同一基线上重复 10 次：CUDA 基线与 Python FP32 各自产生 10 个不同的 `dx` 哈希，输出、核梯度和 bias 梯度则保持唯一哈希。

本地 PyTorch 的 `F.pad(mode="replicate")` 在开启 `torch.use_deterministic_algorithms(True)` 时改用确定性分解。开启后，上述四类张量都各自只有一个哈希。因此新增显式 `--deterministic-algorithms` 验收配置，前后版本同配置比较；生产实现没有强制这个全局开关。

原默认基线的两项失败完整保留。后来独立的默认对照中，基线另有两项、候选有四项 replicate-padding `dx` 非劣失败；所有差异和误差均记录，不把它们改判为通过。默认配置的完整 USRNet 及其余被检查张量保持；严格的全张量保持结论仅属于显式确定性配置。

## 分层性能记录

设备 RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，MSVC 14.44，sm_120。两版均关闭 TF32/AMP/cuDNN benchmark，启用 cuDNN deterministic；确定性验收配置额外开启全局确定性算法。

每场景预热 5 次、5 轮 × 10 次，记录同步 wall time、CUDA event 和额外 allocated 峰值。表中为两组独立前后对照的 wall 中位数比值范围，定义为基线耗时 ÷ 候选耗时；不相乘，不视为统计显著性结论。

算子计时包括外部 activation padding/crop、nearest prior、核准备、全部 FFT 和指定 VJP；不含 optimizer。下表是两项融合的组合收益，没有作单项消融归因。

| 场景 | 输入及实际 FFT 尺寸 | 完整前向＋VJP 加速比 |
|---|---|---:|
| USRNet prior，全梯度 | B1/C128/LR32×40，k3/pad2；FFT36×44 | 1.077–1.149× |
| s2 无广播，全梯度 | B2/C64/LR32×40，k7；FFT64×80 | 1.220–1.267× |
| s2 无广播，仅核梯度 | 同上 | 1.307–1.375× |
| s2 无广播，核+bias | 同上 | 1.271–1.292× |
| s2 无广播，全梯度 | B1/C64/LR32×40，k7；FFT64×80 | 1.144–1.203× |
| s2 无广播，仅核梯度 | 同上 | 1.369–1.391× |
| SRResNet 第二级上采样，全梯度 | B1/C64/LR48×56，k2/s2/replicate pad2；FFT104×120 | 1.110–1.157× |
| USRNet prior，仅输入梯度 | B1/C128/LR32×40，k3/pad2；FFT36×44 | 0.962–1.009× |

无广播 s2/k7 全梯度场景的额外分配峰值降低约 29.1%；B2 仅核梯度降低约 35.5%。两个 SRResNet 上采样全梯度场景降低约 28.4%–28.7%。这些是调用期间额外 PyTorch allocated 内存，不是进程总显存；完整 USRNet 步的额外分配峰值没有下降。

**完整模型。** 同一预训练 USRNet，完整 5 次迭代、7 个 prior block，LR8×10/s2；DataNet FFT16×20，35 次 prior FFT20×24。Adam 计时含 zero_grad、forward、RGB MSE、全部 backward 和更新；每轮恢复同一初始状态，数据已在 GPU。

| 配置 | 基线 Adam 步 | 候选 Adam 步 | 加速比 |
|---|---:|---:|---:|
| 全局确定性，第一组 | 109.666 ms | 105.542 ms | 1.039× |
| 全局确定性，第二组 | 110.044 ms | 105.929 ms | 1.039× |
| 默认算法，独立额外对照 | 97.945 ms | 95.549 ms | 1.025× |

完整模型推理为 0.995–1.006×，基本持平。默认算法配置与确定性配置的绝对时间不可混算；默认 replicate-padding 的加速也不能替代其数值验收。没有把短程一步更新称作收敛或模型质量验证。

**准备阶段证据。** 独立、非计时的 CPU profiler 中，USRNet prior 全梯度例的 PSF `constant_pad_nd` API 事件从前后向合计 2 次降为 0，`roll` 从 6 次降为 0；外部 activation padding 保留。全部 41 个场景中，核仍是前向第一个 FFT 输入，s1 保留 2 个、s>1 保留 3 个前向 `fft_fft2` 调用。反向 worker 的事件可能记在 `other` scope，因此保留并汇总所有 scope。这是 API 工作量证据，不是 Nsight DRAM、寄存器或真实 GPU launch 计数。

## 复现与证据

[fp32_psf_results.json](fp32_psf_results.json) 保存全部场景的原始计时轮次、逐张量指标/哈希、失败记录、profiler 计数和构建指纹；重复记录按 ID 复用，未舍弃较慢样例。原始 JSON、探针与日志在 `artifacts/fp32_psf/`，文件 SHA256 已记录。上一轮 P0 报告和历史失败不变。

基线为 `d37e963` 的独立检出。本轮复用其已通过指纹验证的构建，候选重新构建。Windows 每次在独立 PowerShell 进程初始化编译环境：

```powershell
$env:CONVERSE_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
pwsh -NoProfile -File tools/run.ps1 -m unittest discover -s test -p 'test_*.py' -v
pwsh -NoProfile -File tools/run.ps1 tools/benchmark_fp32_psf.py --root '基线检出目录' --build --include-model --profile --deterministic-algorithms --output artifacts/fp32_psf/repro_before.json
pwsh -NoProfile -File tools/run.ps1 tools/benchmark_fp32_psf.py --build --include-model --profile --deterministic-algorithms --output artifacts/fp32_psf/repro_after.json
pwsh -NoProfile -File tools/run.ps1 tools/benchmark_fp32_psf.py --compare artifacts/fp32_psf/repro_before.json artifacts/fp32_psf/repro_after.json --output artifacts/fp32_psf/repro_comparison.json
```

默认配置另跑同样命令并移除 `--deterministic-algorithms`，使用不同输出文件；不覆盖确定性结果。更换环境、GPU 或形状须重新验证。本轮不包含 s3 专用融合、LayerNorm 融合或 FFT-free 后端。
