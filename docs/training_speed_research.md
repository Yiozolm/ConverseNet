# Converse2D 训练加速调研（2026-09-17）

建议顺序：**先接回已有半谱融合训练后端，再针对小尺寸做训练 CUDA Graph，随后减少 FFT 次数和准备/反向搬运。** 大 batch 再考虑梯度归约与分块；原生低精度 FFT、cuFFTDx、warp specialization 放在后面。

本次按当前项目与本机配置调研，保持 FP32。检查源码、历史结果和官方文档，并补做四组一阶训练实验；未修改生产算子、模型或安装包。

## 1. 最先解决的是实现没有接入当前训练入口

当前 HEAD：`03470cb6724cbb076a8481cae5ff0d85b03e4e6f`。

| 资产 | 实际状态 |
|---|---|
| 当前 `converse2d.cpp` | 自定义 CUDA 校正仅在关闭 autograd 时使用；训练仍走 ATen 展开路径 |
| 当前 `setup.py`、`test/extension_loader.py` | 只构建主 C++ 和 `converse2d_kernels.cu` |
| `converse2d_training.h/.cu` | 已有半谱前向与解析 VJP，但未被当前主 C++ 包含或构建入口接入 |
| 历史 FP32 训练实验 | 冻结源码与补丁可重建融合快照；不等于当前 checkout |
| `.build/cuda` | 构建描述包含当前入口未构建的训练/低精度源，不能直接当当前源码的性能依据 |

源码定位：[训练分派及 ATen 路径](../Converse2D/torch_converse2d/converse2d.cpp)、[构建入口](../test/extension_loader.py)、[已有训练实现](../Converse2D/torch_converse2d/converse2d_training.h)。主 C++ 第 176 行要求 `!GradMode::is_enabled()`，180–193 行的训练路径包含 `full_spectrum`、alias reduction、`repeat` 及其 autograd 中间量。已有融合后端通过半谱 Q/D 和解析梯度，消除这些整谱展开。

恢复时需一起核对 dispatch、CUDA 源列表、内部算子注册、FP32 主参数契约及模型包装，不能只加一条分支。训练头还引用 native FFT namespace，必须明确依赖。`test_training_fusion.py` 和 `test_fp32_training.py` 期待的 `_training_spectral`、`forward_nearest` 目前均未注册。

历史文档不能整段视作当前状态：[9/16 报告](../artifacts/fp32_training/report.md) 中“setup 选择旧 v1/v2”“USRNet 缺少 backend 参数”已经不符合当前文件；但当前训练融合未接线和 AMP 参数 dtype 契约仍需处理。

## 2. 本次补测：融合有效，Graph 收益取决于形状

环境：RTX 5060 Ti 16GB，Windows WDDM，PyTorch 2.11.0+cu130，隔离构建使用 CUDA Toolkit 13.2。按源码校验 JIT 加载当前 checkout；校验历史原始 SHA256 后应用保存补丁重建融合快照，分别使用独立 namespace。没有加载 `.build/cuda` 的旧扩展。

FP32，同输入/参数/上游梯度，预热后 7 轮 × 30 次，轮换路径顺序；下表为 CUDA Event 中位数。测量算子前向 + 所请求的全部输入一阶梯度；nearest 的插值及反向计入。**不含 loss、优化器、数据拷贝、独立输出拷贝及 CUDA Graph 捕获成本**，Graph 输出复用固定缓冲区；eager 每次 autograd 建图计入。编译和预热在计时外。

checkout 与融合快照包含不同阶段的源码集合，这是两套实现的比较，不能将全部收益归给某一个 kernel 改动。实验未测图私有显存池开销；参数只有 batch 共享的正随机 3×3 核、eps=1e-3、一个种子，其他配置见后续验收矩阵。

| B×C×H×W | scale / prior | 当前源码 eager ms | 融合快照 eager ms | 融合收益 | 融合快照 Graph ms | Graph 相对融合 eager |
|---|---|---:|---:|---:|---:|---:|
| 1×32×64×80 | 1 / 同输入 | 1.0190 | 0.5962 | 1.71× | 0.0995 | 5.99× |
| 1×32×128×128 | 2 / 独立 | 2.4472 | 0.9975 | 2.45× | 0.8545 | 1.17× |
| 8×32×64×64 | 2 / nearest | 4.4303 | 2.0071 | 2.21× | 1.8433 | 1.09× |
| 1×32×127×129 | 3 / 独立 | 10.3972 | 6.2572 | 1.66× | 6.1263 | 1.02× |

当前源码也可捕获，四组 Graph 分别为 0.1507 / 2.1787 / 4.0146 / 10.1113 ms。因此融合与减少主机提交开销是两种可以叠加的措施。

全部八个图捕获成功。每个图测试三次原地改变输入、weight、bias 和 upstream gradient，重放输出及各输入梯度与其对应 eager **最大绝对差均为 0**。这验证了固定分配下的数值更新；没有验证优化器状态、参数替换、不同 shape、多次梯度累积或整网 AMP。

四组融合结果相对独立 full-FFT FP64 参考，输出最大绝对误差 `2.05e-5`，被测梯度最大相对 L2 误差 `1.66e-6`，均有限。这里只记录有限样本误差；FP64 比较未设置统一逐点验收门槛，不代表所有精度测试或训练收敛已通过。

小图融合 eager 七轮范围 0.556–0.632 ms，Graph 为 0.098–0.106 ms，调度收益明显；奇数大图仅约 2% 收益仍需多时段复测，不宜视作稳定承诺。本实验在返回输出时 detach 已完成反向的计算图，避免保留 AccumulateGrad 的捕获 stream 给后续 eager 人为增加同步。

原始轮次、wall time、源指纹、误差与 profile：[results.json](../artifacts/training_research_20260917/results.json)。复现脚本依赖本工作区已有的精度审计工具和冻结源码：

```powershell
& artifacts/fp32_training/run.ps1 artifacts/training_research_20260917/study.py --iters 30 --rounds 7
```

## 3. 融合之后，瓶颈也随配置变化

另行使用 PyTorch profiler 采集各路径三次 F+B，按 CUDA 事件核名汇总。下表是**插桩后的设备累计时长分类**，不包含 CPU 提交间隙，不是无插桩端到端占比，也不是 DRAM/occupancy 硬件计数器。

| 配置，与上表同序 | 设备事件数：当前→融合 | 融合 cuFFT / µs | 融合谱求解与 VJP / µs | 其他 ATen、拷贝 / µs |
|---|---:|---:|---:|---:|
| 小图 s1 | 73→38 | 36.3 | 18.4 | 38.4 |
| 128² s2 | 120→43 | 315.9 | 221.6 | 279.5 |
| B8 nearest s2 | 127→48 | 610.0 | 623.2 | 538.5 |
| 127×129 s3 | 144→67 | 3780.4 | 658.2 | 1706.9 |

由此得到的判断：小图先处理提交与 autograd 调度；规则大图同时减少谱计算和外围搬运；奇数大图 cuFFT 已约占本次累计设备时间的 62%，仅优化谱 kernel 的总收益有限。B8 样例 `adjoint_filter` 约 165 µs，广播归约值得实验，但并非唯一主要开销。

## 4. 建议实施路线

| 优先级 | 工作 | 预期作用 | 证据等级 |
|---|---|---|---|
| P0 | 统一源码、构建、dispatch，接回半谱融合 forward/VJP | 去掉 full spectrum/repeat 和大量反向中间量 | 本次与历史算子实测支持 |
| P1 | 固定 shape 的训练 CUDA Graph | 减少 Python/C++/驱动提交开销 | 本次算子级验证；整网待测 |
| P1 | scale=1 专用前后向 | 合并逐频点计算，减少 Q/R/gd 中间量及 launch | 源码分析，尚未实现 |
| P1 | nearest prior 可微频域合成 | 删除 HR 插值和一次 HR RFFT 及其反向 | 数学与推理原型依据；训练待实现 |
| P2 | 同一次模型 forward 内复用可微 kernel spectrum | 减少重复 pad/roll/kernel FFT 与其反向 | 共享模块结构支持；待测 |
| P2 | PSF 准备与反向、广播梯度归约融合 | 减少搬运；提高大 B/C 的并行度 | profile 和源码支持；待测 |
| P2 | FFT batch 分组、模型外围 compile | 改善工作集和小算子开销 | 历史推理/官方机制依据；训练待测 |
| P3 | 原生 FP16/BF16 FFT、callbacks、cuFFTDx | 特定尺寸的进一步优化 | 精度与平台适配成本高 |

**训练 CUDA Graph。** 当前 `models/cuda_graph.py` 是推理 runner，不能直接用于训练。新路径需固定输入、参数、梯度缓冲地址，稳定 shape/control flow，并按实际训练方法处理 `.grad` 清零/累积。先捕获前向、loss、backward；普通 AMP 的 `GradScaler.step/update` 留在捕获外。图本身不消除 FFT 算术开销，适合先验证小 patch、多层小 kernel 的训练任务。[PyTorch 2.11 CUDA Graph 训练说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)

**scale=1 专用路径。** 现有候选训练对所有 scale 都启动 `solve_alias`、`solve_output`，反向先保存 R/gd。s1 没有跨频率 alias 依赖，可逐频点合并前向、合并输入反向，再处理共享核和 lambda 的归约。`x0 is x` 时可进一步合并两条形式参数梯度，避免让 autograd 在后面合并。需保持稳定残差公式；不能只凭代数等价改变弱正则下的数值行为。[实现位置](../Converse2D/torch_converse2d/converse2d_training.cu)

**nearest 的训练专用消 FFT。** 现有候选训练仍执行 `upsample_nearest2d → HR rfft2`。按 DFT 定义，整数倍率 s 的 nearest 保持插值满足：

```text
P[k,l] = Y[k mod H,l mod W] · d_s(k;Hs) · d_s(l;Ws)
d_s(k;N) = Σ(r=0..s−1) exp(−2πikr/N)
```

这是对本项目插值定义的推导。把合成融合进谱求解，有机会删除 HR prior 张量、HR RFFT 及对应反向变换。反向必须实现该映射的伴随，将 prior 分支贡献加回 observation 分支；Hermitian 半谱的共轭反射、DC/Nyquist、奇偶尺寸不能直接按 full spectrum 公式硬套。独立 prior 不适用。保留高阶梯度的可微回退。[已有推理实验](../experiments/nearest_spectral/README.md)、[二阶梯度要求](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)

**同一步内的可微复用。** `ConverseUSRNet.forward` 多次调用同一个 `self.p`，其中每个 Converse2D 的权重及该步空间尺寸不变。可以将每层 kernel FFT 等准备结果作为本次 forward 的可微中间量复用，各次调用的梯度汇合后只反传一次准备图。不能 detach，也不能跨 optimizer.step 复用失效频谱；现有 inference cache 不适合直接开放给训练。数据分支 `self.d` 的核来自不同 `self.convs[i]`，不能按同一个核跨轮共享。[模型调用](../models/converse_usrnet.py)

**准备与归约。** PSF 的 pad/roll 可以融合，其反向只需按真实 kernel 支持区 gather；优先减少整张 HR 临时量。广播 `adjoint_filter` 每个频点线程串行遍历 B/C，可比较分片 partial sum 与两阶段确定性归约；小 batch 保留简单路径。不要用巨大逐样本滤波器梯度缓冲或非确定性 atomic 换取表面吞吐。lambda 梯度也可与已有计算融合分段归约。

**FFT 布局和外围编译。** 当前 FFT 已经批处理 B×C，新增工作应是按实际工作集调优分组大小，并同时设计反向归约和保存张量生命周期。记录 padding 后的真实 FFT 尺寸，例如 padding=2 将 64 变成 68；允许调整训练裁剪策略时，可以选择因子更友好的实际尺寸。不能在算子内部任意补零再裁剪，因为这改变当前周期边界求解。[cuFFT 性能建议](https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance)

LayerNorm、GELU、残差乘加可作为 `torch.compile` 的候选。先核查本机编译器后端支持及 graph break；自定义 CUDA 边界需要符合 FakeTensor/autograd 注册要求。compile 不会自动进入并优化外部 cuFFT 或已有 CUDA kernel 内部，不能当作消除 FFT 开销的方案。[自定义算子与编译支持](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)

## 5. 暂不优先做的事

**不要把 AMP 当作 FFT 自动提速。** 当前包装层可能把低精度激活与 FP32 参数直接传入，而主 C++ 要求所有张量同 dtype。应先统一 FP32 master parameters、FP32 FFT/分母的契约，再让外围卷积使用 AMP。TF32 主要影响卷积/matmul，不能视为 FP32 cuFFT 的加速开关。

**原生低精度 FFT 缺少稳定收益证据。** [历史基准](../artifacts/native_fft_benchmark.json) 中真正命中原生 FFT 的六个 s1/s2 训练配置仅 1.008–1.163×；s3 BF16 为 0.876×。68×76 全 fallback 的样例却显示 1.149×/1.302×，说明噪声已足以覆盖很多所谓收益。必须记录原生/回退次数和完整训练步，先保护 kernel FFT、分母和参数梯度精度。cuFFT 低精度还有二次幂尺寸与 FP16 溢出限制。[CUDA 13 cuFFT 低精度限制](https://docs.nvidia.com/cuda/archive/13.0.1/cufft/index.html#half-precision-cufft-transforms)

**callbacks / cuFFTDx 更适合作为后续专项。** LTO callbacks 可尝试 IFFT 输出归一化等局部融合，但不能假定跨 block 执行顺序以实现任意 alias 归约；Windows 要核对 LTO callback 工具链。[cuFFT callbacks](https://docs.nvidia.com/cuda/cufft/index.html#lto-load-and-store-callback-routines) cuFFTDx 可以把 FFT 嵌入 kernel，适合先做固定小尺寸 s1 原型；大二维图仍可能需要全局交换，官方主机编译器要求也需先与本机 MSVC 环境核实。[cuFFTDx 示例](https://docs.nvidia.com/cuda/cufftdx/examples.html)、[构建要求](https://docs.nvidia.com/cuda/cufftdx/requirements_func.html)

**当前不优先做 warp specialization。** 本项目训练谱 kernel 没有现成 producer/consumer 跨 warp 流水；cuFFT 内部也不在项目中。已有 warp 实验主要是推理且整体收益受 FFT 限制。先删除 FFT/中间量和调度间隙，更有直接依据。

## 6. 后续验收应面向真正训练

最小性能矩阵覆盖 s1/2/3/4、same/nearest/独立 prior、B1/4/8/16、规则与奇数尺寸、四种核广播、实际 padding 后 shape。性能分别测全梯度、参数-only、输入-only；一阶和 `create_graph=True` 分开，因为候选高阶路径仍回退 ATen。

正确性复用现有 gradcheck/gradgradcheck、15 种梯度需求、非连续/共轭 view、stream 和原地更新测试。进一步检查弱 lambda、带符号核及近零梯度；[更广精度审计](../artifacts/accuracy_fp32_20260917/findings.md) 已发现新旧路径都有弱正则误差放大，不能为了速度放宽门槛。

最终报告真实 ConverseBlock 和完整模型的 optimizer step、samples/s、峰值 allocated/reserved、达到同等 loss/PSNR/SSIM 所需时间；统一初始化、数据顺序、优化器和精度策略。历史合成短 USRNet 的 1.26–1.31× 只能作先例，不能代替完整训练验证。小收益采用多时段配对复测，构建/预热/捕获开销单独列出。

本次最适合立即开展的实现任务是 **P0 训练融合接入与接口回归**；随后对真实训练 patch 做 **CUDA Graph 与 eager 对照**，再根据 scale 分布选择 **s1 专用融合或 nearest 可微频谱合成**。
