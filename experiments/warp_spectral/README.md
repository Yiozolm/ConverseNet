# Converse2D 如何进入 warp 级优化：一次实际实验

日期：2026-09-16。结论是先围绕 alias 组建立数据所有权、融合重复访问，再比较线程与 warp 的组织方式。本次大尺寸样例的频谱阶段获得约 1.2–1.4× 加速；专门分配 loader/compute warp 没有普遍超过简单融合。实验没有接入生产 dispatch。

## 当前算子与三个层次

当前 [alias_correction](../../Converse2D/torch_converse2d/converse2d_kernels.cu) 每线程处理一个低分辨率半谱频点，串行计算 s² 个 alias，写出 q；apply_correction 再读取 prior、filter、q 写输出。warp 内相邻线程在多数内部区域访问相邻频点，但没有跨线程数据交换，也没有跨 warp 角色分工。

三个层次应分开理解：

| 层次 | 本实验中的具体动作 | 成本与目标 |
|---|---|---|
| 线程内融合 | 一线程读入 4 对 prior/filter，保留到 q 算完并直接更新输出 | 减少全局访问，增加暂存寄存器 |
| Warp 协作 | 32 lanes 处理 8 个相邻低频点 × 4 个 alias；shuffle 汇总、广播 q | 分摊数据和地址工作，支付 shuffle 成本 |
| Warp specialization | 一个 warp 搬数据，另一个 warp 求解/写出；双缓冲交接 | 尝试重叠，新增 shared memory 和同步 |

第三种才是本次指定技能所分析的流水线。NVIDIA 的定义同样强调把同一 CTA 内不同 warp 分配给不同工作；双缓冲需要分别表示“可填充”和“已填充”的同步状态。[CUDA 异步 barrier 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#producer-consumer-pattern-using-barriers)

## 本次实现的边界

独立实验命名空间 `torch.ops.warp_spectral`，限定 **FP32/complex64、scale=2、独立 prior、动态计算功率分母、推理**。基线直接 include 当前生产 CUDA 文件，调用原始 spectral 函数；所有版本在同一次构建中编译。没有复制后手工简化基线。

- `mode=0`：现有两遍 kernel。
- `mode=1`：线程内融合，256 threads/block。
- `mode=2`：warp 协作融合，128 threads/block，`lane%8` 选择低频点，`lane/8` 选择 alias。按原顺序汇总四个贡献，所有尾部 lanes 参与 shuffle。
- `mode=3`：64 threads/block，一加载 warp、一求解/写出 warp，每 CTA 处理 8 个 tile，每 tile 32 组，两个 shared buffer。使用普通同步 global load 与 `cuda::barrier`，未使用 cp.async/TMA。

所有融合版本额外启动一个边界 kernel，因此本轮仍为两次 launch，收益不是简单“少启动一次 kernel”。

Hermitian 所有权是实现核心。只让 LR 内部列 `0<w<W/2` 参与主融合；一个 `(h,w)` 拥有四个 HR alias。若 HR 列超出存储半谱，同时反射行列并共轭结果。LR 的 DC/Nyquist 对应 HR 边界列独立逐输出重算，避免多个所有者争写。奇数 W 的边界是 `{0,W}`，偶数 W 是 `{0,W/2,W}`。W=1/2 全部由边界 kernel 处理。

NVCC/ptxas 本轮报告：thread fused 40 registers/thread，warp cooperative 36，specialized 40；三者没有 register spill。specialized 的 ptxas smem 报告为 4896 B，最终二进制的 cuobjdump SHARED 字段为 5920 B；后续 occupancy 预算应使用运行时资源查询，不能直接只算源代码数组大小。减少寄存器并未自动决定速度胜负。

## 预测 → 测量 → 解释

测量前的 [prediction.md](prediction.md) 保存了资源模型和可证伪假设。内部组的逻辑 global 访问从约 **212 B 降至 108 B**；specialization 另增加 **152 B shared 访问/组**。这些是源码访问量，不是实测 DRAM 字节；缓存、广播及边界重算会改变实际流量。212/108 的比值不能当作预计加速比。

因为未校准本机单 SM 的有效吞吐，测前保留 `work/throughput` 公式，没有伪造绝对周期和 idle 百分比。预测是：融合最可能获益；warp 协作未必超过线程融合；显式分工不会普遍超过简单融合。

### 关闭打点的多 CTA A/B

硬件 RTX 5060 Ti（SM120、36 SM）；PyTorch 2.11.0+cu130；实验编译 CUDA Toolkit 13.2、MSVC 14.44，目标 sm_120。CUDA 13.0 在本机发生 cudafe++ 内部 ACCESS_VIOLATION，13.2 构建成功。运行库版本与编译器版本分别记录，未把驱动显示的 CUDA 版本作为编译器版本。

相同输入、7 轮轮换执行顺序、CUDA Event 中位数；每个 CUDA Graph 含 16 次求解，每轮 replay 20 次，除以总调用数。编译、FFT plan 初始化和 capture 不计入稳态。桌面 WDDM 环境未锁频，原始轮次全部保留，部分样例波动明显。小幅领先不能用于稳定排名。

**仅频谱求解，单位 μs，越低越好：**

| B×C×H×W（LR） | 原两遍 | 线程融合 | Warp 协作 | Warp 分工 |
|---|---:|---:|---:|---:|
| 1×3×32×40 | 2.70 | 3.80 | 3.35 | 6.93 |
| 1×64×128×128 | 245.85 | 174.89 | 172.94 | 184.24 |
| 1×32×127×129 | 45.70 | 42.38 | 45.96 | 41.08 |
| 8×32×128×128 | 1018.61 | 790.53 | 826.81 | 853.73 |

第二行 warp 协作相对基线约 1.42×，但与线程融合只差约 1.1%，小于本轮波动。第四行简单线程融合中位数最好。第三行各候选差距和波动相近，不能据此宣布分工版稳定更优。小图三个新版本均退化。

**三次 RFFT（Y/prior/filter）→ 求解 → IRFFT，单位 μs：**

| B×C×H×W（LR） | 原两遍 | 线程融合 | Warp 协作 | Warp 分工 |
|---|---:|---:|---:|---:|
| 1×3×32×40 | 18.87 | 20.28 | 20.67 | 24.22 |
| 1×64×128×128 | 557.83 | 515.23 | 535.03 | 528.98 |
| 1×32×127×129 | 622.26 | 606.20 | 631.27 | 609.15 |
| 8×32×128×128 | 4274.67 | 4039.29 | 4039.42 | 4159.16 |

这张表是合成的数学流水线，不是公开 Converse2D API、固定核缓存路径或完整 USRNet 计时；不含小 PSF 的准备、bias→lambda、模型其他层。输入是实数随机张量及完整空间随机 filter，测试独立 prior。FFT 库调用仍占主要耗时，频谱加速缩小到流水线中位数约几个百分点；需要更稳定复测才能判断这些小差距。

### 单 CTA 时间线

![Measured warp pipeline](../../artifacts/warp_spectral/timeline.png)

Profile 使用 B=C=1、LR=16×65，共 512 个内部组，单 CTA、16 iterations。每个 warp 仅 lane 0 写 `clock()`；两角色在同一个 SM 上。普通版用编译期 `PROFILE=false` 移除打点。

事件 0/1 包围 `arrive_and_wait()`，2/3 包围工作；loader 事件 4 表示 filled 发布后，consumer 事件 5 表示 ready 返回后。蓝箭头准确连接 tile i 的生产→消费，橙箭头连接 tile i 消费→tile i+2 缓冲复用。绘图按迭代配对，使用 uint32 位模式和模差处理 clock 回绕，没有把不同 SM 的原始时钟相减。

5 次 trace 的总跨度变异系数为 **0.50%**，稳态约 **2573–2577 cycles/tile**。图选择总跨度居中的 trace_2：

| 角色 | 测前绝对周期预测 | 实测中部工作区间中位数 | 等待区间 / 整段 trace |
|---|---|---:|---:|
| Loader | 有效吞吐未校准 | 2201 cycles/tile | 4.61% |
| Compute/store | 有效吞吐未校准 | 915 cycles/tile | 54.86% |

两角色工作区间的交集覆盖 loader 工作区间的 **27.87%**。消费者多数时间等 filled，而 loader 很少等缓冲复用，说明**这个原型、这个单 CTA 样例**的节拍主要由加载侧决定。加载区间还包括 Hermitian 地址计算、global load 延迟及 shared store；本次不能进一步断言其中哪一项主导，更不能将其等同于整 GPU DRAM 带宽饱和。

模型方向与 A/B 相符：减少重复访问有益，新增 shared/同步的分工没有普遍优势。测量也给模型补上了此前未量化的索引和交接成本。2201/915 是测后标定，不能冒充测前预测；其与 2574 cycles/tile 的差距还含发布、等待调用、打点、未覆盖指令及 fill/drain 的影响。

灰色斜线是等待调用内部的**经过时间**，包含 barrier 本身与打点开销，并非可直接消除的硬件 stall；浅灰仅表示没有成对打点。工作区间重叠也不是“两个 warp 同周期发射指令”的证明。发布点写在 arrive 之后，可能晚于真实解锁时刻；本次选中的 trace 未出现这种标记倒序。

## 正确性与尚未覆盖的范围

152 项数值比较通过，覆盖 singleton、奇偶尺寸、尾部 warp、四种 filter B/C 广播组合和 λ=1e-5/1e-2。FP64 参考从实数输入独立构造 full FFT 闭式解。最大频谱 relative L2 误差 **2.39e-7**，最终图像最大绝对误差 **1.55e-6**；新版本相对 FP32 基线的最大频谱差 **4.58e-5**。较大性能样例另外与 FP32 基线逐点核对。

这不是全域精度证明。未覆盖 FP16/BF16/FP64 kernel、scale=3/4、训练反传及 gradgrad、近退化滤波器、次正规数、非连续输入、自定义 stream 生命周期或完整预训练模型。实验入口明确拒绝非连续/需梯度输入。未运行 Compute Sanitizer，未在本轮重新采集 NCU DRAM 字节及 occupancy。

## 下一步应如何推进

1. **先把融合候选做扎实。** 扩展 cached denominator 与实际模型输入；按尺寸比较线程融合/warp 协作，保留小图现有路径。扫描 block size 和每 warp 组数，配合新 NCU 的 DRAM bytes、sector 合并、寄存器与实际 occupancy 决定 dispatch，而非只追求高 occupancy。
2. **再研究加载侧。** 对本原型先拆出地址计算、global→shared 及 barrier 成本，确认是延迟、访存合并还是指令开销。只有可重叠工作和复用能覆盖 staging 成本时，再试 cp.async/TMA、增加 loader 或调整 tile；不能由 55% 等待直接推导需要更多计算 warp。
3. **scale=3 重新设计 lane 分配。** 9 个 alias 不能照搬 4-lane 组。可试每 warp 8 个相邻频点、4 个 worker 分别处理 3/2/2/2 个 alias，并与线程内展开对照。
4. **训练作为单独问题。** 训练前向需保存 q/d，反向还有 Hermitian 边界重数和共享滤波器的 B/C 梯度归约。可独立研究 warp 合作归约广播梯度，但不能直接移植本推理融合并声称完成训练优化。

SM120 与数据中心 SM100 的资源不同，硬件预算应读取设备属性；本实验没有套用其他 GPU 的 Tensor Core/SM 吞吐数字。[NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)

## 复现与文件

在 CUDA/PyTorch 开发环境中运行：

```text
python experiments/warp_spectral/study.py
python experiments/warp_spectral/plot_trace.py
```

本机 PowerShell 入口（参数默认值对应本机路径，可覆盖）：

```powershell
& .\experiments\warp_spectral\run.ps1 -TaskArgs @('experiments/warp_spectral/study.py')
& .\experiments\warp_spectral\run.ps1 -TaskArgs @('experiments/warp_spectral/plot_trace.py')
```

构建输出隔离在 `.build/warp_spectral`。原始 [results.json](../../artifacts/warp_spectral/results.json) 保存源码 SHA256、环境、误差、全部计时轮次；[timeline_stats.json](../../artifacts/warp_spectral/timeline_stats.json) 保存 5 次 trace 统计；同目录保留原始 NPZ。`artifacts/` 被仓库忽略，分享实验时需要一并带上这些结果。

方法依据用户指定的 [warp-specialization-report 技能](C:/Users/Boyce/.codex/skills/warp-specialization-report/SKILL.md)。绘图脚本沿用其工作/等待/未测量三态规则，并针对本实验增加精确 tile 依赖及 uint32 回绕处理。
