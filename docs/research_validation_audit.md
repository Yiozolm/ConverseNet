# 研究数值门槛的独立只读审计

> 中途快照：2026-09-25 06:26:55 UTC。下文保留当时的报告、未完成项及覆盖缺口，不回写为后来结果。最终候选状态、后续 GPU 验证及生产准入见 [全清单最终报告](fp32_roadmap.md)；研究分支的 `research/RESULTS.md` 和 `research/evidence/index.json` 保存完整归档。

审计对象为本次八小时活动已完成的研究报告和 `tools/summarize_fp32_roadmap.py`。没有运行 GPU、重编译或修改任何已测 candidate/harness。CPU 校验器 [audit_research_reports.py](../artifacts/fp32_roadmap/audit_research_reports.py) 重新计算 **18 组报告**的逐指标门槛、case/张量计数和顶层状态，结果为 **0 处不一致**；[research_validation_audit.json](../artifacts/fp32_roadmap/research_validation_audit.json) 保存原始报告 SHA、重算结果及源码对应关系。这是重算报告中的误差数值，不是重新执行张量计算。

## FP64 输入与门槛

已读路径均从已生成的 FP32/complex64 输入提升到 FP64/complex128，并让候选与 Python FP32 对照使用同一个高精度结果，没有另用独立随机 FP64 输入。各路径的实际覆盖和限制如下。

| 路径及源码位置 | 同输入的 FP64 参考 | 数值判定 |
|---|---|---|
| `research/run_algorithms.py:32,84,92` | CPU 默认 FP32 生成 x/prior/kernel/bias/upstream；三次 capture 分别提升同一 raw/upstream，FP64 调用独立 `models/converse_core.py:37` | 每个 output、dx、独立 dprior、dweight、dbias 分别要求 max_abs 与 relative_l2 均不大于 Python FP32；无额外裕量 |
| `research/fftfree_inference/study.py:35,99,109` | padding 后的同一 FP32 LR、核、bias `.double()`；nearest 只复制输入值 | 完整 padded output 与 crop 各自双指标；还检查重复性、输入不变及 CUDA 残差原型逐字节一致 |
| `research/nearest_coefficient.py:45` | 复用上行同输入参考 | 新 coefficient 公式有独立 admission；要求双指标、重复性、输入不变和 crop，不要求与另一个残差公式逐字节相同 |
| `research/fft_backends/cufftdx2d_study.py:28,68,108` | primitive 使用同一 complex64 升 complex128；shared-s1 使用同一 padded FP32 x/weight/bias 升 double | primitive forward/inverse/roundtrip 各自双指标；算子 padded/crop 各自双指标；记录中有重复性条件 |
| `research/pointwise_wgrad/study.py:100,112` | 同一 activation、checkpoint weight/bias、grad_output 升 double，独立 native `F.conv2d` 前后向 | 每个 output/dx/dweight/dbias 双指标、零额外裕量；高阶另用已有 ATen `3e-5` 容差，不将其称为零裕量 FP64 门槛 |
| `research/fft_backends/probe_lto.py:101` | 已量化 complex64 频谱升 complex128 后做 inverse FFT；不是重新从 FP64 原图生成频谱 | 每个执行 route 的 max_abs/relative_l2 单独记录；有失败时 primitive 计时仍保留为研究诊断，不授予模型资格 |
| `research/fft_backends/probe_cufftdx.py:58` | 同一 complex64 输入升 complex128，独立 `torch.fft.fft` | 每个 1D primitive 的两个误差指标各自非劣；没有模型或 VJP 资格 |

`run_algorithms.py:100` 的旧门槛显式检查 candidate finite，但没有显式把 Python FP32/reference finite 同时并入布尔条件；本次记录中的相关误差和 Python FP32 均有限，CPU 采用更严格的双侧 finite 检查后也没有任何结果改变。它也只记录三个研究文件的源 SHA，没有把 `models/converse_core.py` 的当时 SHA 和 raw/upstream hash 写入报告。本次这三个记录 SHA 与现有文件相同，但不能事后补称旧报告已有完整参考/输入指纹。若将来扩展至溢出等输入，需另开版本加强有限性与 provenance，不能静默修改已测 harness。

没有发现用“更接近参考的张量占多数”替代逐张量门槛的行为。FP64 只出现在独立参考或离线误差计算，不是这些候选的生产计算精度。

## 原始失败数逐项核对

这里区分 case 和 output/VJP 张量。所有未通过的候选继续为未通过；数学等价、模型激活样本通过或某尺寸更快均未覆盖掉失败。

| 研究候选 | case 数 | 失败 case | 受检张量数 | 失败张量 |
|---|---:|---:|---:|---:|
| disjoint k2/s2 ATen | 16 | 11 | 72 | 15 |
| shared s1 transfer ATen | 8 | 8 | 32 | 23 |
| nearest 频谱生成 ATen | 24 | 22 | 96 | 45 |
| 直接小核 DFT | 48 | 46 | 216 | 140 |
| 半谱训练 | 48 | 43 | 216 | 114 |
| 混合空间训练求解 | 48 | 42 | 216 | 119 |
| 同 forward 内核频谱复用 | 48 | 33 | 264 | 33（全部 dweight） |
| 原 full-K GEMM 1×1 wgrad | 22 | 5 | 80 | 5（全部 dweight） |

半谱训练和训练频谱复用仍只属于用户授权的隔离研究例外，不能依据上述运行或数学推导进入发布分支。

| 推理/FFT 研究报告 | 受检 case | 失败 case | 进一步拆分 |
|---|---:|---:|---|
| checked CUDA FFT-free nearest | 180 | 17 | synthetic 144 中失败 17；checkpoint 激活 36 中失败 0 |
| 新 nearest coefficient ATen | 180 | 14 | synthetic 144 中失败 14；checkpoint 激活 36 中失败 0 |
| cuFFTDx 2D 32×40 | 126 | 53 | 72 primitive 中失败 16；54 shared-s1 算子中失败 37 |
| cuFFTDx 2D 36×44 | 126 | 41 | 72 primitive 中失败 20；54 shared-s1 算子中失败 21 |
| cuFFTDx 2D 100×100 | 126 | 67 | 72 primitive 中失败 26；54 shared-s1 算子中失败 41 |

coefficient 是明确命名的新公式。它有 **144 个 case** 不与旧残差原型逐字节相同，这些原始检查仍保留；其自身按同一 FP64/Python FP32 门槛有 14 个失败。不能把它的 `check.passed`（包含另一个公式的 byte 条件）与 `coefficient_admission` 混算，也不能将其称为旧 FFT-free 17 个失败已经修复。

cuFFT LTO 的四个尺寸中，plain route 均通过两指标；LTO 在 **32×40 的 relative_l2 失败**，256²、260² 两指标通过，381×387 按已声明的质因数限制没有执行 LTO。未执行不是通过，也不是数值失败。cuFFTDx 的 1D 64/36/44/100 各只有一个 `[32,N]` primitive fixture，两指标通过，均不能覆盖上述 2D 失败。

原 GEMM 的 5 个失败为 `synthetic/B1/c128_to_64/dweight`、`synthetic/B4/head_c3_to_64/dweight`、`captured/conv1/dweight`、`captured/p.m_body.0.conv1.1/dweight`、`captured/conv2/dweight`。22 个 case 中只有 **18 个真正启用候选**；B1/B2 的小 head 和两个 input-only case 回退 native，不能把回退通过当成优化路径通过。全部门槛失败使完整模型替换和正式计时保持阻断。

## 输入矩阵与未覆盖范围

- 七个 ATen 算法原型固定 `B2,C3,LR5×7`，四种 kernel broadcast、正常/弱正则；scale 集合依候选为 s1、s2、s2/3/4 或 s1/2/3。除 transfer 明确 shared 外，其余多为 independent/nearest prior。所有需要的叶梯度一起请求，没有穷举 mask；主要是连续输入、无外围真实模型 padding、无真实模型激活。没有从这份 numerical sweep 得到多流/Graph 生命周期、高阶或数据集质量结论。其他研究单元测试若有覆盖，不能混为这次 gate 的 fixture。
- 两个 nearest 扩展 sweep 用 17/29/43 三种子；synthetic 的 LR 为 1×1、5×6、7×5，四种广播和 positive/signed/zero/weak；布局循环分配到 contiguous/transpose/strided/channels_last/negative_view/expanded，并非所有形状、布局和模式的笛卡尔积。共 no_grad 108、inference_mode 72；例如 channels_last/expanded synthetic 只分到 inference_mode，negative_view 只分到 no_grad。checkpoint 36 个样本来自完整 SRResNet 的 up1/up2 激活，但刺激是 24×28、13×17 的 seeded synthetic RGB，**不是数据集质量**；只覆盖 no_grad 和三种布局。实际 replicate padding 后 LR/HR 尺寸已记录。候选只做推理，梯度 mask、高阶训练不属于资格范围。
- checked CUDA FFT-free 有独立 side-stream/Graph 更新及拒绝 GradMode 的 contract 记录，但 coefficient 原型没有同一组独立 lifecycle contract 报告。两者均因总 gate 失败而没有执行允许替换 up1/up2 的完整模型阶段；36 个 checkpoint 局部激活通过不能改称完整模型通过。
- cuFFTDx 2D primitive 覆盖三种子、complex/real/weak/impulse、contiguous/transpose、forward/inverse/roundtrip。算子部分只覆盖 **shared prior、s1、k3、C128、B1/B2、circular padding=2**；实际 FFT 尺寸为报告名中的尺寸，核 FFT 仍逐调用用 ATen。它没有独立/nearest prior、训练 VJP/mask、高阶、完整模型、Graph/多流资格；算子部分也没有复现 primitive 的非连续布局矩阵。
- 1×1 GEMM 对真实通道 64→128、128→64、64→3、3→64 做 B1/B2/B4；transpose/channels_last 只覆盖 B2 的64→128；subset 仅 B4 两主方向的 weight-only/input-only，不是全部七种非空 mask。另有当前 USRNet B4 捕获的四个模块首次调用样本，未覆盖 147 次 wgrad 的全部调用、每个 seed/训练阶段或数据集分布。二阶小 fixture 通过既有容差，不能当作全 mask 三阶/Graph 验证。
- LTO 仅 C2R inverse+normalization，clone 输入以隔离 cuFFT 可能的输入覆盖；每 route 的必要 clone 在计时内，但没有 Converse 准备、求解、VJP、Graph 或模型。1D cuFFTDx 是预分配输出，而 Python `torch.fft` 为函数式分配；报告已明确这个分配差异，不能把它当完整算子公平加速比。

## 汇总脚本复核

`tools/summarize_fp32_roadmap.py` 是证据索引，不是新的数值或质量准入器。CPU 实跑初版及修正版分别保存在 `research_audit_index_probe.json` 和 `research_audit_index_probe_v2.json`；13 项已完成研究、8 组比较均可读，缺少的 `fixed_transfer_numeric.json` / `full_layernorm_numeric.json` 显式列入 missing，未改判通过。

审计提出并由主任务修复了三处摘要问题：失败计数增加单位，pointwise 同时报 80 个张量和失败 case；继承 algorithm/FFT-free 的 scope/provenance，明确 pointwise 是算子/VJP 门槛；不再把未来固定 affinity 的比较一概标作未固定。修正版已再次 CPU 执行确认。原七算法数量保存在 `tensors` 字段，其余统一条目的 `tensor_count` 只在适用时填写，读取者须结合 `failure_unit`。

索引仅收录明确登记的报告，不自动证明研究目录所有文件均被列入。初始 `lto_compile_01`、`dx36_compile_01`、`dx44_compile_01` 的 compile_failed 记录仍在隔离目录，不能将后续成功编译称作旧失败通过。新的 split-sum 或其他报告产生后，需要按实际输出名登记；缺报告不是未执行分支的数值证明。外层 affinity/campaign 协议也须与性能记录一起读取。

## 未执行的 split-sum 假设

另只读审查了 `research/pointwise_wgrad/split_sum_candidate.py`：Cout/Cin 维分别固定后以同一 BHW 顺序展平，按 K=256 分块，尾部在两矩阵同时补零，`bmm` 得 `[blocks,Cout,Cin]` 后以 FP32 sum 合并。必要 reshape/copy/pad 均在 VJP 内，高阶在此前回退完整 ATen，没有训练谱复用或 FP64 生产路径。静态审查未发现数学/布局阻断；这是新舍入假设，**尚无 GPU 数值结果**，不覆盖原 GEMM 的失败。

其 dbias 使用 ATen `sum((0,2,3))`，并非原生 convolution_backward 的 bias 分支，仍须逐张量门槛。原复用 study 的 BHW 全能被 256 整除，不能据此覆盖新 tail-padding 分支；已建议加入满足候选规模但不整除的尺寸及非连续 upstream。此建议不构成执行或通过记录。
