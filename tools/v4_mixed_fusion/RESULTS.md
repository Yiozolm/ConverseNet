# FP16 输入转换与 circular padding 融合实验

本轮选择 256 threads 的实验实现。它在限定输入域中通过精度、完整算子性能和两套预训练模型的验证，最终 **203 项 release 测试通过**；生产 FP32 接口、默认模型路径及数值门槛均未改变。

融合的范围是读取 FP16 激活并直接写入 padded FP32 缓冲区，后续 FFT、复数运算、分母、输出及参数仍为现有 FP32 实现。这不是 AMP，也没有引入低精度权重或输出。准入限于原始 Converse2D/default forward、CUDA backend、连续 NCHW FP16 输入、v7/scale=1、有限 `eps >= 1e-5`、FP32 weight/bias，以及 no_grad、inference_mode 或 frozen GradMode。circular padding 必须为正且不超过两个输入空间维度，索引须在 INT32 范围内，并关闭 autocast/TF32。合法但未命中范围的输入使用原来的转换和模块路径；不支持的激活 dtype 或非 FP32 参数仍报错，frozen GradMode 不会被强制关闭。

机器可读证据在 [results.json](results.json)，分支选择记录在 [selection.json](selection.json)，原始报告均保留在 `artifacts/v4_campaign/`。JSON 保存报告 SHA256、实际依赖源码和 checked/research build 身份、所有失败行、每轮配对时间与峰值、模型输入/输出/权重哈希；不复制所有通过的 523 个 gate 行。

## 精度与失败记录

生成候选前的原始混合输入基线通过了 16 个 shape/padding 检查。候选 primitive 覆盖 FP16/BF16、四种 padding、有限值位模式、signed zero 和 subnormal；完整模块检查还覆盖布局、广播、缓存、stream/graph、可导与 frozen GradMode、fallback 及调用方状态。

| 报告 | 完成数 | 完整矩阵 | 实际融合模块数 | 明确限定域准入 |
| --- | ---: | --- | ---: | --- |
| `mixed_fusion_gate_b256_001.json`，未限制 eps | 523 | 失败，8 行 | 66 | 未准入 |
| `mixed_fusion_gate_b256_002.json` | 523 | 失败，8 行 | 33 | 通过 |
| `mixed_fusion_gate_b128_001.json` | 523 | 失败，8 行 | 33 | 通过 |
| `mixed_fusion_gate_b512_001.json` | 523 | 失败，8 行 | 33 | 通过 |

每份报告均包含 108 个 primitive 位精确检查、384 个完整模块检查、18 个其他 fallback、8 个 training fallback、2 个 cache、2 个 stream/graph 和 1 个 invalid-contract 检查。全部 384 个模块输出与当前未修改的量化输入 FP32 模块逐位一致。

首次报告中 8 个 weak/`eps=1e-8` 样本继承了原 FP32 实现相对于冻结参考与独立 FP64 参考的 max-abs 预算失败。下表每项都在 no_grad 和 inference_mode 各失败一次；max-abs 比值上限仍为 1.50，没有调整。

| 失败样本族 | max-abs 比值 | relative-L2 比值 |
| --- | ---: | ---: |
| FP16 circular/g0 | 1.669252 | 1.158274 |
| FP16 reflect/g2 | 1.516095 | 1.058495 |
| FP16 broadcast2x3 | 2.223870 | 1.061619 |
| BF16 reflect/g2 | 1.898737 | 1.028542 |

后续适配器将弱正则化留在原模块路径。三份受限 gate 的 33 个 active 模块样本全部严格通过预算；351 个 fallback 模块样本保持有限、逐位相同、路由和状态正确。8 个旧预算失败的 baseline/candidate 完整预算字典完全相同，失败行及顶层 `passed=false`、`full_matrix_passed=false` 仍保留。`active_domain_admitted=true` 只表示这个显式分区通过，不能解释为完整矩阵通过。四次 gate 的 32 条失败观测全部收入结果，未改写历史。

## 完整算子性能与分支选择

固定门槛为两次独立重复，每次 9 轮 AB/BA，每路每轮 10 次调用：wall 和 CUDA-event 配对比值的中位数均至少 1.03，至少 7/9 个 wall 正收益轮；B4/C128 大样本还必须降低增量峰值分配。cold 路径在每个计时样本前清缓存，清缓存本身在计时外。所有 Python 调用、转换、padding、solver 和 crop 均在完整调用计时内。

首次 `mixed_fusion_perf_b256_001.json` **失败且保留**：大样本两路峰值相等，小样本 cold 正收益轮数也不足。另查明其共同 forward wrapper 改变了基线 FP32 转换缓冲区的生命周期。weakref 探针显示，原始 `module(xlow.float())` 在 solver 执行时仍持有该缓冲区，增量分配 39,845,888 B；共同 wrapper 内调用原 `forward` 时缓冲区已释放，增量分配 20,971,520 B。差值正好为 18 MiB。原测量源码完整保存在 `history/study_wrapper_control.py`，其 SHA 与失败报告一致。该失败未改名为通过，也不与后续结果合并。首次简短探针 `_001` 及随后由 `lifetime_probe.py` 独立复现、附源码 SHA/checked build 的 `_002` 均保留，两次观察相同。

修正后的基线是**未修改的** `module(xlow.float())`；候选为计时外安装 forward wrapper 后的 `module(xlow)`。两路均支付真实 `Module.__call__` 开销，候选还支付自己的 wrapper/dispatch。保留原始调用的生命周期是测量协议修正，没有改 kernel 或门槛。

| block | 完整 8 个 shape/cache 行 | wall 比值几何均值 | CUDA 比值几何均值 | B4 warm 两次 wall 比值 |
| ---: | --- | ---: | ---: | --- |
| 128 | 失败；B1/C64/96 cold 第二次仅 6/9 正收益 | 1.232764 | 1.216353 | 1.336303 / 1.250903 |
| **256** | **全部通过，选用** | **1.254803** | **1.255684** | **1.289084 / 1.254094** |
| 512 | 全部通过，未选用 | 1.277843 | 1.279604 | 1.262504 / 1.128098 |

三种 checked build 使用相同源码和工具链，仅 block 宏不同。256 与 512 的跨样本 wall 几何均值相差不足 2%；保留原来的 256 默认值，因为主要 B4 样本两次收益更强、更一致。这是有限分支选择，不是全局最优 block 的证明。

选中 256 的全部形状结果如下。表中均为两次配对比值中位数；每行均满足固定正收益轮数门槛。

| 输入及 kernel/padding | cache | wall 比值 | CUDA 比值 |
| --- | --- | --- | --- |
| B4/C128/96×96，k3/p2 | warm | 1.289084 / 1.254094 | 1.295095 / 1.272754 |
| 同上 | cold | 1.331214 / 1.238487 | 1.305827 / 1.241279 |
| B1/C64/96×96，k3/p2 | warm | 1.620628 / 1.412391 | 1.630462 / 1.417217 |
| 同上 | cold | 1.178723 / 1.124945 | 1.143583 / 1.157924 |
| B1/C64/31×37，k3/p2 | warm | 1.178060 / 1.337766 | 1.191367 / 1.246286 |
| 同上 | cold | 1.152270 / 1.265958 | 1.165837 / 1.279849 |
| B1/C64/96×96，k7/p6 | warm | 1.366190 / 1.190421 | 1.291325 / 1.175940 |
| 同上 | cold | 1.145090 / 1.090700 | 1.158179 / 1.199323 |

B4 warm 增量峰值由 123,732,480 B 降至 104,858,112 B；cold 由 131,566,080 B 降至 112,691,712 B，均减少 **18 MiB**。三份修正协议报告保留 432 个配对轮；首次失败的历史协议另保留 144 轮。

## 真实预训练整网

`mixed_fusion_models_b256_001.json` 测试 ConverseDnCNN 和 USRNet s1，各使用 6 张固定真实留出图片的 96×96 crop，并按已声明方式合成退化。两者实际 Converse2D 输入均为 **C128**：DnCNN 20 次 k7/p6 调用；USRNet 35 次 prior 调用，其 5 次 DataNet 保持 FP32。单算子 C64/k7/p6 fixture 只代表 DnCNN 的 kernel/padding 几何，不能称为其实际内部通道数。

整网比较包含原始 FP32、每个 Converse2D 输入量化后立即转回 FP32 的 mixed baseline，以及在相同位置量化后调用融合候选三条路径。共同的 FP32→FP16 cast 在整网计时内；质量探针验证真实 extension 调用数后移除，计时中无 hook/counter/profiler。

全部 12 个样本的候选与量化基线输出 SHA256 完全相同，PSNR/SSIM 完全相同，参数、调用方输入及原始 forward 恢复检查通过。每个模型在第一张固定真实样本上完成两次 9×10 整网配对计时。

| 模型 | mixed/fused wall | mixed/fused CUDA | wall 正收益轮 | 原始 FP32/fused wall，仅诊断 |
| --- | --- | --- | --- | --- |
| DnCNN | 1.206426 / 1.107143 | 1.206443 / 1.107680 | 9/9、8/9 | 1.131683 / 1.013277 |
| USRNet s1 | 1.131191 / 1.142363 | 1.130962 / 1.143151 | 8/9、9/9 | 0.974929 / 1.061590 |

两套模型均通过相对其自身量化基线的固定性能门槛。原始 FP32 对照没有显示任一模型稳定达到两次均 1.03× 的收益，因此不宣称原始 FP32 整网加速。整网峰值同样没有下降：DnCNN mixed/fused 均为 42,717,696 B，USRNet 为 38,895,616 / 39,141,376 B。整网在 forward 内量化，与单算子公开调用入口的缓冲区生命周期不同，不能将单算子的 18 MiB 节省套用到整网。

FP16 输入量化本身的质量变化仍仅为观察值，没有建立生产质量阈值：

| 模型 | 原始 FP32 平均 PSNR | mixed−FP32 平均 PSNR | 最差单样本 PSNR 差 | mixed−FP32 平均 SSIM |
| --- | ---: | ---: | ---: | ---: |
| DnCNN | 33.038680 dB | −0.000276 dB | −0.000998 dB | −0.000014696 |
| USRNet s1 | 39.139540 dB | −0.000242 dB | −0.001557 dB | −0.000000337 |

这些固定本地 crop 不是官方数据集评测，也不是训练或收敛证据。当前 USRNet 只量化 prior 的 35 次调用，不能与上一轮同时量化 5 次 DataNet 的质量结果直接混合。

## 同协议流量证据与 SASS

最终 NCU 对照使用完全相同的 `study.py` worker、B4/C128/96×96、固定 seed 和 256 checked binary，每路各一次 application replay capture，5 次 warmup 后 profile 1 次，cache-control/clock-control 均为 none。两路实际 worker 的源码、checked/research build、scope、环境和输出哈希相同，没有 FillFunctor。worker 未单独保存输入哈希，这是此 NCU 对照的记录限制；输入由同一固定源码/fixture/seed 构造，完整性能报告另保存 fixture 输入哈希。

| 完整调用 | kernel 数 | DRAM read | DRAM write | DRAM 总字节 |
| --- | ---: | ---: | ---: | ---: |
| 未修改混合输入基线 | 15 | 189,114,368 | 87,107,328 | 276,221,696 |
| 融合候选 | 10 | 156,279,296 | 66,242,304 | 222,521,600 |

完整调用减少 5 个 kernel，DRAM 字节减少 53,700,096 B，约 **19.44%**。这是单次同协议硬件计数证据，不用 profiler duration 作性能准入，也不推断整网流量。生成候选前另有 15-kernel / 271,172,608 B 记录，来自不同 baseline worker 的前置流程；原样保留，不拿它与最终候选组成对照。

选中 binary 的 SASS SHA 和原文均保留：8 个 dtype/mode specialization 中合计 8 条静态 U16 global load、10 条 global store，未发现 FP64 算术。FP16 widening 在反汇编中以 `HADD2.F32 ... -RZ` 表达，BF16 通过整数移位形成 FP32 位模式；数值与输出类型契约由独立 gate 验证，不能仅凭指令数量声称加速。

## 复现与边界

在与报告一致的 CUDA/MSVC/PyTorch 环境中使用新目录和输出名，避免覆盖历史。完整运行方法及预声明门槛见 [README.md](README.md)。一轮 256 复现顺序如下；128/512 使用独立 build 目录及对应 gate/report：

```powershell
$env:CONVERSE2D_SKIP_BUILD='1'
./tools/run.ps1 tools/v4_mixed_fusion/baseline.py --output NEW_BASELINE.json
$env:TORCH_CUDA_ARCH_LIST='12.0'
./tools/run.ps1 tools/v4_mixed_fusion/loader.py --artifacts .build/mixed_fusion/NEW_256 --build --block-threads 256 --load-production-first
./tools/run.ps1 tools/v4_mixed_fusion/gate.py --baseline NEW_BASELINE.json --artifacts .build/mixed_fusion/NEW_256 --block-threads 256 --output NEW_GATE_256.json
./tools/run_affinity.ps1 -Mask 0xC03C03 -MetadataPath NEW_PERF.affinity.json tools/v4_mixed_fusion/study.py --artifacts .build/mixed_fusion/NEW_256 --gate NEW_GATE_256.json --block-threads 256 --output NEW_PERF_256.json
./tools/run_affinity.ps1 -Mask 0xC03C03 -MetadataPath NEW_MODELS.affinity.json tools/v4_mixed_fusion/model_study.py --artifacts .build/mixed_fusion/NEW_256 --operator-gate NEW_GATE_256.json --block-threads 256 --manifest artifacts/v4_campaign/dataset_absolute.json --samples 6 --output NEW_MODELS_256.json
./tools/run.ps1 tools/v4_mixed_fusion/profile_ncu.py --worker study.py --output NEW_NCU_BASELINE --profile baseline --case b4_c128_96 --artifacts .build/mixed_fusion/NEW_256 --gate NEW_GATE_256.json --block-threads 256
./tools/run.ps1 tools/v4_mixed_fusion/profile_ncu.py --worker study.py --output NEW_NCU_CANDIDATE --profile candidate --case b4_c128_96 --artifacts .build/mixed_fusion/NEW_256 --gate NEW_GATE_256.json --block-threads 256
```

完整 gate 保留旧失败时会非零退出；继续测量前核对 `complete`、精确 scope 和 `active_domain_admitted`，下游脚本还会独立核对，不能把顶层 `passed` 改成 true。其他主机需重新选择有效 affinity mask、记录环境并单独构建测量；当前 checked build 针对 SM120。

最终 `.build/mixed-fusion-release-001.log` 记录 `Ran 203 tests in 29.979s / OK`，日志 SHA256 为 `576f9924f228f61e5f75a012c2db52c0271ca5cdd10f6a41eb46a2f2d6bc4b41`。已有生产源码及冻结 FP32/数值策略文件未修改，当前实验依赖哈希与原始报告逐项核对一致。

实验准入没有修改生产参数格式、FP32 默认路径、数值预算或原始失败分类。BF16/module 其他 padding、训练混合精度、低精度权重和生产模型质量准入均不在本轮结论内。
