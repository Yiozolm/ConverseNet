# 共享输入 s=1 的训练优化

2026-09-18，继续原生 `ConvTranspose2d(groups=1)` 训练速度目标。这里研究实际完整USRNet中的39个s1调用，不以先前k3/s3特例代替整网范围。生产默认尚未改变，原数据、λ、seed和数值预算全部保留。

**基准变更（用户最新指令）**：数值验收改为此前训练对照中的原 `backend=pytorch` 全谱FP32实现，用户进一步确认“精度和训练质量不劣于Python FP32”，不要求逐点贴合其舍入结果。用同一高精度参考度量两者误差，另检查真实训练质量；旧逐点预算和失败保留为诊断，不再作为必须通过的发布门槛。新验收使用独立新报告，不改写这些结果。

## 精确公式与固定精度筛选

仅在 `s=1 && prior is x` 时，原频域解可写为 `H*FFT(x)`，其中 `H=(conj(K)+λ)/(|K|²+λ)`。先计算H再广播到batch，避免逐batch求残差和分母；FFT、核准备、λ保持可微，无跨步参数缓存。独立prior不适用该改写。

固定原先失败的预训练权重、B4/C128/96²、pad2、eps=1e-5，对同一完整FFT FP64参考比较输出及dx/dw/db：

| 路径 | 输出最大预算比 | dw最大预算比 | 所有张量通过 |
| --- | ---: | ---: | --- |
| 原生产CUDA | 0.866 | 1.534 | 否，原dw一点 |
| FP32传递系数 | 0.774 | 0.880 | 是 |
| FP64传递系数，输入FFT仍FP32 | 0.840 | 1.052 | 否 |
| FP64传递系数及输入FFT，分别转回FP32边界 | 0.555 | 0.541 | 是 |
| 全部内部FP64，输出转FP32 | 0.00195 | 0.00112 | 是，仅精度诊断 |

预算未变：普通输出3e-5/3e-5，梯度5e-5/5e-5；只对通过路径计时。固定预训练fixture中FP32传递系数约4.295ms，提升输入FFT的版本约8.273ms，全部内部FP64约11.700ms。此表没有原生分母，不能跨表相除。

原始记录：[shared_s1_transfer.json](../artifacts/native_deconv_target/shared_s1_transfer.json)，入口：[probe_shared_s1_transfer.py](../test/probe_shared_s1_transfer.py)。额外24个CPU案例覆盖全部梯度需求、广播、非连续/退化尺寸及二阶方向导数；两候选共384组检查通过。GPU原7 normal、3 weak及固定预训练模块共11案例、两候选全部通过；这些GPU案例的一阶全VJP通过，不等同全部CPU高阶覆盖已在GPU执行。[扩展门槛](../artifacts/native_deconv_target/shared_s1_transfer_validation.json)。

## 同fixture原生速度对照

再次使用原native基线`s1_module`的seed17完整fixture哈希，B4/C128/96²/s1/k3、eps1e-5。此fixture与上面预训练失败样例不同，两项分别核验。三条Converse路径各自通过独立FP64后，warm5、六轮轮换、每路30次完整forward+dx/dw/db；TF32关闭、cuDNN deterministic=True、global deterministic=False，与这一原生基线设置一致。没有编译或CUDA Graph。

| 路径 | 完整前后向中位数 | peak allocated |
| --- | ---: | ---: |
| 生产FFT | 5.291 ms | 216.28 MB |
| FP32传递系数，原pad/crop | 4.753 ms | 203.22 MB |
| FP32传递系数，cat pad/view crop | **4.141 ms** | **203.22 MB** |
| 原生dense | 3.899 ms | 164.76 MB |

组合/原生耗时配对中位数1.062，范围1.048–1.180；生产/组合配对加速中位数1.275，范围1.266–1.303。原生第三轮3.514ms明显快于其余轮，保留全部数据。参数量与数学不同，原生仅作速度标杆；本次空间模块候选结果不能与先前s1 kernel比率相乘。[完整对照](../artifacts/native_deconv_target/shared_s1_transfer_benchmark.json)，[入口](../test/benchmark_shared_s1_transfer.py)。

将传递函数改为实虚分解后交Inductor融合，并未可靠通过原门槛：FP32 eager/compiled的dw最大预算比1.555/1.104，较高精度compiled仍为1.031；所有失败均停止该路径计时，没有调整数值尺度或门槛。说明“某种运算顺序使一个样例通过”仍不足以直接发布。记录：[compiled_s1_transfer.json](../artifacts/native_deconv_target/compiled_s1_transfer.json)。

## 完整模型精度定位

严格重建原seed9214、full5/7、alpha=.1、LR4×5/s2的压力fixture，全部初始参数、x、kernel、upstream与历史归档逐位核对。不是预训练模型或真实图训练。输出及全部135个梯度（共136张量）使用原3e-5/3e-4预算，记录所有失败点。

| 路径 | 失败张量 | 失败元素 | 最大预算比 |
| --- | ---: | ---: | ---: |
| 当前生产 | 16 | 26 | 41.123 |
| 39次s1改为FP32传递系数 | 13 | 23 | 35.017 |
| 39次s1提升系数/输入FFT精度后返回FP32边界 | 14 | 26 | 35.990 |
| 39次s1全部内部FP64、输出FP32 | 15 | 28 | 40.654 |
| 全部40个Converse内部FP64、输出FP32 | 14 | 25 | 39.148 |

全部路由覆盖均核对：35个prior k3、4个DataNet s1/k7以及1个DataNet s2/k7。旧生产控制在独立诊断中精确复现。提升所有求解器内部精度仍保留大部分失败，不能把整网差距单独归因于CUDA谱核；边界舍入、生成核的网络和其他模型层仍参与误差传播，尚未证明某个其他算子有bug。

[传递系数整网门槛](../artifacts/native_deconv_target/shared_s1_transfer_model.json)、[全FP64求解诊断](../artifacts/native_deconv_target/shared_s1_model_precision.json)。这些检查正常返回失败状态，不跳过或标为预期通过。先前s1专用核+边界的三种子训练通过不适用于这里新传递系数的训练轨迹。

后续隔离诊断显示：同时将KernelNet、五个核投影层及五次DataNet求解提升为FP64，保留prior为原FP32路径，原136项检查全部通过、最大预算比0.747；再提升35个prior求解后最大预算比0.682。这个实验不能单独归因于KernelNet，也没有证明相应性能收益。[生成支路诊断](../artifacts/native_deconv_target/kernel_generation_precision.json)。用户随后指定原Python FP32作为验收基准，因此不再把这项额外精度提升当作优化的前置条件。

下一步按新Python FP32基准验证共享系数CUDA候选及完整模型，并检查真实训练质量与同条件速度后决定生产分派。真实训练收敛及time-to-quality要求继续保留。

## 新Python FP32基准下的解析CUDA候选

新增隔离的[共享系数CUDA核心](../experiments/training_shared_s1/bindings.cpp)：按核batch共享H和分母，手写一阶频谱VJP，B/C和λ归约确定次序并以FP64累加，高阶回退可微ATen。FFT、核准备和λ仍由autograd处理，无参数/图跨步缓存。原GPU正常/弱样例通过，旧FP64预训练逐点检查仍有1点dw超限（最大预算比1.466），保持失败状态。[原FP64诊断](../artifacts/native_deconv_target/shared_s1_cuda_validation.json)。

另完成45项内部契约检查：4种核广播的complex128 gradcheck/gradgradcheck、FP32全部7种梯度需求的手写核分派、选择性高阶、共享输入别名、非默认stream、保存输入版本异常及三次逐位重复全部通过。任意频谱的数学/求导检查不等同完整空间算子或训练质量验收。[契约报告](../artifacts/native_deconv_target/shared_s1_cuda_contract.json)。

用户确认精度非劣后，新协议在计时前写明：对同一FP64参考，output/dx/dw/db逐张量的**最大绝对误差和相对L2均不大于原Python FP32**；不添加百分比余量，FP32表示舍入下限及旧逐点差异仅作诊断。这个严格筛选保留全部未通过路径；只对同fixture独立通过候选计时。[协议](../experiments/training_shared_s1/python_fp32_noninferiority_protocol.json)。

原native s1_module/seed17六轮×30次中，FP64核准备的共享CUDA候选通过上述精度筛选；改为FP32核准备的候选未通过，未计时：

| 路径 | 完整前后向热中位数 | peak allocated |
| --- | ---: | ---: |
| 生产FFT | 4.974 ms | 216.28 MB |
| 共享H CUDA，FP64核准备，cat/view边界 | **3.639 ms** | **190.08 MB** |
| 原Python FP32 | 11.225 ms | 483.28 MB |
| 原生dense ConvTranspose2d | 3.528 ms | 164.76 MB |

这是单个固定形状的直接同条件结果，非整网训练或全部形状的精度声明。完整逐轮记录：[shared_s1_cuda_noninferiority.json](../artifacts/native_deconv_target/shared_s1_cuda_noninferiority.json)。先前对Python FP32逐点对齐失败的报告另行保留，不改写为通过：[空间差异](../artifacts/native_deconv_target/shared_s1_python_fp32.json)、[完整模型差异](../artifacts/native_deconv_target/shared_s1_python_fp32_model.json)。

完整模型的136张量采用同一FP64参考、与Python FP32误差直接比较后，当前生产有6张量、ATen共享H有4张量、CUDA共享H有8张量至少一个误差指标高于Python；CUDA路径其中只有1张量的相对L2较高，其余主要是最大绝对误差的局部峰值。大部分输出/梯度精度改善，但仍不能声称每项均非劣，候选尚未接默认，也未把旧组合方案的真实训练结果挪用过来。[逐张量精度证据](../artifacts/native_deconv_target/shared_s1_accuracy_relative_python.json)、[12组算子精度筛选](../artifacts/native_deconv_target/shared_s1_accuracy.json)。

## 后续分解及真实训练

保留原Python的全谱FFT路径，能使共享H在大多数固定算子样例中的误差降低，但同fixture完整前后向为8.850–13.585ms（原生3.838ms），未达到速度目标；整网仍有5–8张量某项误差指标较Python高。因此暂不采用这一替换，记录全部负结果：[全谱算子精度](../artifacts/native_deconv_target/shared_s1_full_fft_accuracy.json)、[全谱整网精度](../artifacts/native_deconv_target/shared_s1_full_fft_model.json)、[同条件计时](../artifacts/native_deconv_target/shared_s1_full_fft_benchmark.json)。空间恒等项分离的残差表达也未在全部12样例严格非劣：[残差实验](../artifacts/native_deconv_target/shared_s1_residual_accuracy.json)。

核生成精度进一步分解得到更小的修改：KernelNet和5个核投影用可微FP64生成核，核FFT保持原FP64→c64准备，**DataNet的激活FFT、λ、频谱求解和IFFT仍为FP32**。生成核及其回传保持FP64，最终模型参数/梯度和激活/输出保持FP32。这条路线的完整136张量max_abs与relativeL2均不高于Python FP32；若在DataNet入口先把生成核转回FP32，则有12张量某项指标较高，不能提前截断这条精度路径。[分解记录](../artifacts/native_deconv_target/kernel_solver_boundary.json)。它仍有5个旧FP64逐点失败，只作诊断，不影响按用户新基准记录的非劣结果。

进一步组合“5次mixed DataNet + 35次共享prior CUDA”，对同一完整模型的136张量同样全部满足两项零余量非劣比较，且实际调用数与梯度dtype均检查通过。[组合精度记录](../artifacts/native_deconv_target/mixed_kernel_shared_prior_accuracy.json)。该组合尚待完整训练质量与成本验证，不能继承仅共享CUDA方案的训练速度。

仅共享CUDA方案已完成seed17真实图250步：与Python和新current控制的逐批/初始化/配方/验证payload全部核对一致，RGB/Y PSNR及SSIM通过原门槛；对Python的Y PSNR差为+0.00002063dB。新同适配器current/候选训练循环135.872/126.864秒，单对观察约1.071×；训练peak allocated 9.736→8.928GB。no_grad验证全部使用同一原生产路径。旧历史Python时间不参与这次配对速度比。[独立质量审计](../artifacts/native_deconv_target/shared_s1_quality_seed17_audit.json)。不能把这一单种子结果当作完整收敛或新mixed组合的质量证明。

## mixed组合三种子结果与接入状态

最终mixed组合及同适配器current控制各完成seed17/29/43、250步，总计1500个新的optimizer updates。对历史Python FP32和新current共6对比较全部通过原0.05dB/0.001质量门槛，完整性错误0；每对250个训练batch、100张验证图、初始化和配方一致。每次训练forward核对35个SharedTransfer、5个mixed DataNet和35个边界变换；五个生成核VJP为FP64，133个主参数及最终梯度为FP32。no_grad的300次验证forward保持原生产路径。

对Python的三种子Y PSNR差分别为+0.00003060、−0.00000352、−0.00000783dB。新current/组合的完整循环配对比分别1.111、1.085、1.093，中位数1.093；预热step中位配对比约1.12–1.13。这里是三次顺序配对观察，包含验证/I/O的loop与step分开记录；没有用历史Python耗时跨表计算加速比，也没有证明完整收敛时间。[六对最终审计](../artifacts/native_deconv_target/mixed_shared_three_seed_final_audit.json)、[训练入口](../test/train_usrnet_mixed_shared_candidate.py)。

可重用函数已抽取到[models/converse_training.py](../models/converse_training.py)，不依赖test或experiments。CPU24项及实际CUDA33项的输出、普通VJP和方向HVP与冻结实现逐位一致，包含dtype、非连续、广播、边界与输入guard。[抽取验证](../artifacts/native_deconv_target/training_helpers_cuda.json)。共享CUDA核心另以仅命名空间/宿主符号/注册名变换复制到主库的`converse2d_shared_s1.cpp/.cu`，逆向映射后与隔离源逐字节相同；**这些新文件尚未加入主库构建，模型主调用也尚未改默认分派**。

为防止将单层结果外推，另外冻结原Python FP32实际MSE训练的40个调用，保留7组prior参数各5次引用和DataNet bias的5次引用。正式重放一次性完成40个forward及60个唯一目标的VJP，不按单算子计时相加；原生动态权重分别测per-sample loop和batch-folded groups=B，全部输出/VJP等价检查通过。新零余量局部筛选中，current有32/100个张量、仅共享prior候选有36/100个张量至少一项误差指标较Python高；它们被排除计时。实际只得到Python约430.0ms、native loop159.6ms/folded154.2ms。因此**尚无通过该局部筛选的优化路线40-op/native整体速度比，不能声称整体目标已完成**。这一重放切断了层间依赖与核生成网络，不能替代上面的完整mixed模型精度/训练质量结论。[捕获](../artifacts/training_research/operator_workload40/python_seed17_step0/capture.json)、[40-op报告](../artifacts/training_research/operator_workload40/benchmark.json)。
