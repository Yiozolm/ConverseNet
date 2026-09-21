# Converse2D 对齐原生转置卷积性能：实验记录

2026-09-18。用户已明确目标为 `torch.nn.ConvTranspose2d`。默认以 `groups=1` 的常规转置卷积为主要速度标杆，并单列 `groups=C` 的深度转置卷积。两者与 Converse2D 的数学定义、边界、参数量、bias 含义不同，速度对照不能作为数值替换依据。保留 Converse2D 的独立 FP64、弱正则和训练质量门槛；本轮不涉及低精度。

**后续用户指令**：以原Python算子的FP32精度作为数值基准，并明确选择“精度和训练质量不劣于Python FP32”，不要求逐点贴合其舍入结果。接下来的验收用此前训练对照的`backend=pytorch`全谱FP32作基线，以同一高精度参考比较两者误差，并验证真实训练质量；旧固定逐点门槛继续作为诊断。本文下列FP64门槛、失败状态和当时的发布判断是历史实验记录，不回写成通过。最新进展见[共享s1后续研究](shared_s1_training.md)。

本轮前的10个生产/模型/构建源文件已冻结在 `artifacts/native_deconv_target/source_before/`，带字节SHA清单，不能覆盖。既有 `artifacts/training_refinements/source_before/` 是更早基线，不混用。

## 本轮结果

后续针对真实s1形状的共享传递系数研究见[共享输入s1训练优化](shared_s1_training.md)：同fixture模块4.141ms、原生3.899ms，配对差距约6.2%；完整模型严格门槛仍未通过，不替代下方已完成的原组合候选训练证据。

- **B32/C32/64×80、s3/k3、nearest先验**：包含自动高阶回退封装的Converse完整前后向4.519ms，同轮生产FFT33.325ms、原生dense eager7.908ms、原生dense compiled8.119ms。这个形状已达到速度目标；是保持Converse数学的精确空间特例，不是用转置卷积替换输出。
- **真实prior模块B4/C128/96²、s1/k3**：s1专用分派和pad/crop组合4.381ms，同轮原生dense3.892ms，配对耗时比1.142。完整USRNet三种子各250步，组合候选/控制最终参数、Adam状态和验证指标全部相同，完整循环配对加速中位数1.107×。
- 两路仍是显式实验候选，**生产默认未修改**。预训练局部fixture的1点核梯度超限、完整模型固定压力fixture的16张量/26点超限均保留；后者不是加载预训练权重的实验。k3/s3空间特例不命中当前fullUSRNet，不能把其算子速度当作该模型收益。

## 原生基线

同一 RTX5060Ti，FP32、TF32关闭、cuDNN deterministic=True/benchmark=False。完整前向和输入/核/bias三个VJP，warm5、四轮交替、每轮20次，关闭profiler；每次只保留一个GPU fixture。Converse每次重新做可微核准备，s3含nearest先验，s1模块case还包含circular pad2和crop2。固定seed17合成数值匹配真实层形状，未声称是训练中捕获的实际激活。

| 情形 | Converse完整前后向 | 原生groups=1 | Converse/native配对比 | 对depthwise配对比 |
| --- | ---: | ---: | ---: | ---: |
| 实际prior模块 B4/C128/96²/s1/k3，含pad/crop | 5.449 ms | 3.993 ms | **1.365×耗时** | 6.372× |
| 不含外围pad/crop的public算子 B4/C128/100²/s1/k3 | 3.569 ms | 4.355 ms | 0.817×耗时 | 3.750× |
| DataNet B4/C64/32²/s3/k7，每样本动态核 | 1.960 ms | 1.653 ms | 1.188×耗时 | 1.861× |
| 大batch B32/C32/64×80/s3/k3 | 29.327 ms | 8.918 ms | **3.283×耗时** | 5.551× |

每种native使用自己的独立配对轮次，表中Converse/native绝对时间来自groups=1配对；depthwise的Converse计时是另一次fixture，不能混用。耗时列是各路轮次中位数，比例是逐轮比率中位数。动态核的原生API不接受batched weight，明确使用逐样本 `conv_transpose2d + cat`，含循环/拼接成本；这一行不能称单个原生批次kernel的性能。

原生奇数kernel取 `padding=(k-1)/2`、`output_padding=s-1`，精确给出输入高宽乘s的输出，全部形状和VJP大小检查通过。depthwise与Converse共享相同核元素，常规dense有C倍核参数。Converse是带先验的周期正则逆问题，原生是零边界转置卷积和加性bias，结果本来就不应比较为数值相等。

原始结果：[native_baseline.json](../artifacts/native_deconv_target/native_baseline.json)；入口：[benchmark_native_deconv.py](../test/benchmark_native_deconv.py)。

## 已执行的Converse候选

### 真实小空间的s1专用分派

独立namespace中仅把 `use_scale1` 的空间门槛临时改为 `s==1`；kernel正文、准备精度和数学均不改。7个常规shape和3个既有弱正则fixture，两路共80个输出/VJP比较全部通过原算子FP64预算；另外用独立profiler核名确认每种shape实际分派。计时关闭profiler。

| public算子case | 当前 | 强制s1 | 说明 |
| --- | ---: | ---: | --- |
| B4/C128/100²/shared3 | 3.472 ms | 3.155 ms | 本次实际prior内部形状 |
| B4/C64/96²/batched7 | 2.134 ms | 1.989 ms | 动态DataNet形状 |
| B1/C128/100² | 1.208 ms | 1.039 ms | 小batch |
| B8/C128/100² | 7.726 ms | 7.032 ms | 大batch |
| B1/C32/256²，原本已走s1 | 1.720 ms | 1.851 ms | 同逻辑控制项仍有约7%差异，保留噪声/构建差异风险 |

这说明旧空间门槛漏掉了值得测的真实shape，但控制项提示仍需更长匹配计时。不把局部比率与其他实验相乘，不据此无条件启用所有s1。原始结果：[s1_shapes.json](../artifacts/native_deconv_target/s1_shapes.json)；入口：[probe_training_s1_shapes.py](../test/probe_training_s1_shapes.py)。

### 保持ATen自动求导的padding/crop

在B4/C128/96²、p2、相同预训练prior权重上测试：`cat`构造circular pad、单次非重叠`as_strided`表达二维crop，以及两者组合。没有手写空间反向。30个CPU边界检查覆盖退化尺寸、非方形、转置/非连续、channels-last、storage_offset、一阶/二阶和完整FP64参考图，支持域内全部通过。泛化实现必须处理p=0，限制p≤输入高宽，并使用crop的正确stride/storage_offset；原型固定p2不能直接无条件上线。

真实形状的四种GPU路径在这个固定fixture的输出和dx/dw/db **全部逐位一致**，但独立FP64检查发现当前路径及每个候选的同一个核梯度点超限。最大dw绝对误差4.8469e-4、relativeL2约3.49e-7、1个逐点失败；没有调整seed、输入幅度、λ或容差。原始门禁停止计时的报告保持原样：[boundary_probe.json](../artifacts/native_deconv_target/boundary_probe.json)。

另立只用于筛选研究方向的诊断计时，明确 `release_gate_passed=false`，先核对全部逐位一致和源/输入SHA，得到当前4.970ms、cat+view组合4.343ms，四轮配对中位约1.139×。这是**带有继承精度失败的诊断潜力**，不是精度通过/可接入默认/正式收益声明；也不能与上面原生基线的不同fixture混比。记录：[boundary_diagnostic_timing.json](../artifacts/native_deconv_target/boundary_diagnostic_timing.json)、[CPU数学检查](../artifacts/native_deconv_target/cpu_boundary_check.json)。

## 后续实验与保留门槛

### 完整模型三种子真实训练

控制与组合候选各做 seed17/29/43、250步，全部保持原 pretrained full5/7、HR96/s3/B4、Adam1e-5/MSE及900/100数据协议。三个种子的逐批哈希、100张验证输入、调用覆盖和全部有限值检查通过；每个种子的最终模型参数及Adam状态均与控制组完全一致，RGB/Y PSNR/SSIM差为0。

| seed | 控制循环（含验证/I/O） | 组合循环 | 控制/组合 | 控制/组合预热步中位数 |
| --- | ---: | ---: | ---: | ---: |
| 17 | 151.386 s | 135.781 s | 1.115× | 495.459 / 458.256 ms |
| 29 | 145.959 s | 131.897 s | 1.107× | 497.735 / 439.424 ms |
| 43 | 146.206 s | 132.130 s | 1.107× | 500.377 / 438.698 ms |

完整循环配对比中位数1.107×；这是本次实际连续短程训练观察，不能当作完整收敛或稳定独立性能分布。allocated峰值仍约9.736GB。seed29执行顺序反转，其余为控制→候选；全部记录保留。审计0个完整性错误：[training_summary.json](../artifacts/native_deconv_target/training_summary.json)、[表格](../artifacts/native_deconv_target/training_summary.md)。用户可通过 `test/train_usrnet_converse_candidate.py --candidate combined` 复现实验，生产源码未被该适配器修改。

### 同fixture组合结果

在原生基线已固定的 `s1_module` fixture 上，current、仅s1、仅边界、组合四条路径均通过原FP64输出/全部VJP预算（此fixture与前述预训练权重失败fixture不同，旧失败仍保留）。随后完成六轮轮换、每路50次，直接测量同轮分母：

| 路径 | 完整前后向中位数 | current/该路径配对比中位数 | 该路径/原生dense配对耗时比 |
| --- | ---: | ---: | ---: |
| current | 5.336 ms | 1.000× | 1.371× |
| 仅强制s1 | 5.031 ms | 1.060× | 1.291× |
| 仅cat pad + view crop | 4.648 ms | 1.144× | 1.193× |
| 组合 | **4.381 ms** | **1.211×** | **1.142×** |
| 原生dense | 3.892 ms | 独立速度标杆 | 1.000× |

组合/native逐轮范围1.125–1.195；current本轮第一轮4.780ms、后续约5.29–5.48ms，原生第四/五轮也较快，全部原样保留。这不是把前两项独立倍率相乘，亦不是已接入生产或整网训练加速。Converse四路局部peak allocated均216.28MB，native164.76MB。报告：[combined_candidates.json](../artifacts/native_deconv_target/combined_candidates.json)。

### s3的纯ATen频域nearest原型：暂不采用

12个常规/弱正则case、3条路径的输出及全部VJP共144项比较通过原FP64门槛；CPU公式、一阶/二阶检查也通过。实际计时却表明，消除显式nearest和HR FFT后，HR半谱gather/where/multiply及其scatter反向、每调用phase生成抵消了节省：

| 形状 | 当前生产 | 同准备/solve的ATen显式nearest控制 | ATen频域nearest |
| --- | ---: | ---: | ---: |
| B1/C32/64×80/s3 | 1.386 ms | 1.324 ms | 2.957 ms |
| B8/C32/64×80/s3 | 7.036 ms | 7.006 ms | 7.494 ms |
| B32/C32/64×80/s3 | 26.601 ms | 26.625 ms | 26.665 ms |
| DataNet B4/C64/32²/s3 | 1.891 ms | 1.811 ms | 3.125 ms |

此probe的固定eps/核/输入与原生baseline不同，不直接计算跨报告native比率。三次相同B32fixture的output/dw/db逐位相同，但dx有9.31e-10的重复差异；cuDNN deterministic=True不代表ATen scatter反向也确定。结论是**公式可用，当前实现没有速度价值且引入dx不确定性，不接默认**。下一候选需要在频谱CUDA核心现场生成prior、反向确定次序gather，从根本上去掉HR prior/gp物化，重新验证全部门槛。报告：[nearest_training.json](../artifacts/native_deconv_target/nearest_training.json)。

### 固定精度失败的进一步定位

保持原预训练fixture和预算，失败点为 `dw[0,91,0,0]`，current=0.27729180455、FP64=0.27719384185，误差9.796e-5，预算6.386e-5。ATen c64虽在此fixture通过，但全局最大dw误差反而略大，因此不能按这个点选择运算顺序当修复。仅提升输入FFT或局部solve精度仍失败，两者同时提升才通过这一个fixture；结果支持多段舍入误差传播/抵消，没有证明单一FFT实现错误。

给定各自incoming gK的4条ATen路径，最后FP64 kernel-prep伴随重放均匹配转float后的dw；未直接导出当前CUDA内部gK，不能据此声称CUDA每段误差为零。后续需固定边界频谱和上游G，按线性伴随重建每段误差对失败坐标的有符号贡献。记录：[precision_diagnostic.json](../artifacts/native_deconv_target/precision_diagnostic.json)、[解释与下一试验](../artifacts/native_deconv_target/precision_interpretation.md)。

预训练权重的固定FP64失败仍需隔离activation FFT、谱VJP、广播累加及核准备反向，不以“与当前相同”替代独立参考通过。同fixture的s1组合对照及三种子短程训练已完成，结果见上文。s1共享x/prior时原实现已只做一次激活RFFT，没有删除重复FFT的收益。

### 已验证的频域nearest融合候选

隔离CUDA候选在读取时生成nearest频谱，在反向按固定次序gather回LR频谱，不物化HR prior/gp，不用atomic scatter。FFT/IFFT、FP64核准备与λ继续自动求导。初版高阶ATen参考因index_select反向不reentrant而失败；失败源码/记录已归档，随后仅将高阶图改为 `full_spectrum→repeat→slice`，未放宽nondet_tol。修复后全部原空间FP64/弱正则门槛、任意复数/广播/梯度mask、gradcheck/gradgradcheck通过，另外13项选择性高阶/非默认stream/保存输入版本异常检查通过；3次普通前后向逐位重复。

进一步仅在一个fixture内复用依赖几何尺寸的不可训练phase，核准备仍每步执行：

| case | 同轮生产 | 融合且每步生成phase | 融合且复用几何phase（热） |
| --- | ---: | ---: | ---: |
| B1/C32/64×80/s3 | 1.428 ms | 2.617 ms | 1.069 ms |
| B8/C32/64×80/s3 | 7.288 ms | 4.828 ms | 4.185 ms |
| B32/C32/64×80/s3 | 26.628 ms | 15.348 ms | 14.571 ms |
| DataNet B4/C64/32²/s3 | 1.851 ms | 2.308 ms | 1.651 ms |
| s1控制 | 0.925 ms | 2.339 ms | 0.952 ms |

这是另一组固定fixture，不能拿本表热值除以前面的原生baseline时间。phase首次生成约1.7–2.6ms单列；常驻phase约数KB，包含在热显存里，未缓存任何可训练参数或核谱。B32 allocated约1.570→0.849GB。s1不应走此路径。

源码：[training_nearest](../experiments/training_nearest/README.md)；报告：[动态phase](../artifacts/native_deconv_target/nearest_fused_training_repeat_reference.json)、[几何phase](../artifacts/native_deconv_target/nearest_geometry_training.json)、[契约检查](../artifacts/native_deconv_target/nearest_candidate_contract.json)。新NSYS捕获kernel累计15.163ms/次，已重新核对热点；NCU共享核反向为80寄存器、45.48%实际occupancy、DRAM44.0%、SM51.32%，不能继续沿用旧实现“约81% DRAM”的结论。原始采集在 `nearest_nsys_verified/`、`nearest_ncu_filter/`。

### 未采用的共享核工作区

27个独立FP64 case通过，但B32共享s3的4轮配对中位数仅0.978×，范围0.950–1.085，没有稳定收益。逻辑工作区新增181.5MiB，整体allocated峰值未增加，但reserved由约2.192→2.382GB。该测试用独立prior及4类VJP，不能混作nearest的原生对照，且不命中当前fullUSRNet的s3动态核。保持实验状态：[shared_workspace_probe.json](../artifacts/native_deconv_target/shared_workspace_probe.json)。

### 精确非重叠空间解

当 `kh<=s && kw<=s` 时，周期卷积后phase0抽样的算子A满足 `AAᵀ=sum(k²)I`。所以原解精确等于 `p+Aᵀ[(x-Ap)/(sum(k²)+λ)]`，没有额外s²因子。独立CPU矩阵/FFT证明292个case通过；k7/s3等重叠情况有明确反例，必须回退。此结论不代表任意转置卷积与Converse等价。

直接组合卷积/转置卷积的实现通过10组原CUDA门槛，却在B32变为38.09ms，故未采用。k3/s3的nearest情况可进一步只在LR上计算四项邻域预测，再构造9个输出相位；117个独立CPU case及52组二阶核验通过，CUDA五路原门槛也通过。但eager的LR+transpose/Tensor版本仍约15.16/16.86ms（同轮FFT26.40ms），大量逐点操作与临时张量继续占用时间。

`torch.compile` 已完成纯ATen精确公式及自动反向的融合验证，没有手写空间VJP。9个k3/s3编译case及1个k1非编译控制的原FP64/弱正则门槛、三次逐位重复检查全部通过；正式12轮热测unique_graphs均保持7→7，CUDA Graph关闭。B1/B8/B32的裸编译完整前后向为0.504/0.996/5.197ms，同轮FFT为1.762/9.490/35.903ms。此实验启用了global deterministic，且fixture不同于原生基线，**不能用这些数除以前表的native时间**。首次冷缓存完整执行含编译22.114秒，另有callable创建0.442秒；后续已有缓存的首次执行也不是纯编译时间。

裸AOT路径明确不支持double backward。新增[自动求导封装](../experiments/training_nonoverlap/compiled_autograd.py)在普通反向使用编译图，高阶时从原始可微输入重建eager ATen；不缓存输入或可训练参数。最初GPU契约检查因为编译器donated buffers不支持retain_graph而失败，旧源码和报告完整保留。修复为**编译前和整个实验期间统一设置`torch._functorch.config.donated_buffer=False`**，并使用新callable和独立缓存；不是吞掉异常或放宽门槛。

修后GPU三case的一阶及非默认stream、small_odd的二阶VJP均通过原FP64预算，重复retain_graph反向逐位一致；7个梯度mask、共享输入别名、版本修改异常和图释放另有CPU eager替身检查，不声称全部在GPU编译图上覆盖。封装的实际时间和显存必须重新测量，不能继承裸编译的5.197ms。

[PyTorch编译接口](https://docs.pytorch.org/docs/stable/generated/torch.compile)、[AOTAutograd语义说明](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html)说明了编译及额外求导契约。原始记录：[空间原型](../artifacts/native_deconv_target/spatial_nonoverlap.json)、[LR实现](../artifacts/native_deconv_target/spatial_nonoverlap_fast.json)、[编译结果](../artifacts/native_deconv_target/compiled_nonoverlap.json)、[冷缓存smoke](../artifacts/native_deconv_target/compiled_nonoverlap_smoke.json)、[原失败](../artifacts/native_deconv_target/compiled_autograd_contract.json)、[修后契约](../artifacts/native_deconv_target/compiled_autograd_contract_nodonation.json)。

这个k3/s3 nearest分派**不命中本次完整USRNet**：真实prior是k3/s1，DataNet是k7/s3。上面的真实训练1.107×来自s1和边界组合，与这个空间特例无关。

### 最终同条件原生对照

沿用最初原生基线seed17的完整fixture SHA、eps=1e-5、输入和上游梯度，不改核或数值尺度。所有路径统一global deterministic=True、cuDNN deterministic=True/benchmark=False、TF32/AMP关闭、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、donated_buffer=False。空间候选和原生均有Inductor/fullgraph/dynamic=False对照，CUDA Graph关闭；普通执行的生产FFT也使用相同确定性设置。

每路warm5，六轮轮换，每轮30次完整forward+dx/dw/db；关闭profiler和所有冷准备计时，热测禁止重编译，六轮unique_graphs均2→2。Converse三路各自对独立FP64原预算通过，编译原生对eager原生通过，全部五路三次逐位重复通过。

| 路径 | 完整前后向中位数 | peak allocated |
| --- | ---: | ---: |
| 生产FFT | 33.325 ms | 1.561 GB |
| 裸编译空间候选 | 4.576 ms | 0.482 GB |
| **编译空间候选，含高阶回退封装** | **4.519 ms** | **0.482 GB** |
| 原生dense eager | 7.908 ms | 0.420 GB |
| 原生dense compiled | 8.119 ms | 0.420 GB |

封装/native eager逐轮耗时比中位数0.576，范围0.553–0.614；封装/native compiled中位数0.579，范围0.517–0.580；封装/生产FFT中位数0.139，范围0.131–0.149。空间封装的allocated较FFT减少69.1%，仍比native大约15%。不要从各列中位时间之商代替配对统计；裸路径与封装路径的小幅反向差异不代表封装自身加速。

新缓存目录中的首次裸空间FWD/VJP为12.928秒，callable创建0.207秒；随后封装首次执行8.782ms命中同进程编译缓存（unique_graphs未增加），**不是独立冷编译时间**。原生compiled首次完整执行0.921秒。源码、编译设置和首次执行均在报告里单列，冷启动不得并入热速度结论。

Converse核288元素、native dense核9216元素，均有32个bias参数但含义不同；输出同为B32/C32/192×240。这里对照的是用户指定的groups=1速度标杆，没有证明与depthwise持平、算子数学相等、任意形状都更快或完整训练收敛更快。所需Triton/Inductor运行环境、显式入口与高阶约束见[实验使用说明](../experiments/training_nonoverlap/README.md)。原始报告：[final_comparison.json](../artifacts/native_deconv_target/final_comparison.json)；入口：[benchmark_deconv_target_final.py](../test/benchmark_deconv_target_final.py)。

生产默认尚未在此次target实验中改变。独立FP64门槛、完整模型已有的16张量/26逐点失败、复用压力门槛、真实多种子质量及time-to-quality要求均保留。目标是否达到应按各实际形状的完整前后向/整网训练分别判断，不能只看一个谱核或相乘不同加速比。

```powershell
# 先确保test/extension_loader.py已成功构建；此入口严格核对现有库SHA，避免无关cl版本探测。
& ./experiments/training_speed/run.ps1 test/run_verified_cuda.py test/benchmark_native_deconv.py --output artifacts/native_deconv_target/new_native.json
& ./experiments/training_speed/run.ps1 test/run_verified_cuda.py test/probe_training_s1_shapes.py --output artifacts/native_deconv_target/new_s1.json
& ./experiments/training_speed/run.ps1 test/check_compiled_autograd_contract.py --cuda --output artifacts/native_deconv_target/new_contract.json
& ./experiments/training_speed/run.ps1 test/run_verified_cuda.py test/benchmark_deconv_target_final.py --output artifacts/native_deconv_target/new_final.json
```
