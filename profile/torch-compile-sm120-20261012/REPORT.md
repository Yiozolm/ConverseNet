# RTX 5060 Ti / SM120：torch.compile 训练试验

2026-10-12，PyTorch 2.11.0+cu130，CUDA 13.0，当前 FP32 分支。

**torch.compile 可以用于该模型的周边计算，但当前模型不能直接整体编译。** 原 C++ `converse2d.forward` 在 FakeTensor tracing 时访问真实数据指针，直接调用 `torch.compile(model)` 的试验报错。将 Converse2D 和 ConvReverseDataNet 的 forward 设为编译边界后，Inductor 能编译周围的卷积、LayerNorm、GELU 和 residual。

**严格 FP32 设置下，本次完整训练步收益约 3–5%。**

| 路径 | eager median wall | compiled median wall | 吞吐倍率 | 训练步耗时减少 |
|---|---:|---:|---:|---:|
| 原 production USRNet | 502.618ms | 478.904ms | **1.0495×** | 4.72% |
| 当前 Flash prior 研究接入 | 391.509ms | 380.067ms | **1.0301×** | 2.92% |

两条编译路径的输出与全部 133 组参数梯度，都通过 CUDA 冻结 Python FP32 基线 / CPU 独立 Python FP64 参考预算。**compiled LayerNorm 的二阶梯度试验仍失败**，报 `torch.compile with aot_autograd does not currently support double backward`；显式 force_eager 的 LayerNorm 检查通过。因此这次提供研究 harness，不将编译默认接入 release 模型。

## 工作负载与测量

- RTX 5060 Ti 16GB / CC 12.0 / 36 SM / Windows WDDM。
- 与上一轮相同的预训练 USRNet：B4、LR32×32 → HR96×96、scale 3、5 iterations、35 次 prior reverse 和 5 次 data-term reverse。
- 固定 seed 72001 的合成 LR、kernel、target；数据已经驻留 GPU。
- 每步包含 forward、MSE、backward、Adam；optimizer 保持 `foreach=False, fused=False`，optimizer.step 和 loss 没有单独编译。
- eager 和 compiled 为两个从相同 checkpoint 初始化的模型，分别使用独立且相同配置的 Adam。数值 fixture 在任何 optimizer update 前捕获。
- 排除首次编译、FFT plan 准备和 warmup；6 轮 **ABBA**，每个位置 2 次 step，两个路径各 24 次 timed steps，CPU affinity 为 `0xC03C03`。
- primary speedup 使用无 profiler wall-time median 比值；CUDA event 原始轮次一并保留。
- 此处的“完整训练步”不包含 DataLoader、H2D、评估与 checkpoint 写盘。固定输入与短训练更新仅用于测量，不是收敛证明。

严格设置：FP32 / complex64，AMP 与两个 TF32 flag 关闭；`config.use_fast_math=False`、`config.cuda.use_fast_math=False`、`config.cutlass.use_fast_math=False`；另设 `config.eager_numerics.division_rounding=True` 和 `config.eager_numerics.disable_ftz=True`；CUDA Graphs 关闭。

`analysis/codegen_audit.json` 保存 40 个带严格 FTZ metadata 的缓存 module SHA，其中 10 个包含 `div_rn`；没有 FP16/BF16 tensor signatures。运行中的 Inductor counter 为各试验 41 个 generated kernels；缓存 module 数与运行 kernel 数不是同一口径。

## 编译方式

研究 harness 中的主要处理如下：

```python
for module in model.modules():
    if type(module).__name__ in ("Converse2D", "ConvReverseDataNet"):
        module.forward = torch.compiler.disable(module.forward)

compiled = torch.compile(
    model, backend="inductor", fullgraph=False, dynamic=False
)
```

这是主动划定的 eager 边界。它避免 fake tracing 进入原 C++/CUDA 内核，让 compiler 在这些调用前后恢复 tracing；用法见 [PyTorch 2.11 torch.compiler.disable](https://docs.pytorch.org/docs/2.11/generated/torch.compiler.disable.html)。所有 wrapper 只施加于 harness 构造的模型实例。

如果要进一步把这些算子放进单个完整图，需要 opaque custom-op、FakeTensor/meta 和适配训练的 autograd 注册，不能只填一个输出 shape 后就假定支持；参见 [PyTorch custom operator 文档](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html)。本次没有修改生产 dispatch 注册。

## 编译实际做了什么

| 指标 | production eager → compiled | Flash eager → compiled |
|---|---:|---:|
| 单次 diagnostic step 的 GPU kernel 数 | 7284 → **5617** | 7144 → **5547** |
| Inductor generated kernel counter | 41 | 41 |
| Dynamo unique graphs | 7 | 7 |
| `aten::fft_fft2` / step | **40 → 40** | **40 → 40** |
| `aten::_fft_c2c` / step | **131 → 131** | **96 → 96** |

实际编译生效，主要融合了周边 pointwise/reduction 计算并减少启动数。反卷积及其可微 kernel preparation 保持逐调用执行：compiled 和 eager 的 FFT/operator route counts 相同，见 `analysis/fft_route_counts.json`。没有借助训练谱复用加速。

最大的 cuDNN wgrad kernel 仍为 `wgrad2d_grouped_direct_kernel`。本次模型编译没有将反卷积源码重写为更快的 FFT kernel，也未解决这个 wgrad 热点。约 23% 的启动数量减少只换来约 3–5% 的完整步骤收益，符合大量时间仍花在既有 CUDA/cuDNN 工作上的表现。

诊断 trace 只用于看工作和调用路线，不用于速度排名。例如 strict production 的单次 kernel sum 为 427.231ms → 466.747ms，与 ABBA wall-time median 的方向不同；桌面 GPU 波动、profile 影响与采样次序使单次 trace duration 不适合作为性能证据。Flash 的单次 kernel sum 为 363.585ms → 343.879ms，也只作诊断记录。

六轮 ABBA 的几何平均倍率为 production **1.0393×**、Flash **1.0298×**。探索性 round bootstrap 95% 区间分别约 `[1.0206, 1.0584]`、`[1.0225, 1.0386]`；只有六轮，且桌面 GPU 有外部负载，这些区间不代表跨机器或部署稳定性保证。

首次 forward+backward 包含编译和懒初始化：最初 bounded production 为 19.08s；后续 strict production 9.94s、strict Flash 2.65s。后两者已使用编译缓存，不能作为冷启动 compile-time 的比较。

## 数值和梯度兼容性

`harness/gate.py` 在独立进程执行 reference，以避免编译模型占用 reference 的显存预算：

1. GPU FP32 冻结基线使用 `test/fp32_baseline.py` 的 full-spectrum solver。不可变文件 SHA 仍由 `test/numerical_policy.py` 校验。
2. independent FP64 full-model reference 在 CPU 上以 Python/ATen 表达式执行；test-only wrapper 绕过原 module 的 FP32 dtype guards。它不是新增生产 FP64 backend。
3. 在相同输入、checkpoint、MSE 上比较 output 和 133 组参数 gradients，通过 `numerical_policy.comparison` 的 normal rel-L2 1.25× / max-abs 1.50× 及原 floors；未修改任何预算。

| 严格配置 gate | eager | compiled |
|---|---:|---:|
| production | 134/134 pass | **134/134 pass** |
| Flash prior 接入 | 134/134 pass | **134/134 pass** |

原始误差和每张量预算保存在 `analysis/gate_strict_production.json`、`analysis/gate_strict_flash.json`。这是一个 seed 的完整模型 gate；没有声称覆盖所有 seed、动态 shape、广播与梯度子集。

首次采用 CPU FP32 / CPU FP64 模型参考的 `bounded_production` lane 有 **97/134 candidate gate failures**，保留在原 JSON 中。后续另建 CUDA FP32 基线 lane；不修改、不重判这 97 个旧 verdict，也不将它们混入新的 GPU lane。不同 FP32 执行 backend 的 FFT/卷积和梯度累积舍入不同，预算基线必须明确标注执行设备。

高阶 probe：

- 原 eager LayerNorm 二阶梯度通过。
- compiled LayerNorm 二阶梯度失败，AOTAutograd 明确报 double-backward 不支持；原失败保存在 `analysis/higher_order_probe.json`。
- 显式 `with torch.compiler.set_stance("force_eager"):` 的 LayerNorm 二阶检查与 eager bitwise 相同，max_abs 0；记录在 `analysis/higher_order_eager_fallback.json`。它是一个边界 probe，不是整个 compiled USRNet 的高阶验证。
- 需要高阶梯度的训练继续使用原 production eager 模型和其 ATen fallback；FlashS1 研究接入本身也只测了一阶训练，不能外推其高阶能力。

相关 checked release lanes **49/49 通过**，见 `reports/release_checks.log`，覆盖 FP32 route/数值、默认全频谱、numerical policy、gradient mask、PSF preparation 与原高阶 fallback。它们验证原 release 行为；编译路径的一阶值另由上述完整模型 gate 验证。

## 环境准备与保留的失败

本机原环境没有 Triton，最小 `torch.compile` GPU forward/backward 报 `TritonMissing`。为本次试验将 **triton-windows 3.6.0.post26** 安装到工作区 `.build/torch_compile_deps`，没有修改共享 Python 环境的安装目录。版本选择与 SM120 支持依据 [triton-windows 的兼容表](https://github.com/triton-lang/triton-windows#3-pytorch)：PyTorch 2.11 对应 Triton 3.6。后续最小编译 smoke 通过。

最初依赖下载受网络限制失败；经权限范围允许的同一下载重试成功。日志 `dependency_install.log` 与 `dependency_install_retry.log` 都保留。直接整个模型编译的 FakeTensor failure 同样保留在 `analysis/raw_production.json` 和 `reports/raw_production.log`。没有开启 `suppress_errors` 将编译失败隐藏为 eager。

默认 Inductor numerical options 的 bounded 试验也保留：production 1.0395×、Flash 1.0365×，另有 GPU FP32/FP64 reference gate 通过。最终建议引用严格设置的 1.0495× / 1.0301×，不把不同桌面负载时刻的绝对时间直接横向相除。

## 文件与复现

- `harness/environment.py`：隔离依赖/缓存、checked binary 环境、严格 compiler options。
- `harness/study.py`：训练模型、opaque 边界、ABBA 计时、fixture 和独立 trace。
- `harness/gate.py`：CUDA frozen FP32 / CPU FP64 full-model gate。
- `analysis/summary.json`：四个成功 timing lane、每轮倍率、primary medians。
- `analysis/strict_*_fixture.pt`：数值测量时的实际输入与输出/梯度。
- `analysis/strict_*_trace.json.gz`：单步诊断。
- `harness/*snapshot.py`：早期 CPU-reference/default-numerics 的 harness 快照。

示例重新测量 Flash 路径，必须给一个新输出名：

```powershell
$env:CONVERSE_PYTHON = 'H:/Python/ConverseNet/.venv/Scripts/python.exe'
$env:CONVERSE_MSVC_VERSION = '14.44'
./tools/run_affinity.ps1 -Mask 0xC03C03 `
  profile/torch-compile-sm120-20261012/harness/study.py `
  --path flash --policy bounded `
  --output profile/torch-compile-sm120-20261012/analysis/another_flash_run.json
```

后续做部署或新 shape 试验应另建 run 目录；初次编译耗时不能纳入 steady-state speedup。继续提升训练速度时，pointwise wgrad 的完整 VJP 和 layout 开销仍值得优先测量。
