# 可微 non-overlap 编译实验

这是隔离的 FP32 Converse2D 候选，未接入生产默认路径。当前编译专用条件是
**scale=3、3×3 核、prior 精确等于同一个输入 x 的 nearest 上采样**，并保持原有
周期边界、核居中和采样相位。核支持 batch/channel 的原有广播；λ 仍为
`sigmoid(bias - 9) + eps`。不能把这个条件推广为任意 prior 或任意卷积核。

在核支持不超过 stride 的条件下，降采样卷积各行没有重叠，因此
`A Aᵀ = sum(kernel²) I`。实验用低分辨率残差求解和 `pixel_shuffle` 表达同一
数学解，核系数及能量仍以可微 FP64 累加后转回 FP32。空间反向全部由 ATen /
AOTAutograd 自动生成，没有手写空间 VJP，也没有缓存参数或参数更新前的核值。

**默认完整 ConverseUSRNet 不命中这个 k3/s3 专用条件。** 它的首个 s3 DataNet
使用 7×7 核，其余调用是 s1；因此本实验算子收益不能作为整网加速结论。
基础 probe 复用原 nearest 的十个 non-overlap 合法样例，k1 仅验证 eager /
production，k7/s3 与 k3/s1 明确排除，不宣称完整形状覆盖。

## 编译与高阶梯度

[基础编译入口](../../test/probe_compiled_nonoverlap.py) 提供 `make_compiled(case)`，
采用 `backend="inductor"`、`fullgraph=True`、`dynamic=False`，显式关闭
`triton.cudagraphs`。编译失败会报错，不静默换成 eager。首次 forward 和全部
VJP 的编译/执行成本单列，不能纳入或隐藏在热训练比率中。

本机 PyTorch 2.11.0+cu130 / Triton 3.6.0 的真实 AOT 检查明确报告不支持直接
double backward。[适配器](compiled_autograd.py) 的稳定 API 为：

```python
wrapped = wrap_compiled(eager_callable, compiled_callable)
output = wrapped(x, weight, bias)
```

适配器在 forward 内启用梯度，用每个形参独立的 view 构建 compiled 内部图；
原输入、代理和内部输出全部通过 `save_for_backward` 保存。普通 backward
调用自动 `autograd.grad(..., retain_graph=True)`，支持外层重复反向；
`create_graph=True` 时从原输入重建 eager 图，再自动求导。独立代理避免同一
Tensor 传入多个形参时重复计梯度。工厂只保留 callable/代码，内部图随外层保存
变量释放；参数值、可微图和输出不跨训练步缓存。

**必须在创建 compiled callable 及其首次执行之前设置**
`torch._functorch.config.donated_buffer = False`，并在整个适配器实验期间保持
False。开启 donation 的 AOT backward 与所需 `retain_graph=True` 不兼容。
不能仅在旧 compiled 实例上事后切换配置；应创建新的 callable，使用独立的
编译缓存。适配器会检查当前配置，但无法证明外部传入 callable 的历史配置。

[contract 入口](../../test/check_compiled_autograd_contract.py) 自动先设置上述配置，
重置进程内 Dynamo specializations，并使用 `.build/inductor_nonoverlap_nodonation`
及 `.build/triton_nonoverlap_nodonation`。所有 CUDA 路线统一 TF32 关闭、全局
deterministic algorithms 开启、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；不使用 Graph。

## 复现

在仓库根目录运行；以下 `_repro` 输出必须是新路径，脚本拒绝覆盖旧记录。

```powershell
# CPU FP64 eager 替身：梯度路由、别名、重复反向、版本和内部图释放。
F:/anaconda3/envs/vllm/python.exe test/check_compiled_autograd_contract.py

# 裸 compiled smoke：B1/C32/LR64×80/s3，完整前后向与原数值门槛。
./experiments/training_speed/run.ps1 test/probe_compiled_nonoverlap.py --smoke --output artifacts/native_deconv_target/compiled_nonoverlap_smoke_repro.json

# 裸 compiled 原十个合法样例；k1 只验证，其余 k3/s3 可编译。
./experiments/training_speed/run.ps1 test/probe_compiled_nonoverlap.py --output artifacts/native_deconv_target/compiled_nonoverlap_repro.json

# 可重复反向/高阶适配器的真实 CUDA contract；独立 nodonation 缓存。
./experiments/training_speed/run.ps1 test/check_compiled_autograd_contract.py --cuda --output artifacts/native_deconv_target/compiled_autograd_contract_nodonation_repro.json

# 五路同条件对照，包含真实高阶适配器；先验证已构建生产库的源/二进制哈希。
./experiments/training_speed/run.ps1 test/run_verified_cuda.py test/benchmark_deconv_target_final.py --output artifacts/native_deconv_target/final_comparison_repro.json
```

CUDA contract 保留原输出/全部梯度及弱正则 FP64 门槛，并检查真实 compiled 的
一阶梯度、适配器高阶对 eager/独立 FP64、重复反向与非默认 stream。CPU 检查还
覆盖全部七种非空梯度需求组合、同输入别名、gradcheck/gradgradcheck、版本错误
和内图释放。CPU eager 替身通过不等于真实 compiled 已通过；两者分别记录。

## 保存的结果与边界

- [裸 compiled smoke](../../artifacts/native_deconv_target/compiled_nonoverlap_smoke.json)：
  本次首次完整 FWD+VJP setup 约 **22.114 秒**，另有编译callable创建约 0.442 秒；
  这是该次缓存状态下的首次调用成本，不是每步延迟或通用冷启动保证。
- [裸 compiled 全样例](../../artifacts/native_deconv_target/compiled_nonoverlap.json)：
  原数值/重复性门槛通过；其热性能不能代替带高阶适配器、关闭 donation 的结果。
- [原适配器失败报告](../../artifacts/native_deconv_target/compiled_autograd_contract.json)、
  [失败日志](../../artifacts/native_deconv_target/compiled_autograd_contract.log) 和
  [失败源码哈希](../../artifacts/native_deconv_target/compiled_autograd_failed_source/manifest.json)：
  保留 donation 与 `retain_graph=True` 不兼容的原始失败，未放宽门槛掩盖。
- [关闭 donation 后的真实 CUDA contract](../../artifacts/native_deconv_target/compiled_autograd_contract_nodonation.json)
  及 [日志](../../artifacts/native_deconv_target/compiled_autograd_contract_nodonation.log)：
  已通过，包含 CPU 项、真实一阶/高阶、重复反向与 stream 检查。

[最终五路对照](../../artifacts/native_deconv_target/final_comparison.json)已完成：
原native seed17 fixture、B32/C32/LR64×80/s3/k3，FP32、global deterministic、
TF32/AMP/Graph/donation全部按同一设置。六轮×30次完整FWD+dx/dw/db，真实
适配器4.519ms、生产FFT33.325ms、原生dense eager7.908ms、原生dense compiled
8.119ms；同轮适配器/native eager耗时比中位数0.576，范围0.553–0.614。
全部FP64及重复性门槛通过，热测unique_graphs保持2→2。适配器peak allocated
0.482GB，比FFT少69.1%，仍比native高约15%。没有沿用裸编译或历史分母。

本次先执行裸编译路径，首次完整执行12.928秒；随后适配器的8.782ms首次执行
复用了同进程编译图，不能称作独立冷编译耗时。编译setup、热延迟、显存和缓存
状态分开保存。不同数学/参数量的native仅是速度标杆；这个特例未接生产默认、
没有当前fullUSRNet的训练收益证据。完整说明见[结果报告](../../docs/converse_deconv_target.md)。
以上`artifacts/`为本地原始记录，Git默认忽略。
