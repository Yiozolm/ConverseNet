# 既有生产 LayerNorm 的 USRNet 部署覆盖

本轮扩展部署形状和测量证据，不增加融合支持范围，不修改生产分派。
控制组仍是 `06a9673` 的原 LayerNorm statistics forward；实验组是已经在
`ff7f8ba` 生产使用的完整 LayerNorm。两者共用同一个当前 USRNet 实例、预训练
checkpoint、Converse solver、checked CUDA binary 和 Graph runner。旧 forward 与
共享 affine 依赖的固定 SHA256 检查复用
[`benchmark_production_layernorm.py`](../tools/benchmark_production_layernorm.py)。

当前阶段：CPU 真实模型 meta 检查和获分配独占窗口内的完整16项 GPU 矩阵均通过。
所有结论只适用于报告中实际完成的 case/mode/layout。

这些是根据既有模型和 evaluator 支持范围选择的代表性形状；本轮没有取得线上部署
流量、shape 频率或 batch 分布。真实 forward 的调用数及 padding 后尺寸，只描述
该输入实际执行的模型路径，不能外推为线上常用尺寸、实际占比或整体部署收益。

## 有依据的形状集合

[`inference_cases.py`](../tools/training_followup/inference_cases.py) 声明八个 case，
每个分别运行 `no_grad` 与 `inference_mode`。kernel 为 FP32 的 7×7 blur kernel。

| Case | B | LR | scale | HR / DataNet 输出 FFT | prior padding 后 FFT | caller layout / kernel batch |
|---|---:|---|---:|---|---|---|
| b1_even_s3 | 1 | 32×32 | 3 | 96×96 | 100×100 | NCHW / 1 |
| b4_even_s3 | 4 | 32×32 | 3 | 96×96 | 100×100 | NCHW / 4 |
| b1_odd_s3 | 1 | 31×33 | 3 | 93×99 | 97×103 | NCHW / 1 |
| b4_odd_s3 | 4 | 31×33 | 3 | 93×99 | 97×103 | NCHW / 4 |
| b1_medium_s3 | 1 | 64×64 | 3 | 192×192 | 196×196 | NCHW / 1 |
| b1_scale4_generic | 1 | 24×25 | 4 | 96×100 | 100×104 | NCHW / 1 |
| b1_odd_strided | 1 | 31×33 | 3 | 93×99 | 97×103 | 非连续 image/kernel / 1 |
| b4_channels_last_shared | 4 | 32×32 | 3 | 96×96 | 100×100 | channels_last image / shared kernel 1 |

前两项连接既有正式测量，奇数矩形此前仅用于 Graph LRU miss，本轮补完整计时。
B1 HR192 跨过旧 affine 的 `2**21` 元素阈值，规模约等于已有 B4 HR96 feature；
不扩大到未经容量验证的 B4 HR192。s4 属于已有 SR evaluator 接受的 scale，首次
DataNet 求解走既有 generic scale 分支，不把这一分支称为新融合。

真实 `ConverseUSRNet` 的 14 个 LayerNorm 模块在 5 次迭代中共调用 70 次，输入均
为 `(B,64,HR_H,HR_W)`；35 个 prior 调用在 C128 feature 上作每侧 padding2，
因此其 FFT 尺寸是 `HR+4`。另有 5 次 DataNet，首次从 LR 按 scale 放大，随后四次
scale1。CPU 工具实际执行生产模型 forward 的 meta 路径并以 hooks 核对这些尺寸和
调用数；它只证明形状，不把 meta stride 当作真实 CUDA layout。

[`inference_cpu_shapes_v2.json`](../artifacts/training_followup/inference_cpu_shapes_v2.json)
记录了全部 16 个 case/mode 的真实 meta 调用，`cuda_initialized=false`。
可选扩展导入在该进程中被阻止，没有加载 CUDA extension、编译或分配 GPU 内存。
GPU 工具还会在正式计时外记录 CPU dispatcher trace，核对真实 custom operator
接收到的 padding 后输入尺寸及生产 LayerNorm dispatch 次数。

## 数值、布局和生命周期边界

[`inference_deployment.py`](../tools/training_followup/inference_deployment.py) 对每个
case/mode 先执行数值与生命周期检查，失败则保留独立失败报告并停止，不能用容差
或较快的计时覆盖失败：

- 原 caller tensor/mode 的完整 eager old/production 输出须 FP32、有限且逐字节相等。
- Graph 内部始终使用 inference-mode、contiguous clone 的 static image/kernel。
  同 route 的 eager control 完整复制这一 tensor flag、layout 和缓存资格；Graph
  对该 control 逐字节检查，再比较 old/production。与原 caller eager 的相等性另存，
  不把不同控制条件下的差异改写成通过。
- 一次真实首个 LayerNorm feature 被转换为 contiguous、channels_last、strided
  三种布局；比较同参数 old/production 数值，并用 dispatcher 事件检查既有融合/
  fallback。caller image 为 channels_last 不意味着内部 LayerNorm 一定 fallback，
  因为前面的卷积和 FFT 可能改变内部布局。
- 检查首次 capture、另一个 H/W 的 LRU eviction/recapture、同 shape 改 stride 不
  重新 capture、image 与 kernel 值同时变化时的 copy、返回输出存储独立性，以及
  `clear()` 后重新 capture。两个 route 的公共标记必须区分模型 signature。
- 明确 `enabled=False` 的 runner 与同输入同 mode 的 eager 输出逐字节相等、capture
  数为零。它是既有显式 eager fallback，不是让 enabled runner 对无效输入静默回退。

hooks/profiler 仅用于计时外检查，并在 Graph 构造前移除；生产 dispatcher 未替换。
仅 historical control 临时替换各 LayerNorm 实例的 forward，离开作用域即恢复。
模型参数值、version、临时绑定、源码和 checked binary 在结束时重新核对。

## 计时口径与运行

固定亲和性 `0xC03C03`，checked 库；TF32、AMP 关闭，cuDNN deterministic 与
deterministic algorithms 开启。默认 3 次 warmup、7 个 AB/BA paired rounds、每轮
5 次调用，Graph 自身 warmup2。按 case/mode 顺序释放 Graph，避免把全部部署形状
同时留在显存。三条完整公共路径分别计时：

| 路径 | 包含范围 |
|---|---|
| eager_warm | 完整 `model(x,k,s)` 调用 |
| disabled_runner_warm | 完整 `runner(enabled=False)` 调用，包括已有 wrapper lock |
| graph_runner_hit | signature、lock、input/kernel copy、replay、独立 output clone |

首次 runner 调用另存，包括 Graph warmup、capture、copy 和 output clone；LRU miss
与 clear/recapture 也分别保留。它们是已经初始化进程中的观测，不是冷 CUDA 进程、
冷 cuFFT/cuDNN plan 或首次部署启动时间。空 eager spectrum cache 的调用也单列，
不能与 warm 路径混合。fixture/H2D、源检查、CPU 数值快照、profiling 和 route 安装
不在正式计时内；同步 wall 和 CUDA event span 均保留，不能只报局部 LN kernel 时间。
显存项是 live state 之上的 PyTorch allocated peak 增量，不是总进程显存。

CPU 检查：

```powershell
.venv/Scripts/python.exe tools/training_followup/inference_cpu_check.py --output artifacts/training_followup/inference_cpu_shapes_NEW.json
.venv/Scripts/python.exe tools/training_followup/inference_deployment.py --validate-source-only --output artifacts/training_followup/inference_deployment_NEW.json
```

以下 GPU 命令只能在主任务明确分配的独占窗口执行，不能与其他 GPU 工作并发：

```powershell
$env:CONVERSE_MSVC_VERSION='14.44'
$env:TORCH_CUDA_ARCH_LIST='12.0'
pwsh -NoProfile -File tools/run_affinity.ps1 -Mask 12598275 `
  -MetadataPath artifacts/training_followup/inference_affinity_v1.json `
  tools/training_followup/inference_deployment.py `
  --output artifacts/training_followup/inference_deployment_v1.json
```

输出已存在会拒绝覆盖。`CONVERSE_MSVC_VERSION=14.44` 与
`TORCH_CUDA_ARCH_LIST=12.0` 是现有 checked manifest 的环境身份，不据此重新构建。
可用 `--cases` 指定已声明 case 的子集、`--modes` 指定一个 mode 做隔离诊断，但必须
另存输出并准确缩小结论范围，不能覆盖完整矩阵中的失败。
每个 case 的 seed 使用完整声明矩阵中的固定索引；选择子集或更改执行顺序不会改变
该 case 的输入字节，便于独立复现失败。


实测证据：[完整原始报告](../artifacts/training_followup/inference_deployment_v1.json)、[进程日志](../artifacts/training_followup/inference_deployment_v1.log)、[亲和性与退出记录](../artifacts/training_followup/inference_affinity_v1.json)、[CPU汇总JSON](../artifacts/training_followup/inference_summary_v2.json)。完整矩阵汇总器为 [`inference_summary.py`](../tools/training_followup/inference_summary.py)；子集诊断只能保留其原始报告，不能伪装为完整矩阵。

## 完整 GPU 测量结果

本次独占 GPU 进程正常退出，wrapper wall 236.065 秒；16/16 个 case/mode 完整完成。
模型值/version/绑定、source/binary、每轮亲和性检查通过。所有配对 eager 输出、capture-context controls、布局 fallback 与 Graph 生命周期门槛通过。

每个完整 forward 均实际记录70次既有生产 LayerNorm dispatch、35次padding后的prior和5次DataNet。
原始 caller 比较保留了8个no_grad case在两个route中的Graph/eager字节差异（16项）；8个inference_mode case则在两个route中均相等。
因此不能把Graph表述成原no_grad eager的逐字节替代；这里的严格Graph gate使用相同inference tensor、contiguous clone及缓存资格。

下表时间均为毫秒，`旧→生产`是各route的7轮wall中位数；倍数是逐paired round旧/生产比值的中位数，所以不必等于两列中位数直接相除。

| Case / mode | 完整 eager 旧→生产 | disabled runner 旧→生产 | Graph hit 旧→生产 | paired eager / Graph 倍数 |
|---|---:|---:|---:|---:|
| b1_even_s3/no_grad | 28.615→21.834 | 27.980→21.370 | 15.549→13.734 | 1.316 / 1.130 |
| b1_even_s3/inference_mode | 26.429→20.219 | 25.859→19.491 | 15.641→13.696 | 1.315 / 1.132 |
| b4_even_s3/no_grad | 70.513→63.721 | 70.782→63.310 | 65.854→59.355 | 1.114 / 1.107 |
| b4_even_s3/inference_mode | 70.686→62.662 | 70.727→62.656 | 64.740→59.236 | 1.130 / 1.093 |
| b1_odd_s3/no_grad | 36.266→30.209 | 32.123→30.181 | 23.872→22.684 | 1.206 / 1.052 |
| b1_odd_s3/inference_mode | 36.550→29.739 | 33.606→27.567 | 23.856→22.681 | 1.224 / 1.050 |
| b4_odd_s3/no_grad | 102.496→95.543 | 102.706→95.208 | 96.990→90.743 | 1.072 / 1.070 |
| b4_odd_s3/inference_mode | 103.200→94.793 | 102.173→94.964 | 95.979→90.909 | 1.086 / 1.057 |
| b1_medium_s3/no_grad | 81.026→73.732 | 81.161→73.392 | 74.799→69.826 | 1.095 / 1.073 |
| b1_medium_s3/inference_mode | 80.469→72.932 | 80.788→72.892 | 74.935→69.233 | 1.106 / 1.092 |
| b1_scale4_generic/no_grad | 28.518→21.770 | 28.698→21.097 | 16.809→14.954 | 1.310 / 1.130 |
| b1_scale4_generic/inference_mode | 27.348→20.952 | 26.309→19.828 | 16.889→15.076 | 1.321 / 1.120 |
| b1_odd_strided/no_grad | 36.223→30.077 | 33.321→29.176 | 24.002→22.705 | 1.199 / 1.056 |
| b1_odd_strided/inference_mode | 36.329→29.911 | 33.669→29.829 | 23.801→22.703 | 1.216 / 1.058 |
| b4_channels_last_shared/no_grad | 70.187→63.565 | 70.414→62.818 | 65.175→58.826 | 1.107 / 1.098 |
| b4_channels_last_shared/inference_mode | 70.318→62.703 | 70.062→63.138 | 64.169→59.771 | 1.121 / 1.071 |

这些逐case的paired wall中位数在完整eager为 1.072–1.321×，Graph hit为 1.050–1.132×。
这是所测shape的观测范围，没有线上流量权重，不计算或声称整体部署收益；完整逐轮比值、CUDA event span和allocated peak增量保留在JSON。

### 初始化进程中的 setup 观测

下列每项是一次完整调用wall观测，单位毫秒，旧→生产。cuFFT/cuDNN/CUDA进程已初始化；不称为冷进程启动时间，也不与warm中位数合并。

| Case / mode | 空 eager spectrum cache | 首次 runner capture | LRU miss | clear/recapture |
|---|---:|---:|---:|---:|
| b1_even_s3/no_grad | 60.435→24.157 | 94.225→79.713 | 109.773→104.087 | 96.427→75.878 |
| b1_even_s3/inference_mode | 55.648→20.021 | 95.047→75.556 | 122.692→95.033 | 92.239→74.574 |
| b4_even_s3/no_grad | 71.903→63.860 | 243.782→207.443 | 345.890→317.960 | 239.006→209.546 |
| b4_even_s3/inference_mode | 71.389→63.637 | 238.723→200.338 | 330.317→318.883 | 239.079→208.749 |
| b1_odd_s3/no_grad | 56.841→29.282 | 114.445→101.495 | 123.576→99.925 | 121.105→103.420 |
| b1_odd_s3/inference_mode | 53.063→28.733 | 114.135→101.895 | 120.946→99.509 | 122.486→105.768 |
| b4_odd_s3/no_grad | 106.026→97.202 | 334.266→310.824 | 341.485→319.804 | 334.097→304.813 |
| b4_odd_s3/inference_mode | 103.749→96.705 | 331.824→305.126 | 341.665→315.309 | 330.786→301.718 |
| b1_medium_s3/no_grad | 84.168→76.954 | 273.708→248.604 | 345.020→328.806 | 278.305→246.490 |
| b1_medium_s3/inference_mode | 84.202→76.328 | 250.735→217.500 | 313.857→289.247 | 283.228→249.773 |
| b1_scale4_generic/no_grad | 61.070→22.096 | 103.205→73.950 | 99.968→79.105 | 100.018→79.874 |
| b1_scale4_generic/inference_mode | 54.375→20.524 | 96.358→73.684 | 95.803→79.306 | 101.341→74.793 |
| b1_odd_strided/no_grad | 55.833→28.987 | 112.771→103.219 | 120.642→100.196 | 116.551→98.412 |
| b1_odd_strided/inference_mode | 54.150→28.637 | 117.755→102.313 | 122.486→101.418 | 121.091→101.520 |
| b4_channels_last_shared/no_grad | 71.073→63.429 | 237.616→206.317 | 345.223→314.859 | 237.166→209.227 |
| b4_channels_last_shared/inference_mode | 69.584→63.394 | 237.145→208.060 | 342.896→317.303 | 242.592→209.132 |

首次runner时间包含warmup2、capture、copy、replay及output clone；它与已命中的runner是不同口径。显存数据是live state之上的allocated peak增量，不能当作总VRAM或跨shape容量上限。
