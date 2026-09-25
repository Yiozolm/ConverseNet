# 训练耗时、真实热点与单一浮点失败定位

本轮从 `1d804b1` 新建 `codex/training-hotspot-followup`，完成三个聚焦任务及既有生产 LayerNorm 的形状覆盖。生产源码仍为 `ff7f8ba`，没有接入新的训练候选，也没有扩大推理分派条件。全部 GPU 任务串行执行；FP32/complex64、确定性比较、TF32/AMP 关闭，原发布契约保持。

## 结论

| 任务 | 完成的产出 | 验收结果与范围 |
|---|---|---|
| 解释完整训练未加速 | 七个原 session 的互斥时间账本；八次 ABBA/BAAB 短完整流程对照 | 历史新增时间主要在前向/loss/backward。新短训练步中位比 1.026×，但原 4000 步全程未加速，历史时变因素的成因未确定。 |
| 针对真实训练热点 | 正式数据协议 B4 module/call/kernel 归因；一个仅改变 wgrad 调用布局的受限候选 | wgrad 占真实 GPU kernel 总时间 23.28%，LayerNorm 8.99%。候选 54 个张量逐字节与 FP64 门槛通过，完整层却慢 18–20%，停止推进。 |
| 做透一个路线 B 失败 | nearest_spectral 的原案例与确定性最小输入；共同阶段比较、ULP 和单变量干预 | 固定 k3/s2 下最小 B1C1 LR2×2，首次共同偏离为 prior 频谱 P；相位角度量化及其传播留下 2^-25 输出误差，原 Python FP32/FP64 则精确。未提供修复或 VJP 准入。 |
| 扩展推理侧证据 | 八种代表形状 × no_grad/inference_mode；真实 padding 尺寸、布局回退及完整 Graph runner | 16/16 数值与生命周期检查通过。既有 LayerNorm 组件在完整模型中的 eager 配对中位比范围 1.072–1.321×，Graph hit 为 1.050–1.132×；仅适用于所测形状和上下文。 |

## 训练时间的具体解释

历史 seed17/29/43 的完整进程分别增加 **10.487 / 64.806 / 141.331 秒**。其中 forward/loss/backward 增加 **18.593 / 73.599 / 132.269 秒**，而评估主体分别减少 **11.969 / 8.814 / 7.599 秒**。初始 checkpoint 从 setup 扣除后单独计入一次；原失败 session 与恢复成本保留。数据列为同步 CPU 准备/校验/哈希，不冒充异步等待。CUDA event span 与 kernel busy 时间不同。

新短对照每进程 64 个相同更新、0/32/64 三次完整 100 图评估，原 worker、初始化、配方和 checked 二进制未改。八次的全部数据步、loss、梯度范数、逐图指标、最终 133 个模型及 399 个 Adam 张量一致。四组训练步配对比为 **1.008–1.039×**，中位数 **1.025722×**；process 为 **1.039–1.069×**，中位数 **1.055689×**。

短流程的评估密度高于原 4000 步/17 次评估，process 收益不能外推到长期训练。同一历史轨迹内的快慢比例也会改变，已有记录不能判定是频率、温度、竞争还是其他时变因素导致。新采集的 1 Hz GPU 状态只描述本轮，不能补出历史遥测。这次解释确定了“时间在哪一阶段”，没有制造已证实的硬件因果或长训练收益。

完整账本、分块图与短流程边界见 [训练计时报告](training_followup_timing.md)。

## 热点候选为何停止

正式数据 B4 训练步中，64→128 的 70 次 1×1 wgrad 占 kernel 总时间 **17.95%**，128→64 的 70 次占 **5.30%**。本次候选只覆盖一个 128→64 模块的五次调用，保留原 forward、input/bias VJP 和高阶回退；仅 weight VJP 在内部转换成 channels_last 后调用 cuDNN。

完整层 AB/BA 含全部转换后加速比仅 **0.831–0.847×**，额外 allocated 峰值从 **28,344,832** 增至 **56,656,896 bytes**。诊断 trace 证实 NHWC wgrad 内核变快，但新增两个布局 copy 及其他工作抵消了收益。没有把内核片段收益当作完整层收益，也没有把这个负结果泛化为整个布局方向不可行。未扩大到全部模块或替换完整训练流程；训练 LayerNorm 本轮只完成占比归因。

层级路径、逐张量门槛和全部轮次见 [wgrad 报告](training_followup_wgrad.md)。

## 一个可独立重现的浮点失败

最小输入为单点 `x=[[1,0],[0,0]]`、中心为 1 的 3×3 单位核、bias=0、nearest prior、s2。原路径 `FFT2(nearest(x))` 与候选通过重复 LR 频谱和两轴 phase 构造 P 的路径首次在 P 分叉。候选的 Nyquist 相位含约 `8.74e-8` 虚部，最终零像素残差为 `2.9802322387695312e-8`；原 FP32 与独立 FP64 的误差均为零。

只换回原 P 就恢复原输出字节；只注入候选 P 就复现候选字节。kernel FFT、求解公式、归约、正则与缓存均未修改。形状最小性只针对固定 k3/s2，原 45/96 失败矩阵不重判。见 [首次偏离分析](training_followup_route_b.md) 和随分支保存的 [输入字节包](training_followup_route_b_inputs.json)。

## 推理覆盖与证据

代表形状来自现有模型/evaluator 支持范围，包含 B1/B4、LR32²、31×33、64²、s4 矩形、非连续输入和 shared kernel。真实 forward 核对 prior padding 后 FFT 尺寸，包括 100²、97×103、196²、100×104。未获得线上 shape/batch 流量分布，不主张总体部署收益。

原 caller eager 与 Graph 内部 inference tensor 的缓存资格不同，其不等观察仍保留；相同 capture context 的控制全部逐字节通过。输出独立性、LRU、值/stride 更新、clear、disabled runner、模型状态/版本/绑定、源码/二进制/亲和性检查均通过。完整 eager、disabled runner、Graph hit、首次 capture、空缓存调用分别计时，不把已初始化进程的首次 capture 称为冷进程启动。

全部 16 行及范围见 [部署覆盖报告](training_followup_inference.md)。机器汇总见 [training_followup_results.json](training_followup_results.json)，原始报告、失败与诊断 traces 的无损归档见 [证据索引](training_followup_evidence/index.json)。大型 checkpoint/activation fixture 保留在本地 artifacts 并列出 SHA，不加入 Git。

新增四项 CPU 账本测试通过；反例 CPU/CUDA/独立 CPU 回放均成功复现失败。第二人独立复算历史账本、八进程配对，以及推理全部 672 条正式轮次计时和 96 个配对数组，未发现阻断项。本轮未改生产、冻结训练 helper 或旧研究证据，未把短轨迹称为稳定性或收敛证明。
