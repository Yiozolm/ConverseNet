# nearest 相位的定向修复

本轮在 `cad5d97` 上新建 `codex/nearest-phase-repair`，只处理上一轮最小反例中已经定位的相位系数问题。**CPU/CUDA 最小反例修复通过，原完整矩阵仍有 45/96 个张量失败，因此停止性能与生产推进。** 候选作为隔离实验保存，不接入生产分派；旧 `45/96` 失败记录、旧研究源码和发布源码保留。

## 唯一修改

对长度 n、放大倍数 s 的单轴系数

```text
S_s(f) = sum(a=0..s-1, exp(-2*pi*i*a*f/n)),  0 <= f < n
```

先完整执行旧 FP32 `angle → polar → 顺序累加`，然后只覆盖以下可精确判定的最终系数。普通频点的完整复数系数字节不动，不改变单个一般角度项，不做整体 Hermitian 对称化。

- DC：f=0 时为 s。当前测试 s2/s3/s4 均可在 FP32 精确表示；辅助函数对更大整数只在 s≤2^24 的保守范围赋精确 DC。
- 四次单位根：`4*f` 可被 n 整除时，令 q=`4*f/n`，利用周期不超过 4 的整数根和。
- 非 DC 的几何零点：`f*s` 可被 n 整除时，整个几何和为 0。实现用 `g=gcd(n,s)`、`f=j*(n//g)`，j=1..g-1 枚举，避免浮点角度比较。

| q | s2 | s3 | s4 |
|---:|---|---|---|
| 0 / DC | 2 | 3 | 4 |
| 1 | 1−i | −i | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 1+i | i | 0 |

位置判定使用 Python 整数；实际赋值张量仍为 complex64，不将 FP64 用作候选计算。零分量采用明确的正零表示。边界检查不代表扩大算子可支持的 scale、尺寸或上下文。

[`candidate.py`](../tools/nearest_phase_repair/candidate.py) 的 `nearest_spectral` 与冻结旧函数相比只插入一行 `z = apply_exact_coefficients(z, n, s)`。检查、kernel FFT、观测 FFT、两轴乘法顺序、求解公式、广播/归约、IFFT 和几何缓存的 key/容量/淘汰代码保持原样。共享 helper 的 AST 与旧研究函数核验一致。原几何缓存并未因此取得新的流、Graph 或跨上下文生命周期准入。

## 几何与最小反例

独立 CPU 检查使用 Fraction 约分和 Gaussian integer 根和作为判定依据，覆盖 s2/s3/s4、LR 每轴 1..129：387 个轴、75,465 个频点，其中 1,417 个被覆盖点精确，74,048 个普通 complex64 系数的原字节全部保留。另有 27 个稀疏大整数边界检查；没有分配这些巨大尺寸的张量。CUDA 未在该检查中初始化。

固定 k3/s2 的最小 B1C1、LR2×2 单点输入配单位核，在 CPU/CUDA 均得到：

| 最小例输出相对同一独立 FP64 | 旧候选 | 定向修复 | Python FP32 |
|---|---:|---:|---:|
| max absolute | 2.9802322387695312e-8 | 0 | 0 |
| relative L2 | 1.4901161193847663e-8 | 0 | 0 |

修复后的两轴均为 `[2, 1-i, 0, 1+i]`，最终输出与 Python FP32 逐字节相同；相同上下文的第二次缓存调用也相同。这一结果只关闭最小反例，完整输出/VJP 矩阵另行验收。

原输入字节包沿用 [上轮最小反例包](training_followup_route_b_inputs.json)。新增证据为 `artifacts/nearest_phase_repair/geometry_v1.json` 和 `minimal_v1.json`，旧报告没有被覆盖。

## 原完整矩阵重新验收

[`gate.py`](../tools/nearest_phase_repair/gate.py) 从固定提交 `cc244e3` 提取原 `specifications`、CUDA `capture`、误差 `record` 和 fixture 生成语句，逐项核对源码 SHA/AST。使用原 24 个案例及次序：B2/C3、LR5×7、k3、s2/s3/s4、四种 KB/KC 广播、正常/弱正则。种子是 `41191+index`，未使用的 raw prior 仍按原顺序消耗随机数，上游梯度及 weak 缩放顺序保留。

每个案例分别执行同一量化输入的独立 Python FP64、Python FP32、旧候选和修复候选。nearest prior 在各路从该路 x 构造；dx 包含 interpolation 分支，没有单独 dprior。每例的 output0、dx、dweight、dbias 各自要求有限，且 max-absolute 与 relative-L2 均不得超过 Python FP32 对同一 FP64 的误差，零额外裕量。

GPU 为 RTX 5060 Ti，PyTorch 2.11.0+cu130；AMP/TF32 关闭，确定性算法开启。本次重跑旧候选的 **96 个误差字典、finite 与通过标记全部和旧记录精确相同**，原 45 个失败保留。随后修复候选完成全部 96 项，结果如下：

| 张量 | 旧失败数 | 修复后失败数 |
|---|---:|---:|
| output0 | 17/24 | 17/24 |
| dx | 17/24 | 15/24 |
| dweight | 7/24 | 8/24 |
| dbias | 4/24 | 5/24 |
| 合计 | 45/96 | 45/96 |

失败总数相同不代表失败集合未变。48 项保持通过，42 项保持失败；3 项从失败变为通过，另有 **3 项新失败**：

| index（从0起） / seed | scale | KB/KC | weak | 张量 | 变化 |
|---|---:|---|---|---|---|
| 8 / 41199 | 3 | 1/1 | 否 | dweight | 通过 → 失败 |
| 10 / 41201 | 3 | 1/3 | 否 | dweight | 通过 → 失败 |
| 11 / 41202 | 3 | 1/3 | 是 | dx | 失败 → 通过 |
| 16 / 41207 | 4 | 1/1 | 否 | dbias | 通过 → 失败 |
| 18 / 41209 | 4 | 1/3 | 否 | dweight | 失败 → 通过 |
| 23 / 41214 | 4 | 2/3 | 是 | dx | 失败 → 通过 |

例如原 seed41191 的 output 最大误差从 `2.3393899724055700e-6` 降至 `1.7593395371662268e-6`，但 Python FP32 为 `1.7303519825873082e-6`；relative-L2 也仍高于 Python FP32，因此该项仍失败。index8 的 dweight 最大误差则从 `2.7113493341612838e-6` 升至 `2.830558623712065e-6`，超过 Python FP32 的 `2.7113493341612838e-6`，即使 relative-L2 仍通过也必须拒绝。

特殊系数的数学修复已成立，完整 FP32 算子与 VJP 的零裕量非劣条件仍未满足。本次不改变其他频点、FFT、公式、归约或缓存以追逐通过，也不推断所有剩余失败都来自某一个未经定位的机制。按照继续条件，**未做性能测试、模型替换、训练质量或生产集成**。矩阵工具返回退出码 4，affinity wrapper 保留 `task_failed`；这是完整测完后的数值拒绝，不能改写成候选通过。

## 证据身份与复现

本次新增每项输入、上游、派生 prior、四路输出/VJP 的 dtype、shape、原始字节与 SHA。旧研究报告当时没有这些张量哈希，当前字节包只标为本次测量，不能倒填成历史字节证明。reference 整文件在 Git 与工作树中分别使用 LF/CRLF 换行，原始字节 SHA 因此不同；规范化全文及所用三个参考函数 AST 一致。本次从固定 Git 源取独立参考；旧报告没有 reference 整文件 SHA 的限制如实保留。

所有原始报告、输入/输出字节包、日志和 wrapper 记录见 [无损证据索引](nearest_phase_repair_evidence/index.json)。本轮的生产源码、旧 reproducer、旧研究矩阵均未修改。

```powershell
.venv/Scripts/python.exe -B tools/nearest_phase_repair/check_geometry.py --output artifacts/nearest_phase_repair/geometry_NEW.json
.venv/Scripts/python.exe -B tools/nearest_phase_repair/minimal.py --output artifacts/nearest_phase_repair/minimal_NEW.json
.venv/Scripts/python.exe -B tools/nearest_phase_repair/gate.py --stage source --output artifacts/nearest_phase_repair/source_NEW.json
$env:CONVERSE_MSVC_VERSION='14.44'
$env:TORCH_CUDA_ARCH_LIST='12.0'
pwsh -NoProfile -File tools/run_affinity.ps1 -Mask 12598275 -MetadataPath artifacts/nearest_phase_repair/gate_NEW_affinity.json tools/nearest_phase_repair/gate.py --stage matrix --device cuda --output artifacts/nearest_phase_repair/gate_NEW.json
```

使用全新输出名；source 检查不初始化 CUDA。matrix 入口会再次通过 CPU/CUDA 最小例后再运行原矩阵；仍有任何张量失败即返回 4。这里没有新 CUDA extension、构建或发布后端，以上工具执行纯 ATen 隔离研究。
