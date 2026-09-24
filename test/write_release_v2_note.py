"""Generate the 2.0.0 draft from the measured release comparison artifacts."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'artifacts/release_v2'
op=json.loads((RAW/'operators.json').read_text())
model=json.loads((RAW/'models.json').read_text())
quality=json.loads((RAW/'quality.json').read_text())
assert len(op['results'])==11 and len(model['results'])==5
assert len(quality['runs'])==12 and len(quality['quality_gates'])==18
assert op['source_hashes']==model['source_hashes']==quality['current_source_hashes']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def shape(row):return '×'.join(map(str,row['shape']))+f" / s{row['scale']}"
def median(row,key,name):return row[key]['medians'][name]['wall_ms']
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])
def timing_table(rows,key):
    result=[]
    for row in rows:
        a,b,c=[median(row,key,n) for n in ('original_python','v1','v2')]
        result.append([shape(row),f'{a:.4f}',f'{b:.4f}',f'{c:.4f}',f'{a/c:.2f}×',f'{b/c:.2f}×'])
    return table(['B×C×H×W / scale','原 Python ms','1.0.0 ms','2.0.0 ms','2.0 / 原 Python','2.0 / 1.0'],result)
def noninferiority(data):
    failures=[]
    for key,v in data['v2'].items():
        ref=data['stable_python_fp32'][key]
        assert v['finite'] and ref['finite']
        if any(v[m]>ref[m] for m in ('max_abs','relative_l2')):failures.append(key)
    return failures

normal_failures=[dict(shape=shape(r),failures=noninferiority(r['training_errors'])) for r in op['results']]
stress_failures=[dict(scale=r['scale'],weak=r['weak'],failures=noninferiority(r['errors'])) for r in quality['stress']]
full_failures=noninferiority(quality['full_model_precision'])
assert not any(r['failures'] for r in normal_failures+stress_failures) and not full_failures
fixed=[r for r in op['results'] if r['kind']=='fixed']
dynamic=[r for r in op['results'] if r['kind']=='dynamic']
mi=[r for r in model['results'] if r['kind']=='usrnet_inference']
mt=[r for r in model['results'] if r['kind']=='usrnet_adam_step']

infer_rows=[]
for r in mi:
    v=r['timing']['medians']
    infer_rows.append([shape(r),*[f"{v[n]['wall_ms']:.3f}" for n in ('original_python','v1','v2','v1_graph','v2_graph')],
                       f"{v['original_python']['wall_ms']/v['v2']['wall_ms']:.2f}×",
                       f"{v['original_python']['wall_ms']/v['v2_graph']['wall_ms']:.2f}×"])

inference_errors=[]
for r in (fixed[0],fixed[1],*mi):
    data=r.get('inference_errors',r.get('errors_vs_fp64'))
    inference_errors.append([shape(r),*[f"{data[n]['max_abs']:.3e}" for n in ('original_python','v1','v2')]])
grad_rows=[]
for r in op['results']:
    entries=[]
    for n in ('original_python','v1','v2'):
        values=[v for k,v in r['training_errors'][n].items() if k!='output']
        entries.append(f"{max(v['max_abs'] for v in values):.3e} / {max(v['relative_l2'] for v in values):.3e}")
    grad_rows.append([('固定 ' if r['kind']=='fixed' else '动态 ')+shape(r),*entries])
full_rows=[]
for n in ('original_python','v1','v2','stable_python_fp32'):
    data=quality['full_model_precision'][n]
    gradients=[v for k,v in data.items() if k!='output']
    full_rows.append([{'original_python':'原 Python','v1':'1.0.0','v2':'2.0.0','stable_python_fp32':'稳定 Python FP32（仅校验）'}[n],
                      f"{data['output']['max_abs']:.3e}",f"{max(v['max_abs'] for v in gradients):.3e}",f"{max(v['relative_l2'] for v in gradients):.3e}"])
stress_rows=[]
for r in quality['stress']:
    if not r['weak']:continue
    stress_rows.append([f"s{r['scale']}",*[f"{max(v['relative_l2'] for v in r['errors'][n].values()):.3e}" for n in ('original_python','v1','v2')]])

quality_rows=[]
for seed in (17,29,43):
    runs={r['backend']:r for r in quality['runs'] if r['seed']==seed}
    for space in ('rgb','y'):
        quality_rows.append([f'{seed} / {space.upper()}',*[f"{runs[n]['after']['mean'][space]['psnr']:.6f} / {runs[n]['after']['mean'][space]['ssim']:.8f}" for n in ('original_python','v1','v2')]])
passed=sum(r['passed'] for r in quality['quality_gates'])
gates={control:dict(worst_delta_psnr_db=min(r['delta_psnr_db'] for r in quality['quality_gates'] if r['control']==control),
                    worst_delta_ssim=min(r['delta_ssim'] for r in quality['quality_gates'] if r['control']==control))
       for control in ('original_python','v1','stable_python_fp32')}
equivalence=[v for r in op['results'] for v in r['original_fp64_vs_stable'].values()]
normal_count=sum(len(r['training_errors']['v2']) for r in op['results'])
stress_count=sum(len(r['errors']['v2']) for r in quality['stress'])
full_count=len(quality['full_model_precision']['v2'])

text=f'''# ConverseNet 2.0.0 Release Notes（草稿）

2.0 将生产算子收敛为 **FP32 训练全谱、推理半谱**：训练使用融合 CUDA 前后向，FFT、核准备与 λ 参数化保留自动求导；推理沿用稳定残差形式、实数 FFT 与缓存。已有 checkpoint 的参数名称和形状保持兼容。

本说明参照 [1.0.0 release note](https://github.com/Yiozolm/ConverseNet/releases/tag/v1.0.0) 的结构。比较对象为上游原 Python、正式 1.0.0 标签和当前 FP32 分支，所有下面的数字均为本轮同机重新测量。源码目标为 `codex/fp32-clean` 的 `{op['v2_ref'][:7]}`；本文不表示已创建或发布 v2.0.0 标签。

## 全部已落地变化

### 1. 训练改为全谱 FP32 CUDA 融合前后向

- 1.0 的训练是可微半谱 ATen；2.0 采用 complex64 全谱求解核心和手写解析一阶 VJP。
- 核 pad/roll/FFT、输入与先验 FFT、`λ = sigmoid(bias - 9) + eps`、输出 IFFT 仍在逐调用 FP32 自动求导图中。没有 detach 缓存、参数更新前的谱复用或省略梯度。
- 高阶导数回退到可微 ATen。按 GradMode 与输入梯度需求选择路径，`eval()` 本身不切换频谱。

### 2. s1、s2 融合与数值顺序

- s1 合并前向预测、分母、输出与反向逐点 VJP；共享输入不再分配独立 gy，但保留共享梯度累加顺序。
- 仅核 batch/channel 均不广播时，将最终核梯度逐点合成并入 VJP；广播核保留原归约。
- s2 在 LR 宽度大于 1、HR complex 张量字节数不超过 INT32_MAX 时使用四别名有序融合；其余 s2 与 s≥3 使用通用全谱实现。
- 保留 complex 分母梯度实部视图的 stride、ATen 广播归约与原 FMA 边界；不启用 fast-math 或降低精度。

### 3. 推理继续采用半谱及版本化缓存

- 保留 1.0 的残差公式、rfft2/irfft2、Hermitian 索引、PSF 补零/移位融合和动态功率谱累计。
- 固定叶子核可复用准备后的频谱与分母；动态核逐调用准备。缓存按张量身份、版本、存储、形状、倍率、设备、stream 等条件失效。
- 保留可选 CUDA Graph 推理及图拥有的频谱生命周期；没有训练 Graph。本轮不把基本持平的 Graph 结果包装为新的加速。

### 4. FP32 产品范围与工程整理

- 生产张量仅接受 float32，频谱为 complex64。移除低精度和旧半谱训练实现、实验兼容入口及训练谱缓存 API；CUDA 只实例化 float 内核。
- CPU 扩展和无扩展 auto 回退也采用训练全谱、推理半谱。显式 `backend="pytorch"` 保留稳定全谱参考；FP64 仅作独立数值校验。
- 构建输入从 19 个翻译单元精简为 13 个，保留源码/头文件/二进制指纹校验与 Windows 构建入口。历史成功和失败记录保存在清理前快照 `1b579ea`。

## 性能记录与测试条件

**“原 Python”沿用 1.0 发布说明的定义：未经算数化简的上游原始实现，不是当前稳定残差 `backend="pytorch"`。**

- 原 Python：[`368aa39`](https://github.com/cszn/ConverseNet/commit/{op['original_ref']}) 的 `util_converse.py` 和 `converse_usrnet.py`，保留原 CPU 零张量创建、type_as 搬运、零插值和完整 FFT，只重定向模块导入。
- 1.0：[`v1.0.0 / e795a38`](https://github.com/Yiozolm/ConverseNet/tree/v1.0.0)，从该标签提取并重新编译；只改注册命名空间以与 2.0 同进程加载，未改计算代码。
- 2.0：`{op['v2_ref'][:7]}` 当前 FP32 算子，实测绑定源码及二进制 SHA256；不是之前 dev 或半谱训练实验二进制。
- RTX 5060 Ti，PyTorch {op['torch']}，CUDA Toolkit 13.0，MSVC 14.44，sm_120，CPU 线程数 24；FP32，关闭 TF32/AMP/cuDNN benchmark，启用 cuDNN deterministic。
- 同进程、同 GPU、同输入/权重/checkpoint。每轮预热 5 次，轮换执行顺序。算子推理 7 轮×50 次、训练 VJP 7 轮×20 次；整网推理 5 轮×10 次、Adam 步 5 轮×5 次。取同步 wall time 中位数；原始 CUDA-event 时间及每轮数据一并保留，不使用 profiler 计时。
- 加速比定义为“基线耗时 ÷ 2.0 耗时”，小于 1 表示 2.0 更慢。各类收益分别计算，不相乘。首次编译、CUDA 初始化、Graph 捕获及外部数据加载不计入稳态时间。

### 固定 PSF：算子级完整推理

三版均为 7×7 核、padding=0、eps=1e-5，包含各自的完整前向和先验处理。1.0/2.0 固定核缓存预热；原 Python 保持无缓存。最后三行是本轮新增 batch=4 样例，其余形状沿用 1.0。

{timing_table(fixed,'inference')}

### 动态核 DataNet：完整推理

C=64、7×7 核、padding=0、eps=1e-3，三版都包含 kernel clone 和先验上采样。clone 在 inference_mode 内创建，1.0/2.0 每次重新准备动态核谱。

{timing_table(dynamic,'inference')}

推理并非全面快于 1.0：所测大形状多数接近持平，较小形状存在回退；表中保留全部结果，不筛掉较慢样例。缓存首次填充耗时没有冒充热缓存性能。

### 算子训练：完整前向＋全部输入/参数 VJP

固定核对 x、weight、bias 求梯度；动态核对 x、k、alpha 求梯度，包含 nearest 先验及 s1 共享输入的完整链式反向。使用相同固定上游梯度，`autograd.grad`，每次重建图与核 FFT；**不含优化器**，也不是只计自定义 CUDA 核。数值检查另算，不在计时循环中。

固定 PSF：

{timing_table(fixed,'training_vjp')}

动态 DataNet：

{timing_table(dynamic,'training_vjp')}

2.0 相对 1.0 的训练收益与形状有关。s1/s2 和动态核样例有收益；所测部分固定核 s3 的全谱训练比 1.0 半谱 ATen 更慢。不能将单个最佳比值外推到所有倍率、batch 或网络。

### 预训练 USRNet：普通推理和可选 Graph

三版加载同一完整 checkpoint，5 次迭代、7 个 prior block、batch=1、s2。原 Python 整网使用原恢复块。Graph 包含状态检查、GPU 输入复制、replay 和输出 clone；不含首次捕获。

{table(['LR 输入','原 Python ms','1.0 eager ms','2.0 eager ms','1.0 Graph ms','2.0 Graph ms','2.0 eager / 原 Python','2.0 Graph / 原 Python'],infer_rows)}

eager 轮间波动明显，原始轮次见结果文件；不能据这里较小的中位数差异认定稳定的版本收益。Graph 在这两个形状下两版接近持平。

### 完整 USRNet 训练步

计时包含 zero_grad、完整 forward、RGB MSE、全部参数 backward 和 Adam 更新；lr=1e-5，foreach=False，fused=False。每轮预热后恢复相同 checkpoint 和已分配的零 Adam 状态，模型、输入、目标已在 GPU。不包含数据读取、验证、保存或收敛时间。

{timing_table(mt,'timing')}

**本轮所测小形状中，2.0 完整训练步快于 1.0，但仍慢于上游原 Python。** 算子前后向收益不能直接作为整网训练加速结论；本说明不宣称已达到完整训练的整体速度目标。

## 精度单独核对

所有版本分别对照同一稳定全谱 FP64 参考，不以原 Python FP32 舍入误差作真值。11 组正常算子中，原 Python FP64 与稳定 FP64 的输出/梯度最大绝对差为 `{max(v['max_abs'] for v in equivalence):.3e}`、最大 relative-L2 为 `{max(v['relative_l2'] for v in equivalence):.3e}`，支持同数学定义的比较。

### 推理输出

下表为相对 FP64 的 max_abs。两个 USRNet 样例使用同 checkpoint 的稳定 Python FP64 整网作为精度参考，不计入速度基准。

{table(['样例','原 Python FP32','1.0.0 FP32','2.0.0 FP32'],inference_errors)}

### 训练梯度

每格为“该样例全部梯度张量中的最大 max_abs / 最大 relative-L2”，两种最大值可能来自不同张量；逐张量值保存在结果文件中。输出误差另存，未混入本表的梯度最大值。

{table(['训练样例','原 Python FP32','1.0.0 FP32','2.0.0 FP32'],grad_rows)}

完整预训练 USRNet 单个真实 HR48/s3、batch1 fixture 的输出、输入梯度及全部参数梯度共 {full_count} 张量：

{table(['版本','输出 max_abs','梯度最大 max_abs','梯度最大 relative-L2'],full_rows)}

2.0 在本轮 11 组正常算子的 {normal_count} 个输出/梯度张量、{stress_count} 个普通/弱正则附加张量和 {full_count} 个完整模型张量的 max_abs、relative-L2 均不高于稳定 Python FP32，**没有添加裕量**。这不表示每个指标都优于 1.0；例如完整模型的梯度最大绝对误差在 1.0 更小，2.0 与稳定 Python FP32 相同。

### 弱正则检查

附加固定核 s1/s2/s3，B2/C3、LR7×9、7×7 核；弱正则条件为 eps=1e-8、bias=-40、核幅度缩小至 1e-6、输入幅度 1e-5，未临时增大 λ。下表为输出及梯度张量中的最大 relative-L2：

{table(['倍率','原 Python FP32','1.0.0 FP32','2.0.0 FP32'],stress_rows)}

弱正则下也不是某个版本在每项误差指标上均最小；2.0 的验收基准仍为稳定 Python FP32，原始数据完整保留。

### 真实图片短程训练质量

沿用已有 1000 张图片的固定 900/100 拆分及五个 7×7 核，先核对图片/manifest 哈希。本轮三种子 17/29/43，各版本每种子 20 步，完整预训练 USRNet、FP32、HR48/s3、batch1、Adam1e-5/MSE；合成退化为循环模糊、相位零下采样和 σ=0.01 高斯噪声。另跑稳定 Python FP32 作精度控制，共 240 updates。

每次验证覆盖全部 100 个固定保留中心图块。按既有流程截断/舍入为 uint8、裁去 3 像素边界，计算 RGB 与 MATLAB Y 的 PSNR/SSIM。下表为训练后 100 图块均值，每格为 PSNR dB / SSIM：

{table(['种子 / 色彩空间','原 Python','1.0.0','2.0.0'],quality_rows)}

运行前固定每种子、每色彩空间的质量门槛：2.0 相对控制的 PSNR 下降不超过 0.05 dB，SSIM 下降不超过 0.001。与原 Python、1.0 和稳定 Python FP32 的 {len(quality['quality_gates'])} 组配对检查通过 {passed} 组；训练批次哈希逐项一致，loss/梯度/参数有限值检查通过。门槛未因结果修改。

这里只验证固定拆分图块上的短程微调，**不代表完整图像/独立测试集表现、长期收敛或达到相同质量的总时间**；旧的 250 步历史报告没有混入这些新结果。

## 安装与迁移

当前为源码分支草稿，使用已配置的匹配 PyTorch/CUDA/MSVC 环境构建。Windows 可通过 `tools/run.ps1` 运行 Python；其他平台使用对应 Python 命令。

```sh
git checkout codex/fp32-clean
python -m pip install ./Converse2D --no-build-isolation
```

- 1.0 的六参数 `torch.ops.converse2d.forward(x, x0, weight, bias, scale, eps)` 继续有效。默认 variant 为 v7，只支持 v7；不必额外传版本字符串。
- 输入与模型参数保持 float32；FP16/BF16/FP64 生产调用会被拒绝。删除研究代码中的 `reuse_training_spectra` 和私有训练缓存调用。
- 当前 `USRNetCUDAGraph` 构造不再接受 1.0 的 `enabled` 参数；需要可选开关时，在应用端选择普通 model 或显式创建的 runner，停用时调用 clear()。
- checkpoint 名称与形状兼容。更新后重新编译，并重启已加载旧二进制的 Python 进程。
- 本轮只准备 release note 和测量材料，尚未建立 v2.0.0 tag；源码包的 setup.py 元数据目前仍为 0.3.0，正式发布时须与 2.0.0 同步。

## 发布验证与复现材料

整理版已有 32 项回归全部通过，CPU-only 独立构建通过；清理前后 6,140 个张量哈希一致。覆盖全谱/半谱分派、共享/广播/非连续/单宽输入、梯度子集、高阶导数、缓存、跨流、Graph 生命周期及预训练权重。详见 [整理版验证](fp32_release.md)。本轮新增三版本性能、误差与短程质量检查，不重判历史失败。

- [三版本算子/整网基准](../test/benchmark_release_v2.py)
- [FP64、弱正则及真实图片质量脚本](../test/benchmark_release_v2_quality.py)
- [表格数据、全部计时轮次和精度明细](release_v2.0.0_results.json)
- 原始 operators/models/quality JSON、固定源文件和构建保存在 `artifacts/release_v2/`；原始文件 SHA256 已写入表格数据。

```powershell
./tools/run.ps1 test/benchmark_release_v2.py --phase operators
./tools/run.ps1 test/benchmark_release_v2.py --phase models
./tools/run.ps1 test/benchmark_release_v2_quality.py --data-root '数据所在仓库' --manifest 'split_900_100.json'
./tools/run.ps1 -m unittest discover -s test -p 'test_*.py' -v
python test/write_release_v2_note.py
```

速度脚本要求本地 Git 含原始提交及 v1.0.0；质量脚本还需既有 1000 图及固定 manifest。更换 GPU、PyTorch、编译器或 workload 后须重新测量；本轮结果不是全尺寸性能保证。
'''

summary=dict(schema=1,original=op['original_ref'],v1=op['v1_ref'],v2=op['v2_ref'],
             raw_sha256={p.name:digest(p) for p in (RAW/'operators.json',RAW/'models.json',RAW/'quality.json',RAW/'quality_protocol.json')},
             operators=op,models=model,
             quality={k:v for k,v in quality.items() if k!='runs'},
             quality_runs=[{k:v for k,v in r.items() if k not in ('before','after')}|
                           {'before':r['before']['mean'],'after':r['after']['mean']} for r in quality['runs']],
             noninferiority=dict(normal_tensors=normal_count,stress_tensors=stress_count,full_model_tensors=full_count,
                                 failures=normal_failures+stress_failures,full_failures=full_failures),
             quality_worst_deltas=gates)
(ROOT/'docs/release_v2.0.0.md').write_text(text,encoding='utf-8',newline='\n')
(ROOT/'docs/release_v2.0.0_results.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(f'Wrote release note; quality gates {passed}/18; independent noninferiority tensors {normal_count+stress_count+full_count}')
