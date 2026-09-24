# FP32 整理版（2026-09-24）

分支：`codex/fp32-clean`。起点是开发分支 `4c6314d` 加整理时工作区中尚未提交的全谱训练等改动，已先完整保存为快照 `1b579ea`。

## 运行契约

- 生产输入、参数、输出只接受 FP32，频谱为 complex64。CUDA 只实例化 float 内核；不再提供 FP16/BF16/FP64 算子或 v2–v6。
- 开启梯度且任一输入需要梯度：全谱训练。CUDA 保留 s1、受限 s2 专用融合及其余通用实现；核准备与 FFT/IFFT 保持逐调用 FP32 自动求导。高阶梯度保留 ATen 回退。
- `no_grad()` / `inference_mode()` 或全部输入冻结：半谱推理。按梯度需求分派，`eval()` 本身不切换频谱。
- 固定核推理缓存、版本失效、跨流与 CUDA Graph 生命周期管理保留。旧训练谱缓存 API、`reuse_training_spectra` 参数已移除。
- CPU 扩展与无扩展的 `auto` 回退也遵循训练全谱、推理半谱。显式 `backend="pytorch"` 保留为原全谱 FP32 对照。独立 Python reference 的 FP64 仅用于误差度量。

没有更改 CUDA FP32 内核算术、归约或共享梯度的累加顺序，没有新增融合、跨调用训练谱复用、fast-math、TF32 或 AMP。清理了未使用的旧训练、低精度、研究内核、兼容导出和实验脚本。构建输入从 19 个翻译单元缩减为 13 个。

## 验证

设备为 RTX 5060 Ti，PyTorch 2.11.0+cu130，CUDA Toolkit 13.0，MSVC 14.44，sm_120。验证使用本轮源码重新构建的扩展和 source/binary 指纹。训练对照关闭 TF32、AMP 和 cuDNN benchmark，启用 cuDNN deterministic。

| 检查 | 结果与范围 |
| --- | --- |
| 最终 CUDA 构建测试集 | 32 项测试全部通过，无失败、无跳过 |
| Python FP32 训练对照 | 保留的 s1/s2/s3 输出、全部 VJP、共享输入、广播、非连续布局、梯度子集、弱正则逐字节检查全部通过 |
| 独立 FP64 非劣 | s1–s4，正常及两种弱正则配置，共 60 个输出/梯度张量；max_abs 与 relative-L2 均不高于 Python FP32，零额外裕量 |
| 高阶、缓存、流及 Graph | FP32 二阶导数对照、缓存参数更新、跨流、图捕获/失效/生命周期检查通过 |
| CPU 行为 | 全谱训练输出及 VJP 与 Python FP32 一致；无扩展回退的全谱/半谱分派及 dtype 拒绝检查通过 |
| 独立 CPU-only 构建 | 编译通过，1 项 CPU 测试通过；15 项 CUDA 测试按设计跳过，与上面的 32 项 CUDA 构建测试分别计数 |
| 预训练权重 | DnCNN、SRResNet 的 FP32 半谱推理与 Python 参考在原 smoke 容差内一致，严格加载权重成功 |
| 整理前后对照 | 6,140 个张量 SHA256 全部相同：128 个推理输出，以及完整预训练 USRNet 的 6,012 个训练/评估张量 |

推理对照覆盖 s1–s4、四种核广播组合、固定叶子核与动态非叶子核、冷/热调用，以及 no_grad/inference_mode。USRNet 对照保留完整 5 次迭代、7 个 prior block 的架构；三个种子 17/29/43 各做 3 步 Adam1e-5，使用仓库真实图片的 HR24 裁剪、s3、batch1。逐步核对输出、loss、全部梯度、更新后参数、Adam step/一二阶矩及半谱评估输出。

这是**整理前后数值保持检查**，不是新增的数据集 PSNR/SSIM、长期收敛或速度证明。先前研究中的失败仍保持原结论，不借此重判为通过。

机器可读的源文件/二进制指纹、原始结果文件 SHA256 与检查汇总见 [fp32_validation.json](fp32_validation.json)。本地原始数据保存在 `artifacts/fp32_release/`；不纳入 Git。

## 复现

构建和 API 见 [Converse2D/README.md](../Converse2D/README.md)，检查范围见 [test/README.md](../test/README.md)。

```powershell
$env:CONVERSE_PYTHON = '路径/python.exe'
$env:CONVERSE_MSVC_VERSION = '14.44'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
./tools/run.ps1 -m unittest discover -s test -p 'test_*.py' -v
./tools/run.ps1 test/release_snapshot.py --root '清理前的工作树' --output artifacts/fp32_release/before.json
./tools/run.ps1 test/release_snapshot.py --output artifacts/fp32_release/after.json --compare artifacts/fp32_release/before.json
```

Linux 在匹配的 PyTorch/CUDA/C++ 环境中直接执行对应 `python` 命令。源码哈希不是跨平台数值保证；更换 GPU、编译器或 PyTorch 后须重新验证。

## 历史与迁移

原开发工作树及未提交改动保留。完整的旧源码、实验入口、报告和失败记录也保存在 `1b579ea`；例如 `git show 1b579ea:docs/training_scale2.md`。需要重放旧实验时，另建该提交的工作树，避免混用整理版构建。

旧调用移除 `reuse_training_spectra` 参数，variant 使用默认 v7，张量保持 float32。state_dict 的参数名与形状不变。必须重建扩展并重启已加载旧二进制的 Python 进程。
