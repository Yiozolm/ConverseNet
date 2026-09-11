# 数值简化分支缺陷修复

目标分支：`codex/numerically-stable-closed-form`，基于 `dcc9896`。

修改位于附着该分支的独立 Git worktree：
`D:/Python/ConverseNet/.build/simplified_fix`。当前主工作区仍为 main，生产代码未改；简化分支的修改尚未提交或推送。

## 已修正

1. **独立 x0**：仅当 x 与 x0 是同一个张量对象时复用 FFT(x)，scale=1 也保留独立先验及其梯度。
2. **权重缓存过期**：保留原始权重的张量身份，核对版本、存储地址、计算精度、设备、频谱尺寸与 CUDA 流；原地更新权重后不再命中过期频谱。
3. **inference-mode 到反传**：所有 autograd 启用的调用均绕过缓存。inference tensor 没有版本号时也绕过缓存；缓存区分 inference-mode 状态。
4. **低精度输入**：C++、Python Converse2D 和 ConvReverseDataNet 内部以 float32 处理 float16/bfloat16，结果转回输入精度，转换保留梯度。
5. **构建与导入**：支持 MSVC 编译参数；可选择 CUDA 链接以支持设备保护和按流缓存；CPU-only 构建对 GPU 调用绕过缓存；包导入先加载 PyTorch 的动态库依赖。

仍使用原分支的全 FFT 残差公式和六参数算子接口，没有引入 main 的融合 CUDA/rFFT 实现。

## 验证结果

- 分支原有 12 组测试体分别在 CPU/GPU 复跑：**24/24 通过**。
- 上轮发现问题的边界对照：**32/32 通过**，其中简化分支自身 16 项、当前 main 对照 16 项；原来失败的 10 项全部转为通过。
- 新增 `test/test_regressions.py`：**9 项通过**，包括稠密空间域求解、四类梯度、gradcheck/gradgradcheck、缓存更新、inference tensor、连续反传、半精度、Python 模块、非默认 CUDA 流和非法输入。
- 无 CUDA 链接的 C++ 构建再次运行上述测试：**9 项通过**，同时验证 CPU 运算和 GPU 的安全缓存回退。
- `setup.py build_ext --inplace` 构建成功；实际安装入口生成的包通过 float64/float16/bfloat16 模块前向与梯度检查。
- 直接 `import torch_converse2d`、已加载扩展时的回归入口复核通过。

关键复现对比（float64）：

| 原失败项 | 修复前 GPU 结果 | 修复后 GPU 结果 |
|---|---|---|
| scale=1、独立 x0，对稠密求解最大绝对差 | 2.463e-2 | 1.776e-15 |
| x0 梯度 | None | 存在且与独立参考一致 |
| 权重更新后，清缓存前后输出最大差 | 1.0989 | 0 |
| inference 缓存 → 输入反传 | 报错 | 通过且梯度正确 |
| 非规则尺寸 float16/bfloat16 | 报错 | 前向、反传通过 |

CPU 上独立 x0 的最大绝对差从 5.825e-2 降至 1.954e-14，权重更新后的缓存差异同样为 0。scale=1/2/3/4 的 CPU/GPU 稠密对照最大误差小于 6e-14。

同输入 GPU 推理/训练测速也已复跑，保留在结果 JSON 的 `benchmarks` 中。完整频谱缓存上限设为 512 MiB，以容纳 64 通道、512×512 的 FB/FBC/F2B（合计约 320 MiB），避免初设 256 MiB 上限导致大尺寸每次重算。性能仍是全 FFT/ATen 路径的水平；本次不将不同轮次的时钟/调度变化解读为修复带来的加速。

## 文件与复现

- [C++ 算子与缓存](../.build/simplified_fix/Converse2D/torch_converse2d/converse2d.cpp)
- [Python Converse2D](../.build/simplified_fix/models/util_converse.py)
- [ConvReverseDataNet](../.build/simplified_fix/models/converse_usrnet.py)
- [新增回归测试](../.build/simplified_fix/test/test_regressions.py)
- [分支构建说明](../.build/simplified_fix/Converse2D/README.md)
- [复测 JSON](simplified_branch_fixed_results.json)

本机从主工作区运行：

```powershell
& ./.build/run.ps1 .build/simplified_fix/test/test_regressions.py
& ./.build/run.ps1 test/test_simplified_branch.py --source-tree .build/simplified_fix --output analysis/simplified_branch_fixed_results.json
git -C .build/simplified_fix diff
```

`.data` 或裸指针写入若绕过 PyTorch 版本计数，仍需显式调用 `torch.ops.converse2d.clear_cache()`；本次没有声称缓存能自动检测未被 PyTorch 跟踪的外部写入。
