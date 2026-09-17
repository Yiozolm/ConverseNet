# 测量前的假设（2026-09-16）

对象是当前 `converse2d_kernels.cu` 的 FP32、scale=2、独立 prior、动态功率分母路径。
不包含缓存 denominator、nearest prior 特化、训练反传，也不修改生产 dispatch。

现有实现每个线程串行读取 4 个 alias，生成 q；下一 kernel 再读 prior/filter/q，更新输出。
实验的半谱内部组保留这 4 对 prior/filter，算出 q 后直接写 4 个物理半谱频点。
边界列单独按输出频点拥有者计算；不把共轭边界的处理成本隐藏进理想模型。

每个内部组的 FP32 逻辑访问（不是 DRAM 计数器读数）：

| 实现 | 逻辑字节 / 内部组 |
|---|---:|
| 两遍 kernel | alias: 64(K/P)+8(Y)+4(lambda)+8(q); apply: 64(K/P)+32(q)+32(output) = 212 |
| 寄存器融合 | 64(K/P)+8(Y)+4(lambda)+32(output) = 108 |

模型假设 4 个输出均属于内部组；缓存、lambda 广播、sector 对齐、边界开销会改变实际 DRAM 字节。
仅由这些逻辑字节得到的带宽理想比为 212/108=1.96；这不是预计端到端加速比。

每组约 84 次普通标量浮点操作（复乘、加和、功率、缩放、输出）以及 2 次实数除法。
实际指令数受 FMA、除法展开及索引整数运算影响，不能把普通 FLOP 峰值当实际吞吐。
对一轮 32 组的 specialization tile：

- 全局访存下界：`T_global = 32*108 / B_global_per_SM`。
- 计算下界：`T_arith = 32*84 / F_effective_per_SM`，另加除法/整数索引依赖。
- 若 loader 交付每组 4 对复数及 Y/lambda，则 shared 写入再读取至少 `32*76*2=4864` 字节；`T_shared=4864/B_shared_per_SM`。
- 还要计入 ready/filled barrier、两 warp 分工的调度，以及 pipeline fill/drain。

这些是不同共享资源的下界，不直接相加；steady-state 至少受其中最大项限制。
未校准本机单 SM 有效吞吐前，不给出伪精确的 cycles/iteration 或 idle 百分比。

**可证伪预测：** 融合减少全局访问最可能带来收益；warp 协作未必胜过同线程保留寄存器数据；
loader/compute 分工新增 shared 访问和同步，且两边仍争用部分 load/store、地址计算资源，预计不会普遍胜过简单融合。
单 CTA 时间线应揭示是哪一方等数据或等缓冲复用，不能拿它代表全 GPU 带宽。
最终是否优化由关闭打点的多 CTA A/B 决定；时间线只解释结构。
