# 负视图修复与跨缓存资格比较的边界

本审查只读源码、原始 CUDA 报告和日志，并执行一个极小的 CPU dispatcher 探针；没有运行 CUDA 或编译。结论是：将 cold test 的 `resolve_neg()` 放进每个目标 context，能让对照与实际输入到达算子时具有相同缓存资格。逐字节断言不变，这是修正对照条件，不能解释为把原失败改判为通过。

## 80 项旧版探针的 56 个差异应如何归属

原始 [negative_before_cuda.json](../artifacts/fp32_roadmap/negative_before_cuda.json) 和对应旧源码、manifest、binary 身份保持不变。旧报告 `correctness_passed=false` 仍然为 false。

| 原始差异类别 | 数量 | 可支持的结论 |
|---|---:|---|
| `cache_metadata`，冻结输入且 GradMode enabled，s1–s4、四广播、正负标志双向切换 | 32 | 真正的缓存失效遗漏：同 identity/pointer/version/shape/stride/offset，仅 lazy negative 标志变；cached 与 clear 后 fresh 不同 |
| `cold_psf/no_grad`，s1–s4、四广播 | 16 | 真正的 PSF 原始指针读取问题：contiguous 负视图仍带 lazy flag，旧 CUDA PSF 未 resolve 就读实际存储，读到相反符号 |
| `cold_psf/inference_mode`，s3/s4、四广播 | 8 | 比较跨了两种既有准备路径；单凭这 8 个逐字节差异不能归为错误符号或新增缓存 bug |
| 其余 old probe case | 24 | 原始探针逐字节相同，不扩张为整个 API 已正确 |

前两组 48 个明确 bug 的输出 max_abs 差异范围均为 15.78076171875–91.7415771484375。第三组 8 个差异范围为 1.1920928955078125e-6–1.9073486328125e-6。差异较小不等于已证明准确、无害或满足 FP64 非劣；分类依据是派发/准备路径，而不是误差大小。

原 [negative_fixed_cuda_tests.log](../artifacts/fp32_roadmap/negative_fixed_cuda_tests.log) 的 99 项测试仍记录 8 个 failure，全部是旧写法对照在 `inference_mode` 的 s3/s4 子项。不得覆盖旧日志或把旧测试运行标成 99/99 通过。修后匹配对照的通过结果必须来自另一份新运行。

## 为什么调用前的 Tensor 元数据不足以说明实际分派

`converse2d::forward` 注册在 CompositeImplicitAutograd。安装包 `ATen/native/MathBitsFallback.h:113` 的 math-bit fallback 对 out-of-place 参数执行 `at::clone(tensor)` 后再 redispatch。InferenceMode 会影响这个新 clone 是否是 inference tensor。

原 GPU [negative_context.json](../artifacts/fp32_roadmap/negative_context.json) 开头有 profiler warning，读取时应从 JSON 数组开始解析，不能把它误判为纯 JSON 文件或静默丢失。该记录显示调用前 normal leaf 状态相同，但在 inference_mode 中，outside-resolved 对照有 2 次 `aten::square`，lazy-negative 路径为 0；no_grad/enable_grad 两边均有 2 次 square 且输出逐字节相同。该代表性 s3 的 inference_mode 跨路输出差异为 1.5497207641601562e-6。

独立 [negative_context_cpu_dispatch.json](../artifacts/fp32_roadmap/negative_context_cpu_dispatch.json) 使用相同 CompositeImplicitAutograd 注册方式，**直接在一个 CPU 自定义算子内部**记录元数据，结果如下。这支持 Torch dispatch 解释，不冒充 CUDA 算子内部的直接测量。

| 模式与输入 | 调用前 | 进入 Composite 算子时 |
|---|---|---|
| no_grad/enable_grad，lazy negative | negative=true，inference=false | negative=true，inference=false |
| inference_mode，lazy negative | negative=true，inference=false | negative=false，inference=true |
| inference_mode，模式外 resolve 的 control | negative=false，inference=false | negative=false，inference=false |
| inference_mode，模式内 resolve 的 control | negative=false，inference=true | negative=false，inference=true |

这与 `inference_preparation.cpp:20` 的 `cacheable = real_fft && !source.is_inference() && source.is_leaf()` 一致。两次都是 clear-cache 后的 cold call；差别是**源是否允许缓存**，不是一次命中了热缓存而另一次未命中。

可缓存源在 `inference_preparation.cpp:77` 预计算 ATen `real(fb).square()+imag(fb).square()` 及 alias mean。不可缓存且 fused_prepare 的源省略该预备张量，由现有融合路径形成分母。s3/s4 的两条既有路径并不承诺逐字节相同。此次只读核对确认 generic/scale3/dispatch CUDA 源和 `common/spectrum_ops.h` 与归档基线相同；negative 修复没有重写这些舍入路径。

## 合理的修复门槛和 FP64 边界

`test_negative_metadata.py` 新写法在每个 no_grad/enable_grad/inference_mode context 内 materialize control。no_grad/enable_grad 得到普通 leaf，inference_mode 得到 inference tensor，分别匹配 lazy-negative 实际 dispatch 后的资格；随后两边各自清缓存并保留**零字节裕量**比较。它验证“同一条准备路径下，lazy 值与已物化值一致”。另外，cached-vs-fresh flag-toggle 测试验证存储元数据未变时的缓存失效。Graph 签名的 lazy negative/conjugate 检查有独立 CPU before/after 证据。

对这个只恢复逻辑值、缓存失效和同路数值保持的修复，匹配资格的逐字节比较是直接门槛，不必新增“不同既有路径必须逐字节相同”的要求。若要进一步声称两条旧路径都满足 Python FP32 准确性，或声称那 8 个差异数值无害，则**必须另跑**同一量化输入的独立 FP64 参考，分别记录 cached-preparation、uncached-fused、Python FP32 的 max_abs/relative_l2，并逐指标判断。

这样的 FP64 补充应是单列的旧路径数值诊断，不可用“其中一条更接近 FP64”代替同路 byte gate，也不可把旧路径已存在的非劣失败重新归为本次修复通过。当前 probe 没有这些 FP64 结果，因此本页不作两路径准确性排名、训练质量或收敛结论。
