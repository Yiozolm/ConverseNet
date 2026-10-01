# Preserved first admission attempt

`adapter_unrestricted_eps.py` and `gate_unrestricted_eps.py` preserve the exact
Python sources used for `mixed_fusion_gate_b256_001.json`. They are source
archives; historical execution requires their original relative locations in
an isolated checkout. The CUDA source and checked binary are unchanged by the
subsequent scope restriction.

The first report completed 523 cases, including 384 exact complete-module
comparisons and 108 exact padding cases. Eight weak-regularization comparisons
failed the old FP32 solver budget against the frozen Python reference, on both
the original module and the candidate. Four were already unfused reflect
fallbacks. No nonfinite tensor records or new output differences were found.
That report remains failed and no performance timing was used to promote it.

The following attempt restricts accelerated module dispatch to `eps >= 1e-5`.
Weak regularization calls the original module. The whole numerical matrix and
every failed row retain their original acceptance definition. Only the named
normal-regularization active domain can receive separate admission, subject to
unchanged budgets, verified routing, exact finite fallback behavior and the
predeclared complete-call/model performance requirements.

`study_wrapper_control.py` preserves the exact first operator timing frontend
used for `mixed_fusion_perf_b256_001.json`. Both routes went through
`module.__call__(xlow)`, with the baseline wrapper calling its saved bound
`forward(xlow.float())`. This changed the baseline's FP32 temporary lifetime:
the upcast could be released after padding, before the solver. The pre-generation
baseline instead calls the unchanged public `module(xlow.float())`, whose
`__call__` arguments retain that tensor. A weak-reference diagnostic confirmed
the difference (39,845,888 versus 20,971,520 incremental bytes at solver entry).

The first timing remains failed as recorded: its large-case peak reduction
requirement was not met, despite latency improvements. The following timing
restores the exact original public baseline and uses a candidate-only forward
wrapper; both still pay real `nn.Module.__call__` dispatch. Kernel code, numerical
budgets and performance thresholds are unchanged. Whole-model timing uses its
separately documented identical input-quantization locations in both routes.
