# Initial profiler protocol retained

`profile_worker_deterministic_fill.py` is the exact worker used for
`artifacts/v4_campaign/mixed_ncu_{fp32,fp16,bf16}_001/`. The raw captures and
their source/build identities remain preserved locally.

This initial worker enabled global deterministic algorithms. PyTorch's default
uninitialized-memory fill then added FillFunctor kernels to the complete call.
The unprofiled cost study does not enable that setting. These initial counters
are therefore excluded from representative traffic comparisons; their profiler
durations must not be used as latency benchmarks. The corrected worker matches
the cost study's global deterministic flag and releases each warm output before
the next call. Its separately named reports do not replace these captures.
