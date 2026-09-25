# FP32 roadmap evidence

`index.json` records original and compressed hashes. Each `.gz` expands to
the exact original bytes, including failed sessions and rejected comparisons.
Read with Python `gzip.open(path, "rt", encoding="utf-8")`.
Numeric research reports are separately archived in commit cc244e3 under
`research/evidence/`. Local checkpoints and raw profiler captures remain
under `artifacts/fp32_roadmap/`; their identities are recorded in these reports.
