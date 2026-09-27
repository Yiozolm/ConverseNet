# Real/crop experiment and diagnosis evidence

Lossless gzip copies preserve the original release/build results, negative single-epoch comparison, subsequent fixed-weight controls, GPU traces, telemetry, setup failure and scope-text correction. `index.json` records original, archived and decompressed content SHA256 hashes. The archived runtime scripts may predate the final reproducibility tools; their hashes are recorded by each run. Full checkpoints/binaries remain in local artifacts, with identities in the run/build records.

The single-epoch result remains 110.128 -> 116.228 seconds (+5.54%). Later fixed-weight forward/backward controls show roughly 4% benefit, but are not a replacement training epoch and do not establish its historical slowdown trigger. No failed or negative result is relabelled as a success.
