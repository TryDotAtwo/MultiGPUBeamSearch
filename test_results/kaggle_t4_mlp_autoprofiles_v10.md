# Private Kaggle v10 replication result

- Kernel: `trydotatwo/cayley-beam-2xt4-mlp-autoprofiles`, version 10, `COMPLETE`.
- Hardware/source: two Tesla T4 GPUs, world size 2; solver commit `6f95bd6bdb32b5f6ef7cca32b96967bce6036503`; canonical registry SHA-256 `3785713329f0a8dac13e5455f8e5acc58a760f21dbd447e544b08f3338948a50`.
- Harness: fresh deterministic rendezvous port, 10-second inter-attempt cooldown, 120-second per-attempt cap with process-group termination. Result: 31/32 successful; no timeout cleanup was required.
- p16 move: canonical sh4 median `0.110985 s` from `[0.112333, 0.110678, 0.110985]`. Alternate sh2 supplied only `[0.119420, 0.111373]`; r2 failed with rank-1 `SIGABRT`, stderr `double free or corruption (!prev)`. It is incomplete and not materially faster; canonical retained.
- p19 move: canonical sh4 median `0.709567 s` from `[0.698852, 0.709567, 0.718287]`; alternate sh8 median `0.742183 s` from `[0.742183, 0.723934, 0.758800]`. Both tuples completed, and sh8 is 4.6% slower; canonical retained.
- Same-tuple control medians: output1 p19 `15.7436 s`; move p17 `0.212249 s`; move p20 `1.4438 s`; move p21 `2.98881 s`. Controls do not update profiles.
- Profile runtime tuples are unchanged from the canonical registry. There is no justified v11 profile sweep.
- Raw private logs were scanned: no credentials/tokens were found. Model manifests include a private local source path, so raw output directories stay ignored/private. The generated JSON artifacts in v10 ended in literal `\\n`; this export-only defect was fixed locally and regression-tested, but v10 records are not used to update the registry.
