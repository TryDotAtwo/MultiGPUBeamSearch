# Private Kaggle v8 replication result

- Kernel: `trydotatwo/cayley-beam-2xt4-mlp-autoprofiles`, version 8, `COMPLETE`.
- Hardware: two Tesla T4 GPUs; world size 2. Source pin verified as `6f95bd6bdb32b5f6ef7cca32b96967bce6036503`; embedded registry SHA-256 `3785713329f0a8dac13e5455f8e5acc58a760f21dbd447e544b08f3338948a50`.
- Protocol: eight tuples, one warmup + three fresh measured depth-8 runs each; 32 total attempts. Result: 26 successful, 6 failed.
- p19 move head: canonical sh4 median `0.700020 s` from `[0.692649, 0.700020, 0.709851]`; alternate sh8 median `0.755211 s` from `[0.747377, 0.761301, 0.755211]`. Alternate is slower, so canonical remains selected.
- p16 move head: canonical sh4 supplied only one valid measured run (`0.107055 s`) because r1/r2 failed; alternate sh2 median `0.115384 s`. The all-success rule is not met, so canonical remains selected and no timing/config is updated.
- Six failures were rank-1 `SIGABRT` with `double free or corruption (!prev)` during independent torchrun attempts; rank 0 then received SIGTERM. They are reported as harness instability, not a candidate performance result.
- Exported artifact hashes: output1 `af1f21cb1cee8d4a6d811aae17bddf88115417fd1da4c6bf0015de14dd821419`; output_move_count `b4c8d70bd081be63b7941d2ed8db5d49c6ccf25f117e82af081e3230ab2e5ad2`.
- Raw logs include private source path/identity fields from model manifests. They are retained only in the ignored/private `test_results/kaggle_t4_mlp_autoprofiles_v8/` directory and are not staged or published.
- Follow-up: private v9 full rerun, config-only. It assigns a deterministic unique rendezvous port per attempt and sleeps 10 seconds after each torchrun teardown. CUDA/C++ and beam-search architecture remain unchanged.
