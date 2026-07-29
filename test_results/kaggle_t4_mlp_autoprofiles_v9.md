# Private Kaggle v9 replication result

- Kernel version 9 completed on two Tesla T4 GPUs with source pin `6f95bd6bdb32b5f6ef7cca32b96967bce6036503` and canonical registry SHA-256 `3785713329f0a8dac13e5455f8e5acc58a760f21dbd447e544b08f3338948a50`.
- Result: 22/32 successful, 10 failed. No profile registry update.
- p16 move comparison incomplete: canonical sh4 valid `[0.111351, 0.111824]`; alternate sh2 valid `[0.120710]`.
- p19 move comparison incomplete: canonical sh4 valid `[0.733606, 0.734617]`; alternate sh8 valid `[0.794131, 0.800849, 0.811461]` but its warmup failed. Even its valid median `0.800849 s` is slower than the available canonical values.
- Same-tuple controls also failed intermittently; p21 produced no valid measured timings and caused the long run. This is harness instability, not timing evidence.
- v10 is a config-only follow-up: unique rendezvous, 10-second cooldown, and a hard 120-second timeout per torchrun with forced process-group cleanup. Raw logs remain ignored/private because model manifests include private local source paths.
