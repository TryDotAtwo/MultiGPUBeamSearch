# Final Exchange Scale Default

Date: 2026-07-08

Reason:

MEPhI 8xA100 Megaminx transformer run failed on rank 7 with:

```text
what(): exchange recv total exceeds device capacity
```

The 2x final materialize exchange scale is not reliable. The default policy is now:

```text
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM = WORLD_SIZE_EFFECTIVE * 1000000
```

For 8 ranks this defaults to `8000000`. Explicit environment overrides are still honored.

Changed:

- `hpc/mephi_8xa100_common.sh`: common helper and manual-config fallback.
- `hpc/bench_8xa100_megaminx_transformer.sh`: Megaminx transformer benchmark default.
- `hpc/start_8xa100_libtorch_megaminx.sh`: LibTorch Megaminx launcher default.
- `cuda/runtime_config.cpp`: native runtime-config default from `world_size`.
- `tools/stream_pipeline_benchmark.cu`: smoke benchmark default follows its configured world size.
- `hpc/README_8xa100.md`: documented 8-rank value.

Verification:

```text
bash -n hpc/mephi_8xa100_common.sh hpc/bench_8xa100_megaminx_transformer.sh hpc/start_8xa100_libtorch_megaminx.sh
cmake --build build-final-exchange-policy-check --target production_runner stream_pipeline_benchmark -j2
```

Result:

```text
[100%] Built target production_runner
[100%] Built target stream_pipeline_benchmark
```