# Stream1 Transformer Final CLS-Only Block51 Smoke

Date: 2026-07-07

Scope:

- Optimize only the exact p900 `piece_transformer` native `block51` path (`seq=51,d_model=256,nhead=8,head_dim=32,layers=4,ff_dim=1024,output_dim=24`).
- Keep the MLP path and generic transformer path untouched.
- Keep attention exact: the final layer still runs full 51-token QKV and full FMHA attention. The optimization gathers only the CLS row after attention and runs the final layer attention-output projection, LN2, FFN, output LN, and logits on CLS only.
- No fallback or distillation behavior was added.

Implementation notes:

- Added `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY` as an explicit env flag.
- The fast path preserves the old block51 half-rounding order: `linear_residual -> bias_add -> layernorm_copy`.
- Earlier CLS-only attention experiments were rejected because checksum/digest did not match full attention; the committed path does not use CLS-only attention.
- `hpc/bench_8xa100_megaminx_transformer.sh` now passes/logs `BEAM_STREAM1_TRANSFORMER_BLOCK51` and `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY` for pipeline smoke and full target runs.
- `production_runner` logs the effective transformer block51/final_cls_only flags.

Local Docker build and tests:

```text
docker run --rm --gpus all -v D:/100XH100/.worktrees/stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-final-cls-check -j2 && ctest --test-dir build-final-cls-check --output-on-failure"
```

Result:

```text
100% tests passed, 0 tests failed out of 13
```

Real-weight exactness and speed, local RTX 3070 Laptop, `weights/megaminx_vlad_transformer_fp16`, `B_MICRO=512`, graph benchmark:

| mode | concurrency | ms/group | candidates/s | checksum | digest | exact vs full |
|---|---:|---:|---:|---:|---:|---|
| final_cls_only=1 | 1 | 34.3202 | 358039.7 | 420844336 | 17044003705417566170 | yes |
| final_cls_only=0 | 1 | 42.8616 | 286690.4 | 420844336 | 17044003705417566170 | baseline |
| final_cls_only=1 | 2 | 242.0127 | 101548.4 | 841858064 | 821400116975659197 | yes |
| final_cls_only=0 | 2 | 247.6100 | 99252.8 | 841858064 | 821400116975659197 | baseline |

Interpretation:

- The safe speed win is clear for `512x1` on the local GPU: about `1.25x` over the full final-layer path.
- The local `512x2` result is not representative for A100 because the 8GB laptop GPU spills/contends badly, but exactness still holds.
- Cluster/A100 speed must be re-measured because earlier 700M runs were dominated by the full final layer and macro/micro scheduler behavior.

Pipeline smoke:

```text
BEAM_PIPELINE_BENCH_MODE=stream123 BEAM_RING_GRAPH_EXECS_PER_LANE=32 BEAM_B_MICRO=512 BEAM_STREAM1_TRANSFORMER_MICRO=512 BEAM_STREAM1_CONCURRENCY=2 BEAM_STREAM3_RING_SLOTS=8 BEAM_STREAM3_BATCH_CANDIDATES=98304 ./build-final-cls-check/stream_pipeline_benchmark
```

Result:

```text
stream_pipeline_benchmark mode=stream123 window=32 b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 graph_window_jobs=64 physical_jobs=256 frontier_size=131072 ring_slot_jobs=256 stream3_jobs=32 stream4_jobs=0 candidates=3145728 depth_like_ms=3741.18 candidates_per_sec=840839 shard_capacity=1048576 allocation_bytes=3281187584 status=OK
```

Status:

- Safe to test on MEPhI A100 with `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1`.
- If a non-p900 transformer is used, this path remains gated by exact block51 shape and will not silently apply.