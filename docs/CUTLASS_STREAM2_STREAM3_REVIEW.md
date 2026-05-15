# CUTLASS Stream2/Stream3 Review

entity_id=cutlass_stream2_stream3_review; type=technical_review; state=current

## Scope

- stream1_inference: dense embedding/GEMM/ReLU/output scoring; CUTLASS is applicable and already used for `fullbeamnice_static`.
- stream2_expand_dedup_prune: permutation scatter, hash insert/update, histogram updates, threshold pruning, compaction; CUTLASS is not a fit because operations are irregular memory/hash workloads, not dense matrix multiplication.
- stream3_nccl_threshold: fixed all-to-all candidate records, histogram allreduce, threshold broadcast/synchronization; CUTLASS is not a fit because work is communication/reduction/control-flow dominated.

## Optimization Direction

- stream2_possible_optimizations: reduce full-buffer clears through logical limits/touched ranges, improve candidate record coalescing, tune `K_EXPAND_TILE`, tune `PROBE_LIMIT`, tune `HASH_LOAD_FACTOR`, reduce histogram update frequency in prepass.
- stream3_possible_optimizations: tune `BUCKET_CAP_PER_PEER`, reduce allreduce frequency through `HISTOGRAM_PERIOD_MICRO`/`PREPASS_HISTOGRAM_PERIOD_MICRO`, keep NCCL calls batched and static, avoid dynamic message growth.
- cutlass_decision: do not replace Stream2/Stream3 kernels with CUTLASS unless a future dense top-k/GEMM-like operation is introduced.

## Verification

- benchmark_source: use `DEPTH_TUNING_LOG=1` for per-depth wall time, counters, microbatch count, tile count, logical limits, and parameter snapshot.
- stream1_benchmark_source: use `scripts/benchmark_inference_backends_2gpu.py` / notebook benchmark cell for TorchScript vs static CUTLASS inference speed.
