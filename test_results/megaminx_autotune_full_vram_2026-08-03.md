# Megaminx autotune full-VRAM policy ? 2026-08-03

A100x8 job 33363 measured the exact-scale seed at beam 770,883,178 on puzzles 900, 950, and 1000 with peak 39,745 MiB and stable depth-8 execution. The prior performance gate rejected it only because of an unrelated fixed 85% threshold.

The performance and capacity probes now share a 100% hard classification ceiling. Capacity search still stops and refines at the explicit 98% target. Replay, exactness, CUDA, NCCL, timeout, scratch-capacity, and full-frontier checks remain mandatory.

## Verification

```text
python -m pytest tests/portable -q
210 passed, 3 skipped in 12.10s
```
