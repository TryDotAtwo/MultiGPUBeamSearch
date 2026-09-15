# Native Hopper FP8 checkpoint

Hardware: Vast51120383, H200143771MiB, driver595.71.05, nvcc12.8.93,
CUTLASSv3.9.2 ad7b2f5. Source branch codex/hopper-stream1-fusion.
QKV-only full binary built from58bda7c0 (later changes at that point were test/scripts).

## Completed evidence

- Offline v1 artifact:60 files, three E4M3 column-major QKV matrices, all remaining
  operators FP16. SHA256 a83d9c1073bb5b50f9df69237970fd8bb15a45788194606d8b24bc792728717f.
- All60 source/payload hashes verified, source manifest
  03f55e5b617874cbbfaace5796f7e348588eb2c0b5ef079e45c06b05d84d0cb0.
- Frozen architecture: Cube4, D256, FF1024, heads8, four blocks, ReLU, CLS,
  sequence57, output24. Move ordering matches saved Kaggle puzzle_info.
- Independent LN, bias/unrounded-LN/rounded-residual and dense dequantized QKV
  oracles pass M1/31/128/131. Both memcheck and synccheck report0errors.
- Explicit loader opt-in required; three one-byte QKV and final two-byte QKV
  stored once. Loader reports5781600 weight bytes. No forward weight conversion.
- FP8 LN emission reuses dead attention_context scratch. No extra scratch.

Seven alternating unprofiled full-Stream1 pairs, micro384/concurrency8,
compact57, fused-input-LN, final-CLS attention/splitQKV, Hopper FP16 FF1,
FF1epilogue128x64, FF2m128n128:

| Path | Median parents/s | Min | Max |
|---|---:|---:|---:|
| Native FP16 control |597632.1|596538.8|598337.5|
| Native FP8 QKV, other operators FP16 |625906.9|624713.7|626053.8|

Median-rate ratio1.0473; all seven paired gains4.53–4.94%.
This is full Stream1, **not** full pipeline. Scratch1109016576bytes both.
The benchmark's printed FLOP estimate counts the old full-layer graph and must
not be interpreted as exact effective TFLOP/s for the CLS-reduced graph.

Component LN+QKV medians FP16/FP8 (same warp-LN algorithm, not original full-model
LN) at rows21888/43776/87552/175104:
40.34/34.99,73.76/63.69,136.91/115.32,260.89/216.25 microseconds.

CSV puzzles1–10 and1000: each3072 rows contains only6distinct baseline score
vectors, not3072 independent puzzles. Argmin agrees65/66 variants; puzzle7 has
one changed best move. Maximum score-key/1024 deviation0.234375. This is a small
numerical drift diagnostic, **not** solution-quality qualification; no training.

## Still running / not yet established

- 100M production pair through completed depth8. FP16 depth6=121.945s for
  69452694parents; depth7=175.276s for100007936parents. Final pair pending.
  CPU compilation of the next prototype overlapped part of this run; no other
  GPU workload was run concurrently. Treat small timing differences cautiously.
- Whole-model sanitizer smoke remains pending.
- FP8 FFN extension: direct ReLU->E4M3 and FP8 FF2+residual compiled, but not yet
  executed. Offline v2 has9packed matrices, FF1bias multiplied16offline,
  hidden inverse scale16, unqualified/calibration-not-performed. Full FFN binary
  is building separately; this does not change the running QKV-only binary.

Raw initial evidence: fp8_stream1_results.tar.gz (pipeline portion is only an
incomplete snapshot). Reproduce/parse via tools/summarize_hopper_native_fp8.py.
