# Cube555 native 2xT4 integration — 2026-09-28

Branch: `codex/cube555-blend-2xt4` in TryDotAtwo/MultiGPUBeamSearch.
Base: `f679504baddbab3765c91af526e57ec9360cf309` (the existing public 444 launcher).
Verified runtime commit: `c2c04abac3fab189831f456d2f18b0fbacdfad9a`.

## Real GPU evidence

Private Kaggle kernel `trydotatwo/cube555-native-2xt4-blend-smoke`, version 3:
`COMPLETE`, two Tesla T4 GPUs, PyTorch 2.10.0+cu128, Python 3.12.
Both rank logs are present and nonempty for all four native runs.
Cube555 150-facelet / 160-byte specialization and the LibTorch runner compiled.

| Synthetic puzzle | Scramble length | Returned length | Exact replay |
|---|---:|---:|---|
| 0 | 1 | 1 | passed |
| 1 | 2 | 2 | passed |
| 2 | 3 | 3 | passed |
| 3 | 4 | 4 | passed |

Beam 4096, max depth 6, touch radius 0. Original states generated with the actual
Cube555 generators. Sample solutions are only submission baselines, never search
hints. The common launcher reported 202.38 seconds including build/export.

On each physical T4, 128 legal states gave maximum FP16 vs FP32 Q-blend error
0.0433311 and best-action agreement 127/128. Script vs eager FP16 was exactly
equal at batch sizes 1, 7 and 128, including padded input. JIT graph optimization
must remain disabled: v1 detected drift up to 0.05 before this correction.
Version 2 passed GPU parity but failed launcher preflight on a missing optional
reflection-path value; v3 includes the fix and its regression test.

Raw files: `smoke_v3/`, including `cuda_parity.json`, `provenance.json`, native
build logs, per-puzzle CSVs, both rank logs, `run_summary.json`, and submission.

## Local evidence

- Actual checkpoints vs supplied independent JAX FP32 implementation on seven
  legal states (depths 0,1,2,5,10,20,40): blend max absolute error 4.57764e-5,
  best-action agreement 7/7. Transformer and ResMLP individually also passed.
- C++ state storage, padding and target-index tests passed for 120/128 and 150/160.
- C++ LibTorch adapter compiled; a linked CPU executable loaded the exported
  script and produced finite `[1,30]` output.
- 42 initial Python tests passed. After the launcher correction, all 9 Cube555
  tests passed (43 total when combined with the same unchanged 34 old tests).
- Linux regression run: 95 tests passed; one existing exception-note test required
  Python >=3.11 and failed on the container's 3.10. That test passed separately
  on Windows Python 3.11 with a supported 1-GiB history-disk budget.
- Final notebook JSON and all Python cells parsed successfully.

## Scope

This is a native beam integration with a scripted LibTorch parent-Q scorer,
not a custom CUDA Transformer kernel and not a TPU beam port. Stream4 retains
threshold/compact/sort/reduce behavior. No merge into main was performed.

The final notebook uses real competition tasks 1020 and 1034, initial beam 65536,
depth 200, touch radius 2. Smoke does not establish full-puzzle success, this
larger beam's performance, maximum capacity, or the reported TPU lengths 102/100.
