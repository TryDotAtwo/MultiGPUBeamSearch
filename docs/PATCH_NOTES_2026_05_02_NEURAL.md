# Patch notes: neural scorer ensemble and ring dependencies

## Added

- `scorers.py`:
  - `MLPActionScorer`: `120 -> 1024 -> 256 -> 24` MLP;
  - `QuantizedActionScorer`: converts scorer output to int16 raw uint16 score bits;
  - `make_default_mlp_scorer()`.
- `scripts/export_mlp_scorer.py`:
  - exports N TorchScript copies;
  - prints required env variables.
- C++ `TorchScriptEnsembleBackend`:
  - loads multiple TorchScript modules;
  - dispatches micro-batches to scorer copies;
  - accepts output `[B, 24]` as `int16` or `float32`;
  - converts row-major `[B, 24]` into action-major `score_ring[slot]`.
- Config keys:
  - `INFERENCE_BACKEND=torchscript_ensemble`;
  - `INFERENCE_PARALLELISM=N`;
  - `TORCHSCRIPT_SCORER_PATHS=path0:path1:...`;
  - `NN_SCORE_SCALE`, `NN_SCORE_BIAS` for float32 output.

## Fixed

- Added `score_consumed[slot]` event. Stream1 cannot overwrite `score_ring[slot]` before Stream2 finishes reading that slot.
- Added `net_consumed[net_slot]` event. Stream2 cannot reset/reuse bucket slot before Stream3 exchange and Stream2 remote ingest complete.
- Added multiple inference streams under Stream1 role. CUDA Graph captures the event graph; CUDA Graph does not serialize Stream1/Stream2/Stream3.

## Correctness runner

`scripts/kaggle_correctness_check.py` now checks:

```text
central_hamming depth0/depth1/depth2
CSV one-step expansion
2-GPU NCCL stream3 remote packing
TorchScript MLP ensemble depth1
CUDA Graph capture for normal scorer and TorchScript scorer
```
