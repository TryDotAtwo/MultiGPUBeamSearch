# Cube555 on two T4 GPUs

This integration runs the existing distributed C++/CUDA beam pipeline. Stream1
loads a scripted, checkpoint-exact PyTorch implementation through the existing
LibTorch executor. It does not port the TPU beam engine.

Models from [Artgor's CC0 dataset](https://www.kaggle.com/datasets/artgor/cube555-tpu-artifacts):

- `q555_f1_bell2k.pt`: PieceTransformerQ555, 98 piece-head tokens + CLS,
  six layers, width 384, six heads, SiLU, 30 parent-Q outputs.
- `q555_2k_BEST.pt`: ResMLPQ, embedding width 24, ten width-1024 residual blocks,
  30 parent-Q outputs.
- Default score: `0.8 * Q_transformer + 0.2 * Q_resmlp`, mixed in FP32 from
  FP16 model outputs, then quantized once by the existing native score ring.
  QV consistency is off. Script graph optimizations are disabled to preserve
  eager FP16 rounding on T4; this is checked on device rather than assumed.

The 150 logical uint8 facelets occupy 160 bytes of aligned storage. The Zobrist
alphabet is 256, so IDs 128..149 never wrap or index beyond the hash table.
Persistent padding is zero; bytes 150..153 carry a target index only in a
transient FinalResponse. The default 120/128 build remains unchanged.

## Run

```sh
python -m tools.cube555.run \
  --assets /kaggle/input/datasets/artgor/cube555-tpu-artifacts \
  --competition /kaggle/input/competitions/cayley-py-555-cube \
  --output /kaggle/working/cube555_run \
  --pids 1020 1034 --beam 65536 --depth 200 --touch-radius 2
```

The output directory must be new. Both GPUs must be T4. A dedicated conservative
profile uses microbatch 128, one inference lane and four shards. It is not a
Cube4-derived capacity claim. The native memory and history preflights may reject
a larger beam. Logs for both ranks, exact replay, model hashes and a merged
`submission.csv` are preserved. No result upload or competition submission occurs.

Build a pinned notebook:

```sh
python -m tools.cube555.notebook --solver-commit FULL_40_CHARACTER_SHA --output kaggle/cube555
```

Use `--smoke` for a private, four-scramble test (depths 1..4, beam 4096,
search depth 6, touch radius 0), including CUDA eager/script parity on both GPUs.
The sample solutions are not used as a search hint. Successful smoke checks
plumbing and replay, not long-puzzle solution quality or maximum beam capacity.

The T4 arithmetic is FP16 and the native search differs from the TPU/JAX engine.
The reported TPU lengths 102/100 are not reproduced or promised by this port.

## Reference validation

```sh
python -m tools.cube555.validate --assets ASSET_DIRECTORY --output parity.json
```

This checks original FP32 checkpoints against the dataset's independent JAX
implementation on legal states, plus variable script batch sizes and ignored
padding. It does not require or execute a GPU search.


## Public notebook audit

Presets `safe`, `balanced`, `throughput` use parent/model batches 128/256/512;
beam remains user-controlled. They are experimental unless measured on the exact
two-T4 blend. `python -m tools.cube555.benchmark` compares the full depth-8 loop
on the same input and beam. Telemetry samples native+LibTorch device memory each
second; its high-water value is a lower bound on instantaneous peak allocation.

Publication uses replay-validated archives, preserves both checkpoint hashes,
blend weights, script hash, original state and generator proof. Exact gzip
requests survive network failures in each puzzle directory. A 200/202 transport
acknowledgment does not establish final promotion in cayleypy-beam-results.
Synthetic smoke puzzles never enable publishing. Real puzzle 35 is a short
end-to-end delivery check before the two original target puzzles.
