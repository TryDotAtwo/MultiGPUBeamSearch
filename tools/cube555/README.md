# Cube555 on two T4 GPUs

This integration runs the existing distributed C++/CUDA beam pipeline. Stream1
loads a scripted, checkpoint-exact PyTorch implementation through the existing
LibTorch executor. It does not port the TPU beam engine.

Models originally from Artgor's CC0 dataset, restored with identical checkpoint hashes in
[trydotatwo/cube555-transformer-resmlp-artifacts](https://www.kaggle.com/datasets/trydotatwo/cube555-transformer-resmlp-artifacts):

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
  --assets /kaggle/input/datasets/trydotatwo/cube555-transformer-resmlp-artifacts \
  --competition /kaggle/input/competitions/cayley-py-555-cube \
  --output /kaggle/working/cube555_run \
  --pids 1020 --beam 4000000 --b-micro 8192 --model-micro 512 --depth 140 --touch-radius 5 \
  --solution-mode collect --max-collected-solutions 2000
```

The output directory must be new. Both GPUs must be T4. Runtime plans use the
existing Transformer registry as a seed; exact Cube555 native memory and history
preflights remain authoritative. The current command uses model microbatch512
and outer parent batch8192. Logs for both ranks, exact replay, model hashes and a merged
`submission.csv` are preserved. No result upload or competition submission occurs.

Build a pinned notebook:

```sh
python -m tools.cube555.notebook --solver-commit FULL_40_CHARACTER_SHA --output kaggle/cube555
```

The default is collect mode with at most 2000 stored solutions. Supported beam tiers
are 65536, 131072, 262144, 524288, 1048576, 2097152 and 4000000;
the final tier aligns to 4005888. Depth is capped at 140 and BFS radius defaults
to 5. The generated public configuration selects PID1020; the standalone CLI
default puzzle list remains 1020/1034.

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

Measured on 2026-09-28, one full depth-8 comparison at beam 65536:

| Preset | Parent/model batch | Depth 8 seconds | Max sampled MiB/GPU |
| --- | ---: | ---: | ---: |
| safe | 128 | 5.864 | 551 |
| balanced | 256 | 6.604 | 659 |
| throughput | 512 | 7.397 | 1207 |

In the historical coupled-batch run, `safe` was fastest. These are single-run
measurements, not exhaustive optima or a guarantee at a different beam width.
The Artgor source currently uses beam 16777216, 256 times this validation width;
reported solution lengths are not an equal-budget comparison.

## Historical large-beam audit (2026-09-28)

The following p22-p26 settings are superseded by the current 4-million cap.
Historical commands and allocation results do not describe current defaults.

The old 65K table above varied both batch levels at once and is historical only.
It does not select an inference microbatch for an 8192-parent outer transaction.
Enter numeric `BEAM_WIDTH` directly; the existing p22-p26 settings are selected automatically. The default requests 33,554,432. Existing
Transformer p25/p26 pipeline profiles seed shard counts, Stream4 buffers and
final exchange; exact Cube555 native memory checks remain authoritative.
`B_MICRO=8192` and `MODEL_MICRO` are independent, including the C++ LibTorch
launcher which now writes each model chunk into its original score-ring slice.
History uses the same RAM/disk budget and explicit MAX_DEPTH cap as the existing
public notebooks. Requested/effective depth and unchanged beam are reported.

`python -m tools.cube555.capacity_audit` tunes only model microbatches first,
then builds the native runner once and checks near-limit allocation plus an
outer-batch depth loop. A depth2 allocation pass is not a saturated depth8
performance/capacity validation. New measurements are pending.

Kaggle audit v2 completed: selected model micro512; native legal replay smoke
and29,360,128 depth5 loop passed, sampled device high-water8475MiB. Allocation
probes29,360,128 and33,554,432 passed;58,720,256 and67,108,864 failed the
GPU budget gate. History at29,360,128 admitted depth177. These are bounded
allocation/transaction checks, not a fully saturated depth8 ceiling.

The first notebook cell exposes explicit MODEL_ROOT, CHECKPOINT_PATH (Transformer), RESMLP_CHECKPOINT_PATH, inclusive puzzle range, BEAM_WIDTH, MAX_DEPTH, and TRANSFORMER_WEIGHT. Compatible replacement checkpoints are forwarded to export and hashed in provenance. Current BFS radius5 and other runtime defaults are below the explanation.


## Readiness checked 2026-09-30

Public notebook v12 pins `90b5d7884dc1f88577101cb617b5114ac4ff5297`.
Its current output inventory is empty; Kaggle COMPLETE is not a full-run proof.
Private collect acceptance v4 at this pin retained 2000 replay-valid PID35
solutions at requested/effective beam 4000000/4005888, outer8192/model512,
depth140 and BFS5. The native run completed, but the notebook failed its
publication assertion after HTTP503. The exact saved archive is available for
a separately authorized delivery retry; no new delivery was attempted here.

Private width audit v2 at `c5f1b15962b5958ccd138a257324049d753df91d`
passed allocation and saturated depth9 loops for all seven tiers, using
outer8192/model128. At the final tier depth8 took399.134s; sampled memory
high-water was4461/4481MiB. These timings do not establish model512 throughput.
Historical smoke v3 is also complete at its original pin.

Open gates: successful ingest and promotion for the current collect archive;
saturated validation of current model512/collect settings; full real target
validation at the current defaults. Historical PID1020 length132 and PID1034
unsolved results remain historical. See
[test_results/cube555_readiness_2026-09-30.md](../../test_results/cube555_readiness_2026-09-30.md)
for exact versions, pins and evidence.


## Offline retry preparation

`python -m tools.cube555.prepare_retry --archive ORIGINAL.json.gz
--accepted-snapshot D1_SELECT_EXPORT.json --output NEW_DIRECTORY` validates
original envelopes, filters accepted semantic keys and writes bounded retry
parts with unchanged identities. Default100 records per part; no network/send
option exists. Refresh the scoped D1 snapshot before any separately authorized
send. A verified receipt confirms ingest handling; confirm downstream state and
promotion separately. Never use mock receipt ledgers as remote acceptance proof.
See [HTTP503 diagnosis and retry evidence](../../test_results/cube555_retry_2026-09-30.md).

Cloudflare transport uses at most100 results and4MiB decoded JSON per gzip part. The2000 solution collection limit remains independent of request batching. Saved legacy2000-result archives stay readable for idempotent repackaging/retry. DO ingest live acceptance is verified for all2000 preserved originals; staging/promotion and long GPU acceptance remain pending. Public v13 pins10824003c11693aad36ed467cb93183a14d44d93 with the bounded transport client. See test_results/cube555_live_gate_2026-09-30.md; source-only saves do not establish full hardware readiness.
