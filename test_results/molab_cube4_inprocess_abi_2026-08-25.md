# Molab Cube4 in-process production runner

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`)
- CUDA reported by PyTorch: 13.0
- Model: Cube4 piece Transformer, ReLU, fp16, `output_dim=24`
- Solver commit: `4dd0c41838e71463f0db855b189470ecf66c2675`
- CUTLASS commit: `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`
- Invocation: `ctypes.CDLL` in the live marimo kernel via
  `beam_production_runner_to_log`; no worker, detached process, `Popen`, or
  `torchrun` was used for the solver.

## Build and smoke

- Native `120a` shared build passed in 91.33 s.
- Produced `/marimo/storage/cayleypy-cube4/inprocess_4dd0c418/libbeam_production_runner.so`
  (13,323,024 bytes).
- In-process `beam=2**16`, `depth_limit=4` smoke returned `rc=0` and completed
  `depth_done=3` in 0.121182 s on the default CUTLASS backend.
- An immutable cuBLASLt profile smoke also returned `rc=0`; its depth-3
  frontier size and threshold matched the CUTLASS smoke.

## Full `2**25` measurements

Both runs used external `B_MICRO=3584`, internal Transformer microbatch 896,
concurrency 4, two inference rings, 24 accumulation slots, 2,064,384 Stream3
batch candidates, 16 shards, Stream4 262,144/786,432 with four slots, and final
materialization chunk 88,064.

| FP16 backend | depth 6 | depth 7 | depth 8 | Stream3 jobs | Result |
|---|---:|---:|---:|---:|---|
| CUTLASS, incomplete profile | 127.897 s | 127.947 s | 128.046 s | 391 | `rc=0` |
| cuBLASLt immutable profile | 150.030 s | 150.077 s | 150.125 s | 391 | `rc=0` |

cuBLASLt is rejected for this workload. The first CUTLASS run was not the
accepted 80.2952 s profile: its log explicitly showed
`final_cls_only=0`, `final_cls_attention=0`, and
`final_cls_split_qkv=0`, with only six graph execs per lane.

## Exact accepted profile recovered

`configs/molab_sm120_cube4_transformer_profiles.json` records the accepted
profile as CUTLASS with:

- `BEAM_RING_GRAPH_EXECS_PER_LANE=12` (two-ring graph window);
- `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1`;
- `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1`;
- `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1`;
- `BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY=q32k64`.

The exact in-process rerun was prepared after identifying these omissions, but
the Molab endpoint stopped accepting connections before it could start. No
claim is made that commit `4dd0c418` has yet reproduced 80.2952 s in-process.

## Follow-up on the recovered Molab session

The same endpoint was later recovered and the exact profile completed
in-process three times.  All runs returned `rc=0`, reached exactly
`depth_done=8`, retained frontier `33,554,432`, Stream3 jobs `391`, and final
threshold `10,294`, with no OOM or CUDA error.  The measured depth-8 times were:

- original shared runner with RAM history: `103.374 s`;
- original shared runner with disk history: `103.441 s`;
- Release build without compile-time debug, requested through a second shared
  object: `103.427 s`.

The disk A/B proved that pruning/history placement was not the performance
difference.  Binary inspection then proved that the original debug shared
object contained the `depth_done=` logging string while the new nodebug object
did not (`SHA-256 50096b...` versus `cfb747...`).  Nevertheless the second
run still emitted depth logs: the first `RTLD_GLOBAL` production library had
interposed the duplicate C++/CUDA symbols in the long-lived Python process, so
that apparent nodebug timing was not an isolated binary A/B.

`RTLD_DEEPBIND` was insufficient.  A follow-up `dlmopen(LM_ID_NEWLM, ...)`
smoke was rejected because duplicating the CUDA runtime into a new linker
namespace hung during initialization.  The scratchpad was stopped through the
official marimo kernel interrupt endpoint; the Molab sandbox, persistent
weights, libraries, build logs, and solve logs were preserved.  The marimo
server then reported kernel state `stopped`; a browser reconnect is required
before further in-process work.  Do not use `dlmopen` for CUDA production
libraries.  A clean kernel must load exactly one production shared object.
