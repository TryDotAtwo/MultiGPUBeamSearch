# Neural scorer contract

## Runtime contract

`Stream 1` accepts a scorer backend and writes action scores to `score_ring[slot]`.

Required scorer I/O for `torchscript_ensemble`:

```text
input:  states_u8: torch.uint8 CUDA tensor [B, 120]
output: scores: torch.int16 or torch.float32 CUDA tensor [B, 24]
```

If output is `torch.int16`, C++ copies raw int16 bits into `score_ring` and later interprets the memory as `uint16_t`.
If output is `torch.float32`, C++ applies:

```text
q = clamp(round(score * NN_SCORE_SCALE + NN_SCORE_BIAS), 0, 65535)
```

Then C++ stores `q` as `uint16_t` in action-major layout:

```text
score_ring[slot][action][parent_index_inside_microbatch]
```

## Default MLP

`scorers.py` contains the default MLP scorer:

```text
120 -> 1024 -> 256 -> 24
activation: GELU
output wrapper: sigmoid(logits) * 65535 -> int16 raw uint16 bits
```

Export command:

```bash
python scripts/export_mlp_scorer.py --copies 2 --hidden 1024,256
```

The script prints:

```text
TORCHSCRIPT_SCORER_PATHS=<path0>:<path1>
INFERENCE_BACKEND=torchscript_ensemble
INFERENCE_PARALLELISM=2
```

On Windows, `TORCHSCRIPT_SCORER_PATHS` uses `;`; on Linux/Kaggle, `:`.

## Multiple inference copies

Config key:

```text
INFERENCE_PARALLELISM=N
```

C++ creates `N` non-blocking CUDA inference streams and rotates micro-batches by:

```text
infer_lane = microbatch_id % INFERENCE_PARALLELISM
module_id  = score_slot % module_count
```

`score_ring` reuse is protected by:

```text
score_ready[slot]    : Stream1 -> Stream2
score_consumed[slot] : Stream2 -> Stream1
```

`send/recv bucket` reuse is protected by:

```text
send_bucket_ready[net_slot] : Stream2 -> Stream3
recv_bucket_ready[net_slot] : Stream3 -> Stream2
net_consumed[net_slot]      : Stream2 -> Stream2 reset of same net_slot
```

## CUDA Graph rule

CUDA Graph captures the dependency graph, not a sequential stream.

Captured structure:

```text
root stream records start_ready
all inference lanes wait start_ready
Stream2 waits start_ready
Stream2 clears step state
all inference lanes wait clear_ready
for each microbatch:
    inference lane writes score_ring[score_slot]
    Stream2 waits score_ready[score_slot]
    Stream2 reads score_ring[score_slot]
    Stream2 records score_consumed[score_slot]
    Stream2 packs send_bucket[net_slot]
    Stream3 waits send_bucket_ready[net_slot]
    Stream3 performs NCCL send/recv
    Stream2 waits recv_bucket_ready[net_slot]
    Stream2 ingests remote candidates
    Stream2 records net_consumed[net_slot]
periodic Stream3 histogram all-reduce and threshold kernel
Stream2 prune + compact
Stream3 found all-reduce
root stream waits final event
```

Therefore CUDA Graph does not remove asynchronous work. CUDA Graph removes host launch overhead while preserving stream/event dependencies.

## Current framework status

```text
Orchestration: libtorch + pybind11: implemented
Inference: TorchScript ensemble: implemented
Inference: Transformer Engine FP8: placeholder only in Kaggle/T4 build
Communication: NCCL C++ API: implemented
Algorithms: custom kernels; CUB reserved for later exact top-K/scan kernels
```
