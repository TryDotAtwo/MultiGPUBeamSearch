# Kaggle 2×T4 correctness protocol

## Цель

Notebook `notebooks/kaggle_2xt4_debug.ipynb` теперь является алгоритмическим correctness-check, а не smoke-test.

Проверяются:

1. `puzzle_info.json`:
   - наличие `central_state`;
   - наличие ровно 24 actions в фиксированном порядке;
   - каждая action table является permutation `0..119`;
   - пары `A, -A` восстанавливают `central_state` на CPU reference.
2. `test.csv`:
   - schema `initial_state_id, initial_state`;
   - state length `120`;
   - значения помещаются в `uint8`;
   - первый CSV-state запускается через один GPU-depth step.
3. CUDA extension:
   - сборка без monkey-patches в notebook;
   - `score_ring` хранится как `torch.int16`, C++ читает `uint16_t`;
   - action table загружается в `__constant__` CUDA memory;
   - central state загружается в `__constant__` CUDA memory.
4. Search loop:
   - `reset_search(initial_state, active_owner)`;
   - `search(max_depth)`;
   - остановка при найденном central-state;
   - остановка при достижении `max_depth`.
5. CUDA Graph:
   - `USE_CUDA_GRAPHS=1` по умолчанию;
   - graph captures one full depth step: clear → stream1 inference → stream2 ingest/hash/dedup → stream3 NCCL/threshold → prune → compact;
   - subsequent depths launch `cudaGraphLaunch`.
6. Multi-GPU 2×T4:
   - `torchrun --standalone --nproc_per_node=2`;
   - `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SOCKET_IFNAME=lo`;
   - stream3/NCCL path считается валидным только при `remote_packed > 0` globally.

## Запуск на Kaggle

```bash
cd /kaggle/working/CayleyBeam100H100
export USE_CUDA_GRAPHS=1
export INFERENCE_BACKEND=central_hamming
export GLOBAL_BEAM_WIDTH=32768
export B_MICRO=4096
export SCORE_RING_DEPTH=8
export NET_RING_DEPTH=2
export BUCKET_CAP_PER_PEER=65536
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
python scripts/kaggle_correctness_check.py

torchrun --standalone --nproc_per_node=2 scripts/kaggle_correctness_check.py
```

## Success criteria

Required pass conditions:

```text
validate_inverse_pairs: pass
extension build: pass
CUDA Graph captured: true
central/depth1/depth2 generated cases: found
CSV one-step expansion: candidates > 0
TorchScript MLP ensemble case: found depth1 and CUDA Graph captured
2-GPU stream3 path: remote_packed > 0
bucket_overflow: 0
hash_overflow: 0
```

Failure of any condition is an algorithmic failure, not a cosmetic notebook failure.
