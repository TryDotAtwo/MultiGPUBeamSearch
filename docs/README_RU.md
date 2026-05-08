# GPU-resident distributed beam search engine

## Назначение

Проект содержит рабочий каркас распределённого beam search движка для схемы:

```text
Stream 1: инференс Q-модели
Stream 2: apply_move + hash + local dedup + pack remote buckets + prune
Stream 3: NCCL all-to-all + NCCL all-reduce histogram
```

Основной принцип: **данные не возвращаются на CPU в процессе шага**. Python выделяет память через `torch`, C++ получает указатели на tensors, CUDA kernels и NCCL работают напрямую с GPU memory.

## Документация

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Полная архитектура (три потока, структуры данных, буферы, синхронизация)
- **[FRAMEWORKS.md](FRAMEWORKS.md)** — Описание применяемых фреймворков (LibTorch, NCCL, pybind11, CUB, TE)
- **README_RU.md** — Этот файл (быстрый старт)

## Файлы проекта

| Файл | Смысл |
|---|---|
| `beam_engine_common.hpp` | Общие структуры `BeamMeta`, `HashSlot`, `CandidateRecord` |
| `beam_kernels.cu` | CUDA kernels: dummy inference, ingest, hash insert, remote ingest, threshold, prune |
| `beam_engine.cpp` | pybind11/libtorch/NCCL C++ обвязка, 3 CUDA streams |
| `beam_engine.py` | Python launcher: сборка extension, выделение torch tensors, NCCL init, запуск шага |
| `ARCHITECTURE.md` | Полная документация архитектуры |
| `FRAMEWORKS.md` | Полное описание фреймворков и их интеграции |

## Используемые фреймворки (краткий обзор)

| Фреймворк | Статус | Что делает |
|-----------|--------|-----------|
| **LibTorch** | ✅ Готов | Tensors, GPU alloc, данные из Python |
| **pybind11** | ✅ Готов | Python API: `engine.step()` |
| **NCCL** | ✅ Готов | All-to-all, all-reduce через InfiniBand |
| **CUB** | 📋 Планируется | Сортировка для top-K (нужен для оптимального prune) |
| **TE (Transformer Engine)** | 🔧 Заглушка | FP8 инференс (вместо dummy backend) |

Подробнее смотри [FRAMEWORKS.md](FRAMEWORKS.md).

## Схема памяти

Python выделяет все буферы заранее:

```text
beam_current        [N_LOCAL, 120] uint8
next_state_pool     [K_WORK, 120] uint8
next_meta           [K_WORK * sizeof(BeamMeta)] uint8
hash_table          [HASH_CAPACITY * sizeof(HashSlot)] uint8
active_flags        [K_WORK] uint8
score_ring          [SCORE_RING_DEPTH * B_MICRO * FANOUT] fp16/uint16 storage
send_buckets        [NET_RING_DEPTH * WORLD_SIZE * BUCKET_CAP * 160] uint8
recv_buckets        [NET_RING_DEPTH * WORLD_SIZE * BUCKET_CAP * 160] uint8
send_counts         [NET_RING_DEPTH * WORLD_SIZE] int32
local_hist          [65536] int32
global_hist         [65536] int32
threshold_cell      [2] int32
counters            [8] int32
```

`threshold_cell`:

```text
threshold_cell[0] = 0/1, валиден ли threshold
threshold_cell[1] = threshold_q
```

## Производные размеры

```text
N_LOCAL = ceil(GLOBAL_BEAM_WIDTH / WORLD_SIZE)
K_KEEP = gamma * N_LOCAL
K_WORK = beta * K_KEEP
HASH_CAPACITY = K_WORK / hash_load_factor
```

Для целевого режима `GLOBAL_BEAM_WIDTH = 2^30`, `WORLD_SIZE = 100`, `gamma = 1.05`, `beta = 1.10`:

```text
N_LOCAL ≈ 10.74M
K_KEEP ≈ 11.27M
K_WORK ≈ 12.40M
HASH_CAPACITY ≈ 20.67M
```

## Stream 1: инференс

Текущая реализация вызывает `kernel_dummy_inference`. Реальная схема должна заменить dummy backend на `TEInferenceBackend`:

```cpp
InferenceBackend::forward(
    beam_current,
    score_ring,
    slot,
    start_state,
    micro_size,
    cfg,
    stream_infer
)
```

Контракт Stream 1:

```text
input:  beam_current[start : start+B_MICRO]
output: score_ring[slot, B_MICRO, FANOUT]
```

Stream 1 не делает:

```text
apply_move
hash
dedup
NCCL
prune
```

## Stream 2: local ingest / dedup / pack / prune

Stream 2 ждёт событие `score_ready[slot]`, затем запускает `kernel_process_score_slot`.

Для каждого candidate lane:

```text
score = score_ring[slot, state, action]
если threshold valid и score <= threshold: discard
next_state = apply_move(state, action)
hash = hash(next_state)
owner = hash % WORLD_SIZE
если owner == rank: insert/update в local hash table
если owner != rank: pack в send_buckets[net_slot, owner]
```

Local dedup реализован через lock-free hash table:

```text
atomicCAS(hash_slot.hash, EMPTY, hash)
если hash/fingerprint совпал: update if better
если probe_limit исчерпан: HASH_OVERFLOW counter
```

## Stream 3: NCCL / InfiniBand

Stream 3 ждёт событие `send_ready[net_slot]` и выполняет fixed-size all-to-all:

```cpp
ncclGroupStart();
for peer in ranks:
    ncclSend(send_bucket[peer], fixed_bytes, ncclUint8, peer, comm, stream_net);
    ncclRecv(recv_bucket[peer], fixed_bytes, ncclUint8, peer, comm, stream_net);
ncclGroupEnd();
```

Эта схема не требует CPU чтения `send_count`. Каждый peer получает bucket фиксированного размера. Неиспользованные records имеют `valid=0`.

После сетевой передачи Stream 2 забирает `recv_buckets` и вставляет удалённые candidates в локальную hash table.

### InfiniBand

NCCL использует InfiniBand/RDMA при наличии IB fabric, корректного NCCL build и переменных окружения:

```bash
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# При необходимости:
# export NCCL_IB_HCA=mlx5_0,mlx5_1
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_NET_GDR_LEVEL=PHB
```

## Threshold и histogram

Периодически Stream 3 запускает:

```cpp
ncclAllReduce(local_hist, global_hist, 65536, ncclInt32, ncclSum, comm, stream_net);
kernel_compute_threshold(global_hist, threshold_cell, 65536, GLOBAL_BEAM_WIDTH);
```

CPU не участвует. Все ranks получают одинаковый `global_hist`; threshold считается на GPU.

Пока суммарных уникальных retained states меньше `GLOBAL_BEAM_WIDTH`, `threshold_cell[0] = 0`, фильтрация выключена.

## Physical prune

В конце шага запускается:

```cpp
kernel_prune_by_threshold(next_meta, active_flags, local_hist, counters, threshold_cell, K_WORK);
```

Удаляются active slots с `score_q <= threshold_q`. Это histogram-based prune без sort. На следующем этапе разработки нужно добавить:

```text
CUB DeviceSelect для compact active slots
rebuild hash_table из compacted next_state_pool
final tie-break внутри threshold bin
```

## Запуск

### Однопроцессная отладка

```bash
python -c "
from beam_engine import make_default_config, build_extension
ext = build_extension(verbose=True)
print('Extension собран. Готово к тестированию.')
"
```

### Многопроцессный запуск на Slurm/torchrun

```bash
export GLOBAL_BEAM_WIDTH=$((1<<30))
export B_MICRO=131072
export SCORE_RING_DEPTH=64
export NET_RING_DEPTH=3
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

torchrun --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  your_script.py
```

### Отладка

Включить логирование счётчиков:

```python
from beam_engine_ext import BeamEngine

engine = BeamEngine(cfg, buffers, backend="dummy")
engine.enable_debug(verbose=True, print_counters=True, log_period=8)

for step in range(num_steps):
    engine.step(histogram_period_micro=8)
    engine.sync_and_log_step(step)
```

Для NCCL отладки:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# Запустить программу -> подробные логи в stderr
```

Подробнее о отладке и проблемах смотри в [FRAMEWORKS.md](FRAMEWORKS.md#быстрая-интеграция-checklist).

## Что уже полноценно GPU-resident

```text
score_ring на GPU
candidate generation на GPU
hashing на GPU
local hash table на GPU
NCCL buckets на GPU
NCCL all-to-all на GPU memory
NCCL all-reduce histogram на GPU memory
threshold вычисляется GPU kernel
prune выполняется GPU kernel
```

## Что является точкой дальнейшей замены

1. `kernel_dummy_inference` заменить на TE/FP8 Q-MLP.
2. `apply_move_dummy` заменить на реальные таблицы действий головоломки.
3. Fixed-size all-to-all заменить на two-phase counts + all-to-allv, если waste bandwidth станет главным bottleneck.
4. Добавить CUB compact/rebuild для финального плотного `beam_next`.
5. Добавить parent history slab для восстановления решения.

