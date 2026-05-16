# Конфиг

```text
STATE_LEN
STATE_STORAGE_LEN
STATE_VALUE_PAD
MOVE_COUNT

B_MICRO
INFERENCE_PARALLELISM

STREAM3_BATCH_CANDIDATES
STREAM4_BATCH_CANDIDATES
STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT

RING_COUNT
RING_SLOT_COUNT

WORLD_SIZE
LOCAL_RANK

SHARD_COUNT
GLOBAL_SPILL_CAPACITY

USER_GLOBAL_BEAM_WIDTH
GLOBAL_BEAM_WIDTH_EFFECTIVE
GLOBAL_BEAM_WIDTH_MAX_SAFE
BEAM_WIDTH_ALIGNMENT

SCORE_MAX_Q
SCORE_SCALE
SCORE_MAX_KEY
SCORE_BIN_COUNT
GOAL_SCORE_KEY

GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS

SOLVED_RESULT_CAPACITY

threshold_initialized
```

```text
STATE_LEN = 120
STATE_STORAGE_LEN = 128
STATE_VALUE_PAD = 128
MOVE_COUNT = 24

SCORE_MAX_Q = 300.0
SCORE_SCALE = 256
SCORE_MAX_KEY = SCORE_MAX_Q * SCORE_SCALE
SCORE_BIN_COUNT = SCORE_MAX_KEY + 1
GOAL_SCORE_KEY = 0

RING_SLOT_COUNT =
    STREAM3_BATCH_CANDIDATES / (B_MICRO * MOVE_COUNT)

BEAM_WIDTH_ALIGNMENT =
    WORLD_SIZE * SHARD_COUNT * STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT

GLOBAL_BEAM_WIDTH_EFFECTIVE =
    round_up(USER_GLOBAL_BEAM_WIDTH, BEAM_WIDTH_ALIGNMENT)

GLOBAL_BEAM_WIDTH_EFFECTIVE =
    min(GLOBAL_BEAM_WIDTH_EFFECTIVE, GLOBAL_BEAM_WIDTH_MAX_SAFE)
```

Логи:

```text
USER_GLOBAL_BEAM_WIDTH
GLOBAL_BEAM_WIDTH_EFFECTIVE
GLOBAL_BEAM_WIDTH_MAX_SAFE
BEAM_WIDTH_ALIGNMENT
SCORE_SCALE
SCORE_MAX_KEY
SCORE_BIN_COUNT
SOLVED_RESULT_CAPACITY
```

---

# Структура массивов / типы

```cpp
using StateValue = uint8_t;
```

```cpp
struct alignas(16) State128 {
    StateValue v[STATE_STORAGE_LEN];
};
```

Контракт `State128`:

```text
State128.v[0..119]   = logical_state
State128.v[120..127] = padding / temporary final metadata
```

Persistent frontier contract:

```text
current_frontier_states[*].v[120..127] = 0
next_frontier_states_tmp[*].v[120..127] = 0 before persistent write
```

```cpp
struct alignas(16) Hash128 {
    uint64_t lo;
    uint64_t hi;
};
```

`Hash128`:

```text
один логический 128-битный hash
физически два uint64_t
```

```cpp
struct alignas(32) CandidateMeta {
    Hash128 hash;
    uint64_t parent_idx;

    uint32_t score_key;
    uint32_t route_packed;
};
```

`route_packed`:

```cpp
route_packed =
    (uint32_t(source_rank) << 16) |
    (uint32_t(owner)       << 8)  |
    uint32_t(move);
```

```cpp
source_rank = uint16_t(route_packed >> 16);
owner       = uint8_t((route_packed >> 8) & 0xff);
move        = uint8_t(route_packed & 0xff);
```

```cpp
struct alignas(16) FinalRequest {
    uint64_t parent_idx;
    uint32_t target_local_idx;
    uint16_t return_rank;
    uint8_t move;
    uint8_t pad;
};
```

```cpp
using FinalResponse = State128;
```

`FinalResponse` layout:

```text
FinalResponse.v[0..119]   = child_state
FinalResponse.v[120..123] = target_local_idx little-endian byte-pack
FinalResponse.v[124..127] = reserved
```

`target_local_idx` byte-pack:

```cpp
__device__ __host__ inline void final_response_set_target_local_idx(
    FinalResponse& response,
    uint32_t target_local_idx
) {
    response.v[120] = uint8_t(target_local_idx);
    response.v[121] = uint8_t(target_local_idx >> 8);
    response.v[122] = uint8_t(target_local_idx >> 16);
    response.v[123] = uint8_t(target_local_idx >> 24);
}
```

```cpp
__device__ __host__ inline uint32_t final_response_get_target_local_idx(
    const FinalResponse& response
) {
    return
        uint32_t(response.v[120]) |
        (uint32_t(response.v[121]) << 8) |
        (uint32_t(response.v[122]) << 16) |
        (uint32_t(response.v[123]) << 24);
}
```

Padding cleanup:

```cpp
__device__ __host__ inline void clear_state_padding(State128& state) {
    state.v[120] = 0;
    state.v[121] = 0;
    state.v[122] = 0;
    state.v[123] = 0;
    state.v[124] = 0;
    state.v[125] = 0;
    state.v[126] = 0;
    state.v[127] = 0;
}
```

Score quantization:

```cpp
__device__ uint32_t q_to_score_key(float q) {
    q = fminf(fmaxf(q, 0.0f), SCORE_MAX_Q);
    return uint32_t(__float2uint_rn(q * float(SCORE_SCALE)));
}
```

`stream3_val`:

```cpp
uint64_t stream3_val =
    (uint64_t(score_key) << 32) | uint64_t(payload_id);
```

```cpp
payload_id =
    ring_slot * (B_MICRO * MOVE_COUNT)
  + parent_local * MOVE_COUNT
  + move;
```

Static asserts:

```cpp
static_assert(sizeof(State128) == 128);
static_assert(alignof(State128) == 16);

static_assert(sizeof(Hash128) == 16);
static_assert(alignof(Hash128) == 16);

static_assert(sizeof(CandidateMeta) == 32);
static_assert(alignof(CandidateMeta) == 32);

static_assert(sizeof(FinalRequest) == 16);
static_assert(alignof(FinalRequest) == 16);

static_assert(sizeof(FinalResponse) == 128);
static_assert(alignof(FinalResponse) == 16);
```

---

# Память

## constant memory

```text
generators[MOVE_COUNT][STATE_STORAGE_LEN]
central_state[STATE_STORAGE_LEN]
```

Padding rules:

```text
generators[move][0..119]   = обычная перестановка
generators[move][120..127] = 120..127

central_state[0..119]      = logical solved-state
central_state[120..127]    = 0
```

## read-only VRAM

```text
model_weights_fp16
zobrist[STATE_STORAGE_LEN][STATE_VALUE_PAD] : Hash128
```

Padding rules:

```text
zobrist[0..119][0..127]    = обычная Zobrist-таблица
zobrist[120..127][0..127]  = Hash128{0, 0}
```

## shared memory

Только внутри отдельных kernel:

```text
block-local score histogram
block-local compact/prefix scratch
```

Не используется для:

```text
generators
zobrist
current_frontier_states
Hash128 stream3 buffers
CandidateMeta buffers
sort buffers
remote buffers
survivor_shard
```

## mutable static VRAM вне scratch_pool

```text
current_frontier_states[N] : State128

scratch_pool
```

Stop / solved buffers:

```text
solved_flag      : uint32_t
stop_flag        : uint32_t

solved_count     : uint32_t
solved_overflow  : uint32_t

solved_meta_list [SOLVED_RESULT_CAPACITY] : CandidateMeta
solved_depth_list[SOLVED_RESULT_CAPACITY] : uint32_t
```

Смысл:

```text
solved_flag     = хотя бы одно решение найдено
stop_flag       = сигнал прекращения новых jobs
solved_count    = число goal-кандидатов, которые active kernels успели записать или попытались записать
solved_overflow = число найденных goal-кандидатов превысило SOLVED_RESULT_CAPACITY
```

Инвариант:

```text
current_frontier_states не входит в scratch_pool
solved_* не входят в scratch_pool
stop_flag не входит в scratch_pool
```

## scratch_pool: layout_streams

```text
score_ring
  [RING_COUNT]
  [RING_SLOT_COUNT]
  [B_MICRO]
  [MOVE_COUNT]
  : uint32_t
```

```text
hash_ring
  [RING_COUNT]
  [RING_SLOT_COUNT]
  [B_MICRO]
  [MOVE_COUNT]
  : Hash128
```

```text
parent_base[RING_COUNT][RING_SLOT_COUNT] : uint64_t
count[RING_COUNT][RING_SLOT_COUNT]       : uint32_t
```

```text
stream3_key_a[STREAM3_BATCH_CANDIDATES] : Hash128
stream3_key_b[STREAM3_BATCH_CANDIDATES] : Hash128

stream3_val_a[STREAM3_BATCH_CANDIDATES] : uint64_t
stream3_val_b[STREAM3_BATCH_CANDIDATES] : uint64_t

unique_key[STREAM3_BATCH_CANDIDATES] : Hash128
unique_val[STREAM3_BATCH_CANDIDATES] : uint64_t
unique_count                         : uint32_t
```

```text
local_pending_buffer
remote_send_buffer
remote_recv_buffer

send_count[WORLD_SIZE]      : uint32_t
send_offset[WORLD_SIZE + 1] : uint32_t

recv_count[WORLD_SIZE]      : uint32_t
recv_offset[WORLD_SIZE + 1] : uint32_t
```

```text
survivor_shard[SHARD_COUNT][2 * STREAM4_BATCH_CANDIDATES] : CandidateMeta

clean_count[SHARD_COUNT]     : uint32_t
dirty_count[SHARD_COUNT]     : uint32_t
processing_flag[SHARD_COUNT] : bool

global_spill_buffer[GLOBAL_SPILL_CAPACITY] : CandidateMeta
```

```text
local_score_hist[SCORE_BIN_COUNT]  : uint64_t
global_score_hist[SCORE_BIN_COUNT] : uint64_t

current_threshold : uint32_t
```

## scratch_pool: layout_final

```text
next_frontier_states_tmp[N] : State128

final_request_buffer
final_response_buffer : FinalResponse / State128

final_send_count[WORLD_SIZE]      : uint32_t
final_send_offset[WORLD_SIZE + 1] : uint32_t

final_recv_count[WORLD_SIZE]      : uint32_t
final_recv_offset[WORLD_SIZE + 1] : uint32_t
```

Overlay invariant:

```text
layout_streams и layout_final используют одну физическую память scratch_pool
layout_streams и layout_final не активны одновременно
current_frontier_states не входит в scratch_pool
solved_* и stop_flag не входят в scratch_pool
```

`GLOBAL_BEAM_WIDTH_MAX_SAFE`:

```text
current_frontier_states
+ max(layout_streams_bytes, layout_final_bytes)
+ model_weights_fp16
+ read_only_tables
+ CUDA/NCCL/headroom
```

---

# Scheduler / dispatcher

`Scheduler / dispatcher` находится вне CUDA Graph.

Внутри одной глубины:

```text
Stream 1 / Stream 2 / Stream 3 / Stream 5 / Stream 4
работают параллельно по условиям готовности
```

`layout_streams` активен всю глубину.

## CUDA Graph templates

```text
ring_slot_graph:
    Stream 1
    Stream 2
```

`Stream 1` и `Stream 2` — разные kernel/job внутри одного `ring_slot_graph`.

```text
stream3_ring_graph:
    threshold
    compact
    pack Hash128 + stream3_val
    sort Hash128
    dedup Hash128
    restore CandidateMeta
    compute owner
    group remote by owner
    split local/remote
```

```text
stream4_shard_graph:
    stream4_job_threshold = current_threshold на момент запуска job
    threshold + compact
    sort shard
    dedup shard
    write clean survivor_shard
```

```text
final_materialize_graph:
    FinalRequest
    apply_move
    FinalResponse = State128
    write next_frontier_states_tmp
    copy next_frontier_states_tmp -> current_frontier_states
```

Вне CUDA Graph:

```text
dispatcher
условия запуска
очереди готовности
NCCL exchange
NCCL AllReduce histogram
CPU history transfer
stop propagation
```

## Событийный цикл глубины

```text
while depth_not_drained and stop_flag == 0:

    if solved_flag detected:
        stop launching new work
        propagate stop to all ranks
        wait active jobs to safe completion / early-exit checkpoint
        copy solved_count / solved_overflow / solved_meta_list / solved_depth_list to CPU
        reconstruct solutions
        return

    while frontier_remaining > 0
      and free_ring_slot_exists()
      and stop_flag == 0:

        ring, ring_slot = acquire_free_ring_slot()

        parent_base[ring][ring_slot] = frontier_cursor
        count[ring][ring_slot] = min(B_MICRO, frontier_remaining)

        frontier_cursor += count[ring][ring_slot]

        launch ring_slot_graph for ring, ring_slot


    for each ring:

        if stop_flag == 0
        and score_ready[ring][all ring_slot]
        and hash_ready[ring][all ring_slot]
        and stream3_not_running[ring]:

            launch stream3_ring_graph for ring


    for each remote_send_job:

        if stop_flag == 0
        and remote_send_ready
        and stream5_exchange_slot_free:

            launch Stream 5 exchange


    if stop_flag == 0
    and Stream 3 collector has work:

        consume:
            global_spill_buffer
            local_pending_buffer
            ready remote_recv_buffer

        fill:
            survivor_shard dirty regions

        create:
            ready shard jobs


    for each shard:

        if stop_flag == 0
        and dirty_count[shard] > 0
        and processing_flag[shard] == false
        and clean_count[shard] + dirty_count[shard] >= STREAM4_BATCH_CANDIDATES:

            processing_flag[shard] = true
            stream4_job_threshold = current_threshold
            launch stream4_shard_graph for shard
```

## Осушение глубины

```text
if frontier_cursor == frontier_size
and all Stream 1/2 ring_slot jobs done
and all Stream 3 ring jobs done
and all Stream 5 exchange jobs done:

    Stream 3 collector drains:
        global_spill_buffer
        local_pending_buffer
        all ready remote_recv_buffer

    for each shard with dirty_count[shard] > 0:
        processing_flag[shard] = true
        stream4_job_threshold = current_threshold
        launch stream4_shard_graph for shard

    wait all Stream 4 shard jobs

    depth_drained = true
```

После:

```text
layout_final активируется только после depth_drained == true
```

---

# Stream 1: инференс

Backend:

```text
CUTLASS/custom only
```

Fallback отсутствует.

Вход:

```text
current_frontier_states
parent_base[ring][ring_slot]
count[ring][ring_slot]
model_weights_fp16
SCORE_MAX_Q
SCORE_SCALE
```

Работа:

```text
parent_idx = parent_base[ring][ring_slot] + parent_local
state = current_frontier_states[parent_idx]

CUTLASS/custom neural_network(state, model_weights_fp16)

GEMM epilogue:
    q_float
    -> clamp [0, SCORE_MAX_Q]
    -> multiply by SCORE_SCALE
    -> round to nearest
    -> score_key:uint32_t
```

Запись:

```text
score_ring[ring][ring_slot][parent_local][move] = score_key
```

Готовность:

```text
score_ready[ring][ring_slot]
```

Инвариант:

```text
Stream 1 не пишет q_float в global memory
Stream 1 пишет только score_key:uint32_t
Stream 1 backend только CUTLASS/custom
Stream 1 почти непрерывно загружает GEMM
```

---

# Stream 2: ключи, цель

Вход:

```text
current_frontier_states
parent_base[ring][ring_slot]
count[ring][ring_slot]

generators[MOVE_COUNT][STATE_STORAGE_LEN]
central_state[STATE_STORAGE_LEN]

zobrist[STATE_STORAGE_LEN][STATE_VALUE_PAD]
```

Работа батчем:

```text
Stream 2 обрабатывает:
    B_MICRO parents × MOVE_COUNT moves

parent batch:
    current_frontier_states[parent_base : parent_base + count]

generator matrix:
    generators[MOVE_COUNT][STATE_STORAGE_LEN]

output:
    hash_ring[ring][ring_slot][B_MICRO][MOVE_COUNT]
```

Локальная материализация:

```text
child_state : State128
```

```text
for p in 0 .. STATE_STORAGE_LEN-1:
    child_state.v[p] =
        parent_state.v[generators[move][p]]
```

`child_state` не пишется в global memory.

Goal-check:

```text
found = true

for p in 0 .. STATE_STORAGE_LEN-1:
    if child_state.v[p] != central_state[p]:
        found = false
```

Hash:

```text
hash = Hash128{0, 0}

for p in 0 .. STATE_STORAGE_LEN-1:
    v = child_state.v[p]
    h = zobrist[p][v]

    hash.lo ^= h.lo
    hash.hi ^= h.hi
```

Padding не влияет на hash:

```text
zobrist[120..127][*] = Hash128{0, 0}
```

Запись:

```text
hash_ring[ring][ring_slot][parent_local][move] = hash
```

Goal handling:

```cpp
if (found) {
    uint32_t idx = atomicAdd(&solved_count, 1);

    if (idx < SOLVED_RESULT_CAPACITY) {
        CandidateMeta meta;
        meta.hash = hash;
        meta.parent_idx = parent_idx;
        meta.score_key = GOAL_SCORE_KEY;
        meta.route_packed =
            (uint32_t(LOCAL_RANK) << 16) |
            (uint32_t(LOCAL_RANK) << 8)  |
            uint32_t(move);

        solved_meta_list[idx] = meta;
        solved_depth_list[idx] = depth;
    } else {
        atomicExch(&solved_overflow, 1);
    }

    __threadfence_system();

    if (atomicCAS(&solved_flag, 0, 1) == 0) {
        atomicExch(&stop_flag, 1);
    }
}
```

Готовность:

```text
hash_ready[ring][ring_slot]
```

Ограничение:

```text
Stream 2 не считает owner для обычных кандидатов
Stream 2 не раскладывает кандидатов по owner-буферам
Stream 2 только пишет hash_ring
full child_tmp_global отсутствует
```

---

# Stream 3: локал дедуп, порог, раскладка STREAM3_BATCH_CANDIDATES

Вход:

```text
score_ring[ring]
hash_ring[ring]

parent_base[ring]
count[ring]

LOCAL_RANK
WORLD_SIZE
current_threshold
```

Ожидание:

```text
score_ready[ring][all ring_slot]
hash_ready[ring][all ring_slot]
```

Индекс кандидата:

```text
i = 0 .. STREAM3_BATCH_CANDIDATES-1

ring_slot    = i / (B_MICRO * MOVE_COUNT)
local_i      = i % (B_MICRO * MOVE_COUNT)
parent_local = local_i / MOVE_COUNT
move         = local_i % MOVE_COUNT
```

Пропуск хвоста:

```cpp
if (parent_local >= count[ring][ring_slot]) return;
```

## Threshold + compact

```text
если score_key > current_threshold:
    пропуск

иначе:
    запись в compact stream3_key_a / stream3_val_a
```

## Pack

```text
stream3_key_a[compact_i]:
    hash = hash_ring[ring][ring_slot][parent_local][move]

stream3_val_a[compact_i]:
    score_key << 32 | payload_id
```

Критичный контракт:

```text
payload_id = исходный индекс кандидата внутри STREAM3_BATCH_CANDIDATES
payload_id != compact_i
```

## Sort

Порядок сортировки:

```text
hash.hi
hash.lo
```

Назначение:

```text
одинаковые Hash128 становятся соседними
```

## Dedup

Ключ дедупликации:

```text
Hash128
```

Правило выбора:

```text
оставить min(stream3_val)
```

`min(stream3_val)` означает:

```text
сначала минимальный score_key
при равном score_key — минимальный payload_id
```

## Restore `CandidateMeta` + owner + split

Один проход по `unique_key / unique_val`:

```text
score_key  = unique_val >> 32
payload_id = unique_val & 0xffffffff

payload_id -> ring_slot, parent_local, move

parent_idx =
    parent_base[ring][ring_slot] + parent_local
```

Owner после dedup:

```cpp
owner = owner_from_hash128(unique_key.hi, unique_key.lo, WORLD_SIZE);
```

```cpp
route_packed =
    (uint32_t(LOCAL_RANK) << 16) |
    (uint32_t(owner)      << 8)  |
    uint32_t(move);
```

```text
CandidateMeta:
    hash = unique_key
    parent_idx
    score_key
    route_packed
```

Раскладка:

```text
owner == LOCAL_RANK:
    local_pending_buffer

owner != LOCAL_RANK:
    remote_send_buffer
```

`remote_send_buffer` группируется по `owner` после dedup:

```text
count_by_owner
scan owner counts
scatter CandidateMeta в remote_send_buffer по owner ranges
```

Диапазоны:

```text
remote_send_buffer[
    send_offset[peer] :
    send_offset[peer] + send_count[peer]
]
```

## Collector для Stream 4

`Stream 3 collector` — единственный писатель в shard-буферы Stream 4.

Источники:

```text
1. global_spill_buffer
2. local_pending_buffer
3. ready remote_recv_buffer после Stream 5
```

Для каждого кандидата:

```text
shard = shard_from_hash128(candidate.hash)
```

Если shard свободен:

```text
processing_flag[shard] == false

write:
    survivor_shard[shard][clean_count[shard] + dirty_count[shard]]

dirty_count[shard]++
```

Если shard занят:

```text
processing_flag[shard] == true

write:
    global_spill_buffer
```

Условие запуска Stream 4 shard job:

```text
dirty_count[shard] > 0
processing_flag[shard] == false
clean_count[shard] + dirty_count[shard] >= STREAM4_BATCH_CANDIDATES
```

Действие:

```text
processing_flag[shard] = true
stream4_job_threshold = current_threshold
launch stream4_shard_graph for shard
```

Ограничения:

```text
Stream 3 — единственный владелец заполнения shard-буферы Stream 4
Stream 5 не пишет в shard-буферы Stream 4
atomicAdd на каждый кандидат для Stream 4 не используется
```

---

# Stream 4: сбор из Stream 3 / Stream 5, дедуп, порог

Работает shard job.

Вход:

```text
survivor_shard[shard][0 : clean_count[shard] + dirty_count[shard]]
stream4_job_threshold
```

Работа:

```text
1. применить stream4_job_threshold к входу

2. compact

3. sort по:
       hash.hi
       hash.lo

4. dedup по:
       Hash128

5. оставить лучший CandidateMeta по score_key

6. записать clean результат обратно:
       survivor_shard[shard][0 : new_clean_count]
```

Повторный threshold после dedup не выполняется.

После завершения:

```text
clean_count[shard] = new_clean_count
dirty_count[shard] = 0
processing_flag[shard] = false
```

Периодическое обновление `current_threshold` после каждых:

```text
GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS
```

обработанных shard job:

```text
1. построить local_score_hist[SCORE_BIN_COUNT]
   по clean survivor-ам

2. AllReduce SUM:
   local_score_hist -> global_score_hist

3. if threshold_initialized == false
   and total_survivors < GLOBAL_BEAM_WIDTH_EFFECTIVE:
       current_threshold = UINT32_MAX

4. if total_survivors >= GLOBAL_BEAM_WIDTH_EFFECTIVE:
       new_threshold = histogram_threshold(...)
       current_threshold = min(current_threshold, new_threshold)
       threshold_initialized = true
```

Монотонность:

```text
до threshold_initialized == true:
    current_threshold = UINT32_MAX

после threshold_initialized == true:
    current_threshold не ослабляется
    new current_threshold <= old current_threshold
```

`Stream 4` не делает:

```text
top-k shard
cap shard
```

Размер beam ограничивается:

```text
periodic global threshold
final global threshold
```

---

# Stream 5: обмен GPU

Вход:

```text
remote_send_buffer
send_count[WORLD_SIZE]
send_offset[WORLD_SIZE + 1]

recv_count[WORLD_SIZE]
recv_offset[WORLD_SIZE + 1]
```

Работа:

```text
для каждого peer:

    send range:
        remote_send_buffer[
            send_offset[peer] :
            send_offset[peer] + send_count[peer]
        ]

    recv range:
        remote_recv_buffer[
            recv_offset[peer] :
            recv_offset[peer] + recv_count[peer]
        ]
```

Выход:

```text
remote_recv_buffer
remote_recv_ready
```

Ограничение:

```text
Stream 5 только выполняет GPU exchange
Stream 5 не пишет в shard-буферы Stream 4
```

---

# Финал

Финал начинается только если:

```text
solved_flag == 0
```

и после полного осушения активных работ:

```text
Stream 1
Stream 2
Stream 3
Stream 5
Stream 4
```

## Ожидание всех карт

```text
карта завершила обработку своего current_frontier_states
карта ждёт завершения остальных карт
```

## Финальный flush shard-ов

```text
Stream 3 collector сначала раскидывает global_spill_buffer

для всех shard:
    если dirty_count[shard] > 0:
        processing_flag[shard] = true
        stream4_job_threshold = current_threshold
        launch stream4_shard_graph for shard
```

После завершения:

```text
все shard имеют только clean-регион
локальная финальная дедупликация завершена
```

## Финальный global threshold

```text
1. local_score_hist[SCORE_BIN_COUNT] по clean survivors

2. AllReduce SUM:
   local_score_hist -> global_score_hist

3. найти threshold_score по GLOBAL_BEAM_WIDTH_EFFECTIVE

4. current_threshold = threshold_score

5. локальное финальное отсечение:
   score_key <= current_threshold
```

Хвост одинакового `score_key` разрешён:

```text
final count may exceed GLOBAL_BEAM_WIDTH_EFFECTIVE
```

## Балансировка по картам

После финального отсечения:

```text
local_keep_count на каждой карте
AllGather counts
global_keep_count
prefix counts
```

Для каждого оставшегося `CandidateMeta`:

```text
global_idx = prefix_count[LOCAL_RANK] + local_idx

target_rank
target_local_idx
```

Назначение:

```text
равномерное распределение next_frontier_states_tmp по WORLD_SIZE
```

## Передача истории на CPU

```text
final CandidateMeta со всех карт -> CPU
```

Назначение:

```text
CPU хранит историю
CPU восстанавливает путь решения по CandidateMeta
```

## FinalRequest

Для каждого финального `CandidateMeta`:

```text
source_rank = unpack_source_rank(route_packed)
move        = unpack_move(route_packed)

parent_idx
move
return_rank = target_rank
target_local_idx
```

```text
FinalRequest:
    parent_idx
    target_local_idx
    return_rank
    move
```

Запросы группируются по `source_rank`.

## FinalResponse

На `source_rank`:

```text
parent_state = current_frontier_states[parent_idx]
child_state = apply_move(parent_state, move)

child_state.v[120..123] = target_local_idx, ручной byte-pack
child_state.v[124..127] = reserved
```

Ответ:

```text
FinalResponse = child_state : State128
```

Ответ отправляется на `return_rank`.

## Запись next_frontier

На `return_rank`:

```text
response = FinalResponse

target_local_idx =
    final_response_get_target_local_idx(response)

response.v[120..127] = 0

next_frontier_states_tmp[target_local_idx] = response
```

Большой промежуточный буфер полученных родителей не создаётся.

## Копирование в current_frontier_states

После завершения всех ответов:

```text
copy next_frontier_states_tmp -> current_frontier_states
```

После копирования:

```text
layout_final не активен
scratch_pool снова доступен как layout_streams
```

---

# Stop / solved path

Первый найденный goal:

```text
ставит solved_flag
ставит stop_flag
останавливает запуск новых jobs
```

Active jobs:

```text
могут дойти до checkpoint
могут найти ещё goal-кандидаты
записывают goal-кандидаты в solved_meta_list
```

Dispatcher:

```text
не запускает новые Stream 1/2/3/4/5 jobs
распространяет stop на все rank
ждёт безопасного завершения active jobs
копирует solved_count / solved_overflow / solved_meta_list / solved_depth_list на CPU
восстанавливает найденные решения
возвращает solution
```

Ограничение:

```text
solved_meta_list содержит goal-кандидаты, которые реально успели найти active kernels до остановки
solved_meta_list не обязан содержать все возможные goal-кандидаты всей глубины после stop_flag
```

Goal-кандидат не проходит через:

```text
threshold
dedup
exchange
final threshold
load balancing
```

---

# Итоговые архитектурные инварианты

```text
State128.v[0..119]   = логическое состояние.
State128.v[120..127] = padding / служебная зона.

persistent frontier states:
  v[120..127] = 0

FinalResponse = State128.
FinalResponse.v[120..123] хранит target_local_idx только в финальном обмене.
Перед записью в next_frontier_states_tmp padding очищается.

generators[move][120..127] = 120..127.
central_state[120..127] = 0.
zobrist[120..127][*] = Hash128{0, 0}.

Stream 1/2/3/4/5 не материализуют full next_frontier.
next_frontier_states_tmp существует только в layout_final внутри scratch_pool.

layout_streams живёт всю глубину.
layout_final включается только после осушения Stream 1/2/3/5/4.

Stream 1 / Stream 2 / Stream 3 / Stream 5 / Stream 4
работают параллельно по условиям готовности.

Stream 1 и Stream 2 не fused kernel.
Stream 1 и Stream 2 находятся в одном ring_slot_graph.

Stream 1 использует только CUTLASS/custom backend.
Stream 1 пишет score_key:uint32_t.
Stream 1 не пишет q_float в global memory.

Stream 2 не считает owner.
owner_ring отсутствует.
Stream 2 пишет hash_ring с Hash128.
Stream 2 материализует child_state локально как State128.
Stream 2 делает goal-check по STATE_STORAGE_LEN=128.
Stream 2 делает hash по STATE_STORAGE_LEN=128.
Padding не влияет на hash из-за нулевых zobrist-строк.

Hash128 = один логический 128-битный хэш,
физически два uint64_t.

Stream 3 считает owner после dedup из Hash128.

Stream 3 применяет current_threshold до sort/dedup.
Stream 3 выполняет compact перед sort/dedup.

Stream 3 использует Hash128 как ключ сортировки/дедупликации.
Stream3Key отсутствует.

Stream 3 использует CUB sort/reduce как базовый путь.
Свой sort для Stream 3 не фиксируется.

Stream 3 группирует remote_send_buffer по owner после dedup.

Stream 3 — единственный писатель в shard-буферы Stream 4.
Stream 5 только делает exchange и пишет remote_recv_buffer.

Stream 4 работает по shard-ам.
Stream 4 не делает top-k/cap shard.
Stream 4 делает threshold + compact + sort + dedup + merge clean/dirty.

На shard хранится один survivor_shard с clean/dirty регионами.

global_spill_buffer — общий временный буфер для кандидатов, чей shard занят Stream 4.

threshold_initialized=false до первого достаточного survivor-count.
current_threshold=UINT32_MAX до threshold initialization.
После threshold_initialized=true threshold не ослабляется.

GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS задаёт частоту периодического пересчёта current_threshold.
Финальный global threshold считается только после финального flush и локальной финальной дедупликации.

После финального threshold выполняется балансировка нагрузки по картам.

solved_flag / stop_flag / solved_count / solved_meta_list живут вне scratch_pool.
GOAL_SCORE_KEY = 0.
Goal-кандидат пишет score_key = GOAL_SCORE_KEY.

CUDA Graph используется на повторяемых job-шаблонах.
Dispatcher остаётся вне CUDA Graph.

generators остаётся в constant memory.
zobrist хранится как Hash128[STATE_STORAGE_LEN][STATE_VALUE_PAD].
State хранится как State128.
CandidateMeta имеет размер 32 байта.
route_packed хранит source_rank + owner + move.
GLOBAL_BEAM_WIDTH_EFFECTIVE используется во всех вычислениях порога, балансировки и логах.
```
