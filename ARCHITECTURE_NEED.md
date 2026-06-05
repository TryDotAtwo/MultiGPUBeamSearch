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
STREAM4_TRIGGER_CANDIDATES
STREAM4_BATCH_ALIGNMENT

RING_COUNT
RING_SLOT_COUNT

WORLD_SIZE
LOCAL_RANK

SHARD_COUNT
SHARD_BUFFER_COUNT
STORAGE_SHARD_COUNT
SHARD_CAPACITY_CANDIDATES
SHARD_CAPACITY_SCALE_PPM
GLOBAL_SPILL_CAPACITY
GLOBAL_SPILL_SCALE_PPM

USER_GLOBAL_BEAM_WIDTH
GLOBAL_BEAM_WIDTH_EFFECTIVE
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


## 2026-05-26 Stream5 threshold update contract

```text
WORLD_SIZE == 1:
    threshold scheduling may keep the existing single-GPU host-side periodic path.

WORLD_SIZE > 1:
    GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS must not drive threshold scheduling.
    Stream4 remains local and does not call NCCL.
    Stream4 histogram publication remains per-shard A/B:
        Stream4 writes inactive shard_score_hist buffer.
        Stream4 commits shard_score_hist_active_index only after inactive histogram is complete.
        Stream5 threshold snapshot reads shard_score_hist_active_index, then sums the selected A/B buffers.
    Stream5 owns threshold collective communication.
    Each rank publishes threshold_request_local as a service field after enough completed local Stream4 work.
    Every rank participates in threshold request reduction at Stream5 exchange points.
    Collective request reduction:
        threshold_request_global = ncclAllReduceMax(threshold_request_local)
    If threshold_request_global != 0:
        every rank builds local_score_hist from committed Stream4 histogram A/B buffers.
        every rank participates in NCCL SUM local_score_hist -> global_score_hist.
        every rank computes monotonic current_threshold.
        every rank resets its local processed-work threshold counter.

current_threshold publication:
    current_threshold has two slots.
    threshold_initialized has two slots.
    current_threshold_active_index selects the committed slot.
    Stream5 writes inactive current_threshold slot first.
    Stream5 writes inactive threshold_initialized slot second.
    Stream5 commits current_threshold_active_index after device memory fence.
    Stream3 reads current_threshold[current_threshold_active_index & 1].
    Stream4 reads current_threshold[current_threshold_active_index & 1].
    No new atomic operations are allowed in threshold publication or request fields.
```
```text
STATE_LEN = 120
STATE_STORAGE_LEN = 128
STATE_VALUE_PAD = 128
MOVE_COUNT = 24

SCORE_MAX_Q = 300.0
SCORE_SCALE = 1024
SCORE_MAX_KEY = SCORE_MAX_Q * SCORE_SCALE
SCORE_BIN_COUNT = SCORE_MAX_KEY + 1
GOAL_SCORE_KEY = 0

RING_SLOT_COUNT =
    STREAM3_BATCH_CANDIDATES / (B_MICRO * MOVE_COUNT)

STREAM3_BATCH_CANDIDATES =
    RING_SLOT_COUNT * B_MICRO * MOVE_COUNT

BEAM_WIDTH_ALIGNMENT =
    WORLD_SIZE * SHARD_COUNT * STREAM4_BATCH_ALIGNMENT

GLOBAL_BEAM_WIDTH_EFFECTIVE =
    round_up(USER_GLOBAL_BEAM_WIDTH, BEAM_WIDTH_ALIGNMENT)

N_LOCAL =
    ceil(GLOBAL_BEAM_WIDTH_EFFECTIVE / WORLD_SIZE)

LOGICAL_SHARD_SIZE =
    ceil(N_LOCAL / SHARD_COUNT)

STORAGE_SHARD_COUNT =
    SHARD_COUNT * SHARD_BUFFER_COUNT

SHARD_BUFFER_COUNT =
    2

Each logical shard always has two resident physical shard buffers: A/B.

SHARD_CAPACITY_CANDIDATES =
    round_up(ceil(LOGICAL_SHARD_SIZE * SHARD_CAPACITY_SCALE_PPM / 1_000_000),
             STREAM4_BATCH_ALIGNMENT)

RING_COUNT =
    ceil(LOGICAL_SHARD_SIZE / (B_MICRO * MOVE_COUNT))

GLOBAL_SPILL_CAPACITY =
    0

Config search:
    choose SHARD_COUNT and STREAM4_BATCH_CANDIDATES under memory budget
    allocate all STORAGE_SHARD_COUNT resident physical shard buffers with SHARD_CAPACITY_CANDIDATES slots each
    reject default candidates with too few Stream4 jobs per logical shard
    score candidates by Stream4 waves, Stream4 jobs, batch size, shard count
```

Логи:

```text
USER_GLOBAL_BEAM_WIDTH
GLOBAL_BEAM_WIDTH_EFFECTIVE
BEAM_WIDTH_ALIGNMENT
SCORE_SCALE
SCORE_MAX_KEY
SCORE_BIN_COUNT
SOLVED_RESULT_CAPACITY
N_LOCAL
LOGICAL_SHARD_SIZE
SHARD_BUFFER_COUNT
STORAGE_SHARD_COUNT
STREAM4_JOBS_PER_SHARD
STREAM4_JOBS_PER_DEPTH
STREAM4_WAVES_PER_DEPTH
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

```cpp
struct alignas(32) FinalHistoryRecord {
    CandidateMeta meta;
    uint32_t target_local_idx;
    uint32_t reserved0;
    uint64_t reserved1[3];
};
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

static_assert(sizeof(FinalHistoryRecord) == 64);
static_assert(alignof(FinalHistoryRecord) == 32);
static_assert(sizeof(FinalHistoryRecord) % sizeof(uint64_t) == 0);

static_assert(sizeof(FinalRequestValidationError) == 48);
static_assert(alignof(FinalRequestValidationError) == 16);
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
solved_suffix_list[SOLVED_RESULT_CAPACITY] : uint32_t
```

Solved neighborhood lookup:

```text
BEAM_SOLVED_NEIGHBORHOOD_RADIUS = K1

K1 == 0:
    Stream 2 uses the existing exact central_state comparison.

K1 > 0:
    CPU builds the inverse-move solved neighborhood before depth processing:
        all states that can reach central_state in <= K1 moves
        Hash128 -> packed suffix moves, stored on CPU

    GPU stores a readonly bucket table:
        fingerprint_slots[bucket_count][4] : uint32_t
        hash_slots[bucket_count][4]        : Hash128

    Stream 2 computes Hash128(parent + move) as before.
    Stream 2 checks only two fixed buckets by fingerprint, then confirms full Hash128.
    Stream 2 does not store states or suffixes in VRAM.
    CPU appends the suffix from Hash128 -> suffix after history prefix reconstruction.

Stream2 generated-candidate suffix expansion:
    BEAM_STREAM2_SUFFIX_RADIUS = K2
    BEAM_STREAM2_SUFFIX_BACKEND in {base_generators, composed_permutations}
    BEAM_STREAM2_SUFFIX_MAX_COUNT is an optional startup guard.

K2 == 0:
    Stream 2 behavior matches K1-only behavior.
    solved_suffix_list[idx] is 0 for direct/K1 hits.

K2 > 0:
    CPU builds all suffix move chains with length <= K2 before depth processing.
    GPU stores suffix packed_moves[] and lengths[] in readonly VRAM.
    Backend base_generators stores only suffix move chains; Stream 2 uses the base generator table per suffix move.
    Backend composed_permutations additionally stores composed perm[STATE_STORAGE_LEN] per suffix for one-permutation state projection.
    Stream 2 first computes and writes Hash128(parent + move) to hash_ring exactly as before.
    Stream 2 then checks parent + move + suffix_id for suffix_id >= 1 against K1 table or exact central_state.
    Stream 2 records solved_suffix_list[idx] = suffix_id on hit.
    CPU reconstructs history prefix, appends K2 suffix by suffix_id, then appends K1 suffix by Hash128 -> suffix.
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
survivor_shard[STORAGE_SHARD_COUNT][SHARD_CAPACITY_CANDIDATES] : CandidateMeta

clean_count[STORAGE_SHARD_COUNT]     : uint32_t
dirty_count[STORAGE_SHARD_COUNT]     : uint32_t
processing_flag[STORAGE_SHARD_COUNT] : bool
stream3_write_buffer_index[SHARD_COUNT] : uint32_t

global_spill_buffer[GLOBAL_SPILL_CAPACITY] : CandidateMeta
```

```text
local_score_hist[SCORE_BIN_COUNT]  : uint64_t
global_score_hist[SCORE_BIN_COUNT] : uint64_t

current_threshold : uint32_t
```

## scratch_pool: three overlay layouts

```text
layout_1_streams:
    Stream 1/2 score/hash/parent rings
    Stream 3 threshold/compact/sort/dedup/split scratch
    Stream 5 CandidateMeta send/recv slots
    Stream 4 survivor_shard/dirty/clean/sort scratch

layout_2_final_select:
    final local dedup/filter/threshold/load-balance scratch
    final_candidate_buffer : CandidateMeta[N_LOCAL]
    final_selected_buffer  : CandidateMeta[min(GLOBAL_BEAM_WIDTH_EFFECTIVE, survivor_count)] when WORLD_SIZE > 1

layout_3_final_materialize:
    ring_slots = 3
    chunk_capacity = min(N_LOCAL, STREAM3_BATCH_CANDIDATES)
    exchange_capacity = chunk_capacity * WORLD_SIZE
    key_slot_a[ring_slots][exchange_capacity] : uint32_t
    key_slot_b[ring_slots][exchange_capacity] : uint32_t
    request_slot_a[ring_slots][exchange_capacity] : FinalRequest
    request_slot_b[ring_slots][exchange_capacity] : FinalRequest
    request_recv_slot[ring_slots][exchange_capacity] : FinalRequest
    response_send_slot[ring_slots][exchange_capacity] : FinalResponse
    response_recv_slot[ring_slots][exchange_capacity] : FinalResponse
    history_send_slot[ring_slots][chunk_capacity] : FinalHistoryRecord
    history_recv_slot[ring_slots][exchange_capacity] : FinalHistoryRecord
    fixed CUB temp for request sort by rank key
```

## scratch_pool: layout_final

```text
next_frontier_states_tmp[N] : State128

final_request_buffer : FinalRequest[N_LOCAL] only when WORLD_SIZE == 1
final_validation_error : FinalRequestValidationError
final_response_buffer : omitted when WORLD_SIZE > 1; chunked response slots replace the full buffer

final_send_count[WORLD_SIZE]      : uint32_t
final_send_offset[WORLD_SIZE + 1] : uint32_t

final_recv_count[WORLD_SIZE]      : uint32_t
final_recv_offset[WORLD_SIZE + 1] : uint32_t
```

Final materialization data-plane:

```text
CPU participates only in CandidateMeta history transfer after GPU materialization chunk ownership is fixed.
CPU does not participate in FinalRequest exchange.
CPU does not participate in FinalResponse exchange.
GPU builds FinalRequest directly from selected CandidateMeta chunks.
GPU exchanges FinalRequest / FinalResponse through NCCL only.
source_rank == local_rank request path is local GPU work, not NCCL traffic.
return_rank == local_rank response path writes next_frontier_states_tmp directly, not NCCL traffic.
remote response slots are pre-partitioned by return_rank offsets; no atomic write is required.
request slots are sorted/grouped by source_rank before NCCL exchange.
response work slots are sorted/grouped by return_rank before NCCL exchange.
WORLD_SIZE == 1 final path uses the existing single-GPU request/materialize path.
WORLD_SIZE > 1 final path does not allocate full final_request_buffer or full final_response_buffer.
```

Layout 3 chunk lifetime:

```text
candidate_meta_chunk -> build request/history records on GPU
candidate_meta_chunk -> async CPU history copy may run independently from GPU-GPU materialization
after request/history build and CPU history transfer for the chunk are queued, candidate_meta_chunk memory may be reused
response exchange for a slot may overlap build/sort work for later slots
slot reuse waits for the previous response_done event before response_recv memory is overwritten
slot events: build_done, history_done, request_done, response_ready, response_done
```

Layout 3 alignment:

```text
CandidateMeta: size=32 align=32
FinalRequest: size=16 align=16
FinalResponse: size=128 align>=16 and slot base align>=32
FinalHistoryRecord: size=64 align=32
FinalRequestValidationError: size=48 align=16
uint32_t key slots: align>=4
CUB temp: align=256
host exchange control arrays: fixed capacity WORLD_SIZE<=128, no per-chunk dynamic containers
```

Overlay invariant:

```text
layout_streams и layout_final используют одну физическую память scratch_pool
layout_streams и layout_final не активны одновременно
current_frontier_states не входит в scratch_pool
solved_* и stop_flag не входят в scratch_pool
layout_phase1_streams, layout_phase2_select, layout_phase3_materialize use one static scratch_pool
layout_phase2_select output buffers consumed by layout_phase3_materialize stay in a common prefix
layout_phase2_select temporary filter buffers and layout_phase3_materialize temporary exchange buffers overlay after the common prefix
layout_final_budget_bytes is a reserved scratch_pool prefix; persistent stream storage such as survivor_shard, clean_count, and shard histograms starts after this prefix
scratch_pool_bytes = max(layout_phase1_streams_bytes, layout_phase2_select_bytes, layout_phase3_materialize_bytes)
no cudaMalloc/cudaFree is allowed inside depth-loop final materialization
```

Config search memory objective:

```text
layout_final_budget_bytes = max(layout_phase2_select_bytes, layout_phase3_materialize_bytes)
layout_streams_bytes does not need to fit inside layout_final_budget_bytes
SHARD_COUNT and STREAM4_BATCH_CANDIDATES are searched under total device memory budget
GLOBAL_BEAM_WIDTH_EFFECTIVE is only aligned USER_GLOBAL_BEAM_WIDTH, not capped
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


    for each logical_shard:

        if stop_flag == 0
        and no physical buffer in logical_shard has processing_flag == true
        and some physical_shard in logical_shard has dirty_count[physical_shard] > 0
        and clean_count[physical_shard] + dirty_count[physical_shard] >= STREAM4_TRIGGER_CANDIDATES:

            choose one physical_shard from logical_shard
            processing_flag[physical_shard] = true
            stream4_job_threshold = current_threshold
            launch stream4_shard_graph for physical_shard
            keep sibling physical buffer writable for Stream 3
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

    for each logical_shard with dirty physical buffer:
        choose at most one physical_shard when no sibling is processing
        processing_flag[physical_shard] = true
        stream4_job_threshold = current_threshold
        launch stream4_shard_graph for physical_shard

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
        solved_suffix_list[idx] = suffix_id; // 0 means no Stream2 suffix
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

Owner distribution:

```text
owner_from_hash128 uses deterministic avalanche mixing over Hash128 before modulo WORLD_SIZE.
Dedup key remains raw Hash128.
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
logical_shard = shard_from_hash128(candidate.hash)
```

Shard distribution:

```text
shard_from_hash128 uses deterministic avalanche mixing over Hash128 before modulo SHARD_COUNT.
Shard routing key is separate from owner routing key by domain salt.
Dedup key remains raw Hash128.
```

Physical shard mapping:

```text
current_buffer = stream3_write_buffer_index[logical_shard] % SHARD_BUFFER_COUNT
physical_shard = logical_shard * SHARD_BUFFER_COUNT + current_buffer
```

Stream 3 collector write target selection:

```text
prefer current_buffer when physical_shard has enough free slots
otherwise choose a non-processing sibling physical buffer with maximum free slots
if no A/B physical buffer can accept the candidate group:
    raise fatal error
single-buffer global_spill path is legacy/unreachable under SHARD_BUFFER_COUNT == 2 contract
```

Если physical_shard свободен:

```text
processing_flag[physical_shard] == false

write:
    survivor_shard[physical_shard][clean_count[physical_shard] + dirty_count[physical_shard]]

dirty_count[physical_shard]++
```

Если physical_shard занят:

```text
processing_flag[physical_shard] == true

write:
    global_spill_buffer
```

Условие запуска Stream 4 shard job:

```text
dirty_count[physical_shard] > 0
processing_flag[physical_shard] == false
clean_count[physical_shard] + dirty_count[physical_shard] >= STREAM4_TRIGGER_CANDIDATES
no sibling physical buffer for the same logical_shard has processing_flag != false
```

Действие:

```text
processing_flag[physical_shard] = true
stream4_job_threshold = current_threshold
launch stream4_shard_graph for physical_shard
stream3_write_buffer_index[logical_shard] = non-processing sibling buffer with free capacity
```

Ограничения:

```text
Stream 3 — единственный владелец заполнения shard-буферы Stream 4
Stream 5 не пишет в shard-буферы Stream 4
Stream 4 may process at most one physical buffer per logical_shard at a time
The other A/B physical buffer remains the Stream 3 write target when capacity is available
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

Хвост одинакового `score_key` до финального этапа разрешён.

Финальный этап отрезает хвост после `final_threshold`:

```text
raw_final_count may exceed GLOBAL_BEAM_WIDTH_EFFECTIVE
final count is capped to GLOBAL_BEAM_WIDTH_EFFECTIVE
per-rank next_frontier target count is GLOBAL_BEAM_WIDTH_EFFECTIVE / WORLD_SIZE
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
CPU хранит историю в сжатом формате восстановления:
    parent_idx
    route_packed
CPU может асинхронно чистить dead-branch history:
    live previous entries = parent_idx, используемые следующим history layer
    после compact previous layer CPU remap-ит parent_idx следующего layer
CPU восстанавливает путь решения по parent_idx + route_packed
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
Stream 4 запускает не больше одного physical shard на logical shard одновременно.

На каждый logical shard хранится SHARD_BUFFER_COUNT physical survivor_shard buffers с clean/dirty регионами.
Stream 3 пишет в non-processing physical survivor_shard.

global_spill_buffer — общий временный буфер для кандидатов, чей shard занят Stream 4.

threshold_initialized=false до первого достаточного survivor-count.
current_threshold=UINT32_MAX до threshold initialization.
После threshold_initialized=true threshold не ослабляется.

GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS задаёт частоту периодического пересчёта current_threshold.
WORLD_SIZE > 1 periodic threshold update uses local completed shard histograms only.
WORLD_SIZE > 1 periodic threshold update does not run NCCL AllReduce and does not require a cross-rank barrier.
WORLD_SIZE > 1 final global threshold still uses NCCL/global counts after local final flush.
Финальный global threshold считается только после финального flush и локальной финальной дедупликации.

После финального threshold выполняется балансировка нагрузки по картам.

Final phase pipeline switch:

```text
after Stream 1 and Stream 2 are drained on a card:
    card may use final-local mode

final-local mode:
    residual spill is sorted/partitioned by Stream 3
    clean-only shard regions may be submitted to Stream 4
    Stream 4 applies current_threshold + dedup + compact
    purpose is local final dedup before global threshold

after local final dedup:
    Stream 5 exchanges candidate-count / histogram metadata
    final global threshold is computed
    final cut and load balance are applied

single-rank final completion:
    if dirty_count is zero
    and clean deduped candidate count >= GLOBAL_BEAM_WIDTH_EFFECTIVE
    residual spill may remain outside survivor_shard
    final threshold/cut uses clean deduped survivor set
```

Normal pipeline mode:

```text
clean-only shard jobs are not launched before final-local mode
```

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

---

# Superseding clarification: Stream4 CUB/fixed-temp sort/reduce dedup

```text
approval_id = supersede_stream4_custom_table_with_cub_sort_reduce_2026_05_23

Stream 4 may implement the shard-local:
    threshold + compact + sort + dedup

as:
    threshold + compact + CUB/fixed-temp Hash128 sort/reduce + clean compact

Specialization:
    key   = Hash128
    value = CandidateMeta
    input = survivor_shard[shard][0 : clean_count[shard] + dirty_count[shard]]
    capacity = SHARD_CAPACITY_CANDIDATES
    sort scratch slots = STREAM4_ACTIVE_SORT_SLOTS

Hard requirements:
    no CPU count/read/scheduling in Stream 4 hot path
    no Python data-plane
    no dynamic hot-path allocation
    no top-k shard cap
    fixed buffer capacity = SHARD_CAPACITY_CANDIDATES
    keep best CandidateMeta by score_key
    tie-break deterministic by parent_idx then route_packed
    fixed pre-start CUB temp storage
    fixed pre-start score histogram state per shard
    no atomics outside solved path

Output contract remains:
    survivor_shard[shard][0 : new_clean_count] = clean deduped survivors
    clean_count[shard] = new_clean_count
    dirty_count[shard] = 0
    processing_flag[shard] = false
    shard_score_hist_active[shard] points to a complete histogram for clean survivors

This clarification changes implementation mechanics only.
Stream semantics remain threshold + dedup by Hash128 + best CandidateMeta.
```

---

# Superseding clarification: resident shard capacity is independent from Stream4 batch

```text
approval_id = resident_shard_capacity_decoupled_from_stream4_batch_2026_05_24

All STORAGE_SHARD_COUNT physical shard buffers are resident GPU memory for the full depth.
STORAGE_SHARD_COUNT = SHARD_COUNT * SHARD_BUFFER_COUNT.
SHARD_BUFFER_COUNT is always 2; every logical shard always has resident A/B physical shard buffers.
SHARD_CAPACITY_CANDIDATES is the physical capacity of each resident physical shard buffer.
STREAM4_BATCH_CANDIDATES is the dirty-count launch threshold and tuning knob, not the shard capacity.

SHARD_CAPACITY_CANDIDATES derives from:
    LOGICAL_SHARD_SIZE = ceil(N_LOCAL / SHARD_COUNT)
    SHARD_CAPACITY_SCALE_PPM reserve multiplier
    STREAM4_BATCH_ALIGNMENT alignment

Stream 3 writes into:
    survivor_shard[physical_shard][clean_count[physical_shard] + dirty_count[physical_shard]]
when processing_flag[physical_shard] == false and physical shard capacity has free slots.

Stream 3 raises fatal overflow when:
    no non-processing A/B physical shard for logical_shard has enough free slots.

Stream 3 pre-launch writable-buffer backpressure:
    allowed only when WORLD_SIZE == 1
    forbidden when WORLD_SIZE > 1
    multi-rank Stream 3 must launch and expose capacity defects through fatal overflow

Stream 4 ready queue invariant:
    if any physical shard for logical_shard has processing_flag == true:
        no other physical shard for logical_shard may be launched
    otherwise:
        at most one dirty-ready physical shard for logical_shard may be launched
        stream3_write_buffer_index[logical_shard] points to a non-processing sibling when possible

GLOBAL_SPILL_CAPACITY:
    legacy single-buffer overflow capacity
    unreachable under required SHARD_BUFFER_COUNT == 2 contract
    expected production value is 0

The old capacity equation:
    survivor_shard[SHARD_COUNT][2 * STREAM4_BATCH_CANDIDATES]
is invalid for production config search.
```
