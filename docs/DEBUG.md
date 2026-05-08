# Отладка и оптимизация CayleyBeam100H100

## Верификация архитектуры

### Проверка Stream 1 (инференс)

**Проблема**: Score_ring не заполняется

```bash
# 1. Проверить, что backend запускается
engine = BeamEngine(cfg, buffers, backend="dummy")
engine.enable_debug(verbose=True)

# 2. В логе должно быть: "NEXT_POOL=" и счётчики
engine.step(histogram_period_micro=1)
engine.sync_and_log_step(0)
```

**Что проверить**:
- ✅ Лог: `[Step 0] Counters: NEXT_POOL=...`
- ✅ `NEXT_POOL > 0` (есть новые кандидаты)
- ✅ Нет ошибок CUDA

### Проверка Stream 2 (обработка)

**Метрики в counters**:
- `COUNTER_LOCAL_INSERTED` — кандидаты вставлены в локальную hash table
- `COUNTER_REMOTE_PACKED` — кандидаты упаковано для отправки
- `COUNTER_BUCKET_OVERFLOW` — переполнение fixed-size bucket

```python
# После engine.step()
counters_cpu = engine.counters_.cpu()
print(f"Local: {counters_cpu[1]}, Remote: {counters_cpu[3]}, Overflow: {counters_cpu[4]}")
```

**Если `COUNTER_BUCKET_OVERFLOW > 0`**:
- Bucket слишком мал
- Решение: увеличить `bucket_cap_per_peer` в конфиге

**Если `COUNTER_HASH_OVERFLOW > 0`**:
- Hash table переполнена или probe_limit исчерпан
- Решение: увеличить `hash_load_factor` или уменьшить `probe_limit`

### Проверка Stream 3 (сеть)

**Single-GPU**: Stream 3 пропускает NCCL, копирует гистограмму через cudaMemcpy

**Multi-GPU**: Проверить NCCL инициализацию

```python
import torch.distributed as dist
from beam_engine_ext import get_nccl_unique_id

# На rank 0
if dist.get_rank() == 0:
    unique_id_bytes = get_nccl_unique_id()
    dist.broadcast_object_list([unique_id_bytes], src=0)
else:
    unique_id_list = [None]
    dist.broadcast_object_list(unique_id_list, src=0)
    unique_id_bytes = unique_id_list[0]

# Инициализировать на всех ranks
engine.init_nccl(unique_id_bytes)
print(f"NCCL инициализирован на rank {dist.get_rank()}")
```

**Если hang при NCCL**:
- Проверить переменные окружения:
  ```bash
  echo $NCCL_IB_DISABLE
  echo $NCCL_NET_GDR_LEVEL
  ```
- Убедиться, что все ranks дошли до `init_nccl()`
- Проверить, что все ranks видят одну и ту же NCCL ID

---

## Оптимизация памяти

### Вычисление размеров

```python
from beam_engine import derive_sizes

cfg = make_default_config()
cfg["world_size"] = 100
cfg["global_beam_width"] = 2**30

sizes = derive_sizes(cfg)
print(f"""
N_LOCAL: {sizes['n_local'] / 1e6:.1f}M
K_KEEP:  {sizes['k_keep'] / 1e6:.1f}M
K_WORK:  {sizes['k_work'] / 1e6:.1f}M
HASH_CAP: {sizes['hash_capacity'] / 1e6:.1f}M

Примерный расход памяти на 1 GPU:
  beam_current:        {sizes['n_local'] * 120 / 1e9:.1f} GB
  next_state_pool:     {sizes['k_work'] * 120 / 1e9:.1f} GB
  hash_table:          {sizes['hash_capacity'] * 32 / 1e9:.1f} GB
  send_buckets:        {sizes['send_recv_records'] * 160 / 1e9:.1f} GB
  recv_buckets:        {sizes['send_recv_records'] * 160 / 1e9:.1f} GB
  local_hist:          {65536 * 4 / 1e9:.2f} GB
  Итого:               ~{(sizes['n_local'] * 120 + sizes['k_work'] * (120 + 32) + sizes['send_recv_records'] * 160 * 2 + 65536 * 4 * 2) / 1e9:.1f} GB
""")
```

### Out-of-memory?

**Решение 1**: Уменьшить batch

```python
cfg["b_micro"] = 65536  # вместо 131072
cfg["global_beam_width"] = 2**29  # вместо 2**30
```

**Решение 2**: Уменьшить ring depth

```python
cfg["score_ring_depth"] = 32  # вместо 64
cfg["net_ring_depth"] = 2    # вместо 3
```

**Решение 3**: Уменьшить bucket capacity

```python
cfg["bucket_cap_per_peer"] = 1048576  # вместо автоматического
```

---

## Производительность

### Профилирование CUDA

```python
import torch

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for step in range(10):
    engine.step()
end.record()

torch.cuda.synchronize()
ms_per_step = start.elapsed_time(end) / 10
print(f"Step time: {ms_per_step:.1f} ms")
```

### Ожидаемые время для production

| Конфигурация | Stream 1 | Stream 2 | Stream 3 | Итого |
|---|---|---|---|---|
| Single H100 | 50 ms | 100 ms | N/A | 150 ms |
| 8 H100 | 50 ms | 100 ms | 80 ms | 230 ms |
| 100 H100 | 50 ms | 100 ms | 120 ms | 270 ms |

**Профилирование по потокам**:

```python
# Синхронизировать после каждого потока для измерения
engine.clear_runtime_state()

# Stream 1
t0 = time.time()
engine.step()  # только инференс
torch.cuda.synchronize()
t1 = time.time()
print(f"Stream 1: {(t1-t0)*1000:.1f} ms")
```

---

## Отладка NCCL

### Проверка NCCL build

```bash
python -c "import torch; print(torch.cuda.nccl.version())"
# Output: (2, 16, 0) или выше
```

### Включить NCCL логирование

```bash
export NCCL_DEBUG=WARN    # Минимум
export NCCL_DEBUG=TRACE   # Максимум (очень много)
export NCCL_DEBUG_SUBSYS=INIT,COLL  # Только инициализация и collective ops

# Запустить
torchrun --nproc_per_node=2 your_script.py 2>&1 | tee nccl.log
```

### Типичные ошибки NCCL

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `ncclUnhandledCudaError` | CUDA kernel ошибка перед NCCL | Проверить CUDA_CHECK перед NCCL calls |
| `ncclInvalidUsage` | ncclGroupStart без ncclGroupEnd | Проверить паренность |
| `ncclInternalError` | Размер буфера неверный | Гарантировать, что все ranks отправляют одинаковый размер |
| Hang на 30 сек | Deadlock между ranks | Убедиться, что все ranks доходят до NCCL call |

### Мониторинг NCCL BW

```python
# Baseline test: какая максимальная BW на InfiniBand?
import torch

def measure_nccl_bandwidth():
    size = 1000000000  # 1 GB
    x = torch.randn(size, dtype=torch.uint8, device="cuda")
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(10):
        x_sum = torch.zeros_like(x)
        torch.distributed.all_reduce(x_sum)  # nccl.allreduce внутри
    end.record()
    
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 10
    gbps = (size * 1e-9 / (ms * 1e-3))
    print(f"NCCL BW: {gbps:.1f} GB/s (Expected: 1300+ on InfiniBand 4x)")
```

---

## Отладка Transformer Engine

### Проверка TE установки

```bash
python -c "import transformer_engine; print(transformer_engine.__version__)"
# Output: 1.2.0 или выше

python -c "import transformer_engine; print(transformer_engine.CUDA_AVAILABLE)"
# Output: True
```

### Тест TE MLP

```python
import torch
from transformer_engine.pytorch import TransformerLayer

# Создать простой TE layer
te_layer = TransformerLayer(
    hidden_size=512,
    ffn_hidden_size=2048,
    num_attention_heads=8,
    dropout=0.1,
    attention_dropout=0.1,
    ffn_dropout=0.1,
    activation='gelu',
    dtype=torch.float8,
)

# Инференс
x = torch.randn(16, 512, device="cuda")
with torch.no_grad():
    y = te_layer(x)
print(f"TE MLP output shape: {y.shape}")
print(f"Output dtype: {y.dtype}")
```

### Интегрировать TE в TEInferenceBackend

```cpp
// В beam_engine.cpp, класс TEInferenceBackend::forward()
// 1. Преобразовать beam_current в query
// 2. Запустить TE forward через C++ API
// 3. Результат -> score_ring

// Шаблон:
auto q_scores = te_model->forward(
    beam_current.slice(0, start_state, start_state + micro_size),
    te::Config{.dtype = te::DType::Float8, .stream = stream}
);
// score_ring[slot] = q_scores;
```

---

## Отладка CUB (когда будет интегрирован)

### Проверка CUB headers

```bash
ls /usr/local/cuda/include/cub/cub.cuh
# Должен существовать
```

### Тест CUB сортировки

```cuda
#include <cub/cub.cuh>
#include <cuda_runtime.h>

int main() {
    const int N = 1000000;
    uint32_t *d_keys, *d_values, *d_keys_out, *d_values_out;
    
    cudaMalloc(&d_keys, N * sizeof(uint32_t));
    cudaMalloc(&d_values, N * sizeof(uint32_t));
    cudaMalloc(&d_keys_out, N * sizeof(uint32_t));
    cudaMalloc(&d_values_out, N * sizeof(uint32_t));
    
    void *d_temp_storage = nullptr;
    size_t temp_bytes = 0;
    
    // Расчет temp storage
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        d_keys, d_keys_out,
        d_values, d_values_out,
        N
    );
    
    cudaMalloc(&d_temp_storage, temp_bytes);
    
    // Настоящая сортировка
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        d_keys, d_keys_out,
        d_values, d_values_out,
        N
    );
    
    cudaDeviceSynchronize();
    printf("CUB sort OK\n");
}
```

---

## Санитайз и верификация

### Проверка консистентности hash table

```python
# После engine.step()

# Все active_flags[i] == 1 должны иметь valid entries в hash table
# Все next_meta[i] должны иметь корректный hash/fingerprint

def verify_hash_consistency(engine, cfg):
    active = engine.active_flags_.cpu().numpy()
    meta = engine.next_meta_.cpu()
    
    num_active = active.sum()
    print(f"Active candidates: {num_active}")
    
    # Проверить, что no duplicates
    hashes = meta.view(np.int64)['hash'][:num_active]
    unique_hashes = len(np.unique(hashes))
    print(f"Unique hashes: {unique_hashes} / {num_active}")
    
    if unique_hashes < num_active * 0.95:
        print("⚠️  WARNING: Possible hash table collisions!")
```

### Проверка threshold корректности

```python
# Проверить, что global_hist суммируется правильно
def verify_histogram(engine, cfg):
    hist = engine.global_hist_.cpu()
    total = hist.sum()
    print(f"Total in histogram: {total}")
    print(f"Expected: ~{cfg['global_beam_width']}")
    
    # Проверить monotonicity
    cumsum = np.cumsum(hist[::-1])
    if cumsum[-1] != total:
        print("⚠️  WARNING: Histogram sum mismatch!")
```

---

## Типичный рабочий процесс отладки

1. **Начать с single GPU**:
   ```python
   cfg = make_default_config()
   cfg["world_size"] = 1
   cfg["global_beam_width"] = 2**20  # Маленький для отладки
   engine.enable_debug(verbose=True)
   engine.step()
   ```

2. **Проверить логи**:
   - Должны быть счётчики: `NEXT_POOL > 0`, `LOCAL_INSERT > 0`
   - Нет CUDA ошибок

3. **Постепенно увеличивать**:
   ```python
   cfg["global_beam_width"] = 2**25
   cfg["b_micro"] = 131072
   ```

4. **Перейти на multi-GPU**:
   ```bash
   torchrun --nproc_per_node=2 debug_script.py
   ```

5. **Включить NCCL логирование**:
   ```bash
   export NCCL_DEBUG=WARN
   ```

6. **Профилировать**:
   - Время per stream
   - NCCL bandwidth
   - GPU memory usage

7. **Оптимизировать**:
   - Настроить hyperparameters
   - Добавить CUB для top-K
   - Добавить TE для FP8

---

## Контакты и вспомогательные материалы

- **NCCL**: https://docs.nvidia.com/deeplearning/nccl/
- **Transformer Engine**: https://github.com/NVIDIA/TransformerEngine
- **CUB**: https://github.com/NVIDIA/cub
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

