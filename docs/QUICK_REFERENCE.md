# Быстрый справочник

Шпаргалка для часто используемых команд и проверок.

---

## 🚀 Быстрый старт

### Однопроцессная отладка

```python
from beam_engine import make_default_config, build_extension
import torch

# 1. Собрать extension
ext = build_extension(verbose=True)

# 2. Создать конфиг
cfg = make_default_config()
cfg["global_beam_width"] = 2**20  # Маленький для отладки

# 3. Выделить буферы (через Python)
# ... allocate torch tensors ...

# 4. Создать engine
from beam_engine_ext import BeamEngine
engine = BeamEngine(cfg, buffers_dict, backend="dummy")

# 5. Включить отладку
engine.enable_debug(verbose=True, print_counters=True)

# 6. Запустить
for step in range(10):
    engine.step()
    engine.sync_and_log_step(step)
```

---

## 📊 Основные счётчики

```
COUNTER_NEXT_POOL_SIZE      0    Всего слотов выдано
COUNTER_LOCAL_INSERTED      1    Новых локальных вставок
COUNTER_LOCAL_DUPLICATE     2    Дублей
COUNTER_REMOTE_PACKED       3    Упаковано в сеть
COUNTER_BUCKET_OVERFLOW     4    ⚠️ Переполнение bucket
COUNTER_HASH_OVERFLOW       5    ⚠️ Переполнение hash
COUNTER_PRUNED              6    Удалено при prune
```

### Интерпретация

```
✅ GOOD:
  LOCAL_INSERTED > 0
  REMOTE_PACKED > 0
  BUCKET_OVERFLOW == 0
  HASH_OVERFLOW == 0

⚠️ WARNING:
  BUCKET_OVERFLOW > 0   → Увеличить bucket_cap_per_peer
  HASH_OVERFLOW > 0     → Увеличить hash_load_factor
  LOCAL_INSERTED == 0   → Threshold слишком строгий?
```

---

## 🔧 Частые правки

### Из памяти на GPU

```python
# Уменьшить batch
cfg["b_micro"] = 65536  # вместо 131072

# Уменьшить глобальный beam
cfg["global_beam_width"] = 2**29  # вместо 2**30

# Уменьшить ring depth
cfg["score_ring_depth"] = 32  # вместо 64
cfg["net_ring_depth"] = 2    # вместо 3
```

### Неправильный threshold

```python
# Если threshold никогда не valid:
cfg["gamma"] = 1.0  # Близче к n_local

# Если threshold слишком строгий (много discard):
cfg["beta"] = 1.0   # Меньше работающего пула
```

### Медленный NCCL

```bash
# Проверить InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN

# Профилировать
torchrun --nproc_per_node=2 benchmark_nccl.py
```

---

## 📚 Документы по теме

| Документ | Для кого | Читать если... |
|----------|----------|----------------|
| [README_RU.md](README_RU.md) | Все | Новичок в проекте |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитекторы | Нужно понять дизайн |
| [FRAMEWORKS.md](FRAMEWORKS.md) | Разработчики | Интегрируешь TE/CUB |
| [DEBUG.md](DEBUG.md) | Отладчики | Что-то не работает |
| [CHECKLIST.md](CHECKLIST.md) | QA | Готов к production? |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Все | Что нового в v1.0? |

---

## 🐛 Быстрая отладка

### "Не собирается"

```bash
# Проверить CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Проверить pybind11
python -c "import pybind11; print(pybind11.get_include())"

# Полный лог сборки
cd beam_engine_ext && cat build/temp.*/beam_engine.cpp.o
```

### "Не работает на multi-GPU"

```python
import torch.distributed as dist
from beam_engine_ext import get_nccl_unique_id

# 1. Инициализировать torch.distributed
dist.init_process_group("nccl")

# 2. Создать NCCL unique ID на rank 0
if dist.get_rank() == 0:
    uid = get_nccl_unique_id()
    dist.broadcast_object_list([uid], src=0)
else:
    uid = [None]
    dist.broadcast_object_list(uid, src=0)
    uid = uid[0]

# 3. Инициализировать engine на все ranks
engine.init_nccl(uid)
```

### "NCCL hang"

```bash
# 1. Проверить, что все ranks на месте
export NCCL_DEBUG=INFO
# Должны быть логи от каждого ранга

# 2. Проверить env vars
echo $NCCL_IB_DISABLE
echo $NCCL_NET_GDR_LEVEL

# 3. Убедиться что весь мир инициализирован
# (torch.distributed + engine.init_nccl)
```

### "Низкий throughput"

```python
# Профилировать per-stream
import time
import torch

torch.cuda.synchronize()
t0 = time.time()

for step in range(100):
    engine.step()

torch.cuda.synchronize()
t_total = time.time() - t0
print(f"Avg: {t_total/100*1000:.1f} ms/step")
print(f"Throughput: {100/(t_total/100):.0f} steps/sec")
```

---

## 📋 Multi-GPU команды

### torchrun (PyTorch elastic)

```bash
torchrun --nproc_per_node=8 \
         --nnodes=$SLURM_NNODES \
         --node_rank=$SLURM_PROCID \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         your_script.py
```

### Slurm

```bash
srun -N $SLURM_NNODES \
     --tasks-per-node=1 \
     --gpus-per-task=8 \
     --job-name beam \
     python your_script.py
```

### NCCL Debug (Slurm)

```bash
srun -N 2 --ntasks=16 --gpus-per-task=1 \
  env NCCL_DEBUG=INFO \
      NCCL_IB_DISABLE=0 \
      NCCL_NET_GDR_LEVEL=PHB \
      python your_script.py 2>&1 | tee nccl.log
```

---

## 🎯 Метрики, которые нужно отслеживать

```
Step time (ms)
├─ Stream 1: 50-100 ms (инференс)
├─ Stream 2: 50-150 ms (обработка)
└─ Stream 3: 50-200 ms (сеть)

Памятью:
├─ GPU: < 80 GB (оставить маржин на H100)
└─ Total cluster: < 100*80 GB (для 100 GPU)

Счётчики:
├─ LOCAL_INSERTED: > 0
├─ REMOTE_PACKED: > 10% LOCAL (для world_size > 1)
├─ BUCKET_OVERFLOW: == 0
└─ HASH_OVERFLOW: == 0
```

---

## 💡 Советы оптимизации

### Если медленно

1. Профилировать: где время тратится?
   ```python
   # Запустить только Stream 1
   # Запустить только Stream 2
   # Запустить только Stream 3
   ```

2. Увеличить batch
   ```python
   cfg["b_micro"] *= 2
   ```

3. Уменьшить hash probe
   ```python
   cfg["probe_limit"] = 16  # вместо 32
   ```

### Если ошибки NCCL

1. Проверить InfiniBand
   ```bash
   ibstat
   ibnetdiscover
   ```

2. Проверить NCCL версию
   ```python
   import torch
   print(torch.cuda.nccl.version())
   ```

3. Проверить конфиг синхронизирован
   ```bash
   # Убедиться, что все ranks имеют одинаковый cfg dict
   ```

---

## ⚡ Production чеклист

Перед development → production:

- [ ] Код компилируется без warning
- [ ] Single GPU тест проходит (10+ шагов)
- [ ] Multi GPU тест проходит (8+ GPU)
- [ ] Нет CUDA ошибок: `CUDA_CHECK(cudaGetLastError())`
- [ ] Нет NCCL ошибок: `NCCL_CHECK(...)`
- [ ] Threshold стабилизируется после 10 шагов
- [ ] BUCKET_OVERFLOW == 0 (всегда)
- [ ] HASH_OVERFLOW == 0 (всегда)
- [ ] Throughput >= 1000 steps/hour на 1 GPU
- [ ] Память не превышает 80 GB на GPU

Всё ОК? **Go to production!**

---

## 📞 SOS: Что-то совсем не работает?

1. Прочитай ERROR сообщение полностью
2. Проверь [DEBUG.md](DEBUG.md) - скорее всего там ответ
3. Включи логирование:
   ```python
   engine.enable_debug(verbose=True)
   ```
4. Проверь счётчики:
   ```python
   print(engine.counters_.cpu())
   ```
5. Прочитай [CHECKLIST.md](CHECKLIST.md) - может что-то не инициализировано

---

## 🔗 Важные ссылки

- NCCL docs: https://docs.nvidia.com/deeplearning/nccl/
- TE docs: https://github.com/NVIDIA/TransformerEngine
- CUB docs: https://github.com/NVIDIA/cub
- PyTorch C++: https://pytorch.org/docs/stable/cpp_index.html
- pybind11: https://pybind11.readthedocs.io/

---

## 📝 Шаблоны кода

### Включить отладку

```python
engine = BeamEngine(cfg, buffers, backend="dummy")
engine.enable_debug(verbose=True, print_counters=True, log_period=8)

for step in range(100):
    engine.step(histogram_period_micro=8)
    engine.sync_and_log_step(step)
```

### Проверить счётчики

```python
counters = engine.counters_.cpu()
print(f"Pool: {counters[0]}")
print(f"Local insert: {counters[1]}")
print(f"Remote packed: {counters[3]}")
print(f"Overflow: {counters[4]}")  # Should be 0!
```

### Проверить гистограмму

```python
hist = engine.local_hist_.cpu()
total = hist.sum()
print(f"Total candidates: {total}")

# Проверить распределение scores
for score in [65535, 65000, 64000, 60000, 50000]:
    print(f"Score {score}: {hist[score]}")
```

### Multi-GPU init

```python
import torch.distributed as dist
from beam_engine_ext import BeamEngine, get_nccl_unique_id

dist.init_process_group("nccl")

if dist.get_rank() == 0:
    uid = get_nccl_unique_id()
    dist.broadcast_object_list([uid], src=0)
else:
    uid = [None]
    dist.broadcast_object_list(uid, src=0)
    uid = uid[0]

engine = BeamEngine(cfg, buffers, backend="dummy")
engine.init_nccl(uid)

for step in range(100):
    engine.step()
```

---

## 💾 Сохраняй знания

Если нашёл баг или оптимизацию:

1. Документируй в [DEBUG.md](DEBUG.md)
2. Добавь в [CHECKLIST.md](CHECKLIST.md)
3. Обнови [IMPROVEMENTS.md](IMPROVEMENTS.md)

Спасибо за улучшения! 🚀

