# Применяемые фреймворки

## Таблица применения

| Компонент | Фреймворк | Зачем | Версия | Статус |
|-----------|-----------|-------|--------|--------|
| Оркестрация | LibTorch | Tensors, GPU alloc | PyTorch 2.0+ | ✅ Интегрирован |
| Инференс | TorchScript ensemble | arbitrary scorer / MLP 1024-256 | PyTorch JIT | ✅ Интегрирован |
| Инференс | TE (планируется) | FP8/Q-MLP on H100 | TE 1.2+ | 🔧 Заглушка |
| Связь | NCCL | All-to-all/reduce | 2.16+ | ✅ Интегрирован |
| Алгоритмы | CUB | Сортировка/редукция | CUDA 11.0+ | 📝 Планируется |
| Интерфейс | pybind11 | Python API | 2.6+ | ✅ Интегрирован |

---

## 1. LibTorch (PyTorch C++ API)

### Назначение
Получение torch.Tensor из Python, управление GPU памятью, синхронизация потоков.

### Используется в
- `beam_engine.cpp`: Приём всех буферов как torch::Tensor
- `BeamEngine` конструктор: Валидация и сохранение tensors
- `check_cuda_tensor()`: Проверка device и contiguity

### Интеграция

```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Проверка и получение указателя
check_cuda_tensor(beam_current, "beam_current");
const uint8_t* ptr = beam_current.data_ptr<uint8_t>();

// Синхронизация
CUDA_CHECK(cudaStreamSynchronize(stream_infer_));

// Логирование на CPU
auto c_cpu = counters.cpu();
int32_t* cpu_ptr = c_cpu.data_ptr<int32_t>();
```

### Конфигурация в setup.py

```python
from torch.utils.cpp_extension import CUDAExtension, build_ext

ext_modules = [
    CUDAExtension(
        'beam_engine_ext',
        sources=['beam_engine.cpp', 'beam_kernels.cu'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
        }
    )
]
```

### Типичные ошибки

1. **Tensor не на CUDA**: `check_cuda_tensor()` выбросит ошибку
2. **Non-contiguous**: Нужен `tensor.contiguous()`
3. **Неправильный dtype**: `data_ptr<float>()` на int32 tensor → UB

### Расширения

- **autograd**: Не используется (инференс, no gradients)
- **JIT компиляция**: Не нужна (все в C++)
- **Distributed**: Используется только для NCCL init, не для data

---

## 1.1 TorchScript ensemble

### Назначение
Приём произвольной PyTorch-нейронки через TorchScript-export. C++ загружает несколько копий модели и ротирует microbatch-и между inference lanes.

### Контракт

```text
input  = uint8 CUDA tensor [B, 120]
output = int16 или float32 CUDA tensor [B, 24]
```

### Конфиг

```bash
export INFERENCE_BACKEND=torchscript_ensemble
export INFERENCE_PARALLELISM=2
export TORCHSCRIPT_SCORER_PATHS=runtime/scorers/mlp_copy00.ts:runtime/scorers/mlp_copy01.ts
```

### Экспорт MLP 120→1024→256→24

```bash
python scripts/export_mlp_scorer.py --copies 2 --hidden 1024,256
```

### CUDA Graph
TorchScript forward участвует в CUDA Graph после warmup. Ring reuse защищён `score_consumed[slot]`, поэтому graph capture не ломает асинхронную работу Stream1/Stream2/Stream3.

---

## 2. NCCL (NVIDIA Collective Communications)

### Назначение
Эффективная GPU↔GPU коммуникация через InfiniBand/RDMA. Нет CPU overhead.

### Используется в

```cpp
// Инициализация
ncclComm_t comm;
ncclUniqueId id;
ncclGetUniqueId(&id);
ncclCommInitRank(&comm, world_size, id, rank);

// All-to-all
ncclGroupStart();
for (int peer = 0; peer < world_size; ++peer) {
    ncclSend(send_ptr, count, ncclUint8, peer, comm, stream);
    ncclRecv(recv_ptr, count, ncclUint8, peer, comm, stream);
}
ncclGroupEnd();

// All-reduce
ncclAllReduce(local_hist, global_hist, 65536, ncclInt32, ncclSum, comm, stream);
```

### Конфигурация NCCL

#### Окружение для InfiniBand

```bash
# Используй InfiniBand вместо Ethernet
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB    # GPU Direct RDMA: Policy High Bandwidth
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

# На кластере с несколькими сетями:
export NCCL_IB_GID_INDEX=3       # GID для RoCE v2
export NCCL_DEBUG=INFO           # Логирование
```

#### Окружение для RDMA optimizations

```bash
# GPU Direct RDMA (самое быстрое)
export NCCL_NET_GDR_LEVEL=PHB    # PHB = Policy High Bandwidth
export NCCL_P2P_DISABLE=0        # Разрешить P2P если есть

# Если проблемы:
export NCCL_NET_GDR_LEVEL=SYS    # SYS = System
export NCCL_P2P_LEVEL=SYS        # Отключить P2P
```

#### Build time: linker flags

```bash
# В beam_engine.py
extra_ldflags = ["-lnccl"]

# На системе нужны:
# - /usr/include/nccl.h
# - /usr/lib/libnccl.so (или /usr/local/cuda/lib/libnccl.so)
# 
# Если custom install:
# extra_ldflags = ["-L/opt/nccl/lib", "-lnccl"]
# extra_cflags = ["-I/opt/nccl/include"]
```

### Типичные проблемы и решения

| Проблема | Причина | Решение |
|----------|---------|---------|
| `ncclUnhandledCudaError` | CUDA error в NCCL kernel | `CUDA_CHECK(cudaGetLastError())` перед NCCL |
| `ncclInvalidUsage` | Group не закрыт | `ncclGroupStart/End` парой |
| `ncclInternalError` | Неверные размеры | Проверить `count * world_size` не превышает bucket |
| Low BW | Нет InfiniBand | Проверить `export NCCL_IB_DISABLE=0` |
| Hang | Разные размеры на ranks | Синхронизировать конфиг через torch.distributed |

### Измерение NCCL BW

```python
import torch
import nccl
import time

def benchmark_nccl():
    comm = nccl.Comm([torch.device("cuda")])
    x = torch.randn(1000000, device="cuda", dtype=torch.uint8)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        comm.allGather([x], [x])
    torch.cuda.synchronize()
    t1 = time.time()
    
    gbps = (x.numel() * 100 * 1e-9) / (t1 - t0)
    print(f"BW: {gbps:.1f} GB/s")
```

---

## 3. pybind11

### Назначение
Экспорт C++ класса `BeamEngine` в Python как чистый Python объект с методами.

### Используется в

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Экспорт класса
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_nccl_unique_id", &get_nccl_unique_id);
    m.def("derive_sizes", &derive_sizes);
    
    py::class_<BeamEngine>(m, "BeamEngine")
        .def(py::init<py::dict, py::dict, std::string>())
        .def("init_nccl", &BeamEngine::init_nccl)
        .def("step", &BeamEngine::step)
        .def("enable_debug", &BeamEngine::enable_debug);
}
```

### Использование из Python

```python
from beam_engine_ext import BeamEngine

# Создание
engine = BeamEngine(cfg_dict, buffers_dict, backend="dummy")

# Методы доступны как обычные Python методы
engine.init_nccl(unique_id_bytes)
engine.step(histogram_period_micro=8)
engine.enable_debug(verbose=True)
```

### Типичные интеграции

| Паттерн | Код |
|---------|-----|
| Параметры | `py::arg("name") = default` |
| Return dict | `return py::dict(); d["key"] = value;` |
| Return bytes | `return py::bytes(ptr, len);` |
| Исключения | `throw std::runtime_error("message");` |

### Ошибки при компиляции

```
error: 'py' was not declared
-> #include <pybind11/pybind11.h>

error: cannot convert to 'PyObject*'
-> Используй py::dict вместо обычного dict
```

---

## 4. CUB (CUDA Unbound)

### Назначение
Оптимальная сортировка и редукция на GPU без ручных kernel написаний.

### Планируемое использование

#### Сортировка для top-K

```cuda
#include <cub/cub.cuh>

// Sortdefaultгу по descending score
cub::DoubleBuffer<uint32_t> d_keys(scores_in, scores_out);
cub::DoubleBuffer<uint32_t> d_values(indices_in, indices_out);

void* temp_storage = nullptr;
size_t temp_bytes = 0;

cub::DeviceRadixSort::SortDescendingByKey(
    temp_storage, temp_bytes,
    d_keys, d_values,
    k_work
);

cudaMalloc(&temp_storage, temp_bytes);
cub::DeviceRadixSort::SortDescendingByKey(
    temp_storage, temp_bytes,
    d_keys, d_values,
    k_work,
    0, 32,  // Начать сортировку со всех 32 бит
    stream  // stream
);
cudaFree(temp_storage);
```

#### Редукция для глобальной гистограммы

```cuda
// Вместо ncclAllReduce на CPU можно использовать локально
// перед передачей:
cub::DeviceReduce::Sum(
    temp_storage, temp_bytes,
    local_hist, global_hist_partial,
    65536,
    stream
);
```

### Интеграция в build

```python
# В beam_engine.py
extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-I/usr/local/cuda/include",  # CUB headers
]
```

### Когда CUB нужен

1. **Сейчас** используется threshold-based filter: `if (score <= threshold) discard`
   - ⚠️ Проблема: если нет точного threshold, выборка может быть неполной

2. **Лучше**: Top-K extraction через сортировку
   - Гарантирует ровно k_keep самых лучших
   - CUB::DeviceRadixSort имеет 300 ГГц пиковую пропускную способность

3. **Место интеграции**: `kernel_prune_by_threshold()` → `kernel_prune_by_topk()`

### Пример: top-K для 10M элементов

```
Input: next_meta[0:k_work] with scores
CUB DeviceRadixSort: Sort по score (descending)
Output: top k_keep элементов отсортированы

Time: ~10ms на GPU для 10M элементов
Memory: ~500MB temp storage (3x input)
```

---

## 5. Transformer Engine (TE)

### Назначение
FP8 precision MLP + Q-head для ultra-fast inference при малой потере качества.

### Планируемая интеграция

#### Setup: установка TE

```bash
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
pip install .

# Или через conda:
conda install -c nvidia transformer-engine
```

#### Header integration

```cpp
// В beam_engine.cpp добавить:
#include <transformer_engine/transformer_engine.h>

// Или если нестандартный путь:
#include "../../../opt/transformer_engine/include/transformer_engine.h"
```

#### Реализация TEInferenceBackend

```cpp
struct TEInferenceBackend final : public InferenceBackend {
    std::unique_ptr<te::Transformer> model_;
    
    TEInferenceBackend(const std::string& model_path) {
        // Загрузить модель TE
        model_ = std::make_unique<te::Transformer>(model_path);
    }
    
    void forward(const torch::Tensor& beam_current,
                 torch::Tensor& score_ring,
                 int slot,
                 int64_t start_state,
                 int micro_size,
                 const EngineConfig& cfg,
                 cudaStream_t stream) override {
        
        // 1. Преобразовать beam_current в query embeddings
        auto query_batch = beam_current.slice(0, start_state, start_state + micro_size);
        // shape: [micro_size, state_size_bytes] -> проекция -> [micro_size, hidden_dim]
        
        // 2. Запустить TE forward (FP8)
        auto q_scores = model_->forward(
            query_batch,
            /* config */ te::Config{.dtype=te::DType::Float8},
            /* stream */ stream
        );
        // Выход: [micro_size, fanout]
        
        // 3. Поместить в score_ring[slot]
        auto slot_scores = score_ring.select(0, slot);
        // shape [fanout*b_micro] -> распаковать в [fanout, micro_size]
        // и скопировать q_scores
        
        // Опционально: fp16 quantize если нужно
        // slot_scores = q_scores.to(torch::kHalf);
    }
};
```

#### Build flags

```python
# В beam_engine.py build_extension():
extra_ldflags = [
    "-lnccl",
    "-ltransformer_engine",  # <-- Добавить
]

# Если custom install:
extra_cflags.append("-I/opt/transformer_engine/include")
extra_ldflags.append("-L/opt/transformer_engine/lib")
```

#### Использование из Python

```python
from beam_engine_ext import BeamEngine

# Создать с TE backend
engine = BeamEngine(cfg, buffers, backend="te")
# или через env:
# INFERENCE_BACKEND=te python script.py
```

### TE FP8 механика

```
Нормальный float32:
  score = MLP(q_proj(x))  # float32 all the way
  Cost: 16 байт на input, высокая память

TE FP8:
  1. Динамическое квантование: float32 -> FP8 (8 бит)
  2. MLP compute в FP8: 8x меньше памяти
  3. Результат: float8 -> float16 (для score_ring)
  Вывод качества: ≤ 0.5% при правильном scale factor

  Cost: 1 байт на input, 8x меньше памяти, ~2x быстрее
```

### Типичные параметры TE

| Параметр | Значение | Назначение |
|----------|----------|-----------|
| `reduce_type` | `ReduceType::Tensor` | FP8 per-token |
| `fp8_format` | `Format::Hybrid` | Dynamic range |
| `scale_method` | `TE_DEFAULT` | Auto-tuning |

---

## Быстрая интеграция: checklist

### ✅ LibTorch
- [x] Включены заголовки
- [x] torch::Tensor параметры
- [x] data_ptr<T>() для GPU memory
- [x] Валидация tensors

### ✅ NCCL
- [x] ncclComm_t инициализирован
- [x] Grouped send/recv
- [x] All-reduce для histogram
- [x] Синхронизация через events

### ✅ pybind11
- [x] PYBIND11_MODULE определён
- [x] py::dict для config
- [x] py::bytes для NCCL ID
- [x] Методы экспортированы

### 🔧 CUB (планируется)
- [ ] Включить #include <cub/cub.cuh>
- [ ] DeviceRadixSort для top-K
- [ ] DeviceReduce для локальной гистограммы
- [ ] Temp storage management

### 🔧 Transformer Engine (планируется)
- [ ] Установить libtransformer_engine
- [ ] Реализовать TEInferenceBackend::forward()
- [ ] Линкер flags (-ltransformer_engine)
- [ ] Тесты инференса

---

## Отладка фреймворков

### LibTorch

```python
import torch
print(torch.__version__)  # Проверить версию
print(torch.cuda.is_available())  # GPU доступна?
t = torch.randn(10, device="cuda")
print(t.is_contiguous())  # Контигуальна?
```

### NCCL

```bash
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
# Запустить программу -> подробные логи NCCL

# Быстрый тест:
python -c "import nccl; print('NCCL OK')"
```

### pybind11

```bash
# При compile errors
g++ -E -dM my_file.cpp | grep pybind  # Макросы

# При import errors в Python
python -c "from beam_engine_ext import BeamEngine; print(BeamEngine)"
```

### CUB

```cuda
// Тест компиляции CUB
#include <cub/cub.cuh>
// Если ошибка: CUB не найден, добавить -I/usr/local/cuda/include
```

### TE

```bash
# Проверить установку
ls /opt/transformer_engine/include/transformer_engine.h
python -c "import transformer_engine; print(transformer_engine.__version__)"
```

---

## Резюме

| Фреймворк | Статус | Что даёт | Как использовать |
|-----------|--------|----------|-----------------|
| **LibTorch** | ✅ Ready | GPU memory, tensors | `torch::Tensor`, `data_ptr<T>()` |
| **NCCL** | ✅ Ready | Fast GPU↔GPU comms | `ncclSend/Recv`, `ncclAllReduce` |
| **pybind11** | ✅ Ready | Python API | `py::class_<T>`, методы |
| **CUB** | 📋 Planned | Fast sort/reduce | `DeviceRadixSort`, `DeviceReduce` |
| **TE** | 📋 Planned | FP8 inference | `TEInferenceBackend::forward()` |

Всё готово для масштабирования на 100×H100!
