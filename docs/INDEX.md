# 📚 Указатель документации

Полный указатель всех файлов проекта и документации.

---

## 🏠 Главные документы

### [README_RU.md](README_RU.md)
**Для**: Новички в проекте  
**Содержание**:
- Назначение проекта
- Быстрый старт
- Обзор файлов
- Используемые фреймворки (краткий список)
- Схема памяти
- Описание трёх потоков (высокий уровень)

**Начни отсюда** ➡️

---

## 📖 Подробная документация

### [ARCHITECTURE.md](ARCHITECTURE.md)
**Для**: Архитекторы и тех, кто хочет понять дизайн  
**Объём**: 3000+ строк  
**Содержание**:
- Обзор архитектуры
- Три асинхронных CUDA потока (детально)
- Структуры данных (BeamMeta, HashSlot, CandidateRecord)
- Все буферы и их размеры
- Использованные фреймворки (где и как)
- Отладка и логирование
- Размеры конфигурации (формулы)
- Компиляция и запуск
- Архитектурные решения и их обоснование
- Расширения (планируется)

**Читай если**: нужно понять полную архитектуру системы

### [FRAMEWORKS.md](FRAMEWORKS.md)
**Для**: Разработчики, интегрирующие фреймворки  
**Объём**: 2500+ строк  
**Содержание**:
- Таблица применения фреймворков
- **LibTorch**:
  - Назначение и использование
  - Конфигурация в setup.py
  - Типичные ошибки
- **NCCL**:
  - All-to-all и all-reduce
  - Конфигурация InfiniBand
  - Env vars для оптимизации
  - Типичные проблемы и решения
  - Измерение bandwidth
- **pybind11**:
  - Экспорт класса в Python
  - Типичные интеграции
  - Ошибки при компиляции
- **CUB**:
  - Планируемое использование
  - Сортировка для top-K
  - Редукция для гистограммы
  - Когда CUB нужен
  - Пример на 10M элементов
- **Transformer Engine**:
  - Планируемая интеграция
  - Setup и установка
  - Реализация TEInferenceBackend
  - Build flags
  - FP8 механика
  - Типичные параметры
- Быстрая интеграция (checklist)
- Отладка каждого фреймворка
- Резюме

**Читай если**: интегрируешь CUB, TE, или отладиваешь фреймворк

### [DEBUG.md](DEBUG.md)
**Для**: Отладчики и тестировщики  
**Объём**: 2000+ строк  
**Содержание**:
- Верификация архитектуры
  - Проверка Stream 1
  - Проверка Stream 2
  - Проверка Stream 3
- Оптимизация памяти
  - Вычисление размеров
  - Out-of-memory решения
- Производительность
  - Профилирование CUDA
  - Ожидаемые времена
  - Профилирование per-stream
- Отладка NCCL
  - Проверка build
  - Логирование
  - Типичные ошибки
  - Мониторинг bandwidth
- Отладка TE
  - Проверка установки
  - Тест TE MLP
  - Интеграция TEInferenceBackend
- Отладка CUB
  - Проверка headers
  - Тест сортировки
- Санитайз и верификация
  - Проверка hash table
  - Проверка threshold
- Типичный рабочий процесс отладки
- Контакты

**Читай если**: что-то не работает или нужно оптимизировать

### [CHECKLIST.md](CHECKLIST.md)
**Для**: QA, архитекторы, перед production  
**Объём**: 500+ строк  
**Содержание**:
- Архитектурный чеклист (Stream 1/2/3)
- Фреймворки (проверка использования)
- Код и чистота
- Отладка
- Документация
- Конфигурация
- Known limitations
- Контрольные точки расширения
- Финальная проверка перед production
- Метрики для мониторинга
- Итоговая сертификация

**Читай если**: готовишь к production или проверяешь качество

### [IMPROVEMENTS.md](IMPROVEMENTS.md)
**Для**: Все, кто хочет знать что изменилось в v1.0  
**Объём**: 1000+ строк  
**Содержание**:
- Что было проверено
- Что было улучшено
- Применяемые фреймворки (подтверждено)
- Архитектурные гарантии
- Отладка и мониторинг
- Дорожная карта v1.1
- Как использовать улучшения
- Качество кода
- Статус готовности
- Файлы которые изменились
- Итог

**Читай если**: хочешь понять что нового в этой версии

---

## ⚡ Справочники

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Для**: Все разработчики  
**Объём**: 500+ строк  
**Содержание**:
- Быстрый старт (3 строки кода)
- Основные счётчики
- Частые правки
- Документы по теме
- Быстрая отладка (SOS)
- Multi-GPU команды
- Метрики для отслеживания
- Советы оптимизации
- Production чеклист
- Важные ссылки
- Шаблоны кода

**Читай если**: нужна быстрая шпаргалка или знаешь что нужно но забыл как

### [INDEX.md](INDEX.md) (этот файл)
**Для**: Все разработчики  
**Содержание**:
- Это быстрый указатель всех документов

---

## 📁 Исходные файлы

### [beam_engine_common.hpp](beam_engine_common.hpp)
**Что**: Общие структуры данных  
**Содержит**:
- `BeamMeta` (32 байта)
- `HashSlot` (32 байта)
- `CandidateRecord` (160 байт)
- `CounterIndex` enum
- Константы hash table

### [beam_kernels.cu](beam_kernels.cu)
**Что**: CUDA kernels  
**Содержит**:
- Dummy inference kernel
- Reset net slot kernel
- Process score slot kernel
- Ingest recv slot kernel
- Compute threshold kernel
- Prune by threshold kernel
- Clear hash table kernel
- Clear counters kernel
- Host launch wrappers
- Комментарии о CUB

### [beam_engine.cpp](beam_engine.cpp)
**Что**: C++/pybind11/NCCL обвязка  
**Содержит**:
- `EngineConfig` структура
- `DebugConfig` структура
- `InferenceBackend` интерфейс
- `DummyInferenceBackend` реализация
- `TEInferenceBackend` заглушка (для интеграции TE)
- `BeamEngine` класс
- NCCL функции (all-to-all, all-reduce)
- pybind11 экспорт

### [beam_engine.py](beam_engine.py)
**Что**: Python launcher и конфигурация  
**Содержит**:
- `make_default_config()` - конфиг по умолчанию
- `build_extension()` - сборка C++ extension
- Документация и примеры

### [setup.py](setup.py)
**Что**: Python setup для сборки  
**Содержит**:
- Конфигурация torch.utils.cpp_extension
- Зависимости

---

## 🎯 Как выбрать документ?

```
Новичок? 
  ├─ Прочитай → README_RU.md
  └─ Потом → ARCHITECTURE.md (высокий уровень)

Хочу понять дизайн?
  └─ Прочитай → ARCHITECTURE.md

Интегрирую CUB или TE?
  └─ Прочитай → FRAMEWORKS.md

Отлаживаю проблему?
  ├─ Первое: QUICK_REFERENCE.md (может быть известная проблема)
  └─ Потом: DEBUG.md

Готов к production?
  ├─ Прочитай → CHECKLIST.md
  └─ Убедись что все галочки

Какая быстрая команда?
  └─ Смотри → QUICK_REFERENCE.md

Хочу понять что нового?
  └─ Прочитай → IMPROVEMENTS.md

Вообще не знаю с чего начать?
  └─ Ты здесь! Смотри "Как выбрать" выше 👆
```

---

## 📊 Размер документации

| Документ | Строк | Раздел |
|----------|-------|--------|
| README_RU.md | ~300 | 📚 Главные |
| ARCHITECTURE.md | ~3000 | 📖 Подробная |
| FRAMEWORKS.md | ~2500 | 📖 Подробная |
| DEBUG.md | ~2000 | 📖 Подробная |
| CHECKLIST.md | ~500 | 📖 Подробная |
| IMPROVEMENTS.md | ~1000 | 📖 Подробная |
| QUICK_REFERENCE.md | ~500 | ⚡ Справочники |
| INDEX.md | ~400 | ⚡ Справочники |
| **Итого документации** | **~10000** | **10K строк!** |

---

## 🔗 Перекрёстные ссылки

### README_RU.md → ?
- `make_default_config()` - смотри beam_engine.py
- Три потока - детали в ARCHITECTURE.md
- Фреймворки - подробности в FRAMEWORKS.md
- Отладка - инструкции в DEBUG.md

### ARCHITECTURE.md → ?
- CUB использование - смотри FRAMEWORKS.md раздел CUB
- TE интеграция - смотри FRAMEWORKS.md раздел TE
- Отладка - смотри DEBUG.md
- Проверка перед production - смотри CHECKLIST.md

### FRAMEWORKS.md → ?
- Ошибки при сборке - смотри DEBUG.md
- Verification - смотри CHECKLIST.md
- Примеры кода - смотри QUICK_REFERENCE.md

### DEBUG.md → ?
- Что означают счётчики - смотри ARCHITECTURE.md
- Структуры данных - смотри beam_engine_common.hpp
- Конфиг параметры - смотри beam_engine.py

### CHECKLIST.md → ?
- Расширение - смотри FRAMEWORKS.md
- Типичные проблемы - смотри DEBUG.md
- Дорожная карта - смотри IMPROVEMENTS.md

---

## 🚀 Типичные пути через документацию

### Путь новичка
1. README_RU.md (что это)
2. ARCHITECTURE.md (как устроено)
3. QUICK_REFERENCE.md (быстрые примеры)
4. Код (beam_engine.cpp, beam_kernels.cu)

### Путь разработчика
1. QUICK_REFERENCE.md (быстрая шпаргалка)
2. beam_engine.cpp (примеры кода)
3. DEBUG.md (если проблема)

### Путь интегратора
1. FRAMEWORKS.md (что интегрировать)
2. Код (см. заглушки TEInferenceBackend, комменты про CUB)
3. DEBUG.md (тестирование)
4. CHECKLIST.md (перед production)

### Путь отладчика
1. QUICK_REFERENCE.md (быстрая диагностика)
2. DEBUG.md (подробная отладка)
3. Код (воспроизведение проблемы)
4. FRAMEWORKS.md (если специфичная проблема)

---

## 💾 Где найти...

| Что искать | Где найти |
|-----------|----------|
| Как использовать engine.step() | beam_engine.py, QUICK_REFERENCE.md |
| Что означает COUNTER_BUCKET_OVERFLOW | ARCHITECTURE.md, DEBUG.md |
| Как интегрировать CUB | FRAMEWORKS.md, QUICK_REFERENCE.md |
| Как интегрировать TE | FRAMEWORKS.md, beam_engine.cpp (TEInferenceBackend) |
| Как профилировать | DEBUG.md (Performance section) |
| Как отлаживать NCCL | DEBUG.md (NCCL debugging section) |
| Как проверить готовность | CHECKLIST.md |
| Быстрые примеры кода | QUICK_REFERENCE.md |
| Описание hash table | ARCHITECTURE.md (Структуры данных) |
| Описание threshold | ARCHITECTURE.md (Stream 3) |
| Дорожная карта разработки | IMPROVEMENTS.md, CHECKLIST.md |

---

## ❓ FAQ (часто задаваемые вопросы)

**Q: С чего начать?**  
A: Начни с [README_RU.md](README_RU.md)

**Q: Как собрать проект?**  
A: `python -c "from beam_engine import build_extension; build_extension()"`  
Подробнее: [beam_engine.py](beam_engine.py)

**Q: Что такое три потока?**  
A: Подробно описано в [ARCHITECTURE.md](ARCHITECTURE.md) в разделе "Три асинхронных CUDA потока"

**Q: Как отлаживать?**  
A: Включи debug: `engine.enable_debug(verbose=True)`  
Подробнее: [DEBUG.md](DEBUG.md)

**Q: Как добавить CUB/TE?**  
A: Смотри [FRAMEWORKS.md](FRAMEWORKS.md) соответствующие разделы

**Q: Что делают счётчики?**  
A: Объяснено в [ARCHITECTURE.md](ARCHITECTURE.md) раздел "Буферы"  
Интерпретация: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) раздел "Основные счётчики"

**Q: Готов ли код к production?**  
A: Почти! Проверь [CHECKLIST.md](CHECKLIST.md)

---

## 📝 История версий

### v1.0 (текущая)
- ✅ Три потока CUDA
- ✅ LibTorch, NCCL, pybind11 интегрирована
- ✅ 10K строк документации
- 📋 CUB планируется
- 📋 TE планируется

---

## 🎓 Для самообучения

**День 1: Основы**
- Прочитай: README_RU.md
- Посмотри: beam_engine.cpp (класс BeamEngine)
- Поймешь: Что такое EngineConfig и buffers

**День 2: Архитектура**
- Прочитай: ARCHITECTURE.md (Stream 1/2/3)
- Посмотри: beam_kernels.cu (kernels)
- Поймешь: Как работают три потока

**День 3: Практика**
- Запусти: `python beam_engine.py` (собрать extension)
- Включи: `engine.enable_debug(verbose=True)`
- Посмотри: Счётчики (COUNTER_*)

**День 4: Расширение**
- Читай: FRAMEWORKS.md (CUB/TE раздел)
- Плануй: Как добавить CUB/TE
- Пиши: Первый прототип

**День 5: Production**
- Проверь: CHECKLIST.md
- Профилируй: DEBUG.md (Performance section)
- Готово!

---

## 🔗 Внешние ссылки

- **NCCL**: https://docs.nvidia.com/deeplearning/nccl/
- **Transformer Engine**: https://github.com/NVIDIA/TransformerEngine
- **CUB**: https://github.com/NVIDIA/cub
- **PyTorch C++ API**: https://pytorch.org/docs/stable/cpp_index.html
- **pybind11**: https://pybind11.readthedocs.io/

---

**Последнее обновление**: 2026-05-01  
**Версия**: 1.0  
**Статус**: Ready for development & testing

## Current correctness entry points

- `notebooks/kaggle_2xt4_debug.ipynb`: main Kaggle 2×T4 correctness notebook.
- `scripts/kaggle_correctness_check.py`: torchrun-compatible correctness script.
- `docs/KAGGLE_T4_DEBUG.md`: exact pass/fail protocol.
- `docs/CHECKLIST.md`: fixed and still-open items.


- `NEURAL_SCORER.md`: TorchScript scorer ensemble, 1024/256 MLP export, inference parallelism, CUDA Graph dependency rules.
- `PATCH_NOTES_2026_05_02_NEURAL.md`: neural scorer ensemble, score/net consumed events, TorchScript correctness.
