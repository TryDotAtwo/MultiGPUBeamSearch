# Как отправить решения CayleyPy

Плагин отправляет результаты без токена через закреплённый Cloudflare Worker и проверяет появление записи в публичном GitHub-репозитории. Нормальный пользователь не вводит URL сервиса и не хранит секреты.

## Самый короткий путь: готовый JSON или JSONL

Если solver уже создал канонический envelope v1 (Kaggle) или v2 (кластер/локальная машина):

```bash
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py preflight results.json
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py submit results.json --wait
```

Поддерживаются один envelope, `{schema_version, results:[...]}` и JSONL. CLI валидирует локально, упаковывает детерминированные gzip-архивы не более 32 MiB и отправляет каждый архив одним запросом. Большой набор автоматически делится на несколько последовательных запросов.

## Простой путь: строка ходов или CSV/TSV

Один раз создайте конфиг среды:

```bash
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py init --source kaggle --output publisher-config.json
# либо для кластера/локального запуска:
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py init --source native --output publisher-config.json
```

Отредактируйте `publisher-config.json`: автора, соревнование, модель, hardware, профиль, runtime, run_id и `puzzle_contexts`. Для каждого puzzle id нужны `puzzle_type`, начальное/целевое состояния и генераторы. Нельзя придумывать неизвестные значения.

Одна строка ходов (в конфиге должен быть ровно один puzzle id):

```bash
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py preflight solution.moves --config publisher-config.json
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py submit solution.moves --config publisher-config.json --wait
```

`solution.moves` содержит, например, `clockwise.counterclockwise`.

Для одного или многих решений используйте `templates/solutions.csv`. Колонки должны быть ровно такими:

```text
puzzle_id,solution,final_orientation,search_mode,collection_index,collection_status,solved_depth,touch_depth,reflected_source_solution,searched_solution
```

```bash
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py submit solutions.csv --config publisher-config.json --wait
```

Для TSV допустима та же шапка с табами. В режиме `first` передайте одно решение и `collection_status=first_solution`; в режиме сбора передайте все найденные решения отдельными строками с последовательным `collection_index`.

## Kaggle

Добавьте папку плагина как dataset/input либо скопируйте один stdlib-only файл `cayleypy_submit.py` в notebook. После solve:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "/kaggle/input/cayleypy-results-publisher/cayleypy_submit.py",
    "submit", "/kaggle/working/results.json", "--wait"
], check=True)
```

Если notebook формирует только CSV, добавьте `--config /kaggle/working/publisher-config.json`.

## PowerShell

```powershell
py .\plugins\cayleypy-results-publisher\scripts\cayleypy_submit.py submit .\results.json --wait
```

## Кластер/Linux

```bash
python3 plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py submit ./results.json --wait
```

При обрыве сети повторите ту же команду: manifest `cayleypy-receipts.json` не даст повторно отправить уже принятые архивы. Отдельная проверка:

```bash
python plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py poll --manifest cayleypy-receipts.json
```

HTTP 202 означает только приём Worker. Успех публикации — `published>0`, `rejected=0`, `unresolved=0`. Manifest содержит только receipt/idempotency/status/GitHub URL и хэши частей, но не исходные состояния или решения.

## Где результат

Worker публикует нормализованные TSV/Parquet-совместимые данные и подробные записи в [TryDotAtwo/cayleypy-beam-results](https://github.com/TryDotAtwo/cayleypy-beam-results). CLI печатает путь к receipt-manifest и проверяет ожидаемый immutable GitHub JSON даже после очистки временной записи Worker.

## Безопасность

- токен GitHub/Cloudflare не нужен и не должен попадать в notebook;
- официальный endpoint встроен в CLI, redirect и другой origin запрещены;
- архив максимум 32 MiB compressed и 64 MiB raw;
- перед сетью выполняется replay решения и вычисляются proof hashes;
- ошибки публикуются как безопасные коды без содержимого envelope.
## Что именно заполнить в publisher-config.json

| Раздел | Что указать |
|---|---|
| `schema_version` | `1` для Kaggle, `2` для native/SLURM/local |
| `common.author` | имя автора; Kaggle username, если применимо |
| `common.competition` | slug соревнования |
| `common.kaggle` | owner, slug, version, run URL и SHA-256 notebook для v1 |
| `common.model` | имя/путь модели, SHA-256, формат, manifest и класс head |
| `common.hardware` | платформа, GPU, их число, world size; для v2 также SM/VRAM/driver, если известны |
| `common.profile` | requested/effective beam, alignment, выбранный профиль и evidence |
| `common.runtime` | solution mode, touch radius, depth/collection limit и эффективные pipeline-параметры |
| `common.timings` | честные solve/wall timings |
| `common.run_id` | стабильный ID конкретного запуска |
| `common.solver_commit` | точный git commit solver |
| `puzzle_contexts` | для каждого ID: puzzle type, initial/central state и генераторы |
| `solution_defaults` | общие orientation/search/collection/depth значения для строк CSV |

В CSV строка может переопределить только поля решения и ориентации из фиксированной шапки. Поля запуска берутся из `common`; произвольные дополнительные CSV-колонки отклоняются.

## Частые ошибки

| Код | Что делать |
|---|---|
| `CONFIG_PUZZLE_MISSING` | добавьте puzzle id в `puzzle_contexts` |
| `INPUT_UNKNOWN_MOVE` | проверьте имя хода и generators |
| `INPUT_FIELD_INVALID` | проверьте целые depth/index и точную CSV-шапку |
| `ENVELOPE_TOO_LARGE` | одна запись превышает лимит и должна быть уменьшена у источника |
| `SUBMIT_PARTIAL` | часть записей отклонена; сохраните manifest, исправьте вход и повторите |
| `HTTP_TRANSPORT` | сеть/DNS недоступны; повторите идентичную команду позже |
| `STATUS_URL_UNSAFE` | manifest повреждён или не от официального Worker; не обходите проверку |
| `unresolved>0` | публикация ещё идёт; выполните `poll` снова |