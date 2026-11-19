Warning: truncated output (original token count: 28238)
Total output lines: 2469

# Конфиг

## Rust продолжение permutation-поиска между глубинами, 2026-11-18

Совместимая Rust-сессия предпочитает optional ABI-2 depth controller и
сохраняет ABI-1 для старых библиотек и клиентов. На каждой полностью
завершённой глубине C++ передаёт Rust только scalar-состояние для существующей
проверки лимита калибровки перед `propagate_frontier_live` и состояния
frontier/debug-stop после этого коллектива. ABI-2 также передаёт solve-bucket
состояние после прежней синхронизации принятых записей. C++ сохраняет текущий
CUDA dispatcher, память, восстановление решения и весь NCCL обмен. Rust не
получает GPU-адреса и не вызывается на кандидата или ядро.

Порядок прежний: калибровочный `stop` по-прежнему пропускает следующий
frontier-коллектив; иначе все ранги вызывают его до решения о пустом фронте и
debug-stop. Solve-bucket сохраняет приоритет причин `capacity → depth → extra
window` и прежнюю арифметику unsigned 32-bit. Сканирование записей, обработка
найденного решения, восстановление пути, история и финализация остаются в C++.
Старая библиотека и прямой C++ CLI сохраняют прежний fallback; новые Rust
CLI/session пути подключают Rust-политику. ABI-1 сохраняет исходные структуры
24/12 байт и два прежних checkpoint; ABI-2 вводит отдельные структуры 64/16
байт и символы с суффиксом `_v2`. Это срез переноса управляющего цикла, не
завершение миграции.

CPU ABI и детерминированная политика должны проверяться через GitHub Actions.
Они не подтверждают сборку CUDA runner или runtime NCCL на нескольких GPU.

## Rust-владелец публичных native-сессий, 2026-11-18

Публичный поиск и `NativeSession()` используют Rust-владельца ранговых
работников, если установлены совместимый PyO3 API, контракт терминального
результата и упакованный `beam-runner`. Для старых установок остаётся Python
совместимый владелец; `engine="python"` выбирает его явно, а `engine="rust"`
завершает вызов ошибкой при отсутствии нужной возможности. Запросы
`backend="torch"` не затрагиваются.

Rust управляет жизненным циклом группы рангов, последовательностью заданий,
отменой и передачей проверенного результата. Подготовка модели, экспорт весов,
калибровка и сборка runtime всё ещё проходят через существующий Python-адаптер.
Внутренний нативный поисковый движок и его CUDA/NCCL вычислительные стадии
этим переключением не переписываются; полная миграция и отдельная проверка
multi-GPU остаются незавершёнными.

### Повторная конфигурация Rust-сессии, 2026-11-19

При изменении Python identity `NativeSession` сохраняет существующий PyO3
`SearchSession` и передаёт ему новый `SessionIdentity` и `Launch`. Rust сравнивает
типизированные identity, закрывает старую группу ранговых workers и запускает
новую; Python закрывает owner только если выполнение завершилось до нативного
launch. Служебный вложенный ключ в `options` включает поля, которые Python тоже
считает частью identity: геометрию, размеры артефактов модели, область binding
и `repr(options)`. Счётчик запроса начинает с 1 только для Rust-результата
`persistent_session.reused=false`; публичные поля результата не меняются.

Это сохраняет Rust owner при смене конфигурации и передаёт решение о повторном
использовании rank-группы Rust runtime. Стадии поиска, CUDA/NCCL вызовы, порядок
кандидатов и multi-GPU топология не меняются. Проверка этого среза выполнялась
на hosted CPU; фактическая PyO3/CUDA/NCCL и multi-GPU runtime-проверка остаётся
отдельной задачей.

## Контракт публичного поиска в beam-core, 2026-11-18

После адаптации сигнатуры установленной CayleyPy версии Python передаёт
нормализованные управляющие поля в `beam-core`: типизированную ширину beam,
лимит шагов, режим, `history_depth`, наличие неподдерживаемых целевого
состояния/BFS-результата и булевы флаги. Rust проверяет диапазоны и
поддерживаемость; PyO3 возвращает различие между некорректным вводом и
неподдерживаемым режимом. Python сохраняет прежний валидатор при отсутствующей
или старой extension, а также для нестандартных объектов с пользовательским
сравнением. Состав поиска, порядок кандидатов и multi-GPU выполнение не меняются.

## Проверка GraphContract в beam-core, 2026-11-18

Python извлекает исходные CayleyPy/NumPy/PyTorch значения и проверяет их
исходные типы без coercion. Затем совместимое PyO3-расширение передаёт
нормализованные целые числа в `beam-core`, где проверяются диапазоны меток,
мультимножество стартового состояния и полные gather-перестановки. Rust
возвращает проверенную структуру и канонический SHA-256, совпадающий с
`cayleypy.models.checkpoint.graph_hash`; тот же Rust helper используется при
проверке persistent-session identity. Имена генераторов остаются в Python-
адаптере и не входят в fingerprint.

Python сохраняет прежнюю проверку и hash для отсутствующего/старого расширения.
Этот шаг переносит построение математического контракта, но не поисковый цикл,
подготовку модели или CUDA/NCCL вычисления. CNOT и multi-GPU протокол не
меняются; полная миграция остаётся незавершённой.

## Rust-валидация NativeOptions, 2026-11-18

`NativeOptions.__post_init__` нормализует path-like поля и устройства в Python, затем передаёт конфигурацию в `beam-core::NativeOptions::validate` через необязательный PyO3 API. При старом, отсутствующем или не загружающемся расширении, а также для значений вне диапазонов Rust, остаётся Python-валидатор. При отказе Rust Python сохраняет публичный контракт ошибок. Дескриптор исключает только явный `false`, поэтому неверные falsey-значения не теряются при сериализации.

Это перенос проверки конфигурации. Управление поиском, вычисления CUDA/NCCL и переключение публичных путей этим изменением не меняются.

## Rust-контракт параметров CNOT-поиска, 2026-11-18

При `validation="auto"` совместимое PyO3-расширение передаёт число проводов,
глобальную ширину, shortlist и режим в `beam-core`. Rust сохраняет прежний
предел `width * alphabet <= i64::MAX`, положительный shortlist и режимы
`qhat`/`proposal`. Python оставляет точную проверку Python-типов, неотрицательный
лимит глубины и резервную проверку для Python-режима и старых расширений.
`validation="rust"` по-прежнему требует только уже существующий матричный
валидатор; отсутствие нового optional API не ломает старые установки.

Последовательность стадий, tensor/scorer/collective-вызовы, порядок ходов и
мульти-GPU обмен не меняются. Проверка сообщает перенос публичного контракта,
а не завершение переноса всего Python CNOT-адаптера.

## Rust-координация ошибок CNOT, 2026-11-18

Для мира из нескольких рангов `World.checked` передаёт callback через
`run_cnot_native_checked`, если установленная PyO3-библиотека экспортирует
этот ABI. Rust выполняет callback на исходном потоке, координирует голос ошибки
через существующий `world.sum` и собирает сообщения только после ненулевого
голоса через прежний `torch.distributed.all_gather_object`. Старые расширения
без этого символа сохраняют прежний Python fallback; однопроцессный путь не
меняется.

Этот шаг переносит владельца контрольной последовательности, но сами
коллективы по-прежнему используют тот же `World`/NCCL транспорт и тот же порядок
вызовов. Раскладка рангов, tensor/scorer операции и поиск не меняются. Полный
перенос CNOT и multi-GPU runtime остаётся незавершённым.

## Rust-сборка распределённых CNOT-строк, 2026-11-18

При нескольких рангах `World.gather_rows` использует Rust ABI для получения
локальных размеров, выделения общего tensor и последовательной рассылки
непустых диапазонов от рангов по возрастанию. Tensor-буферы остаются во владении
PyTorch, а `gather_fixed` и `broadcast` используют тот же выбранный транспорт.
Пустой локальный пакет пропускает только локальную копию, сохраняя участие в
сборе размеров и последующих broadcast чужих данных. Старые PyO3-расширения
используют прежний Python цикл.

Это перенос управляющего цикла без изменения порядка коллективов, рангов,
распределения данных или алгоритма. Проверка идёт через hosted CPU/Gloo; CUDA
NCCL исполнение проверяется отдельно и сейчас остаётся неподтверждённым.

## Rust-диспетчеризация управляющих коллективов CNOT, 2026-11-18

Для многорангового CNOT `World.sum`, `World.gather_fixed`, `World.broadcast` и
`World.exchange` передают выбор операции в PyO3-функции Rust. Rust сохраняет
прежний выбор `world.transport` или `torch.distributed`; сами tensor и вызов
transport остаются во владении PyTorch/Python. `World.exchange` также сохраняет
значение `native_packet`: обычный вызов не выбирает CUDA packet operator,
явный `native_packet=True` оставляет прежний путь. Одноранговая семантика и
старый прямой ABI exchange не меняются. Если расширение старое или нового API
нет, остаётся прежний Python путь. Этот шаг переносит диспетчеризацию, но не
заменяет NCCL/PyTorch transport собственным Rust/CUDA collective.

Hosted Rust/PyO3 и CPU compatibility проверки прошли. Физический NCCL и
multi-GPU CNOT runtime не проверены: текущий Vast workflow только проверил
состояние существующего инстанса и пропустил запуск.

## Rust-выбор CNOT transport, 2026-11-18

`World.start_transport` передаёт выбор транспорта в PyO3 ABI Rust, если он
доступен. При `prefer_rust=True` Rust получает локальную совместимую библиотеку
`RustStream5`, выполняет прежнее суммирование голосов всех рангов и выбирает
Rust transport только при поддержке на каждом ранге; иначе все ранги создают
существующий Python `Stream5`. При `prefer_rust=False` голосование не
выполняется и сохраняется Python transport. Если расширение старое, остаётся
исходный Python путь. Порядок коллективов, состав рангов, построение
коммуникатора и wire protocol не меняются; шаг переносит только выбор и
голосование и не добавляет оптимизаций.

Hosted Rust/PyO3 и CPU compatibility проверки прошли. Они не подтверждают
физический CUDA/NCCL запуск: закреплённый Vast-инстанс был выключен, поэтому
Vast workflow не менял его состояние и пропустил GPU-проверку.

## Rust-голосование за нативные CNOT-стадии, 2026-11-18

Когда доступен Rust CNOT controller, `World.all_ranks_support` передаёт
проверку доступности необязательного нативного stage runner в Rust. PyO3 helper
формирует локальный голос через прежний `World.longs`, суммирует его через
`World.sum` и разрешает runner только если значение совпало с размером мира.
Старые расширения сохраняют прежнее Python-голосование. Выбор между нативными
стадиями и Rust controller с прежними Python stage callbacks не меняет
распределение рангов, порядок обмена, представление тензоров или алгоритм.

Hosted Rust/PyO3 и CPU compatibility проверки прошли. Физические CUDA/NCCL
стадии и multi-GPU CNOT runtime не проверены: Vast-инстанс выключен.

## Rust-проверка совпадения CNOT rank inputs, 2026-11-18

После подготовки CNOT fingerprint Python-адаптер передаёт digest в Rust.
PyO3 функция формирует локальный `int64` tensor через `World.longs`, вызывает
прежний `World.gather_fixed` и сравнивает fingerprints всех рангов. При
различии Rust возвращает тот же `ValueError`; старые расширения сохраняют
прежнее Python-сравнение. Вычисление digest и предшествующая валидация остаются
в Python/Rust адаптере; переносится только межранговая проверка. Порядок
коллективов и multi-GPU топология не меняются.

Hosted Rust/PyO3 и CPU compatibility проверки прошли. Физические CUDA/NCCL и
multi-GPU CNOT runtime остаются непроверенными: текущий Vast-инстанс выключен.

## Самостоятельный CUDA-домен CNOT, 2026-11-17

Отдельная необязательная библиотека предоставляет версионированный C ABI для
существующих ядер генерации детей, распределения владельцев и выбора первых
ранее не встречавшихся матриц. Контекст привязан к одному устройству и потоку,
владеет CUDA-буферами и scratch, а Rust-владелец удерживает библиотеку до его
освобождения. Все слова CNOT-матрицы участвуют в сравнении. Обновление входов
сбрасывает зависимые результаты; переполнения сообщают требуемый и доступный
размер. Отдельные контексты не передают указатели между процессами.

Базовый ABI работает с host-буферами. Необязимое расширение передаёт ядрам
генерации потомков и распределения владельцев заимствованные CUDA-адреса,
размеры и устройство без копирования через host. Контекст только для таких
операций хранит геометрию и stream, не выделяя массивы базового ABI; его
host-операции возвращают ошибку. Вызывающий код гарантирует реальные границы и срок
жизни буферов, отсутствие пересечений и готовность данных на переданном stream;
для генерации он также гарантирует допустимость индексов и физических ходов.
Вызов синхронизирует stream перед возвратом. Указатель GPU не становится
ссылкой Rust и не передаётся другому процессу.

Дополнительный вызов сравнивает заимствованные CNOT-матрицы целиком и выбирает
первое появление каждого ранее не встречавшегося состояния. Вызывающая сторона
выделяет индексы, флаги, счётчик и CUB scratch по запрошенному размеру;
записываемые области не пересекаются друг с другом или входами. Счётчик остаётся
на GPU, пустой пакет записывает ноль. Старые библиотеки без вызова поддерживаются.

По явному `MGBS_CNOT_OWNED_LIBRARY` Rust-поиск использует этот вызов для
сравнения CNOT-матриц в каждом ранге. Вызов получает локальные Torch-буферы
и stream; GPU-адреса между рангами не передаются. Число выбранных элементов
временно читается на host для формирования Torch-представления результата.
Без переменной и на CPU сохраняется прежний путь. Это переходная интеграция:
остальные стадии, владение тензорами и scorer ещё требуют переноса. NCCL,
история и порядок рангов не меняются. Проверки и ограничения описаны в
`test_results/refactor_20261117/STAGE_CNOT_SEARCH_DEVICE_BRIDGE.md`.

Перед borrowed expansion Rust bridge проверяет диапазоны parent index и move
на GPU и читает по одному boolean scalar для каждого descriptor tensor.
Неверный дескриптор должен стать обычной rank-local ошибкой до вызова ядра,
чтобы существующий rank-wide vote мог скоординировать отказ без device-side
assert. Эта проверка добавляет синхронизацию скалярного результата. Тест
двухранговой ошибки и повторного вызова пока не принят; удалённый прогон и
ограничение получения его логов описаны в
`test_results/refactor_20261117/cnot-device-error/GITHUB_STATUS.md`.

## CNOT optional-extension compatibility, 2026-11-18

`rust_control.extension()` tries both supported PyO3 module names. Missing modules and import/dependency failures become `NativeUnavailable`. The public CNOT `validation="auto"` and `engine="auto"` paths can then retain the Python compatibility implementation; explicit `rust` selections fail closed.

With the matching module, all ranks use the Rust CNOT controller and native stage owner when that capability is present. Packed-matrix GPU operations dispatch to existing C++ ATen/CUDA operators, PyTorch continues to own tensor storage, and arbitrary Python scorers remain supported. Older modules keep the compatibility stage adapter. This update changes extension-load error handling only; it does not change the search algorithm or GPU stages.

Hosted validation covered CPU/control flow and extension build/import. CUDA execution, NCCL runtime, and physical multi-GPU search remain unverified.

## CNOT root identity from the typed Rust input, 2026-11-19

The compatibility stage uses `CnotSearchInput` or its `CnotMatrix` owner to
check whether the root is identity. The check compares the packed matrix in
Rust; older extensions and Python-only inputs retain `torch.equal`. This moves
only the root check. Tensor ownership, the distributed search order, CUDA
operators, scorer calls, and collective sequence remain unchanged.

## Переход CNOT на Rust, 2026-11-16

`search_cnot` по умолчанию выбирает Rust-контроллер, когда установлен
совместимый PyO3-модуль; без него сохраняется Python-реализация. Явные
`engine="rust"` и `engine="python"` выбирают соответствующий контроллер.
Rust управляет порядком поисковых стадий, остановкой, историей рангов и
восстановлением пути; проверка пути использует выбранный валидатор. Состояния
остаются в локальных Torch/CUDA-буферах ранга. При доступных новых операторах
локальная проверка цели, генерация CNOT-кандидатов, подготовка уникальных
матриц, упаковка выбранного фронта и разбор полученных записей выполняются
в C++ ATen с прежними CUDA-ядрами;
старый внешний операторный модуль продолжает работать через прежние операции.

Новый Rust-владелец стадий хранит локальный фронт, закрытые состояния,
кандидатов, метаданные и ссылки на записи истории каждого ранга. PyTorch
по-прежнему владеет памятью тензоров; вызовы выполняются в потоке вызывающей
стороны. Перед поиском ранги совместно проверяют наличие нового интерфейса:
если хотя бы у одного ранга установлен старый PyO3-модуль, все используют
совместимый Python-адаптер стадий. На CPU и со старыми библиотеками операторов
остаются прежние резервные реализации пакетных операций.
Для Rust-бэкенда проверка формы, устройства, типа и конечности оценок,
а также последовательность резервных операций генерации и дедупликации
теперь выполняются в Rust через те же тензорные операции. Это сохраняет
совместимость со старыми CUDA-библиотеками и CPU, в том числе при разных
наборах операторов на рангах; PyTorch владеет тензорами и вызывает scorer.

Rust также управляет проходами глобального radix-отбора и определяет cutoff;
подготовка ключей, гистограмма, обновление масок и итоговое упорядочивание на
CUDA выполняются в C++ ATen при наличии операторов. В новом бэкенде Rust
также хранит тензоры и управляет пакетными стадиями этого отбора; прежний
Python-адаптер остаётся доступен для совместимого старого пути. Остальные тензорные
операции, вычисление оценок и часть управления коллективами пока остаются в
Python. Это промежуточная граница переноса.
Подготовка CNOT-записей к межранговому обмену на CUDA также выполняется в
C++ ATen с прежней стабильной группировкой по владельцу; однопроцессный путь
не изменён.
В новом бэкенде Rust управляет подготовкой и отправкой записей: прежний
C++ ATen-оператор либо тензорная резервная последовательность формируют те же
непрерывные пакеты. При доступном транспорте используется тот же Stream 5;
CPU/Gloo сохраняет прежний порядок двух all-to-all, включая пустые отправки.
Сохраняются группа процесса вызывающей программы, отдельный NCCL Stream 5,
порядок обменов, полный packed-CNOT и сравнение всех слов. PyO3 `TQPScorer`
передаёт владение библиотекой, контекстом и прежним циклом scalar-ABI порций в
Rust; тонкий Python-класс сохраняет интерфейс и fallback для старого расширения.
Checkpoint-провайдер, модель, порядок выходов и численная политика не меняются.
PyTorch продолжает владеть буферами; GPU-адреса не превращаются в ссылки Rust
и не передаются между процессами.

На matching PyO3-пути `prepare_cnot_input` возвращает типизированный
`CnotSearchInput`: он хранит проверенную `beam_core::CnotMatrix` и Python-owned
ссылки на snapshot тензора и его байтовое представление. Python передаёт этот
объект дальше без повторной сборки `_InputMatrix`; старое расширение с прежним
четырёхэлементным ответом продолжает работать через совместимую обёртку.
Владение CUDA-буферами остаётся у PyTorch, Rust не создаёт из их адресов ссылки.

Rust-владелец библиотеки и NCCL-коммуникатора CNOT Stream 5 управляет обменом
счётчиков и записей для Rust-поиска. Ранги выбирают этот транспорт совместно;
если хотя бы на одном нет подходящего ABI, все используют прежний транспорт.
Версионированный ABI сохраняет устройство инициализации, восстанавливает
текущее устройство при закрытии и остаётся привязанным к создающему потоку.
PyTorch владеет CUDA-буферами, а FFI проверяет тензоры и отмечает их stream;
safe Rust-контроллер не получает GPU-указатели. PyO3/native-sys передаёт только
заимствованный адрес, размер и текущий stream в C ABI.

Добавочный `cnot_s5_owned_collectives_abi=1` предоставляет на том же
коммуникаторе NCCL `int64` all-reduce, fixed all-gather и broadcast. Rust
выбирает этот путь только после общего голосования и лишь если у каждого ранга
есть оба ABI: обмена записей и управляющих коллективов. Тогда проверки флагов,
статистики и восстановление истории сохраняют прежний порядок на одном
коммуникаторе. Если хотя бы один ранг использует старую библиотеку, все ранги
остаются на прежнем Torch-пути. Создание NCCL ID и failure-only
`all_gather_object` продолжают использовать группу вызывающей программы.
Буферы остаются во владении PyTorch; C ABI не сохраняет адреса после вызова,
а `record_stream` удерживает занятые аллокации до завершения поставленных
операций NCCL на текущем stream.

## Экспериментальный Rust supervisor внешних рангов, 2026-11-16

Режим `engine="rust"` для `run_cluster_rank` и `--engine rust` для
`cluster_worker` передаёт проверку назначения torchrun/Slurm, запуск,
ожидание и завершение дочернего ранга в Rust. Python подготавливает пути и
передаёт одно задание через PyO3; CUDA/NCCL и сам поиск остаются в прежнем
нативном runner. Входное окружение, локальные и глобальные ранги, общий путь
NCCL ID и маркер ошибки продолжают использовать существующий протокол.
Прерывание сообщает об ошибке соседним рангам и освобождает только дочерний
процесс текущего ранга. До сравнительной проверки многонодовых заданий
Python-supervisor остаётся режимом по умолчанию. Переход CNOT-поиска и
публичного поискового цикла в Rust этой работой не завершается.
Экспериментальный PyO3-адаптер пока требует совместимые с UTF-8 аргументы
и значения окружения; это не полная замена старого представления Python/OS.

## Коллективная ошибка поиска цели CNOT, 2026-11-15

Локальная ошибка при проверке завершающего хода передаётся всем рангам через
дополнительный флаг в уже существующем обмене цели. При отсутствии ошибки новый
коллектив и дополнительное чтение на host не добавляются. Сообщения об ошибках
собираются только при сбое. Тесты проверяют ошибки каждого из двух рангов и
повторный поиск в исправной группе; восстановление неисправного устройства,
NCCL-коммуникатора или неудачного самого обмена не гарантируется.

## Экспериментальный пакетный CNOT scorer, 2026-11-15

Опция `experimental_native_batch=True` передаёт весь пакет строк CNOT через
один вызов необязательного ABI `mgbs_tqp_score_many`. Внутри него прежний
нативный scorer последовательно обрабатывает подпакеты заданного размера;
ядра, формат матриц и численная политика не меняются. Старые библиотеки без
символа отклоняются только при явном запросе опции. Обычный путь остаётся
основным: ограниченный тест на RTX 3060 не прошёл порог ускорения.

Строки, результаты и статусы — отдельные заимствованные непрерывные CUDA-буферы
одного процесса и устройства. Rust/PyO3 сохраняет живые PyTorch-объекты на
время вызова; native-sys проверяет dtype, форму, устройство, непрерывность,
непересечение и размеры до FFI. Сам ABI повторно проверяет заявленные длины и
арифметику смещений, но не обнаруживает границы аллокаций или устройство.
Python при необходимости создаёт непрерывную копию входа. Стандартный scalar
цикл теперь выполняется в Rust с тем же разбиением и порядком вызовов; пакетный
ABI остаётся опциональным. Ни копирование, ни численная политика не изменяются.

## Rust/C ABI: выполняемый переход, 2026-11-07

Рабочая ветка и checkout — `main`. Исторические ветки и незакоммиченные
исходники сохранены в архивных тегах; восстановление описано в
`docs/REPOSITORY_RECOVERY.md`.

Введены Rust workspace (`beam-core`, `beam-native-sys`, `beam-runtime`,
`beam-cli`, `beam-python`) и C ABI v1 в `native/beam_native.h`.
Установка Python-пакета через PEP 517 собирает PyO3-модуль maturin и Rust worker,
включает совпадающий комплект исходников C++/CUDA/Rust с SHA-256 манифестом. Сборка из
checkout ведётся во временном каталоге, а source distribution допускает
повторную сборку без исходного checkout и проверяет целостность этого комплекта.
Сборка wheel сама по себе ещё не переключает публичный поиск на Rust и не
включает готовую CUDA-библиотеку.
Rust-работник лениво загружает CUDA-библиотеку в отдельном процессе ранга.
Явный подготовительный `ensure_native_library` собирает библиотеку в каталоге
совместимого управляемого runner и до сборки повторяет его CMake-конфигурацию.
При `inference_backend="auto"` используется backend уже проверенного runner;
явно указанный несовместимый backend и неразрешённое значение в metadata
отклоняются до сборки.
Для LibTorch, Transformer/ensemble и калибровочных runner metadata обязана
содержать идентичность сборки PyTorch; её отсутствие не допускает сборку или
повторное использование библиотеки.
В кэш входят спецификация runner, геометрия, CUDA/CUTLASS/NCCL, toolchain,
PyTorch при его использовании, а также SHA-256 runner, worker и библиотеки.
ABI и фактическое выравнивание проверяются отдельным worker-процессом; CUDA
не загружается в Python-процесс. Повреждённый или несовместимый кэш выдаёт
ошибку. Эта подготовка подключена к публичному поиску при явном выборе
экспериментальной Rust-сессии; путь по умолчанию остаётся прежним.
Контекст принимает подготовку и целые задания, сообщает состояние и ошибки,
поддерживает отмену. Специализированные ядра и диспетчер остаются в C++/CUDA.
Повторные задания сохраняют модель, арены, streams/events и CUDA Graphs.

Текущий ABI разрешает один контекст на процесс: движок использует process-wide
environment, cwd и stdout. Контекст нельзя размещать в Python-процессе, уже
использующем CUDA. `beam-native-sys` удерживает библиотеку до уничтожения
контекста; остальные собственные Rust crates запрещают `unsafe`.
Геометрия проверяется по фактической библиотеке, включая размер и выравнивание
Hash128/CandidateMeta. Указатели GPU через межпроцессный протокол не проходят.

В Rust-работнике история RAM/disk/static hybrid принадлежит `beam-runtime`:
резервы целых глубин, файлы, RAM, fallback и перенумерация предков. C++ владеет
pinned-слотами и событиями D2H; готовый пакет заимствуется одним вызовом ABI до
возврата записи. `beam-native-sys` перехватывает panic на каждом callback.
Одиночный Rust CLI использует тот же провайдер истории.

`ProbeSession`/`ProbeCohort` владеют калибровочными процессами, логами,
дедлайнами и общей отменой. `PlanningSession` проверяет точную ширину и допуск
всех рангов; перед NCCL все ранги подтверждают подготовку. `PipelineProbe`
владеет запуском полных калибровочных шагов, телеметрией и проверкой результатов.
Rust выполняет ограниченный поиск вместимости и выбор из списка профилей.
`InferencePool` сохраняет независимые процессы инференса, перезапускает только
сбойный ранг и различает ошибку качества от подтверждённого CUDA OOM.
Rust-контроллер выполняет грубый поиск микробатча, уточнение, проверку полной
группы рангов, телеметрии и статистики; вычислительные пробы остаются C++/CUDA.
`NativeOptions` имеет типизированный Rust-контракт. Целое задание калибровки
проверяет SHA-256 модели/helper до и после запуска, готовит входы, проверяет
исходные измерения из кэша и атомарно сохраняет выбор. Ключ включает фактические
артефакты, геометрию, оборудование, ABI-зависимые сведения сборки и версию политики.
Набор калибровочных состояний воспроизводится от центра графа и включён в ключ.
Повышение свободной VRAM более чем на 32 MiB требует повторной проверки профиля,
который ранее был ограничен памятью. Попадание в кэш не заменяет допуск памяти.
`DownstreamCalibration` передаёт целое задание одному Rust-владельцу. Он проверяет
геометрию и телеметрию всех рангов, формирует ограниченный список профилей,
повторно допускает каждый вариант нативным планировщиком и сохраняет исходные
замеры. Кэш стадий пересчитывается из исходных данных и требует свежего допуска
с той же геометрией. При скалярном MLP число входных строк и число родителей
различаются множителем по числу ходов; размер инференс-микробатча неизменен.
Сумма времени отдельных стадий — только оценка стоимости работы. Дополнительный
режим полного шага выполняет прогрев и пять повторов на всей запрошенной ширине.
Допуск по VRAM сам по себе не подтверждает отсутствие рабочего переполнения.
Нативная диагностика required/available сохраняется при ошибке ранга.
`FrontierCalibration` объединяет инференс, точный допуск и downstream-калибровку
в одном Rust-вызове. Для числовой ширины допускается замена только измеренного
микробатча; ширина сохраняется. Для `max` отдельно фиксируются граница памяти,
повторная калибровка инференса на точной ширине, полный шаг и рабочая эффективность.
Граница перестановок учитывает кратности меток и исключённые решё…16238 tokens truncated…iter; no following local batch launches before this round's receive
collection completes. Stream4 only reduces the occupancy of its owned bank.
This preserves the receive credit without an additional receive-side collective
or metadata synchronization. The device overflow guard remains diagnostic.

This is a guarantee for each admitted round, not a proof that every arbitrary
score distribution can be admitted by the current average-based planner. Ties
and adversarial shard skew are not silently discarded or handled by shard top-k.

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

## Audit follow-up: final logical union (2026-09-28, integration pending)

The final union primitive operates on the two adjacent clean physical buffers of
one logical shard: threshold, compact, common CUB sort/reduce, compact back into
the same pair, split counts at physical capacity, then publish exact histograms.
The representative is ordered by score, parent index, then packed route. There
is no shard top-k or semantic truncation. The caller must drain all writers and
wait for both histogram publications before final threshold/selection.

`LayoutUnionView` is a separate sequential phase over the beginning of the static
scratch pool. `layout_final_budget_bytes` is the maximum of union, selection and
materialization extents. Survivor storage, clean counts and histogram publications
are beyond that whole extent. Union scratch includes its own two housekeeping
dirty/processing cells; dead Stream3/4 transient counters are not union inputs.
It is sized for one pair (2*physical capacity), reused across logical shards, and
does not allocate a third persistent frontier. At this checkpoint the primitive
and memory plan exist but dispatcher integration/acceptance is still pending.

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

### StaticHybrid history reservations (2026-09-28)

`depth_limit` counts beam expansion iterations. K1/K2 suffix radii do not reduce
the number of history layers required by an unsolved run. The host planner in
`src/history_budget.cpp` reserves every layer before the depth loop, using
`min(local_beam, MOVE_COUNT^(depth+1))` as a conservative entry bound.

Each whole layer has a fixed RAM or disk extent. A smaller actual count does not
move later extents. Preflight must validate placement, not just combined free
bytes; disk-failure fallback may use only RAM beyond all planned reservations.
Failure to reserve or write required history is explicit, never silent pruning.
Pinned/staging bytes are budgeted separately. This contract does not guarantee
physical filesystem reservation, checkpoint resume, or tolerance of disk failure
when the additional fallback reserve is exhausted.

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
WORLD_SIZE > 1 periodic threshold update uses committed conservative local histograms followed by Stream5-owned NCCL SUM when the collective request is set.
All ranks preserve the same collective order; local-only periodic rules are superseded by the Stream5 contract above.
WORLD_SIZE > 1 final global threshold uses NCCL/global counts only after common logical A/B union and local final flush.
Финальный global threshold считается только после финального flush и общей финальной дедупликации обоих physical buffers каждого logical shard.

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
    value = CandidateMeta (default), or uint32 index into stationary CandidateMeta (experimental)
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

The experimental index path keeps the compacted CandidateMeta array immutable
through sort, reduction and gather. Two non-overlapping index arrays occupy the
existing value-B arena; their offsets use the physical capacity, including in
bounded graph bodies. A separate device count bounds every reducer access:
CUB may evaluate inactive lanes with arbitrary index values, so a sentinel-only
check is insufficient. The compacted count stays live until reduction finishes;
the old block-offset slot receives CUB's unique count and is reused only after
its reader completes. The final gather writes to the distinct survivor arena.
This path is opt-in and must pass the full-step promotion gate.
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

## CNOT input admission ownership, 2026-11-19

For the matching Rust extension, `validation="auto"` and `validation="rust"`
must prepare the owned typed CNOT input inside the Rust controller and inside
the existing rank-wide admission error boundary. The Python input-builder
callback remains for explicit `validation="python"` and old-extension
compatibility. The snapshot remains detached, cloned and contiguous; packed
byte encoding, matrix validation, admission digest, collective order, scorer
callbacks, search decisions, and multi-rank topology remain unchanged.

## Rust-owned CNOT identity and construction checks, 2026-11-19

The Rust native stage owner reads identity and construction validity from the
already-owned `CnotMatrix` inside `CnotSearchInput`. The matching typed path no
longer dispatches these two checks through Python tensor/object methods; legacy
Python input objects keep the previous compatibility fallback. Identity checks
compare every packed row and word, and construction replay uses the existing
physical-gate order and full-matrix equality. Tensor ownership, frontier
transitions, scorer calls, rank collectives, stage order, and result semantics
are unchanged. This is a control/domain ownership transfer, not a search or
kernel optimization. CPU tests cover packed identity boundaries and the public
Rust CNOT root-goal path; physical CUDA/NCCL multi-GPU execution remains a
separate unverified gate.

