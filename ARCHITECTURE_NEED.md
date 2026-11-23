## Rust-owned single-host rank supervision — 2026-11-23

`local_launcher.run_local_ranks` now sends matching installed wheels to one Rust runtime call. Rust starts all local ranks with the established `world_size, rank` argv suffix and rank environment, supervises timeout and cancellation, and reaps each worker process group on completion or failure. Python retains the compatibility supervisor for extensions without this API. Search stages, CUDA kernels, NCCL transport, ordering, and numerical policy are unchanged.

Installed-wheel validation ran outside the checkout and exercised the public launcher with two synthetic ranks, per-rank environment, command metadata, and log layout. A failing rank also verified sibling cleanup. The installed-wheel run passed on source SHA `fd42ebca8d7486c92a7d3a26c98a7ea591cc6f8f`: [run 45131297173](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45131297173). This proves CPU process supervision only; physical permutation CUDA/NCCL remains pending.

## Rust-owned permutation solution policy environment — 2026-11-22

The public permutation launch now resolves the fixed solution-collection controls through a versioned `beam-runtime` request: first-solution versus collection mode, extra collection depths, maximum collected solutions, and the result TSV name. `backend._run_native_once` applies that environment after calibration and immediately before the runner launch, so calibration probes retain their existing non-collecting behavior. Older wheels continue to use the same Python compatibility builder. This changes environment ownership only; search ordering, kernel execution, numerical policy, and CUDA/NCCL transport are unchanged.

Installed-wheel validation passed on code SHA `c9b6e10b2f013d7fc243719d1d7316faba78b88b` in [run 45062575238](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45062575238). The installed `test_cluster_rust.py` step exercised the compiled Rust resolver through `backend.search_policy_environment`, checked collection and first-solution values, old-resolver fallback, and rejection of an invalid mode. The same workflow passed the existing saturated permutation CPU acceptance. No physical CUDA/NCCL run occurred.

## Rust-owned permutation search launch environment — 2026-11-22

The public permutation launcher now obtains its fixed puzzle/input/weight/history paths and base BEAM controls from a versioned Rust runtime function. Python supplies the already prepared paths and validated BFS settings, then merges the returned variables with model-specific inference settings. Older wheels keep the previous Python builder. This does not change search controls, model dispatch, kernel behavior, or numerical policy.

## Rust-owned permutation CUDA device admission — 2026-11-22

The public permutation preparation path now sends the observed CUDA availability and normalized device request to one versioned `beam-runtime` plan. Rust rejects a missing CUDA runtime and resolves the same explicit, `num_gpus`, or graph-derived process-local indices; Python keeps the Linux/torchrun boundary checks, collects Torch facts, and retains the prior device selection for wheels without this API. No device binding, worker launch, search order, kernel, or numerical behavior changes.

Installed-wheel validation passed at source SHA `e4e96c452dba8b531c7de262f8f333b088a579c3` in [run 45058727951](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058727951). The real `backend.runtime_devices` test exercised the Rust plan, the old-wheel resolver fallback, and the unavailable-CUDA short-circuit. The same installed run passed saturated permutation CPU acceptance. Rust CPU/Gloo formatting and contract checks passed in [run 45058728166](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058728166); the hosted acceptance harness passed in [run 45058728397](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058728397). Physical permutation CUDA/NCCL remains unverified.

## Current migration priority — 2026-11-22

The active implementation phase prioritizes the public permutation beam-search path through Rust preparation and launch while preserving the existing CUDA/C++ kernels, multi-GPU execution, numerical policy, Python/Kaggle interfaces, CLI, torchrun and Slurm. CNOT implementation and compatibility remain in the branch; further CNOT changes, private checkpoint preparation and CNOT-specific physical acceptance are deferred at the user's request. Permutation CUDA/NCCL acceptance on one and two physical GPUs remains its own required gate.

## Rust-owned public cluster planning — 2026-11-22

The public `plan_cluster` adapter now sends the validated per-node GPU counts, global beam width, shard count and batch alignment to `beam-runtime`. Rust returns the same ordered global/node/local rank mapping, exact global beam, per-rank frontier capacity and alignment. Python preserves the public dataclasses and validation errors, and uses the existing calculation only when an older extension does not export the resolver.

This moves multi-node geometry policy without changing the requested/effective width, rank ordering, shard layout, transport, CUDA execution or hardware-verification status. Hosted Rust CPU/CNOT validation and the installed PEP 517 wheel passed on code SHA `9fa887577d2c1684da8ddb6db29395493ebe45bb`: [Rust CPU run 45050490783](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45050490783) and [installed-wheel run 45050490824](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45050490824). The installed test exercises the exported resolver, the public helper with the Python formula disabled, and the older-wheel fallback. The hosted acceptance harness also passed ([run 45050491058](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45050491058)); its physical GPU job was skipped. Physical CUDA/NCCL verification remains separate.

## Rust-owned permutation microbatch plan — 2026-11-22

The existing integer plan for Stream 3's initial inference rows is resolved in beam-runtime, including the scalar-head row-to-parent mapping, four initial sort slots, 1024-row alignment, capacity scale and metadata fields. The Python adapter supplies typed request values and turns the plan into the same BEAM_B_MICRO environment value consumed by the existing runner. Older wheels retain the Python implementation. This transfers the current policy without changing candidate, search, model, kernel or precision behavior.

The installed-wheel smoke compares exact scalar-head, ensemble/nondefault-row-limit and unadjusted plans, then exercises the real helper with the Python fallback forced to fail and checks legacy-extension behavior. Hosted installed-wheel validation passed on code SHA `5cfff58de62f454fe6b63d8ea61514718e4ed578` ([run 45048372751](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45048372751)), exercising the public helper against the scalar-head, ensemble/nondefault-row-limit, unadjusted, installed-Rust, and legacy-fallback cases. The Rust CPU/CNOT suite passed on the same SHA ([run 45048372758](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45048372758)). Physical CUDA/NCCL verification remains separate.

## Rust-owned permutation device selection — 2026-11-22

The public native preparation path now sends the visible CUDA count, explicit device request, requested GPU count, and normalized graph-device kind/index observations to `beam-runtime`. Rust applies the existing selection rules and returns the concrete device ordinals used by preparation. Python continues to query Torch and translate CayleyPy device objects; it uses the prior Python resolver only when an older extension does not expose this API. CUDA_VISIBLE_DEVICES mapping remains a separate launch-environment step.

The installed-wheel smoke checks the Rust selection table and calls the public preparation helper with the Python fallback forced to fail. It separately checks the old-wheel compatibility path. CPU/GPU search, model export, inference profile selection, kernels and precision are unchanged.

## Keep CNOT history edges typed across checked stages — 2026-11-22

The Rust checked-stage helper now carries generic Rust results through the existing rank-wide vote and diagnostic path. CNOT history decodes the fixed signed wire row directly into the runtime `Edge` inside that checked phase; it does not convert the values back through a Python tuple. Error stage names, error text, vote ordering, history layout and replay checks remain unchanged.

The typed row decoder rejects invalid length and signed or over-wide fields. Hosted CPU/Gloo and installed-wheel checks must cover normal history reconstruction and a coordinated history-stage failure with a healthy retry. This change does not establish physical CUDA/NCCL acceptance.

## One ranked CNOT request reaches the native stages — 2026-11-22

The Rust CNOT controller now passes the same validated RankedCnotRequest it used for admission into the native-stage adapter. The adapter derives rank, world size, mode, audit and search options from that typed value and checks that its topology matches the live World before stage construction completes. The older scalar run_cnot_native arguments remain valid for direct compatibility callers; the public Rust-controller path no longer reconstructs stage configuration from those scalar copies. Tensor layouts, scorer callbacks, collectives, kernels, search ordering and numerical policy are unchanged.

The existing hosted CPU and installed-wheel public CNOT paths exercise the controller through the native-stage boundary. The change is source-level/runtime ownership progress; it does not establish physical CUDA/NCCL acceptance.

## Rust-owned CNOT dispatch fingerprint — 2026-11-22

The public CNOT path now asks `beam-runtime` for the versioned request fingerprint used by rank consensus. The PyO3 adapter converts the public engine and depth inputs once at this boundary; the Rust runtime hashes engine, u64-depth fit, and strict-mode bits using the established byte contract. The Python adapter uses this API when present and retains the identical legacy digest for older wheels. Gloo phase ordering, capability voting, route selection, search stages, scorer, and CUDA/NCCL behavior are unchanged.

The installed-wheel regression compares the Rust digest against the prior Python formula for every engine, the u64 depth boundary, strict/compatibility mode, and invalid engine; it also asserts that the public search calls the installed Rust API. Physical CUDA/NCCL execution remains a separate gate.

## Accept the current panel-v2 producer receipt — 2026-11-22

The consumer accepts producer source revision `eafc0f82f088603a525f428a8fa3d00e9e702fa8` with the exact existing `build_multigpu_beam_handoff` emitter SHA-256. The receipt source revision is distinct from the emitter ABI: the producer function is unchanged from its parent `42ed72f587ab70fc74a85cf0a25001efb29cb994`, whose receipts remain supported, along with the historical `7310b1eb5ec07c76c131ba8c60b105ff4aad7a9c` pin. The hosted no-Torch compatibility fixture checks out the current producer revision, calls its real serializer, and sends the serialized receipt through the consumer resolver. This is CPU-only handoff validation; trained scorer/GPU acceptance remains separate.

## Rust session identity includes the resolved inference plan — 2026-11-22

Permutation preparation identity schema v3 carries a typed `ResolvedInferencePlanIdentity` with the selected backend, policy and inference row limit returned by the Rust resolver used in `prepare_runtime`. A repeated `NativeSession` call can reuse its resident rank owner only while this plan and the other typed preparation inputs match; a changed policy or row limit invalidates the preparation. Schema-v2 extensions keep the plan inside the existing structured options value, and older extensions retain the Python compatibility identity.

The installed-wheel contract calls the real PyO3 resolver through `prepare_runtime`, checks the runner backend and profile fields, and exercises Rust identity reuse and plan invalidation without CUDA. This changes preparation identity only; scoring, inference kernels, numerical policy and rank execution are unchanged.

## CNOT scorer contract selection — 2026-11-22

The acceptance launcher builds two separate Rust scorer libraries: the historical
GL4 `dev_v1` adapter and the sealed panel `panel_search_v2` adapter from its
own pinned upstream commit. The shared launcher helper selects the library and
its SHA-256 by contract and enables strict Rust admission for both paths.

For the panel, `verify_cnot.py` records metadata returned by the loaded scorer
for every dimension. The report aggregator checks the panel contract, upstream
adapter pin, checkpoint and manifest, training pins, dtype, feature registry,
and the launcher-bound panel library digest. It also requires the strict event
receipt to match that report. This proves the selected loaded scorer contract
and binds the invocation to the built artifact digest; it does not constitute
physical CUDA/NCCL acceptance. The installed-wheel workflow reuses one built wheel and records panel endpoint
and inference-policy outcomes independently. Hosted checks for the implementation
revision are pending; physical acceptance remains unverified.


## CNOT Gloo lifecycle receipts — 2026-11-22

The installed-wheel lifecycle check records each phase independently on both
Gloo ranks. For engine mismatch, it validates each rank's requested engine
(`auto` on rank 0 and `python` on rank 1) separately, while both receipts
must record the same consensus rejection. Rank-local scorer failures and their
healthy retries also require matching phase receipts from both ranks.

`verify_cnot.py` owns the global result report on rank 0 only. The aggregate
therefore requires exactly the three rank-0 retry reports
`healthy-after-engine-mismatch-0.json`, `healthy-after-0-0.json`, and
`healthy-after-1-0.json`, each solved with bitwise replay. It does not require
or infer a report from rank 1. The CPU receipt fixture rejects missing phases,
wrong rank identity and missing root replay evidence.

At code SHA `c59d6294c11c411c9c1eaad54115b689b00a10a3`, the installed-wheel
Gloo lifecycle passed in [run 44977925758](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925758); the physical GPU job was skipped.

## CNOT public dispatch ownership — 2026-11-22

The public CNOT entry point obtains its runtime route from typed `beam-runtime` policy. The capability snapshot is rank-agreed before selection. The policy selects the Rust controller, the Rust controller with compatibility stages, or the Python reference path; strict acceptance requires the complete Rust controller/stage/transport capability set and fails closed. Older extension builds retain their ordinary compatibility path. This does not yet move PyTorch's Gloo control-group object out of the Python adapter; that lifecycle remains an open migration boundary. Search order, score semantics and kernels are unchanged.

## Typed CNOT request to runtime search options — 2026-11-22

After the existing collective input-admission vote, the Rust controller keeps one validated `CnotRequest`. Its `search_options(rank, world)` conversion supplies the generic `beam-runtime::cnot_search::search` with the normalized global width, directed-gate alphabet, alphabet-capped shortlist and depth. The Rust compatibility-stage branch uses this conversion directly. The existing `run_cnot` PyO3 signature remains as a compatibility adapter. Search order, scoring, tensors, kernels and collective sequence are unchanged.

The hosted RED fixture at `c383aa9` confirmed that the runtime conversion was absent. The exact green source SHA `c63937944b7752227982b388ee075be26e89c055` passed Rust CNOT CPU/Gloo [run 44976811410](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811410), installed wheel [run 44976811495](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811495), hosted harness [run 44976811467](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811467), and panel-v2 CPU compile/handoff [run 44976815210](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976815210). Physical CUDA/NCCL execution remains unverified.

## Rank-bound CNOT request — 2026-11-22

The public controller binds its validated scalar request to the active
`rank/world` as `RankedCnotRequest` during the existing collective input
admission. It rejects `world=0`, `rank >= world`, and `world > 65536`
before native backend admission or stage allocation. The typed request supplies
normalized controls to the Rust controller while Python `World` still provides
the rank topology. That remaining Python ownership is an open migration
boundary. Search semantics, tensors, collective ordering, scoring, kernels and
precision are unchanged.

On source SHA `25bc43abc1911d6a96c14d7c7d0c766a043f5f4d`, Rust CNOT CPU/Gloo,
installed wheel, hosted acceptance and panel-v2 CPU checks passed in runs
[44980014848](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014848),
[44980014912](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014912),
[44980014854](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014854) and
[44980018575](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980018575).
The physical CUDA/NCCL job was skipped.

## Admitted CNOT topology at native-stage binding — 2026-11-22

The public Rust CNOT controller forwards the `rank/world` values from the admitted `RankedCnotRequest` into native-stage construction. The adapter checks that pair against the Python `World` and places construction inside the existing common Gloo phase before search. Existing direct PyO3 callers may omit the optional argument and retain their compatibility behavior.

This narrows the topology handoff for stage setup; Python `World` still owns the control group and tensor collectives. No search, scoring, ordering, kernel, precision, or data-plane behavior changed. The installed wheel passed on implementation SHA `d9814462b05938f3419aa5c71332f9c3ce6619cd`; physical CUDA/NCCL remains unverified.

# Конфиг

## Paired permutation acceptance receipt contract, 2026-11-22

The CNOT campaign verifier now checks the permutation report as a run-bound acceptance artifact. A valid report must bind source SHA, installed wheel SHA and acceptance run ID; include the existing LRX(4) one/two-GPU replay and the saturated LRX(5), B=3 oracle case; prove the exact global cutoff after cross-rank dedup; and carry two ordered calls from one reused Rust session per topology. Each call must match per-rank PIDs, observed CUDA UUIDs, build/binary hashes, expansions, global cutoff counts, retained frontier counts, complete Hash128 survivor order and independent replay status. The canonical report digest is verified before the final paired receipt is written. Hosted fixtures reject world-size-only JSON, foreign identities, missing/duplicate ranks or GPUs, unsaturated counts, changed winner order, missing session reuse and a body modified after digest. This is a CPU-verifier contract; it does not establish physical CUDA/NCCL execution.


## Installed-wheel проверка штатного выбора CNOT-контроллера, 2026-11-22

Публичная точка `verify_cnot.main` проверяется с `engine=auto` на двух Gloo-рангах. В healthy retry тест подменяет callback Python reference-loop на ошибку; успешный результат подтверждает, что установленный wheel выбрал Rust controller. До каждого retry тест отдельно инъецирует сбой загрузки scorer только на rank 0, затем только на rank 1, и проверяет последующий запуск в той же живой группе.

На tested source SHA `a517f8afec9de82ad3fe1958fe8b83ba5df2fafa` installed-wheel run [44955518273](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518273) прошёл, вместе с прежними PyO3 dtype, strict-admission, event, verifier и permutation CPU gates. Hosted source harness [44955518309](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518309) прошёл. Оба артефакта содержат только redacted status. Физический CUDA/NCCL job был пропущен.

Это подтверждает выбор штатного Rust controller на CPU/Gloo и восстановление живой группы после согласованной ошибки. Оно не подтверждает CUDA/NCCL, обученный checkpoint scorer на GPU, физическую работу рангов или завершение полного переноса. Остаточная граница Python `World`/Torch и scheduler/profile preparation остаётся открытой; общий статус — `test_results/refactor_20261121/rust-cnot-admission/GITHUB_STATUS.md`.

## Typed CNOT controller choice in beam-runtime, 2026-11-22

The Rust runtime now owns the controller selection table for public `auto`, `python`, and `rust` requests. The PyO3 boundary keeps the existing public value checks, then calls the typed `beam_runtime::cnot_search::select_engine` function. If depth exceeds the native `u64` range, `auto` and `python` keep the reference path, while explicit `rust` returns the existing error. Search stages, scoring, kernels and precision are unchanged.

The runtime policy has unit coverage for each selection branch. The installed-wheel two-rank Gloo test uses `engine=auto` for both healthy retries and replaces the Python reference-loop callback with a failing sentinel. Injected scorer-load failures on rank 0 and rank 1 remain explicitly Rust; each is followed by a successful search in the same Gloo group. On tested SHA `0606965a210c9cc2417b729433471e3d5ca6988c`, the Rust CNOT CPU/Gloo gate passed in [run 44956473123](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473123), the installed wheel passed in [run 44956473162](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473162), and the hosted source harness passed in [run 44956473161](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473161). The physical GPU job was skipped. Full context ownership and physical CUDA/NCCL acceptance remain open.

## Независимая проверка reference cutoff CNOT, 2026-11-22

CPU acceptance теперь сравнивает порядок выбранных состояний с независимыми оценками scorer. Отчёт сохраняет, был ли cutoff underfilled, точно равным с полным сравнением path word, отделённым от границы погрешности или неоднозначным внутри допуска. Для неоднозначной границы проверка явно не утверждает устойчивый выбор по эталонным оценкам; для точных ties проверяется прежний полный tie-break по path word.

Проверка относится к acceptance verifier `tools/verify_cnot.py`. Она не меняет порядок поиска, scorer, precision или продуктовую численную политику. На SHA `77cc69c421a640dc0ec32ee2ce68564b1ded67d3` все три новые fixtures прошли в установленном wheel: точная ничья, неоднозначная граница и отделённая граница. [Installed-wheel run 44950126048](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126048), [artifact 14724934157](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126048/artifacts/14724934157), SHA-256 `58f8024b1df53370a5d5292cad9980c34c99e34e88dd8086043612d332e271d9`. Source harness [run 44950126040](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126040) также прошёл. Обе проверки hosted CPU; CUDA/NCCL физический job пропущен.

## Typed options in both Rust session identities, 2026-11-21

The v2 preparation identity already carries options as JSON. The nested identity used by the Rust resident worker now does the same: Python places options_descriptor(options) inside python_session_identity.options instead of storing repr(options). Rust therefore compares the same structured option values in both preparation invalidation and worker reuse.

Hosted RED `97441f6` failed the existing reuse test on world sizes 1 and 2 because the nested object had options_repr but no options. After the fix, Python-owner passed on source SHA `9e6863387861c0a643f0a043acfdbe6892126c13` in [run 44947770658](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770658), artifact [14724530178](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770658/artifacts/14724530178), SHA-256 `74681e4acf1d526dead977f441b6b9d1cc69722819b70cfb185b59c59bf535b3`. The installed wheel passed in [run 44947770665](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770665), artifact [14724367144](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770665/artifacts/14724367144), SHA-256 `9a2cd384509198463d78cc99df2ea0ee5ac7e834fc2aee404be8c4ebe42c27af`.

No search, scoring, precision, or rank behavior changed. This is hosted CPU evidence only; physical CUDA/NCCL acceptance is pending.

## Rust session preparation identity v2, 2026-11-21

The resident permutation session now receives a typed preparation identity from the Rust extension. Schema v2 stores NativeOptions as a JSON object and compares it structurally; it no longer relies on Python repr output. The Rust builder validates that the options value is an object. The requested beam width retains both the numeric form and the public max form.

The identity includes graph and geometry, model artifact hash and sizes, model backend, device list, binding scope, beam width, depth, options, CUDA device visibility/order, and the native runtime build. Matching identities reuse the resident worker group. A change in model or preparation fields invalidates the match. Python uses the typed schema only when the installed extension declares schema v2; an older extension does not receive the new arguments and keeps the compatibility path.

On source SHA 4cad49f3c98fe1bc37895af98bfffd5827a0829f, the hosted source harness passed in [run 44946584217](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584217), artifact [14723403934](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584217/artifacts/14723403934), SHA-256 68a7692500d325d01b683a79564f7ad820ee1ca12178475953eface51998c911. The installed wheel and smoke passed in [run 44946584220](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584220), artifact [14724077257](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584220/artifacts/14724077257), SHA-256 58ec47161cfb381e009412e12b116378b5e277fcb54d7ad29e981d0ecf0c9f93. The CNOT CPU/Gloo suite passed 312 tests with 2 skipped in [run 44946584249](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249), artifact [14723688156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249/artifacts/14723688156), SHA-256 e1fb89a51013395b43b947a9996f8d68dfbb63311d9b1affe7f52a6beb8868b5. The Python session-owner suite passed in [run 44946584203](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584203), artifact [14723997251](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584203/artifacts/14723997251), SHA-256 18ab19696753ad16767ba8fef0cb388b28941d48e9b8724cffbc1e50171a1cea.

These hosted results establish the current source and installed CPU behavior. The physical CUDA/NCCL job was skipped; the Rust migration and physical 1/2-GPU gate remain open.

## Saturated permutation acceptance, 2026-11-21

The installed-wheel acceptance keeps the public LRX(4), B=64 smoke and adds a
separate LRX(5), B=3 saturated case. An independent CPU oracle checks full
Hash128 deduplication, the existing owner/shard and equal-score cutoff order,
rank-local frontiers, cross-rank duplicates, exact global width and independent
replay. The selected LRX(5) witness has the same ordered winners at world sizes
1 and 2 without changing the production tie policy. Each topology runs twice
inside one Rust session; per-rank traces bind source, wheel, build and binary
hashes, PID, observed CUDA UUID, rank work and bounded frontier snapshots.

Hosted source harness `44943664015` and installed-wheel run `44943664035`
passed on source SHA `bcd92d0a404802604cabf28f7a477ffe41c8d7de`. The wheel gate
ran 11 permutation acceptance contracts, CNOT dtype/admission/Gloo checks and
the public CNOT search with a CPU scorer. Its physical job was skipped; this
evidence does not establish CUDA/NCCL execution. The source and wheel artifacts
are recorded in
`test_results/refactor_20261121/rust-cnot-admission/GITHUB_STATUS.md`.

## Новая физическая приёмка Vast: срок и SSH маршрут, 2026-11-21

Разрешённый физический gate использует только один новый, уже работающий
instance с ровно двумя desktop RTX 3060 и остаётся в остатке лимита $10.
Старый instance `61080396` исключён. Контракт требует выбрать целиком одну
пару API полей SSH: `ssh_host`/`ssh_port` либо `public_ipaddr`/`ports[22/tcp]`.

Бюджетный deadline рассчитывается по времени создания instance и фактической
ставке. Preflight, архивация исходников, передача, сборка/поиск и выгрузка
результатов повторно считают оставшееся время от этого одного deadline.
Каждая фаза оставляет резерв остановки; сборка и поиск дополнительно оставляют
окно для извлечения receipt и частичных результатов. При нехватке времени
следующая фаза не стартует. Остановка подтверждается свежим API readback.

Hosted CPU harness `44913195522` подтвердил выбор endpoint-пары, точную
арифметику срока, пересчёт budget для каждой фазы, отказ при нехватке времени
и сбор частичных результатов после ошибки. Installed-wheel и GPU job в этом
push-run были `skipped`.

Физический gate пока не готов к запуску. Workflow ещё получает SHA manifest
GL4 с целевой машины во время preflight вместо сверки с доверенным staging
receipt. До появления проверяемого pin и нового валидированного предложения
2× desktop RTX 3060 физический dispatch запрещён. Hosted проверки не являются
физической CUDA/NCCL проверкой.

Контракт исходников теперь находится в
`integrations/cayleypy_native/multigpubeamsearch/native_source_manifest.json`.
Vast packager создаёт source archive из объявленных roots и отклоняет
отсутствующие обязательные файлы. Wheel source bundle проверяет те же пути,
даже если generated file manifest пропустил один из них; native fingerprint
включает объявленные roots и vendor headers. Hosted RED `44914169925`
подтвердил отсутствие packager, а `44914169994` — отсутствие vendor headers в
fingerprint. Hosted source harness `44916537646` прошёл упаковку, распаковку и
negative fixtures для обоих headers. Installed-wheel run `44916537642` прошёл
проверки двух vendor fingerprint и обязательного bundle header. Их GPU job
был `skipped`; платный запуск не выполнялся.

Remote preflight также проверяет API CayleyPy до сборок. После установки wheel
worker проверяет NCCL header/library и готовит pinned CUTLASS через source
cache с проверкой SHA до нативной GPU-компиляции. Путь CUTLASS передаётся
permutation smoke явно, поэтому поиск не должен запускать скрытую загрузку
зависимости. Для этих проверок hosted GREEN ещё не получен.

## CNOT SIGINT handling across ranks, 2026-11-20

Rust CNOT stage checks now poll Python signals at checked-stage boundaries
before entering the existing Gloo control-group error vote. If any rank reports
a `KeyboardInterrupt`, all ranks receive the rank diagnostics and raise
`KeyboardInterrupt`; ordinary failures remain coordinated
`RuntimeError` values. The Rust root-goal stage also records a signal as a
rank-local failure and includes it in its existing goal/error exchange.
The Python compatibility fallback preserves the same cancellation exception.

Search order, goal selection, score policy, precision, rank ownership, and
NCCL data-plane operations are unchanged. The two-rank CPU/Gloo regression
injects SIGINT on rank 1 during root-goal checking and verifies a subsequent
Gloo all-reduce still succeeds. This does not verify cancellation inside a
physical CUDA kernel or NCCL data-plane collective. Hardware verification is
still pending.
Evidence: `test_results/refactor_20261120/cnot-cancel-gloo/GITHUB_STATUS.md`.


## CNOT setup failures are coordinated before search, 2026-11-20

The Rust top-level controller runs CNOT identity construction, native-stage
runner loading, and compatibility-stage construction through the existing
checked stage protocol. For multi-rank jobs, every rank votes and shares
diagnostics over the stable CPU/Gloo control group before continuing. Native
stage capability is also agreed through `World.all_ranks_support` on that
control group, not by reducing a vote over the search transport.

This prevents a rank-local setup error from leaving peers to enter a later
collective or search stage alone. Search order, candidate selection, scoring,
precision, owner mapping, and NCCL payload exchange are unchanged. Hosted
CPU/Gloo tests do not establish physical CUDA/NCCL failure recovery.
Hosted CPU/Gloo and installed-wheel receipts:
`test_results/refactor_20261120/cnot-setup-gloo/GITHUB_STATUS.md`.

## CNOT checked errors use the Gloo control group, 2026-11-20

For multi-rank CNOT, the new optional PyO3 capability
run_cnot_native_checked_control coordinates failure votes and diagnostics
over the CPU Gloo control group. The capability is agreed by every rank before
use. Older extensions without it use the Python checked fallback, which follows
the same Gloo protocol; their legacy checked callbacks are not called for error
coordination.

The Rust v1 compatibility checker gathers diagnostic objects on the Gloo
group. The v2 and control ABIs gather fixed-size UTF-8 byte tensors on Gloo.
All control tensors are explicitly CPU-resident. Rank order and error text stay
unchanged. The one-rank path still raises the original exception. Search stages,
NCCL payload exchange, owner mapping, scorer calls, search order, and numerical
policy are unchanged. CPU/Gloo and PyO3 checks do not establish physical
CUDA/NCCL failure recovery.


## CNOT selection and gathered-row preflight, 2026-11-20

CNOT selection now coordinates local preparation failures over the existing
CPU/Gloo control group before ranks enter the next search-data collective.
Preflight covers count and histogram reduction setup, local radix-boundary
updates before another selection pass, selected-record preparation before
gather, and row-gather buffer allocation plus per-source slicing and copy
preparation before broadcasts. Stream 5 fixed gathers allocate the input and
output buffers, validate their shape/device/contiguity, and vote on readiness
before entering the existing NCCL all-gather. The Python compatibility path
uses the same control-stage ordering.

The checks add control-group coordination; they preserve the search order,
winner tie-breaks, scoring, numerical policy, candidate IDs, rank ownership,
and existing NCCL payload and collective order. A rank-local preparation
failure can be coordinated while the Gloo control group is usable. This does
not recover process loss or a failed or indefinitely blocked NCCL communicator.
CPU/Gloo and installed-wheel checks passed; physical CUDA/NCCL behavior remains
unverified.
Evidence: `test_results/refactor_20261120/cnot-collective-preflight/GITHUB_STATUS.md`.


## Rust session identity для beam_width="max", 2026-11-20

Запрошенное значение ширины и разрешённая ширина — разные поля контракта.
Preparation wire сохраняет legacy-ключ `beam_width`, но принимает
`RequestedBeam::AutoMax` как строковый токен `"max"`; числовые запросы
продолжают сериализоваться как раньше. После разрешения `SessionIdentity`
содержит фактическую ширину типа `u64`.

PyO3 публикует `PREPARATION_IDENTITY_AUTO_MAX_VERSION = 1`. Python передаёт
AutoMax в typed Rust preparation identity только при наличии этой capability.
Старые расширения сохраняют прежний Python-owned cache identity и не пытаются
десериализовать строку в `u64`. Алгоритм выбора beam и precision не менялись.

## Межранговый выбор CNOT engine и подготовка Stream 5, 2026-11-20

В compatibility пути CNOT запросы `auto`, `python` и `rust` согласуются на
CPU Gloo control group до engine-зависимой ветки. Если ранги запросили разные
режимы, каждый завершает setup ошибкой до поискового конвейера. Общий запрос
`auto` выбирает Rust только при общей поддержке; иначе все ранги переходят
на существующий Python compatibility путь.

Stream 5 обменивает счётчики, проверяет ограничения и заранее выделяет
receive backing и пустой send placeholder. Затем все ранги выполняют OR-vote
готовности через Gloo CPU control group. Ошибка выделения на одном ранге
останавливает payload обмен на всех рангах. Существующий NCCL payload,
форматы записей, выбор кандидатов и оценки не изменены. Python fallback
использует тот же порядок проверок и readiness vote.

Hosted evidence находится в
`test_results/refactor_20261120/cnot-runtime-failure-safety/GITHUB_STATUS.md`.
CPU/Gloo и wheel проверки прошли. Физический CUDA/NCCL запуск не подтверждён:
проверка Vast увидела остановленный instance и не меняла его состояние.


## Rust-владелец CPU CNOT first-unseen, 2026-11-20

Для typed Rust compatibility path CPU-дедупликацию CNOT теперь выполняет
beam-core::cnot::first_unseen_indices. Результат сохраняет первый child index
в generation order; все logical words сравниваются после маскирования только
неиспользуемых tail bits. Совпадение хеша используется лишь для выбора bucket
и подтверждается полным сравнением матриц. CUDA kernels и Python-only engine
не меняются. Hosted CPU/PyO3 evidence: test_results/refactor_20261120/cnot-rust-first-unseen/GITHUB_STATUS.md.

## Rust-владелец CPU CNOT child application, 2026-11-20

Для typed CNOT compatibility path пакетное применение физических ходов к
CPU-состояниям выполняет beam-core::cnot::apply_gate_words. Формат packed
матриц, control-major ID, XOR полной строки, tail mask и результат PyTorch
int64 сохраняются. CUDA child kernels и Python-only fallback не меняются.
Результаты hosted CPU/PyO3 проверки: test_results/refactor_20261120/cnot-rust-cpu-children/GITHUB_STATUS.md.
Этот путь не подтверждает физическое выполнение CUDA/NCCL на нескольких GPU.

## Rust-владелец CNOT CPU/legacy batch fallback, 2026-11-20

Typed `PyCnotSearchInput` compatibility dispatch больше не вызывает
высокоуровневые `cnot._expand`, `cnot._distinct` и `cnot._scores`.
При отсутствии соответствующего fused CUDA hook Rust вызывает существующий
`cnot_native_batch`: он сохраняет порядок scoring, выбор ходов,
generation-order deduplication, score validation и формы выходных tensors.
При наличии fused CUDA hooks они остаются приоритетным путём.

Для typed Rust compatibility path CPU-расширение CNOT теперь применяет
упакованные операции через beam_core::cnot::apply_gate_words; физический
алфавит ходов, XOR контрольной строки и очистка tail padding сохранены.
CUDA dispatch и Python-only engine продолжают использовать прежний
cnot_children путь. CPU first-unseen в typed Rust compatibility path теперь использует
beam-core::cnot::first_unseen_indices: он маскирует tail padding, хранит
первый индекс в порядке генерации и подтверждает совпадение сравнением всех
packed words. CUDA dispatch и Python-only engine сохраняют прежний
cnot_first_unseen путь. Typed CPU owner assignment теперь вычисляет прежний
position-dependent hash в beam-core и возвращает те же owner ranks. Python-only
CPU/reference fallback остаётся доступен, а CUDA dispatch сохраняет прежний
C++/CUDA owner kernel. pack_cnot_route больше не вызывается из
typed Rust compatibility path: Rust проверяет source rank, dtype и диапазон
move ID, затем собирает тот же int64 route lane.

Python-only engine и старые stage providers сохраняют fallback. Алгоритм
поиска, scoring, precision, state layout и transport не менялись.

Hosted CPU suite сравнивает qhat/proposal с Python эталоном и запрещает
высокоуровневые helper callbacks. Он не заменяет физическую CUDA/NCCL
multi-GPU проверку.
Результат: `test_results/refactor_20261120/cnot-rust-domain-fallback/GITHUB_STATUS.md`.

## Rust-владелец CNOT selection, 2026-11-20

Для typed `PyCnotSearchInput` compatibility adapter вызывает существующий
`cnot_native_selection::select` напрямую и записывает его результат в
`ranks`, `local` и `new_lex`. Он больше не создаёт Python
`SelectionStages` и не вызывает `prepare_selection/commit_selection`.
Python-only input и старые stage providers сохраняют callback fallback.

Реализация использует тот же Rust radix schedule, score/word keys, существующие
CUDA selection operators или прежние Torch fallback операции, process-group
reductions, stable ordering и размер глобального beam. Алгоритм, scoring,
точность, transport и collectives не менялись. Проверка с width=5 подтверждает
cutoff и совпадение audit слоёв с Python эталоном для qhat/proposal. Проверка
hosted CPU/PyO3 не подтверждает физический CUDA/NCCL multi-GPU runtime.
Результат: `test_results/refactor_20261120/cnot-rust-selection/GITHUB_STATUS.md`.

## Rust-владелец CNOT redistribution, 2026-11-20

Для типизированного `PyCnotSearchInput` compatibility adapter теперь
выполняет orchestration стадии redistribution в Rust. При наличии полного
набора существующих CUDA hooks используются `winner_packet`,
`sort_received` и `frontier_views`; иначе Rust собирает тот же пакет
Torch-операциями и сортирует принятые записи прежним ключом. Обмен вызывает
существующий `cnot_native_exchange::exchange`. Python callback остаётся для
Python-only input и старых stage providers.

Сохранены состав и порядок wire-записей, обновление `closed` до обмена,
порядок collectives, статистика, поля audit, правило пустого локального
beam и разбиение history/parents/lex. Алгоритм поиска, транспорт и score
не менялись; rank-wide error negotiation не добавлялся. Аппаратный CUDA/NCCL
multi-GPU путь этой CPU проверкой не подтверждён.
Результат: `test_results/refactor_20261119/cnot-rust-redistribute/GITHUB_STATUS.md`.

## Rust-владелец CNOT deduplicate/score, 2026-11-19

Для типизированного `PyCnotSearchInput` Rust compatibility adapter теперь
выполняет dispatch distinct-state и scoring, затем назначает `children`,
`meta` и `values` результату. GPU путь использует прежние CUDA-операторы
`distinct_candidates` и checkpoint/model scorer; CPU или библиотека без
этого optional operator сохраняет `cnot._distinct`. Ошибки обрабатываются
прежним rank-wide протоколом `checked_v2`/legacy `checked`.
Python-only и старые stage providers сохраняют callback fallback.

Полное сравнение CNOT состояний, стабильный порядок кандидатов, метаданные,
оценки и порядок collectives не менялись. Новые копирования не добавлены.
Проверки hosted CPU/PyO3 не подтверждают физический CUDA/NCCL multi-GPU runtime.
Результат: `test_results/refactor_20261119/cnot-rust-dedup-score/GITHUB_STATUS.md`.

## Rust-владелец стадии CNOT expansion, 2026-11-19

Для типизированного `PyCnotSearchInput` Rust compatibility adapter теперь
сам вызывает stage expansion и распределяет результат в `candidates` и
`owners`. GPU-режим повторяет существующую связку scorer → `expand_candidates`;
CPU и CUDA-библиотеки без этого оператора сохраняют прежний helper
`cnot._expand`. Ошибки стадии используют тот же rank-wide выбор
`checked_v2`/legacy `checked`. Python-only и старые stage providers сохраняют
callback fallback.

Формат кандидатов, владельцы, диапазон ходов, scoring, последовательность
collectives и публичный API не изменились; новые копирования не добавлены.
Проверки hosted CPU/PyO3 не являются проверкой физического CUDA/NCCL runtime.
Результат: `test_results/refactor_20261119/cnot-rust-expand/GITHUB_STATUS.md`.

## Rust-владелец инициализации поиска CNOT, 2026-11-19

Для типизированного `PyCnotSearchInput` совместимый Rust-`PyStages`
теперь напрямую выполняет исходную настройку CNOT-поиска. На ранге 0
начальный пакет родителей ссылается на корень, на остальных рангах это
пустой срез корня; начальный `lex` содержит 0 только на ранге 0.
Существующий `cnot_owners` определяет владельца корня. Ошибка этой операции
согласуется между рангами через прежний выбор `checked_v2`/legacy
`checked`; Python-провайдеры без совместимого typed input сохраняют вызов
`CnotStages.initialize`.

Изменён только dispatch инициализации. Правила поиска, rank ownership,
порядок обменов и публичный API не менялись. Hosted CPU/PyO3 проверка прошла;
физические CUDA/NCCL multi-GPU проверки не выполнялись. Результат:
`test_results/refactor_20261119/cnot-rust-initialize/GITHUB_STATUS.md`.

## Rust-владелец выбора профиля инференса, 2026-11-19

`beam-core` реализует статистический выбор профиля с прежними правилами
отклонения, оценкой median/MAD, детерминированным bootstrap, восстановлением
baseline и порядком разрешения равенств Python. PyO3 принимает обычные
`Measurement`; старое расширение и неподдерживаемые Python-значения сохраняют
Python fallback. Алгоритм поиска, оценки, precision, GPU-стадии и транспорт
multi-GPU не менялись. Проверка CPU с PyO3 не является аппаратной проверкой
CUDA/NCCL. Результат: `test_results/refactor_20261119/rust-inference-selection/GITHUB_STATUS.md`.

## Rust-владелец чтения shared inference profile registry, 2026-11-22

Чтение source-pinned `profiles/index.json` и hash-addressed record перенесено в `beam-runtime::shared_profiles::find_profile`. Rust ограничивает размер index/record двумя MiB и индекс 1024 записями, проверяет совпадение compatibility identity, lowercase SHA-256, нахождение record внутри registry, schema, verified evidence и диапазон `parent_batch`.

PyO3 возвращает только проверенную запись и её digest. Если новый extension вернул отсутствие совпадения, Python не повторяет чтение в обход Rust-проверок. Для старого extension без этого метода сохранён прежний Python reader. Формирование hardware/model compatibility identity пока остаётся Python адаптеру; найденный профиль остаётся предложением и проходит обычную проверку target runtime. Search, scoring, kernels, precision и transport не менялись.

На code SHA `0ca418cbbc2757508dc96ff986af75b888cba5e2` installed wheel прошёл в [run 44999384648](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384648); Rust CPU/Gloo прошёл в [run 44999384690](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384690), source harness — в [run 44999384719](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384719), panel-v2 CPU compile/handoff — в [run 44999387783](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999387783). Физические CUDA/NCCL проверки пропущены.

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
Python-адаптер остаётся доступен для совместимого старого пути. Новый
Rust-контроллер CNOT напрямую вызывает существующие Rust-помощники для проверки
согласованности входа между рангами и голосования о доступности нативных стадий.
Низкоуровневые `longs`, `gather_fixed` и `sum` по-прежнему используют process
group вызывающего кода, сохраняя текущую топологию и семантику коллективов.
Rust-контроллер теперь также напрямую вызывает существующий Rust helper настройки Stream 5. Он сохраняет тот же выбор `RustStream5`/`Stream5`, коллективную проверку доступности и назначение `world.transport`; Python `World.start_transport` остаётся для совместимости со старым расширением. Объект транспорта и прежние CUDA/NCCL операции не изменены. Другие тензорные операции и адаптеры пока
остаются на прежних границах; перенос не меняет поиск или численные правила.
Подготовка CNOT-записей к межранговому обмену на CUDA также выполняется в
C++ ATen с прежней стабильной группировкой по владельцу; однопроцессный путь
не изменён.
В новом бэкенде Rust управляет подготовкой и отправкой записей: прежний
C++ ATen-оператор либо тензорная резервная последовательность формируют те же
непрерывные пакеты. При доступном транспорте используется тот же Stream 5;
CPU/Gloo сохраняет прежний порядок двух all-to-all, включая пустые отправки.
Сохраняются группа процесса вызывающей программы, отдельный NCCL Stream 5,
порядок обменов, полный packed-CNOT и сравнение всех слов. Python-scorer и
checkpoint-провайдер не меняют интерфейс или численную политику. GPU-адреса
не превращаются в ссылки Rust и не передаются между процессами.

Rust-владелец библиотеки и NCCL-коммуникатора CNOT Stream 5 управляет обменом
счётчиков и записей для Rust-поиска. Ранги выбирают этот транспорт совместно;
если хотя бы на одном нет подходящего ABI, все используют прежний транспорт.
Версионированный ABI сохраняет устройство инициализации, восстанавливает
текущее устройство при закрытии и остаётся привязанным к создающему потоку.
PyTorch владеет CUDA-буферами, а FFI проверяет тензоры и отмечает их stream;
Rust-контроллер не получает GPU-указатели. Порядок обмена NCCL и исходный
Python-транспорт сохраняются.

## Приёмка CNOT ABI и фазовых ошибок, 2026-11-21

Строгий acceptance-путь согласует до обмена наличие и идентичность Rust
контроллера, checkpoint scorer, DeviceDomain, CNOT ABI/capabilities и Stream 5
на всех рангах. Ошибка ABI, отсутствующая обязательная реализация или разный
набор возможностей завершаются коллективной ошибкой; обычный совместимый
fallback сохраняется вне строгого режима. Dtype входа проверяется по
каноническим объектам `torch.dtype`, включая совместимые `int64`/`uint16`
пути. Подготовка winner packet и history edge получает ранговое голосование
до следующего data-plane collective.

На source SHA `78087abe5c1b8e794c9e16aa934f6afe084c0f8b` hosted installed-wheel
проверки прошли: PyO3 dtype с deprecation warnings как errors, strict admission
на двух Gloo-рангах, инъекции ошибок winner-packet/history-edge на каждом
ранге с повторным поиском в той же группе, публичный CNOT search и Actionlint.
Проверки hosted source harness также прошли. Это CPU/Gloo evidence; физические
CUDA/NCCL и trained-checkpoint вызовы ещё не подтверждены.

Граница миграции остаётся частичной: Rust задаёт последовательность стадий и
владеет историей поиска, но `NativeStages` всё ещё удерживает Python `World`,
scorer и Torch-тензоры, а часть стадий вызывает Python/PyTorch. Полный runtime
контекст CNOT в `beam-runtime` и самостоятельное владение вычислительным
провайдером остаются следующими вертикальными срезами. Текущие проверки и
hardware gate: `test_results/refactor_20261121/rust-cnot-admission/GITHUB_STATUS.md`.

## Rust-контракт CNOT checkpoint scorer, 2026-11-19

`beam-native-sys::TqpScorer` разбирает metadata scorer при открытии,
проверяет наличие всех 102 зарегистрированных `feature_ids` и целое
`max_packet_rows`. Rust вычисляет допустимый размер inference batch и
закрывает нативный handle при ошибке metadata. PyO3 по-прежнему создаёт
публичный Python `metadata`-словарь; ABI scorer, порционирование вызовов,
модель, порядок выходов и численная политика не меняются. Hosted CPU/PyO3
проверка не подтверждает физический CUDA/NCCL runtime. Результат:
`test_results/refactor_20261119/cnot-scorer-metadata-contract/GITHUB_STATUS.md`.

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

### Slurm rank mapping, 2026-11-22

Перед созданием rank-owned каталогов Python и Rust проверяют, что
`SLURM_PROCID = sum(SLURM_STEP_TASKS_PER_NODE[..SLURM_NODEID]) + SLURM_LOCALID`.
Это сохраняет раздачу непрерывных глобальных рангов при неодинаковом числе
задач на узел и отклоняет противоречивое окружение до запуска дочернего
процесса. На source SHA `901a7c630d6be25555fe7a8154bd9bb45b537f33`
hosted CNOT CPU run [44959505649](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959505649)
прошёл; Rust contract и Python compatibility-набор зелёные. Installed-wheel
run [44959505809](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959505809)
прошёл. RED run [44959263103](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959263103)
подтвердил принятие неправильного rank до исправления. GPU acceptance
[44959505792](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959505792)
прошёл только hosted CPU harness; физический GPU шаг был пропущен.

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
одного процесса и устройства. Вызывающий отвечает за реальные размеры,
непересечение и срок жизни хранилищ. ABI проверяет заявленные длины и арифметику
смещений, но не устанавливает физические границы аллокаций или устройство.
Python при необходимости создаёт непрерывную копию входа; сам пакетный вызов
не добавляет копию. Переход CNOT-цикла и коллективных операций в Rust остаётся
отдельной незавершённой задачей.

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
Граница перестановок учитывает кратности меток и исключённые решённые состояния;
она не доказывает размер достижимой орбиты. Прежний параметр
calibration_frontier_max_states ограничивает генерацию калибровочной выборки и не
уменьшает числовую или максимальную ширину поиска. Отдельная опциональная граница
низкоуровневого задания применяется только при её явной передаче в descriptor.
После изменения микробатча или резерва память допускается заново; максимум три
цикла уточнения. Рабочая ширина сравнивается со свежим шагом на половинной ширине,
с тем же профилем и минимум одним полным микробатчем на каждом ранге. Требуется
сохранение не менее 80% пропускной способности; максимум три сравнения. Это
наибольшая подходящая измеренная ширина, а не доказанный эффективный максимум.
`SearchSession` владеет совместимостью ресурсов, процессами рангов, общей отменой
и единым дедлайном подготовки/поиска. Типизированный граф и потоковый разбор
результатов в `beam-core` проверяют путь, полную коллекцию и точную ширину.
Изменение модели/геометрии освобождает старую группу; старт не входит в ключ
совместимости. Крупные PyO3-вызовы поиска и калибровки проверяют Python-сигналы,
передают отмену Rust-владельцу и дожидаются освобождения его процессов.
Для перестановочного `SearchSession` Rust также сверяет объявленный `graph_hash`
с SHA-256 канонического представления центра и упорядоченных генераторов в
формате CayleyPy. Начальное состояние и имена генераторов в него не входят.
При несовпадении задание отклоняется до запуска рангов или обработки тривиального
случая, а уже существующая группа освобождается. Хеш вычисляется потоково, без
второй полной JSON-копии графа. При запуске новой Rust-сессии её проверенный хеш
передаётся каждому работнику через обязательное расширение ABI v1. Перед загрузкой
модели и GPU каждый нативный ранг потоково вычисляет тот же хеш по фактически
загруженным центральному состоянию и упорядоченным генераторам, исключая padding,
и прекращает подготовку при несовпадении. Имена генераторов в хеш не входят.
Работник подтверждает поддержку при запросе возможности; библиотека без ABI-
расширения выдаёт ошибку. Прежний ответ на обычный запрос протокола сохранён.
В привязанной сессии целевое состояние должно совпадать с центром; отдельный
режим repair не допускается через этот контракт. CNOT не затронут.
При подготовке перестановочного native runner генераторы и имена разбираются из
одной последовательности считанных байтов. `NativeMoves` владеет обоими
результатами до завершения runner; внутренние ссылки на векторы не переживают
владельца. Проверка биективности использует индексируемый по позиции массив;
размер результата резервируется заранее. Дополнительного копирования векторов
при передаче внутри runner нет, а необходимая копия таблицы генераторов на GPU
сохраняется. Generator padding содержит тождественные индексы и не меняет
требование нулевого persistent padding состояний. Одно чтение не гарантирует
атомарность относительно внешней записи в тот же файл; Rust-сессия проверяет
именно получившиеся загруженные данные. Непривязанный нативный CLI сохраняет
прежнее поведение. Свой произвольный worker и его ответ о возможностях не
являются аттестацией кода: гарантия относится к соответствующему Rust worker и
нативной библиотеке, используемым управляемым публичным запуском.
Для CNOT отдельно доступен `validation="rust"`: PyO3 заимствует байты только на
время создания объекта, после чего Rust владеет матрицей из строк little-endian
64-битных слов. Проверяются геометрия 2–256, нулевой хвост и невырожденность в
GF(2); временный буфер исключения занимает не более 8192 байт. Возвращённые
физические номера ворот воспроизводятся от единичной матрицы в порядке построения;
сравниваются все слова, без использования хеша как признака равенства. Все ранги
обязаны выбрать одинаковый валидатор; старая библиотека отклоняется без fallback.
По умолчанию остаётся Python-валидатор. Этот этап не переносит цикл поиска CNOT,
его collectives и checkpoint scorer. При запуске CNOT сначала создаётся отдельный
непрерывный тензор на исходном устройстве: и проверка, и сам поиск используют этот
снимок, а не дальнейшие изменения тензора вызывающей стороны. Копия занимает не
более 8192 байт при 256 проводах. Вызывающая сторона должна синхронизировать
сторонние записи во время создания снимка; атомарность копирования относительно
другого GPU-потока не гарантируется.
Экспериментальный `NativeSession(engine="rust")` подключает Rust-владельца
рангов и проверку результатов к существующему публичному `beam_search`.
Проверка формата и экспорт модели пока остаются в Python; обычный
`NativeSession()` сохраняет прежний путь. Требуется управляемая сборка runner,
соответствующие PyO3-модуль и Rust worker. Внешний `runner_path` не поддержан.
Проверенный Rust результат передаётся в публичный адаптер один раз и привязан к
каталогу именно этого задания; ошибочный следующий вызов не раскрывает старый
результат. Rust повторно проигрывает путь и проверяет ширину и подтверждения
рангов. Python сохраняет прежний формат ответа, но не разбирает тот же terminal
log второй раз. Для этого требуется `SEARCH_TERMINAL_RESULT_VERSION=1`; старый
PyO3-модуль отклоняется до сборки нативной библиотеки. Динамические поля terminal
metadata не переносятся из предыдущего задания в текущее.
Необязательный `experimental_parallel_nccl_identity=True` принадлежит сессии:
обычный SHA-256 библиотеки сохраняется вместе с поблочными SHA-256; перед каждым
повторным использованием все байты библиотеки считываются и хешируются заново,
а заголовок NCCL проверяется отдельно. Проверка совпадения блоков предполагает
стойкость SHA-256 к коллизиям; обнаруженное во время чтения изменение файла
вызывает ошибку. Вариант по умолчанию выключен; на синтетическом профиле с двумя
RTX 3060 проведены пять пар повторных полных вызовов. Отдельный полный шаг GPU и
другие профили ещё не проверены; вариант не меняет вычислительный конвейер или
точность инференса. Результаты — в `test_results/refactor_20261114/STAGE_NCCL_IDENTITY.md`.
Для подготовленного runner включение этого варианта требует доступности заголовка
и библиотеки NCCL и их совпадения с записанными при сборке хешами.
Экспериментальный `NativeSession(experimental_reuse_model_snapshot=True)` для
явного MLP `NativeModel` в режиме `backend="native"` сохраняет один проверенный
снимок весов между вызовами; исходные и сохранённые байты повторно проверяются.
Без копирования работают только совместимые уже загруженные worker; при их
перезапуске создаётся новый снимок. Каталоги снимков сохраняются после закрытия.
Для экспериментальной сессии с MLP/CUTLASS каждый новый rank вычисляет тот же
SHA-256 по собственным загруженным host-буферам и тем байтам manifest, которые
использовались для разбора конфигурации. Имена тензоров упорядочены как в Python;
хеширование использует ограниченный scratch, без второй полной копии весов.
До возврата результата контроллер требует подтверждение от каждого rank. При
повторном вызове совместимой сессии сохраняются уже загруженные GPU-веса;
подтверждение относится к первоначальной загрузке, а не к повторному хешированию
GPU-памяти. Старый внешний runner без подтверждения в этом режиме отклоняется.
Проверка подтверждения происходит до возврата результата, но старый внешний
runner может выполнить первое задание и записать его историю до отказа.
Для LibTorch, ensemble и обычного запуска эта привязка пока не действует;
их файловая загрузка не связывает атомарно проверку и последующее чтение.
Новое связывание прошло CPU-тесты и ограниченные проверки на одной и двух
RTX 3060 с Python- и Rust-владельцами сессии. Проверка других GPU, моделей и
полного шага остаётся открытой: `test_results/refactor_20261114/STAGE_MLP_LOADED_BUFFER_BINDING.md`.
Для `engine="rust"` с соответствующим PyO3-модулем Rust теперь владеет записью
снимка, его ключом и повторным использованием: хеширует и копирует файл
ограниченным буфером, проверяет записанные байты и публикует подготовленный
каталог. Python проверяет manifest, размеры и SHA-256 исходной модели; по
умолчанию PyTorch проверяет конечность значений. Отдельный экспериментальный
`experimental_rust_mlp_finite=True` в сочетании с `engine="rust"` и повторным
использованием снимка передаёт неизменяемые блоки FP16/BF16 как заимствованные
байты в Rust и проверяет экспоненту без копии блока в `bytearray` и создания
тензора. Чтение файла, хеширование и проверка размеров сохраняются. Флаг требует
little-endian и подходящий PyO3-модуль; при их отсутствии явно сообщает об ошибке.
Проверка применяется только к явному MLP `NativeModel` в режиме `native` и не
меняет арифметику инференса. Она выключена по умолчанию: пять парных полных
вызовов дали медианное изменение −2,01%, меньше порога включения; память и полный
GPU-шаг отдельно не измерялись. См. `test_results/refactor_20261115/STAGE_RUST_MLP_FINITE.md`.
Перед запуском Rust сверяет тот же SHA-256 и каждый размер, полученный из этой
проверки. Размер manifest и каждого тензора обязателен: общий хеш конкатенации
сам по себе не фиксирует границы файлов. При изменении геометрии графа, старом
PyO3-модуле без size-bound API или отсутствии привязки используется полная
Python-проверка. Сессия включает неизменяемый снимок центра и генераторов в
идентичность worker, поэтому их изменение вызывает перезапуск. На Linux
публикация использует атомарный `RENAME_NOREPLACE`; при отсутствии его поддержки
файловой системой операция завершается ошибкой. Поведение сетевых каталогов
Slurm не проверено. Отмена кооперативная: запрос после последней проверки может
совпасть с успешной публикацией. Снимок остаётся экспериментальным и выключен
по умолчанию; тесты и границы измерений описаны в
`test_results/refactor_20261114/STAGE_RUST_MLP_LAUNCH_VERIFY.md`.
Перенос общего запуска и CNOT-управления, CUDA-оптимизации и переключение
Python API по умолчанию остаются следующими этапами.
Наличие workspace не означает завершения перехода.

Полная 128-битная radix-сортировка CUB для Stream 3/4 и финального A/B union
доступна только через экспериментальный флаг сборки. Порядок ключа — {hi, lo};
сортировка и редукция используют раздельные буферы, размер scratch запрашивается
у выбранной операции. Автоматическая сборка явно оставляет флаг выключенным:
на проверенном профиле 2xRTX3060 полная глубина замедлилась. Отдельный вариант
Stream 4 сортирует полные хеши и индексы; на одной конфигурации он снизил
измеренный с опросом пик памяти без медианного замедления. Явный параметр
`NativeOptions(experimental_index_sort=True)` выбирает этот вариант в публичной
управляемой сборке и привязывает его к ключу кэша runner и Rust-библиотеки.
На двух RTX 3060 публичный путь с CUTLASS и LibTorch прошёл ограниченные
проверки Python/Rust-сессий, включая смену этого параметра; данные и границы
парного замера находятся в `test_results/refactor_20261114/STAGE_PUBLIC_INDEX_SORT.md`.
Автоматический выбор для проверенного профиля пока не включён.

Отдельный CMake-эксперимент `BEAM_EXPERIMENTAL_DIRTY_MERGE` выключен по
умолчанию. После стабильной threshold-компакции отсортированный и уникальный
clean-префикс объединяется с radix-сортированным dirty-суффиксом по полному
Hash128; равные ключи clean идут первыми, затем действует прежний выбор по
оценке, родителю и маршруту. Полная metadata неподвижна до сборки победителей;
индексы занимают часть существующего временного буфера. Сортировка заполняет
хвост sentinel-значениями и использует стабильность CUB; действительный
максимальный Hash128 остаётся допустимым. Число clean/dirty и выбор размера
сортировки условного графа остаются на GPU. Размер scratch запрашивается у CUB
для каждого варианта и повторно используется только в порядке потока. Финальное
A/B-объединение не меняется. Выбор не включён автоматически: на проверенном
полном шаге выигрыш ниже порога, а проверки sanitizer для условного графа
оставили открытые сигналы. Границы и логи — в
`test_results/refactor_20261114/STAGE_STREAM4_DIRTY_MERGE.md`.
Дополнительная диагностика воспроизвела bounded `synccheck` и после исключения
сортировки одного элемента; кандидат не принят. Причина сигналов остаётся
неизвестной; минимальные варианты графа не являются доказательством безопасности
Stream 4. Подробности — в `test_results/refactor_20261115/STAGE_STREAM4_SYNCCHECK.md`.

Экспериментальный Stream 4 scan использует CUB DeviceScan для счётчиков блоков и
BlockScan для стабильной компакции. Число записей остаётся на GPU. Размер scratch
запрашивается для ёмкости и размеров условных графов; в одном потоке scan, sort и
reduce последовательно используют один scratch после завершения предыдущих читателей.
Счётчики, offsets и выходной total не пересекаются. Автоматическая сборка оставляет
этот вариант выключенным до прохождения профильного критерия.

Экспериментальный Stream 3 owner radix группирует кандидатов стабильной сортировкой
по рангу с отдельным sentinel; число разрядов включает sentinel даже при 256 рангах.
Он использует существующие A/B буферы разбиения и scratch этого же потока, затем
публикует непрерывные диапазоны отправки и локальный массив. Исходный candidate ID
по-прежнему определяет родителя и ход. Путь с одним GPU остаётся отдельным;
автоматическая сборка не включает эксперимент до профильной проверки.
При переполнении счётчика до обмена записываются требуемое и доступное число
записей в существующий fatal trace; предыдущая ошибка сохраняется, а выходные
счётчики обнуляются. Этот путь не урезает поиск молча.

Экспериментальный bounded score radix сортирует только разряды, необходимые для
представления всех score bins и sentinel; расчёт CUB scratch использует ту же
ширину. Для текущей конфигурации это 19 бит. Hash128 всегда остаётся полным.
Автоматическая сборка оставляет этот вариант выключенным до профильной проверки.

При создании GPU-арены объединение фактически выделенных диапазонов CUB scratch
инициализируется один раз до запуска вычислительных потоков и CUDA Graphs.
Диапазоны перекрывающихся фаз не очищаются повторно; до передачи арены в другие
потоки завершение инициализации явно ожидается. Пользователь низкоуровневого
API, передающий собственный scratch, отвечает за его инициализацию и срок жизни.
Ошибка построения разметки или инициализации освобождает частично выделенную
арену; отказ в повторном выделении не изменяет уже существующую арену.

Владение буферами, события переиспользования и реестр копирований:
`docs/NATIVE_MEMORY_OWNERSHIP.md`. Результаты каждого этапа сохраняются в
`test_results/refactor_20261107/`; аппаратная проверка не выводится из компиляции.

## Домен CNOT и запуск на нескольких узлах

Для CNOT физический ход — XOR двух строк упакованной матрицы. При 144
проводах состояние занимает 3456 байт, а номер хода доходит до 20591.
Адаптер CNOT хранит матрицы отдельно от перестановочного `State128`,
передаёт 16-битный номер хода и использует общий нативный обмен Stream 5
для записей переменной ширины. Хеш назначает ранг-владелец. Равенство
проверяется по всем словам до оценки детей. Глобальный отбор сравнивает
исходное значение `V`, затем лексикографический порядок цепочки. Описание
алгоритма, этапов обмена и пределов памяти приведено в
[CNOT.md](integrations/cayleypy_native/CNOT.md).

При допуске поиска чтение метаданных scorer и сериализация конфигурации входят
в ту же согласованную между рангами проверку, что и входная матрица. Если один
из этих шагов завершился ошибкой, ранги сообщают её до обмена отпечатками;
дополнительный коллективный вызов к обычному допуску не добавляется.
В новом ранговом backend Rust также координирует проверку цели CNOT: сохраняет
тот же выбор локального кандидата, обмен ключей и передачу ошибок всем рангам.
Вычисление использует существующий C++ ATen-оператор или те же операции Torch,
если оператор недоступен. Владение памятью тензоров остаётся у PyTorch. Старый
Python backend сохранён для совместимых прежних модулей; вызовы scorer и часть
коллективных и тензорных операций всё ещё проходят через Python.
Ранговый backend на Rust также выполняет прежнее голосование об ошибках
инициализации, раскрытия и дедупликации: при ошибке сообщения собираются со
всех рангов до следующего обмена. Проверка допуска задания по-прежнему
использует Python-адаптер.

На кластере ранг коммуникатора адресует обмен и историю, а локальный
номер процесса вместе с маской видимых GPU определяет CUDA-устройство.
Логическая ширина задаётся для всего задания. Примеры запуска нативного
перестановочного движка через torchrun и Slurm приведены в
[CLUSTER.md](integrations/cayleypy_native/CLUSTER.md).

## Логический луч, 2026-10-25

`user_global_beam_width` задаёт точный глобальный top-k.
`global_beam_width_effective` равен ему; выравнивание буферов и shard geometry
не увеличивает семантическую ширину поиска. Физическая ёмкость rank-local
frontier может быть ceil(width/world), но фактическое число состояний
определяется существующим global final exchange; сумма rank quotas равна width.


## Действующая параметризованная специализация, 2026-09-30

Эта секция имеет приоритет над числовыми примерами ниже. Имя `State128`
сохранено как alias `StatePacked`, а не обещание размера128 для всех puzzles.
Источник размера: `cmake/BeamStateSizing.cmake`, `src/config.hpp`, `src/types.hpp`.

| Специализация | STATE_LEN | STATE_STORAGE_LEN | STATE_ALIGNMENT | MOVE_COUNT |
|---|---:|---:|---:|---:|
| Cube4 проверенный native build | 96 | 112 | 16 | 24 |
| Исторический пример ниже | 120 | 128 | 16 | 24 |

Логические байты: `v[0..STATE_LEN-1]`. Persistent padding:
`v[STATE_LEN..STATE_STORAGE_LEN-1] == 0`. Временный `FinalResponse`
хранит target_local_idx в четырёх байтах с `FINAL_RESPONSE_TARGET_LOCAL_IDX_OFFSET`
(равен STATE_LEN); padding очищается перед persistent frontier write.
Числа120..127 и120..123 ниже относятся только к примеру120/128:
для Cube4 это96..111 и96..99. Generator padding — identity; central padding
ноль; padding Zobrist — нулевой Hash128. Конкретные размеры подтверждать
через actual runner `--build-info` и model/input contract, не по имени alias.

`CandidateMeta` остаётся32 bytes/alignment32; `Hash128` — один128-bit key
`{lo,hi}`. Stream3 payload id — original candidate id, не compact index.
Stream4 не использует hash-table dedup, shard top-k или semantic shard cap.

Дедупликация и threshold: physical survivor A/B одного logical shard могут
содержать общий hash. Online lower bound — max(CDF_A,CDF_B), затем сумма
непересекающихся owner/logical domains; суммировать A/B как unique нельзя.
Перед exact final global threshold нужна common A/B threshold/compact/CUB
fixed-temp sort/reduce/compact union с детерминированным winner contract.
Publication A/B histogram slots и physical survivor A/B — разные сущности.

Для WORLD_SIZE>1 scheduling/collective ownership определяется секцией
`2026-05-26 Stream5 threshold update contract`: все ranks участвуют в
request reduction и нужном NCCL histogram SUM в одинаковом порядке.
Исторические local-only periodic правила ниже superseded. Это уточнение
контракта не подтверждает H200-производительность или полноту acceptance.

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
        Stream5 threshold snapshot reads shard_score_hist_active_index (publication slots).
        Online: max(CDF_physical_A,CDF_physical_B) per logical shard, then sum disjoint logical domains.
        Final: sum physical histograms only after a common logical A/B sort/reduce has removed duplicates.
    Stream5 owns threshold collective communication.
    Each rank publishes threshold_request_local as a service field after enough completed local Stream4 work.
    Every rank participates in threshold request reduction at Stream5 exchange points.
    Collective request reduction:
        threshold_request_global = ncclAllReduceMax(threshold_request_local)
    If threshold_request_global != 0:
        every rank builds the conservative local_score_hist from committed Stream4 histogram publications.
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

Automatic eager LibTorch/ensemble physical-pool refinement (2026-10-10):
    RING_COUNT = min(derived RING_COUNT, BEAM_RING_COUNT_LIMIT)
    default BEAM_RING_COUNT_LIMIT = 16, positive uint32 only
This bounds reusable physical buffers and graph executables, not logical
frontier width or shard capacity. Dispatcher frontier_cursor must still reach
every parent by recycling Free rings. Legacy native-graph execution retains the
derived count unless the limit is explicitly configured. Every changed physical
geometry requires fresh native memory admission. The same lower-bound pruning
policy is used in Python; it never replaces native admission or full-depth
performance validation. A pool upper bound is not an overlap guarantee.
Manual configuration with explicit BEAM_RING_COUNT retains its requested
physical count and remains subject to native memory admission.

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

## Multi-rank round admission and receive credits (2026-10-26)

Before launching local Stream3 collection, reserve `B + R` free entries in at least
one non-processing physical bank of every logical shard. `B` is the local Stream3
batch bound; `R` is the admitted remote receive-slot capacity. The automatic
physical capacity floor includes both contributions, not `max(B, R)`.

The existing once-per-round threshold-request collective runs at this admission
point, including ranks with no local send work. If filtering cannot admit the
round, fail before local collection and payload exchange; receiving into a full
bank and waiting for capacity is not a valid execution path.

After local collection, its captured ready-queue may transfer a bank to Stream4
only if a non-processing sibling still has `R` free entries. Stream3 remains the
sole writer; no following local batch launches before this round's receive
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

# Rust/CNOT migration addenda through 2026-11-19

The complete pre-truncation multi-GPU contract is restored above from its parent snapshot. The dated sections below retain later migration changes. Updated descriptions are recorded after the original design where they supersede earlier status text.

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

### Ключ повторного использования Rust-сессии, 2026-11-19

Rust `SearchSession` хранит подготовленный ключ как структуру
`PreparationIdentity` с полями графа, модели, устройств, опций и runtime build.
Совместимое PyO3-расширение собирает и сериализует эту структуру в Rust из
типизированных полей. Python вызывает builder в Rust-owned пути; расширения без
этого optional метода используют прежнюю Python-сборку JSON. Rust сравнивает
десериализованные значения структурно.

Перед Python callback подготовки `NativeSession` вызывает
`SearchSession::begin_preparation`. Совпадающая identity сохраняет resident
rank-группу. При изменении identity Rust закрывает старые workers и очищает
session metadata до следующей подготовки. PyO3 отпускает GIL во время закрытия.
После успешной подготовки Python записывает новую identity через
`remember_preparation`. Старые расширения без нового метода используют
прежний `preparation_matches` и Python fallback. Нативный `SessionIdentity`
отдельно проверяется Rust при запуске поиска.

Rust дополнительно строит и сериализует типизированный `SessionIdentity` для
resident launch через optional PyO3 builder. Расширения без этого метода
сохраняют прежнюю Python JSON сборку. Python пока строит preparation descriptor
и выполняет callback подготовки.

Перенесено только создание launch identity; вычислительные стадии, CUDA/NCCL
операции, порядок поиска и публичные результаты не меняются. Миграция host
orchestration продолжается. Hosted evidence:
`test_results/refactor_20261119/rust-session-identity-builder/GITHUB_STATUS.md`.
CPU/PyO3 проверки не подтверждают физический multi-GPU runtime.
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

## Фиксированный сбор диагностик CNOT, 2026-11-19

Новый optional ABI `run_cnot_native_checked_v2` сохраняет существующий голос
ошибки, а после него передаёт UTF-8 диагностические строки двумя fixed-shape
tensor all-gather: сначала размеры, затем строки, дополненные нулями до общей
длины. Сообщения сохраняют порядок рангов и прежний формат текста. Rust
контроллер вызывает v2 напрямую. Публичный `World.checked` один раз на мир
согласует поддержку ABI между рангами; если хотя бы один ранг не поддерживает
v2, используется прежний ABI/object-gather или его Python fallback. Одноранговый
путь не меняется.

Изменён только failure-only diagnostic transport. Обычные поисковые стадии,
матрицы, scorer, выбор победителя и порядок rank-local вычислений не меняются.
На первом multi-rank `World.checked` добавлен один cached capability vote. Hosted CPU/PyO3 evidence: `test_results/refactor_20261119/cnot-error-message-gather/GITHUB_STATUS.md`; physical CUDA/NCCL error propagation remains unverified.


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

## CNOT optional-extension compatibility, 2026-11-18

`rust_control.extension()` tries both supported PyO3 module names. Missing modules and import/dependency failures become `NativeUnavailable`. The public CNOT `validation="auto"` and `engine="auto"` paths can then retain the Python compatibility implementation; explicit `rust` selections fail closed.

With the matching module, all ranks use the Rust CNOT controller and native stage owner when that capability is present. Packed-matrix GPU operations dispatch to existing C++ ATen/CUDA operators, PyTorch continues to own tensor storage, and arbitrary Python scorers remain supported. Older modules keep the compatibility stage adapter. This update changes extension-load error handling only; it does not change the search algorithm or GPU stages.

Hosted validation covered CPU/control flow and extension build/import. CUDA execution, NCCL runtime, and physical multi-GPU search remain unverified.

## CNOT root identity from the typed Rust input, 2026-11-19

The compatibility `PyStages` backend checks a typed `CnotSearchInput` through
the existing Rust `search_input_is_identity` helper before dispatching the root
goal to Python. This removes the Python stage-method call for Rust-owned typed
inputs. Python-only inputs and older stage providers retain the existing
`root_is_goal` callback. The packed matrix check, tensor ownership, distributed
search order, CUDA operators, scorer calls, and collective sequence are
unchanged.

## CNOT compatibility goal-stage dispatch, 2026-11-19

For a typed Rust `CnotSearchInput`, the compatibility `PyStages` backend now
calls the same Rust `cnot_native_goal::goal` helper as `NativeStages`. It
retains the Python stage-method fallback for Python-only inputs and older stage
providers. The goal predicate, packed tensor operations, error coordination,
rank collectives, stage order, and result semantics are unchanged.

## CNOT typed history-edge broadcast in Rust, 2026-11-19

For typed Rust `CnotSearchInput`, the compatibility `PyStages` backend now
selects and copies the owner-held history record in Rust, then uses the same
`World.broadcast` operation and validates the resulting three-field edge.
Python-only input and older stage providers retain the previous callback.
Record layout, owner rank, broadcast order, path reconstruction, and result
semantics are unchanged.

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

## Current update: Самостоятельный CUDA-домен CNOT, 2026-11-17

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

## Current update: Переход CNOT на Rust, 2026-11-16

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
Python-адаптер остаётся доступен для совместимого старого пути. Новый
Rust-контроллер CNOT напрямую вызывает существующие Rust-помощники для проверки
согласованности входа между рангами и голосования о доступности нативных стадий.
Низкоуровневые `longs`, `gather_fixed` и `sum` по-прежнему используют process
group вызывающего кода, сохраняя текущую топологию и семантику коллективов.
Rust-контроллер теперь также напрямую вызывает существующий Rust helper настройки Stream 5. Он сохраняет тот же выбор `RustStream5`/`Stream5`, коллективную проверку доступности и назначение `world.transport`; Python `World.start_transport` остаётся для совместимости со старым расширением. Объект транспорта и прежние CUDA/NCCL операции не изменены. Другие тензорные операции и адаптеры пока
остаются на прежних границах; перенос не меняет поиск или численные правила.
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

## Current update: Экспериментальный пакетный CNOT scorer, 2026-11-15

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
`beam-native-sys::TqpScorer` теперь также валидирует metadata-контракт из 102
признаков и выбирает effective batch в Rust. `json.loads` используется только
для сохранения публичного Python-словаря metadata; scorer ABI и численная
политика остаются прежними.

## Current update: Rust/C ABI: выполняемый переход, 2026-11-07

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

## Rust-owned preparation result metadata, 2026-11-19

A compatible Rust SearchSession retains the complete result metadata produced by the Python preparation callback together with its typed PreparationIdentity. After the initial terminal outcome is written, Rust atomically finalizes native-outcome.json once callback metadata and persistent-session fields are ready. Repeated searches remove stale terminal fields, merge the current run's terminal record, and return a one-shot outcome bound to its run directory. The Python adapter does not rebuild metadata or rewrite the file on this Rust-owned path. Older PyO3 extensions without this optional finalization method keep the existing Python cache and write fallback. Search semantics, rank order, inference kernels, and multi-GPU exchange are unchanged.

Hosted checks: public-owner CPU tests 79 passed/1 skipped; CNOT CPU compatibility 174 passed/2 skipped; Rust depth-control tests 5 and 2 passed. These checks do not run a physical CUDA/NCCL collective or establish multi-GPU acceptance. Details: test_results/refactor_20261119/rust-preparation-result-metadata/GITHUB_STATUS.md.

## Rust owns native outcome-file persistence, 2026-11-19

On the Rust resident path, backend.py still assembles the public NativeOutcome in memory after parsing the worker result and runtime metadata. It now skips its intermediate file write only when the active NativeSession has a current Rust outcome and the worker exposes remember_preparation_result. The Rust session has already written the initial outcome; after the preparation callback adds persistent-session metadata, Rust atomically finalizes the file.

Without an active capable Rust owner, including standard non-resident calls, Python-owned sessions, older extensions, and Rust sessions without this capability, the Python writer keeps the existing trailing newline. The returned NativeOutcome, search behavior, multi-GPU ordering, CNOT behavior and CUDA stages are unchanged. Hosted RED/GREEN evidence: test_results/refactor_20261119/rust-outcome-file-owner/GITHUB_STATUS.md. Physical CUDA/NCCL multi-GPU execution remains unverified.

## Rust inference candidate ordering, 2026-11-19

The coarse and refinement candidate batch order is now generated in `beam-core` through optional PyO3 methods. Rust implements the CPython-compatible MT19937 sequence used by `random.Random`, including bounded draws and shuffle, so public Python calibration sees the same candidate sequence. `autotune.py` retains the Python implementation for absent or older extensions.

The transferred boundary is only candidate generation. Python still owns calibration orchestration, cache policy, samples, measurements, statistics, ranking, and admission. Inference kernels, numeric behavior, global beam semantics, and multi-GPU rank exchange are unchanged. Hosted run [44543777970](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44543777970) passed Rust/PyO3 CPU checks and 194 Python tests (2 skipped); no physical CUDA/NCCL multi-GPU run was performed. Evidence: `test_results/refactor_20261119/rust-inference-candidates/GITHUB_STATUS.md`.

## Rust inference probe-time allowance, 2026-11-19

The deterministic probe-time estimate is now computed in `beam-core` and exposed through optional PyO3. The public Python helper sends ordinary finite rank-timing lists to Rust and retains the prior Python formula for unusual types or unavailable/older extensions. The arithmetic remains `max(default, 10 * slowest_rank_seconds + 3)` over non-empty timing rows.

Only the helper's ownership changed; the Python calibration loop, candidate order, trial execution, statistical selection, caches, and GPU stages remain as before. Hosted run [44547537516](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44547537516) passed 200 Python tests (2 skipped) and the selected Rust/PyO3 checks. This is not physical CUDA/NCCL or multi-GPU acceptance. Evidence: `test_results/refactor_20261119/rust-inference-probe-time-allowance/GITHUB_STATUS.md`.

## Rust capacity-cache reuse decision, 2026-11-19

Rust now owns the deterministic decision for reusing a capacity-limited inference cache: previous/current GPU UUID keys must match and no current free-memory value may exceed the recorded value by more than 32 MiB. Python still performs the GPU memory query, keeps the non-capacity short-circuit, and falls back to the prior predicate for unusual map values or unavailable/older extensions.

No cache schema, candidate selection, calibration order, GPU kernel, or multi-GPU protocol changed. Hosted run [44550275840](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44550275840) passed 211 Python tests (2 skipped) with CPU/PyO3 checks; no physical CUDA/NCCL run occurred. Evidence: `test_results/refactor_20261119/rust-inference-cache/GITHUB_STATUS.md`.

## Rust CNOT candidate exchange dispatch, 2026-11-19

For typed `PyCnotSearchInput`, Rust `PyStages::exchange_candidates` reads
`world`, `candidates` and `owners` from the compatibility stage object and
calls the existing `cnot_native_exchange::exchange(..., native_packet=true)`.
The returned tensor is assigned to `incoming`. Python-only inputs and older
stage providers keep the existing callback fallback.

The exchange helper retains the established single-rank path, optional CUDA
packet generation, Stream5 transport selection, and PyTorch
`all_to_all_single` fallback. Candidate rows, owner IDs, rank membership,
collective order, and wire semantics are unchanged. No kernel, search rule, or
transport behavior was changed. Hosted CPU/PyO3 checks passed; physical
CUDA/NCCL multi-GPU execution remains unverified. Evidence:
`test_results/refactor_20261119/cnot-rust-exchange/GITHUB_STATUS.md`.

## Rust CNOT construction verification dispatch, 2026-11-19

For typed `PyCnotSearchInput`, compatibility
`PyStages::verifies_construction` now calls the existing
`search_input_verifies_construction` helper. The owned packed matrix remains
the source of truth; Rust `CnotMatrix` checks the reconstructed gate sequence.
Python-only inputs and older stage providers retain the
`CnotStages.verifies_construction` callback.

Only verification dispatch moves into Rust. Path construction, accepted gate
sequences, rank ownership, search semantics and scorer behavior are unchanged.
Hosted CPU/PyO3 checks passed. Physical CUDA/NCCL multi-GPU execution remains
unverified. Evidence:
`test_results/refactor_20261119/cnot-rust-verify/GITHUB_STATUS.md`.

## Rust-owned permutation search task files (2026-11-20)

beam-runtime::search_inputs now writes the legacy puzzle_info.json and test.csv from a Rust-validated PublicGraphContract. The PyO3 boundary rebuilds the contract from normalized input before writing. It preserves Python's existing JSON spacing, synthetic m0 through mN move names in generator order, trailing newline, and quoted single-row CSV format. The Python backend keeps its prior writer for older extensions and legacy contract objects without the Rust fields.

This moves pre-launch task-file serialization only. Search expansion, frontier selection, inference arithmetic, rank setup, collectives, candidate ownership, CUDA kernels, and CNOT are unchanged. Hosted run [44640784456](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640784456) passed the CNOT CPU/PyO3 checks, 240 Python tests (2 skipped), and NCCL probe syntax. Companion depth-control run [44640371591](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640371591) passed. These do not establish physical CUDA/NCCL multi-GPU runtime acceptance. Evidence: test_results/refactor_20261120/rust-search-input-serialization/GITHUB_STATUS.md.

## Rust component pipeline-cache receipt validation (2026-11-20)

The public `component_calibrated` pipeline receipt is now validated by
`beam-runtime::pipeline_cache` through an optional PyO3 API. It checks cache
identity and phase, environment admission, rank-plan coherence, exact beam
width, the complete per-rank timing cohort, and positive finite transfer and
stage measurements. Missing or older extensions retain the Python validator;
full-frontier receipts remain in Python because their bootstrap selector must
remain policy-compatible.

Fresh all-rank native admission and geometry comparison in `read_profile`
remain mandatory after receipt validation. No receipt schema, selected profile,
search control flow, rank ordering, collective, inference arithmetic, or CUDA
kernel changed. Hosted checks and run evidence are recorded in
`test_results/refactor_20261120/rust-pipeline-cache-receipt/GITHUB_STATUS.md`.
Physical CUDA/NCCL multi-GPU execution remains unverified.

## Rust rank-frontier quotas for calibration (2026-11-20)

The Python calibration helper now delegates logical rank quota calculation to
the existing `beam_core::planning::rank_frontier_counts` through optional
PyO3. Rust validates the admitted rank-plan protocol and computes the same
ceiling-difference quotas used by native final exchange; Python retains the
formula for older extensions and non-typed legacy plan shapes.

This changes no beam width, rank membership/order, spare-slot policy,
collective, frontier payload, or CUDA execution. Evidence:
`test_results/refactor_20261120/rust-rank-frontier-quotas/GITHUB_STATUS.md`.
Physical multi-GPU runtime remains unverified.


## Строгая приёмка CNOT-бэкендов и границ стадий, 2026-11-21

Обычный публичный режим сохраняет совместимый выбор CNOT engine и транспорта.
Только профиль `MGBS_CNOT_STRICT_BACKEND=1` запрещает fallback и требует Rust
controller, Rust TQP checkpoint scorer, CUDA DeviceDomain ABI 1 с возможностями
owners/expansion/full-matrix distinct и Rust Stream 5 для мног GPU запуска.
Все ранги сначала согласуют сам флаг strict через существующую CPU/Gloo группу;
при разном включении они завершаются общей ошибкой до NCCL обмена. Затем каждая
ранговая конфигурация сравнивает ABI, возможности и SHA-256 библиотек, scorer,
checkpoint и search manifest до поиска. Strict diagnostics записывают реальные
вызовы стадий после успешного вызова и не копируют CUDA тензоры для трассировки.

Проверки входных CNOT тензоров сравнивают dtype с каноническими
`torch.int64`/`torch.uint16`; это сохраняет отдельный маршрут для 16-битных
направленных ID. Подготовка winner packet и history edge голосуется через Gloo
до соответствующего exchange/broadcast. Следующая глубина и closed frontier
фиксируются только после успешного построения буферов. Эти votes покрывают
ранговые ошибки подготовки, но не восстановление после потери процесса или
ошибки внутри зависшего NCCL collective.

Приёмочный Vast workflow принимает только явный десятичный ID уже запущенного
нового 2× desktop RTX 3060, сверяет владельца/состояние/тариф и считает общий
срок от API `start_date` под лимитом $10 с резервом для подготовки и остановки.
До передачи кода он требует pinned приватный checkout, manifest и checkpoint.
После выполнения JSON receipt связывает source/wheel/scorer/checkpoint/manifest
hashes, события rank/device и readback состояния ресурса. Workflow не создаёт и
не арендует инстанс. Физический прогон остаётся pending до появления доступного
предложения и приватного источника данных.

## Rust validation for calibration rank plans (2026-11-20)

The calibration-side `validate_rank_plans` delegates ordinary typed rank records
to `beam-core::planning::validate_rank_plans` through an optional PyO3 method.
Only exact Python lists of dictionaries with integer values for the recognized
rank-plan fields cross this adapter; unknown fields are omitted. Missing or old
extensions, unsupported Python values, and Rust rejections retain the existing
Python validator and its public errors. This preserves legacy acceptance where
the current Rust contract is stricter, including zero-valued geometry.

The change does not alter beam width, rank ordering, capacity, calibration
selection, collectives, search output, or CUDA execution. Hosted CPU/PyO3
evidence is recorded in
`test_results/refactor_20261120/rust-rank-plan-validation/GITHUB_STATUS.md`.
Physical CUDA/NCCL multi-GPU execution remains unverified.

## CNOT goal failure diagnostics use fixed tensors (2026-11-20)

After the existing rank-wide goal vote reports an error, the Rust goal stage
now gathers diagnostic byte lengths and padded UTF-8 bytes through
`World.gather_fixed`. The existing successful-goal collective and winning
candidate selection are unchanged. The older checked-stage ABI retains its
object-gather fallback; this goal path uses fixed tensors even when
`World.gather_fixed` falls back to the caller's Torch process group.

RED/GREEN CPU/PyO3 evidence is recorded in
`test_results/refactor_20261120/cnot-goal-tensor-errors/GITHUB_STATUS.md`.
No physical CUDA/NCCL multi-GPU goal-error run has been accepted.

## CNOT selection completion errors are coordinated before redistribution (2026-11-20)

After the distributed selection cutoff has finished its own collectives, the Rust
stage now votes on the returned result through the existing Gloo control group
before allowing search to enter candidate redistribution. This reports a
rank-local error in post-selection work to every rank and keeps the next stage
from starting with a missing participant. The check does not cover failures
that occur inside selection before one of its collectives; those still require
separate boundary-level handling.

The change adds a control-group vote at the selection boundary. Candidate
ranking, cutoff width, score ordering, and lexical tie-breaking are unchanged.
The two-rank Gloo regression and installed-wheel evidence are recorded in
`test_results/refactor_20261120/cnot-selection-gloo/GITHUB_STATUS.md`.
Physical CUDA/NCCL multi-GPU execution remains unverified.

## CNOT record exchange preparation errors are coordinated (2026-11-20)

The Rust exchange stage now votes over the existing Gloo control group after
rank-local packet construction and receive-count-buffer allocation, before a
rank can enter the data-plane count exchange. In the fallback all-to-all path,
it also votes after the count exchange and before local receive-size
conversion/allocation can be followed by the payload exchange. Existing
Stream 5 transports keep their own validation and allocation checks.

This guards rank-local failures in preparation around the exchange collectives;
it cannot recover a failure inside a low-level data-plane collective after
some ranks have entered it. Candidate routing, stable packet order, empty
splits, and payload contents are unchanged. The control-flow votes and hosted
evidence are recorded in
`test_results/refactor_20261120/cnot-exchange-gloo/GITHUB_STATUS.md`.
Physical CUDA/NCCL multi-GPU exchange remains unverified.

## CNOT acceptance evidence is invocation-scoped (2026-11-21)

Strict CNOT acceptance records use one versioned JSON schema for Rust and
Python producers. Every event carries the run and case IDs, source and wheel
hashes, rank/world, process ID, device, physical GPU UUID, event kind and
structured fields. The hosted verifier reads only the requested report and
its matching per-rank JSONL receipts, then compares them with the combined
trace. It requires one admission/controller/terminal record per rank, required
DeviceDomain operations, Rust TQP scorer dispatch, and the actual Rust
Stream 5/NCCL phases on every rank of a two-rank run. Backend ABI and artifact
hashes must match from admission through dispatch. A single-rank receipt
declares `single_rank_no_exchange` explicitly.

The installed-wheel CPU workflow exercises the public `verify_cnot.main`
entry under two Gloo ranks. It coordinates scorer setup failure on rank 0 and
rank 1, destroys the entered `World` before process-group teardown, then runs
a healthy verifier search after each failure. The result JSON write is also
checked across ranks. These hosted checks validate control flow and receipt
contracts only; they do not prove CUDA/NCCL execution. The physical gate stays
pending until a matched installed-wheel run on one and two physical RTX 3060s
passes with per-rank backend evidence and independent replay.

For strict acceptance, each worker exposes the selected GPU through its full
physical UUID in `CUDA_VISIBLE_DEVICES`, initializes that CUDA device, then
reads the current context's device UUID through the CUDA driver. Python and
Rust event producers and the result report use the observed UUID and compare
it with the expected rank mapping. Host metadata is gathered over the Gloo
control group. CPU fake-driver checks can validate mapping and rejection
logic, but only physical runs confirm the selected 3060 devices.


## Rust-owned typed CNOT search request, 2026-11-22

The native Rust CNOT controller now creates one `beam_runtime::cnot_search::CnotRequest` during the existing collective admission phase. It owns the wire count, global width, effective shortlist, depth, mode, validation policy and requested engine as typed values. Shortlists are capped to the existing directed-gate alphabet, and the same validated values are used for native stage execution.

Python still supplies the compatible tensor/model adapter and initialized distributed world. The request changes no row-XOR semantics, scoring, precision, kernel, stage order or collective. Hosted Rust/PyO3 and installed-wheel checks are the acceptance boundary for this control transfer; CUDA/NCCL hardware remains separately unverified.


## Rank-consistent CNOT dispatch selection (2026-11-22)

CNOT route selection consumes one snapshot of callable native bindings per rank.
The existing control-group vote intersects those snapshots before the executor
starts a search. A rank-local missing stage therefore cannot cause one rank to
enter a Gloo vote while another enters a different phase. Strict acceptance
rejects an incomplete required native set before the data plane. Public
`auto` selects one shared route, preferring Rust compatibility stages when
available; explicit `rust` returns a shared error if no Rust route is common.
The selected route is passed to execution instead of being independently
resolved again from the user's original engine string.

The installed-wheel test covers missing native callables on either rank,
strict failure before transport, common-route selection, explicit Rust failure,
and a healthy auto retry in the same two-rank Gloo group. The same SHA also
passed wide cutoff parity and separate topology/deduplication/tie fixtures.
These are CPU/Gloo and installed-wheel checks; physical CUDA/NCCL acceptance
remains pending.



## Panel-v2 handoff byte binding — 2026-11-22

The resolver retains each verified JSON artifact's bytes, hashes those bytes, and parses the same byte sequence. It does not reread a manifest or result after digest verification. This applies to the training manifest and result, selected-panel search manifest, and baseline Hamming search manifest. A read-count regression was RED before the change and passed on tested code SHA `57bf4f0deb90e1b0009b1a458968e5ca4096fe23` in the installed-wheel endpoint suite ([run 44983817156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983817156)); the pinned producer compile/handoff CPU gates also passed ([44983817105](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983817105), [44983819836](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983819836)). This is CPU handoff evidence, not a trained GPU replay.

## Per-search CNOT backend-mode snapshot (2026-11-22)

The controller votes the strict or compatibility mode during backend admission and stores that decision in a Rust-owned per-search admission object. Stream 5 startup, native-stage construction and Rust scorer dispatch consume the stored decision. Production execution does not reread the process environment after admission, so a rank-local environment change cannot change the backend requirements partway through a search. Legacy direct extension calls that omit an admission object retain the environment-based compatibility fallback.


## Single CNOT dispatch decision (2026-11-22)

After the existing rank-wide callable-capability vote, the PyO3 adapter converts
the public engine token and depth-range fact into the typed runtime input. It
calls `beam_runtime::cnot_search::select_dispatch` once; the discarded
preliminary engine selection has been removed. The typed route remains the
value passed to the executor. The Python `World` still owns process-group
creation and supplies Gloo/Torch adapters, so this change does not transfer
world or CUDA-handle ownership.

An installed-wheel regression checks the selector boundary and the existing
two-rank Gloo route/failure/retry cases. The test failed on
`edcafd46a1585deb81c4f6007db877015a699fae` and passed on
`9ce5954b6b1d6bd014a2af086dec71ff7686a255` ([run
44996962384](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44996962384)).
CUDA/NCCL behavior remains pending.

## Rust-owned permutation inference backend selection, 2026-11-22

The public permutation runner preparation now sends a typed device/model request to
`beam_runtime::inference_backend`. Python/Torch still observes the selected CUDA
devices and supplies their SM version and device name. Rust selects `libtorch` or
`cutlass` and enforces the existing dtype, ensemble and MLP Q-head compatibility
rules before native runner construction. The Python rule table remains only for
older installed extensions that do not expose this API. The selected route is
stored in the prepared profile and passed to the native build.

This slice moves runner backend selection only. Search stages, model scoring,
specialized inference kernels, precision policy, CUDA/NCCL ownership and public
result formats are unchanged. The installed-wheel test covers direct PyO3 policy
calls, legacy-extension fallback, error translation, and the normal
`prepare_runtime` caller with a discriminating Rust resolver response.


## Panel-v2 consumer binding and pinned producer CI — 2026-11-22

The handoff resolver reads and hashes each referenced artifact's bytes once, parses the same bytes, and now stores the validated frontier expansion bound in the resolved panel object. The public verifier passes that resolved object into search without loading the resolver or revalidating the search contract a second time. Its installed-wheel CPU regression runs two distinguishable synthetic panel receipts and observes the roots sent to search, the selected search width/shortlist/depth, and the scorer's manifest and dimension. Rank consensus and independent replay remain required.

The Vast preflight still validates the private input before build, because it protects the long build stage. At runtime, the public verifier resolves the current receipt once and compares its checkpoint/manifest digests with the preflight pins; the native scorer still checks the pinned manifest itself. The shell no longer hashes that same manifest a third time.

The producer compatibility workflow keeps one stable required PR check on every PR update. It runs the cheap CPU classifier and synthetic handoff tests first. Its input fingerprint includes the pinned producer/adapter files, resolver pins, handoff fixture and workflow. A matching prior success is reused for docs-only changes; a matching failure remains failed; missing or changed inputs run the full pinned build. Manual dispatch always runs full. Status is stored in the Actions cache by fingerprint and uploaded as a small redacted artifact; CPU logs are separate artifacts. Physical GPU behavior is outside this one-off.


## Producer source and emitter ABI compatibility — 2026-11-22

The consumer supports producer source revisions `42ed72f587ab70fc74a85cf0a25001efb29cb994` and legacy `7310b1eb5ec07c76c131ba8c60b105ff4aad7a9c`. Each source revision maps separately to emitter ABI SHA-256 `2a6f579be0b1426a451aadf06709fa027b17eb7170f7b694d945acf81e070bf1`. The older synthetic receipt exercises legacy acceptance; negative tests cover unsupported source revisions and a mismatched emitter ABI. Hosted CPU compatibility uses the current emitter commit. The installed-wheel public verifier also consumes a receipt generated by that pinned emitter, then independently replays the result.


## CNOT panel preflight snapshot — 2026-11-22

The Vast panel acceptance resolves the private producer receipt and sealed dataset once before the build. That preflight emits only the selected panel roots, validated search settings, artifact digests and a bounded expansion value into a compact run-local snapshot. The acceptance wrapper hashes the snapshot once; the one-rank and two-rank public verifier invocations consume those exact bytes and reject a changed snapshot. Workers no longer reopen, hash and parse the full frozen dataset. The Rust scorer still validates the resolved search manifest against the pinned digest.

The CPU read-count regression models the 1-rank plus 2-rank invocation pattern and requires exactly one full sealed-dataset read across preflight plus all three rank-local snapshot loads. The installed-wheel endpoint test exercises the same snapshot CLI path used by Vast. Physical acceptance remains pending.

The producer compatibility fixture now changes the baseline Hamming manifest's source revision and digest before constructing its comparison when it simulates a different producer source pin. This keeps producer source, baseline manifest, comparison receipt pins and emitter ABI aligned; runtime source/ABI admission remains strict.


## Rust-owned inference profile resolution (2026-11-22)

The installed-wheel acceptance check exposed that the Rust extension exported backend selection but not the typed profile plan consumed by the preparation path. Rust now returns the selected backend together with the existing profile name and inference row limit. The Python adapter uses that single plan for runner construction and the prepared profile. Older extensions retain the existing backend-selection fallback and the same fixed profile mapping. No inference kernel, numerical policy, or beam-search behavior changes. CPU/PyO3 verification passed through the installed-wheel policy and preparation path on tested SHA a0f406238515ffab5d0c45b9ae4b0d5dc5011931 (run 45040185404). Physical CUDA/NCCL verification remains pending.

## Rust-owned public scheduler rank resolution

The public `cluster.scheduler_rank` helper delegates torchrun/Slurm environment parsing to `beam-runtime` when the installed PyO3 extension exports the versioned resolver. Rust returns global, node, and local rank plus world/local-world sizes; Python keeps the existing `ScheduledRank` object and uses the original parser only for older wheels. Invalid scheduler input still raises `ValueError` at the Python API. A runtime failure from a present resolver is an error and does not fall back to a different rank assignment. This transfer does not change process launching, CUDA device binding, NCCL, or search behavior.


## Rust-owned torchrun and Slurm scheduler rank resolution

The public `cluster.scheduler_rank` path uses the typed `beam-runtime` resolver for torchrun and Slurm environments. Python preserves `ScheduledRank`, its `ValueError` contract, and fallback only when an older wheel lacks the resolver. Installed-wheel validation on source SHA `a96be795a011174d7f244c06f1593a62b16e6c27` passed in [run 45052314835](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052314835); Rust CPU/Gloo and hosted acceptance passed in [run 45052314834](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052314834) and [run 45052315268](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052315268). The physical CUDA/NCCL job was skipped. This transfers scheduler metadata only; it does not change device binding, process launch, transport, or search behavior.


## Rust-owned scheduler-to-CUDA ordinal mapping

`ScheduledRank.cuda_ordinal` now delegates its existing single-visible-device and local-rank rules to `beam-runtime` through a versioned PyO3 function. The public method, error text, and old-wheel compatibility fallback remain unchanged. This only maps an already assigned scheduler rank into the process-local CUDA namespace; it does not bind devices, alter `CUDA_VISIBLE_DEVICES`, launch ranks, or initialize NCCL. The extension's ordinary rank resolver and CUDA ordinal API use separate version constants.


The scheduler-to-CUDA ordinal transfer was validated on code SHA `935016b513a742067f3be11315b7013a322dbefc`: Rust CPU/Gloo passed in [run 45054019605](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054019605), the installed PEP 517 wheel passed in [run 45054019597](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054019597), and hosted acceptance passed in [run 45054020205](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054020205). The physical GPU job was skipped.


## Rust-owned CUDA visibility translation

`backend.child_environment` obtains the child process's `CUDA_VISIBLE_DEVICES` mapping from a versioned `beam-runtime` function. The Rust rule preserves selected-index order, token trimming, duplicate selections, inherited `CUDA_DEVICE_ORDER`, and the previous rejection of out-of-range, empty, or disabled tokens. The Python API and child environment isolation remain unchanged; older extensions without the resolver use the existing Python rule. This translates the scheduler-provided visibility mask and does not select or bind physical devices.


The Rust-owned child CUDA visibility translation passed Rust CPU/Gloo in [run 45055128201](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128201), installed-wheel validation in [run 45055128252](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128252), and the hosted acceptance harness in [run 45055128570](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128570) on source SHA `db3e1beb0027b4c4ef42c80b87468fdf3630a07f`. The physical GPU and private producer jobs were skipped.


## Rust-owned permutation child environment

The public `backend.child_environment` now delegates reserved-variable filtering, CUDA visibility translation, retention of `CUDA_DEVICE_ORDER`, and the NCCL ID path to a versioned Rust runtime resolver. It returns the same child environment map and keeps the Python builder for older wheels. The resolver removes inherited rank, BEAM, torchrun, CUDA, NCCL, publish/results-ingest, shard and stream controls before adding the selected CUDA mask and run-local NCCL path. This covers the permutation worker's environment preparation; it does not start a child or change CUDA/NCCL algorithms.

Installed-wheel verification passed on code SHA `1e8b88cf6f188171bbe583875a400fca9095c5f8` in [run 45056737762](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45056737762). The tests separate direct `backend.cuda_visibility` mapping from the public `backend.child_environment` path, which exercises the combined Rust resolver, filtering, NCCL ID path, old-wheel fallback and error handling. The same installed run passed the saturated permutation CPU acceptance cases; it does not verify physical CUDA/NCCL.


## Rust-owned permutation search phase selection — 2026-11-23

The public permutation runner now asks a versioned `beam-runtime` resolver which existing preparation phases apply: inference calibration, fixed-width capacity discovery, maximum-frontier refinement, downstream calibration, and working-capacity selection. The returned ordered plan is consumed by `backend._run_native_once`; Python retains the existing calibration functions and launches them in the same order. The resolver preserves the current MLP calibration-binary and ensemble predicates, the `beam_width="max"` behavior, and the delayed unsupported-calibrator errors at their existing phase boundaries. Older wheels without the resolver keep the matching Python compatibility plan; errors from a present Rust resolver are not converted into fallback.

This transfers stage selection only. It does not change model inference, precision, GPU kernels, beam semantics, CUDA device binding, NCCL transport, Python/Kaggle APIs or CLI/scheduler interfaces. Rust CPU phase-plan tests passed on [run 45135569729](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45135569729); the installed PEP 517 wheel passed on source SHA `258cd9f3428edd2fb89212c92290ac59a520527a` in [run 45135335424](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45135335424). The installed-wheel step exercised the real Rust extension and public runner phase-plan helper. Physical permutation CUDA/NCCL on one and two GPUs remains unverified. CNOT continuation and final acceptance remain deferred.


#### Rust-owned permutation beam-width preparation — 2026-11-23

The public permutation runner now obtains the ordered preparation phases and the width used for resource planning from a typed beam-runtime execution plan. Fixed beam requests pass through unchanged. The existing beam_width="max" request resolves to the same 1,048,576 calibration width before microbatch planning; the maximum-frontier search and working-capacity algorithm are unchanged. The installed Rust resolver is authoritative when present, while older wheels retain their matching phase and width compatibility path. Rust CPU execution-plan tests passed (3/3) in [run 45139333020](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139333020), artifact [14800709775](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139333020/artifacts/14800709775). The installed PEP 517 wheel passed on source SHA 565694c013863238bb9ce38e403e35dfd1231221 in [run 45139332961](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139332961), redacted status artifact [14801331950](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139332961/artifacts/14801331950). The installed runner used the Rust width plan before microbatch preparation. Physical permutation CUDA/NCCL was not run.


#### Rust-owned permutation rank-planner session — 2026-11-23

The public \`NativePlanSession\` adapter now uses the matching installed \`beam-runtime::PlanningSession\` when available. Rust owns the rank processes, rank environment, rendezvous identity, request/response exchange, deadline, cancellation, all-rank admission validation, component two-phase handshake, and process cleanup. Python retains the same adapter behavior and the compatibility path for wheels without this API. Errors from a present Rust planner remain errors; capacity rejection maps to the existing Python capacity-rejection type.

The installed PEP 517 wheel was built and its external smoke step passed on code SHA \`7304818d0ab7860869053ef01dda5af71ad44ed1\` in [run 45142440137](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45142440137). The added CPU fixture made two admissions on the same two Rust-owned rank PIDs and compared its first result with the legacy adapter. The broader combined workflow was still running its remaining CPU/Gloo checks when this note was written. Physical permutation CUDA/NCCL was not run.
