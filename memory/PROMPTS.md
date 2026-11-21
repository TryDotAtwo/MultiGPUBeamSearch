2026-11-19: Ivan authorized installing the repository Actions secret `VAST_API_KEY`. The current hosted Actions environment has already authenticated with this secret; the existing value must not be exposed in code, command arguments, or logs.

2026-11-19: Ivan asked to continue the Rust + CUDA/C++ rewrite, including CNOT, without optimization; preserve multi-GPU architecture and Python compatibility, and synchronize verified changes with GitHub.

2026-10-10 user requirement: Update existing public Kaggle notebook and leave only one GitHub branch in MultiGPUBeamSearch. Preserve checked source and historical reproducibility evidence. No additional GPU rentals.

## 2026-10-10 GNN requirements
Implement our own support for the two-level GNN architecture in issue #5, without using the submitted implementation as runtime. Provide both LibTorch and CUTLASS backends and preserve the five-stream search architecture.


GNN acceptance rental authorized: up to $5 total including disk/traffic. Implement our own GNN from issue5 with both LibTorch and CUTLASS; preserve beam architecture and do not use subagents.


2026-10-10: User requested running a large GNN comparison after the tiny fixture; approved a new RTX3060 rental up to $5 total. Compare matched Python/PyTorch, native LibTorch and native CUTLASS inference, archive results via GitHub and remove own rental.


## 2026-10-10: native GNN CUTLASS fusion
Requirement: improve large GNN native CUTLASS by at least 2x, preserving beam architecture and numerical correctness; profiling rental authorized up to $5.


## 2026-10-10: GNN fused rounded bias
User requested separate bias removal, RTX3060 speed/precision verification, an active goal and authorized $5 total for new tests.


## 2026-10-10: further GNN optimization with Astra
Continue improving GNN while measured gains remain, consult Astra as expert, preserve architecture; new GPU-test budget $10 total.

2026-10-11: Fix slow results publication and misleading published status. Keep the checked Cube555 solver and GitHub-only inputs; no new GPU execution.

2026-10-25: User authorized publishing the verified Cube555/API/autotuner changes to GitHub and updating the existing public Kaggle notebook. Use GitHub-pinned source and inputs; preserve requested beam and explicit blend.

2026-10-27: Run two Cube555 Kaggle searches at beam 2,500,000 on 2×T4 using the reference blend notebook settings: a legal scramble around 10–12 moves and a puzzle around 100 moves. Do not silently reduce requested beam. For unstable component-calibration baseline timings, permit one bounded retry and report failure explicitly if still unstable.

2026-10-31: Ivan asked whether the Cube555 helper can be omitted so the notebook is “through CayleyPy” rather than exposing unfamiliar MultiGPUBeamSearch modules. Keep graph search on CayleyPy's graph.beam_search API; puzzle/model loading and result handling may be optional convenience helpers, while native multi-GPU activation/session still requires the current adapter. Do not claim the public interface is CayleyPy-only until CayleyPy or its official optional backend exposes that native integration.


2026-10-31 clarification: Keep `import multigpubeamsearch`. Remove only the puzzle-specific `multigpubeamsearch.cube555` helper; configure other problems with a CayleyPy graph, supported predictor/model, and start state, then use the common beam-search API.


2026-10-31 clarification: Keep `import multigpubeamsearch`. Make the public notebook specifically for Cube555, with a straight-line graph/model flow and no generic input branches or repeated guard code. Retain the useful first-cell puzzle, model/blend, beam, collect, BFS, reflection, GPU and publication settings. Save a Kaggle Quick Save without running the search.

2026-11-06: Ivan requested completion and closure of GNN issue #5. Compare both Level-1 encoders and the existing baseline under matched training and evaluation conditions. Separately assess the architecture required for CNOT issue #6.


2026-11-06: Ivan asked to implement what is feasible for issue #6 and ensure the code works with more than eight GPUs, including GPUs distributed across multiple nodes.

## 2026-11-07 — approved Rust/CUDA refactor
User requested implementation of the accepted unified-main + Rust/CUDA/C++ plan in docs/plans/2026-11-07-rust-cuda-runtime.md. Preserve existing specialized inference, public Python adapters, all supported GPU families and algorithmic semantics. Total paid-compute budget USD 50. Follow-up: builds and tests on Vast, all local artifacts on D. Follow-up: publish necessary source to GitHub and keep the local checkout synchronized with the latest main.


2026-11-07 resource correction: Ivan questioned the 2xRTX3090 choice and preferred cheaper RTX3060 instances. Use the cheapest adequate remote GPU configuration, keep the USD50 total cap, and retain the earlier instruction to publish verified source and synchronize local main.

2026-11-16: «Продолжай переписывание на раст. Пока просто переписываем. Ничего не улучшаем». Continue the Rust migration while preserving existing algorithms, CUDA kernels, numerical behavior, interfaces and multi-GPU semantics; defer performance optimizations and benchmarking.

2026-11-16 follow-up: Fully transfer beam search, including CNOT, to Rust and
CUDA/C++, while preserving the multi-GPU architecture and Python/Kaggle
compatibility. Publish verified changes to GitHub and synchronize local main.
Keep an active goal until the migration and required verification are complete.

2026-11-16: Ivan requested completing the Rust + CUDA/C++ migration, including CNOT, without optimizations or changes to multi-GPU architecture and compatibility. Retain Python bindings for Kaggle and other environments.

2026-11-17 remote-only execution: Ivan instructed that Windows may retain repository code only. Do not download datasets, weights, checkpoints, logs, results, archives or build outputs. Use GitHub-hosted Actions and the already-owned Vast instance; check current status and never start, stop or rent it. No local SSH/SCP; keep raw logs on Vast and expose only sanitized Actions reports. Ivan explicitly approved storing the existing Vast API key as the repository Actions secret.


2026-11-17 follow-up: Windows may contain repository source and small source-controlled instructions only. Run builds and checks remotely through GitHub Actions and the already-owned Vast instance. Do not create local datasets, weights, checkpoints, logs, result archives, dependencies, binaries or build outputs; preserve full logs/results remotely in test_results/ and Actions artifacts, leaving only small verification notes in the checkout. Check instance status before access; do not start, stop or rent compute without a user-set budget. Keep Python compatibility and the multi-GPU architecture; continue Rust + CUDA/C++ migration without optimization.

2026-11-18 follow-up: Ivan confirmed use of the existing `VAST_API_KEY` as the GitHub Actions repository secret. A working status workflow already verifies that the secret is available; do not print or copy its value into source or logs.
2026-11-18: Ivan reaffirmed the full Rust + CUDA/C++ migration including CNOT, with no new optimizations; preserve multi-GPU and Python/notebook/CLI/torchrun/Slurm compatibility, publish to GitHub main and synchronize the local checkout. Replacement AGENTS instructions require source-only local work, hosted GitHub Actions for builds/tests, logs and test artifacts in remote `test_results/` and Actions artifacts, and spending only within the user-set budget.

2026-11-19: Иван попросил продолжать полный перенос beam search на Rust + CUDA/C++, включая CNOT, с сохранением multi-GPU и публичной Python-совместимости; пока переносим существующее поведение без оптимизаций.

2026-11-19: Иван подтвердил запись уже имеющегося Vast API key в зашифрованный GitHub Actions secret `VAST_API_KEY`. Секрет передан через stdin; его значение не добавлялось в исходники и логи. Это разрешение не запускает оплачиваемые GPU-инстансы.

2026-11-20: Иван ответил «Я разрешаю» на запрос об использовании существующего Vast API key в зашифрованном GitHub Actions secret VAST_API_KEY; значение не раскрывать.


2026-11-20: Ivan сформулировал цель довести MultiGPUBeamSearch до проверенного запуска permutation и CNOT на Rust + CUDA/C++ со штатным checkpoint scorer. Сохранять мульти-GPU, Python, CLI, Kaggle, torchrun и Slurm, алгоритмы и precision. Исправлять публичные API и сбои распределённого запуска/обмена/отмены регрессионными тестами; сначала CNOT, где ранги могут выбрать разные коллективы. Проверить установленный wheel и физические 1/2 GPU, публиковать проверенные изменения в GitHub. Оптимизации не входят; соблюдать действующие ограничения ресурсов и бюджета.

2026-11-21: Иван поручил продолжать полный перенос MultiGPUBeamSearch на Rust + CUDA/C++, включая CNOT, без оптимизаций, с сохранением мульти-GPU и Python-совместимости и синхронизацией с GitHub. Текущие обязательные этапы: строгая per-rank CNOT backend admission с ABI/hash/capabilities и фактическими DeviceDomain/Stream5 dispatch evidence; исправление PyO3 dtype проверки; rank-wide Gloo votes на winner packet/history подготовке; безопасная приёмка только нового 2x RTX 3060 под оставшимся лимитом, без использования старого instance. При отсутствии предложения продолжать CPU/Gloo работу.
