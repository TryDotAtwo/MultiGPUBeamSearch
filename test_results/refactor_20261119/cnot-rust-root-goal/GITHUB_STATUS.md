# Rust CNOT root-goal dispatch — GitHub evidence

Scope: The Rust compatibility backend checks typed `CnotSearchInput` with the existing `search_input_is_identity` helper. Python-only inputs and older stage providers keep the Python callback. Search order, tensor ownership, CUDA operators, scorer behavior, and distributed collectives are unchanged.

- RED: [Actions run 44585776893](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44585776893), 1 failed, 218 passed, 2 skipped. The expected failure was the Python `root_is_goal` callback. Artifact ID: 14559869836.
- GREEN: [Actions run 44586229891](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44586229891), Rust checks and PyO3 build passed; selected CPU suite: 219 passed, 2 skipped. Artifact ID: 14561312765.
- [Actions run 44585189616](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44585189616) passed before the test forced the compatibility-stage path; it is not counted as RED/GREEN evidence.
- No physical CUDA/NCCL or multi-GPU run occurred. [Vast run 44581742018](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44581742018) skipped GPU work because instance 61080396 was not running; no start or rent was requested.

Full logs remain in GitHub Actions artifacts. No logs or build outputs were downloaded to Windows.
