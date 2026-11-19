# Typed CNOT goal-stage dispatch — GitHub evidence

Scope: For typed Rust `CnotSearchInput`, the Rust compatibility backend calls the shared `cnot_native_goal::goal` helper used by the native stage owner. Python-only input and older stage providers keep the existing Python callback. Goal semantics, tensor operations, error coordination, stage order, and collectives are unchanged.

- RED: [Actions run 44589295992](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44589295992), 1 failed, 219 passed, 2 skipped. The expected failure was the Python `CnotStages.goal` callback. Artifact ID: 14563611172.
- GREEN: [Actions run 44589852872](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44589852872), Rust checks and PyO3 build passed; selected CPU suite: 220 passed, 2 skipped. Artifact ID: 14562407176.
- The regression test disables native-stage dispatch so it exercises the Rust compatibility backend. No physical CUDA/NCCL or multi-GPU run occurred.

Full logs remain in GitHub Actions artifacts. No build outputs or logs were downloaded to Windows.
