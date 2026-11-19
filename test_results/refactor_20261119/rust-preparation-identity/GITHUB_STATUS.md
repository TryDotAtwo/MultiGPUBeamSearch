# Rust preparation identity — verification

Implementation commit: [dc9ffd4](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/dc9ffd4cde45fcb993bf200e97181645a5fefb61).

## RED

- [Run 44445381559](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44445381559): 2 intended Python failures because the old implementation passed `repr(tuple)`, which is not JSON; 77 passed, 1 skipped.
- [Run 44445580149](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44445580149): the Rust preparation test failed on structural equality for reordered JSON keys, as expected.

## GREEN

- [Run 44446065188 — Rust public owner CPU validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44446065188): 79 passed, 1 skipped. Artifact 14492743778, `rust-session-reuse-cpu-validation`.
- [Run 44446065116 — CNOT CPU compatibility](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44446065116): Rust formatting/checks and PyO3 build passed; 174 Python tests passed, 2 skipped. Artifact 14492993116, `cnot-compat-root-goal-cpu-validation`.
- [Run 44446065115 — permutation depth control CPU validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44446065115): Rust test groups passed (5 and 2 tests). Artifact 14493132685, `permutation-depth-control-cpu-validation`.

Raw logs remain in the remote Actions artifacts. No artifacts were copied to Windows. These are hosted CPU/PyO3 checks; physical CUDA/NCCL and multi-GPU runtime were not exercised.
