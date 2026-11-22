# Rust inference candidate ordering — verification

- RED revision: `a1bdadacc95f617eb81ea02b386aa1cb856f32bf`; Actions run [44540169138](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44540169138) failed on the expected missing PyO3 methods and public routing.
- Implementation: `8f1f20bad1a59b61548b08a03fd4375fd013d780`. Formatting-only follow-ups: `b6019bb557f9e5a6c2e7e881ca81b3ed0a82fceb` and `e6f492bb56df6ed8610c7f14898c0e0045792c0a`.
- GREEN: Actions run [44543777970](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44543777970), commit `e6f492bb56df6ed8610c7f14898c0e0045792c0a`, conclusion success. It passed rustfmt, selected Rust test groups (5 + 9 + 4 + 2 + 2 + 4 = 26 tests), `cargo check -p beam-python`, PyO3 extension build, 194 Python tests (2 skipped), and NCCL probe syntax.
- Actions artifact: [cnot-compat-root-goal-cpu-validation, ID 14540030869](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44543777970/artifacts/14540030869), SHA-256 `5a2ee6e102e6eed259bac31c9d04dcc8732442a1734313e54458d386ff1368ff`.
- Rust now reproduces CPython `random.Random` ordering for coarse and refinement candidate batches. Optional PyO3 builders are called by public Python wrappers; Python remains the fallback for older/unavailable extensions.
- Remaining Python ownership: calibration request loop, cache policy, sample/trial execution, statistics, candidate ranking and admission. No inference kernel or numerical policy changed.
- Limits: CPU/PyO3 only. NCCL probe syntax is not a collective run; physical CUDA and multi-GPU acceptance were not performed. No Vast instance was started.
