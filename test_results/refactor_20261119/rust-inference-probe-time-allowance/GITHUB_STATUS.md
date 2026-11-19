# Rust inference probe-time allowance — verification

- RED: test revision `c8d0182fe32630681e53b39cbb51e59407b88bd2`; Actions run [44545990522](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44545990522) reported the missing Rust method and missing public delegation (5 failed, 195 passed, 2 skipped).
- Implementation: `48515594a6d91d38a469cdf5eaa882c6194eb17e`; rustfmt correction: `429c560dd9a196865ce4718bccfa8f000e70a597`; test expectation correction: `e6b0aa3c83e91e7a712820f50b014aa7ce96671d`.
- GREEN: Actions run [44547537516](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44547537516), success. Rust test groups passed 5 + 9 + 4 + 2 + 2 + 4 = 26; `cargo check -p beam-python`, PyO3 build, 200 Python tests (2 skipped), and NCCL probe syntax also passed.
- Artifact: [cnot-compat-root-goal-cpu-validation, ID 14541208218](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44547537516/artifacts/14541208218), SHA-256 `49c4cc1ca852f3df92eefa24d9195e202d4b3621cc6fc94ddfa68543bea9a0cf`.
- The Rust helper computes the same allowance from ordinary finite timing rows. Public Python dispatch preserves the old implementation for unavailable/older extensions and unsupported types.
- Calibration orchestration, selection, candidate ordering, cache policy, kernels, and rank protocol are unchanged.
- Limits: CPU/PyO3 only. NCCL probe syntax is not a collective run. No physical GPU, multi-GPU acceptance, or Vast compute was used.
