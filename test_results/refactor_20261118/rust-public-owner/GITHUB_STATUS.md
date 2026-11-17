# Rust public owner migration — verification

- RED: [Actions run 44181078596](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44181078596) failed as intended: 2 missing Rust-default behaviors failed and 2 compatibility cases passed.
- GREEN: [Actions run 44182404997](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44182404997) passed 78 tests in 5.19 s across `test_rust_default_owner.py`, `test_dispatch.py`, `test_persistent_search.py`, and `test_rust_persistent_search.py`.
- Artifact: [rust-public-owner-cpu-validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44182404997/artifacts/14372272952), archive SHA-256 `40342afa5079cfc7121667be570a7d818c02f2f2c7fae52f9f1af2b34790f73f`.
- The hosted runner used CPU PyTorch. Rust capability selection was mocked in the new tests; the run did not build or start the Rust worker and did not test CUDA, NCCL, or multi-GPU execution. No performance claim follows from it.
