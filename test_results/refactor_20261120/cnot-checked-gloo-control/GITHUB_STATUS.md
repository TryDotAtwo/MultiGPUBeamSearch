# CNOT checked-error Gloo control

## Scope

The CNOT Rust compatibility dispatcher now recognizes the new
`_checked_error_gather_control` marker. Rust-owned initialize, expand, and
deduplicate/score stages remain enabled with the new checked-control ABI.
The two-rank controller fixture records the control vote and no longer expects
the old failure vote through the search-world `longs + sum` path.

Search payload exchange, NCCL transport, beam order, scoring, and numerical
policy are unchanged by this fix.

## Verification

- RED, [run 44744338479](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44744338479), commit `ace432595f287eaedc7f1b7644dd688cf4e32ebc`: 12 Python compatibility tests failed. Eleven Rust compatibility checks still tested the retired v2 capability marker; one two-rank fixture lacked the new control vote method.
- RED, [run 44745437675](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44745437675), commit `0645a0ffcaecb6efeb76a58ef162bc9c5206b089`: Rust checks, PyO3 build, and 296 Python tests passed; one controller test still expected the old search-world error vote.
- GREEN, [run 44746148112](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746148112), commit `84acfce3c24546a698c5796f1199fd40c41542e2`: Rust CNOT checks, PyO3 build, 297 Python compatibility tests passed, 2 skipped, 104 warnings, and NCCL probe syntax passed.
- Artifact: [cnot-rust-domain-fallback-cpu-validation.zip](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746148112/artifacts/14632339297), SHA-256 `db25f1da847170b2bf06d64736966023bbb2fff28a3e39bce14e60ae8ee11f26`.

## Acceptance boundary

This verifies hosted CPU/Gloo behavior, Rust tests, and PyO3 compilation.
It does not verify physical CUDA/NCCL behavior, multi-GPU failure recovery,
or installed-wheel runtime behavior. The configured Vast instance was exited;
no instance was started or rented.
