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

## Main commit revalidation

- Main commit: `d7fd23dda968ae47cb2346f2c96e2c26b3d2b75c`, with the same code tree as the tested feature commit.
- [CNOT CPU run 44746884936](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884936) passed Rust CNOT checks, PyO3 build, 297 Python compatibility tests (2 skipped, 104 warnings), and NCCL probe syntax. Artifact [14632634476](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884936/artifacts/14632634476), SHA-256 `596a1c081a1e8f48b0a692d50a6a16d50a4a8fb8170b407ad0a73f269ddb4dd2`.
- [Installed wheel run 44746884980](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884980) built and installed the PEP 517 wheel, passed `pip check`, and ran smoke checks outside the checkout for package exports, Rust ABI, worker protocol, AutoMax preparation identity, and two resident-session requests with a fixture worker. Artifact [14633756534](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884980/artifacts/14633756534), SHA-256 `23973b113206b82b9e9bbe32ee286088a8324b13621eee7e00e97b00a8801089`. This is an installed CPU wheel smoke, not model inference.
- [Vast status run 44746884926](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884926) found instance 61080396 exited. It skipped CUDA/NCCL execution and made no start, stop, or rent request. Artifact [14633197432](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884926/artifacts/14633197432), SHA-256 `f3858af2cb34b6b6f2c7c38f12109403e78bc793b7668ca8a81c562790ea5cc9`.

## Acceptance boundary

Verified: hosted CPU/Gloo behavior, Rust CNOT checks, PyO3 compilation, installed-wheel CPU smoke, and NCCL probe syntax.
Unverified: physical CUDA/NCCL execution, two-GPU runtime behavior, and rank-wide physical failure recovery. The configured Vast instance was exited; no instance was started or rented.
