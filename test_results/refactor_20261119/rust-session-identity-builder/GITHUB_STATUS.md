# Rust resident SessionIdentity builder

## Scope

The optional Rust/PyO3 builder now assembles and serializes the typed launch `SessionIdentity`. `RustResidentWorkers` uses it when present; older extensions retain the previous Python JSON payload. No search algorithm, ranking, numerical policy, rank order, CUDA stage, or NCCL protocol is changed.

## RED verification

- Test-only commit: `420d39f47d35bc64f214e45e068af0f0097f3984`.
- Python owner run: [44524014350](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44524014350), concluded failure.
- CNOT CPU run: [44524013917](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44524013917), concluded failure.
- These are run-level metadata only. Full logs were not downloaded, and no assertion-level cause is claimed.

## GREEN verification

Pending hosted Actions for the implementation commit.

## Limits

No local build or test was run. These checks do not establish physical CUDA/NCCL execution or multi-GPU CNOT behavior.