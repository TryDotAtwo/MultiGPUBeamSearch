# CNOT distributed setup preflight — GitHub evidence

## Tested source

- Repository: `TryDotAtwo/MultiGPUBeamSearch`
- Branch commit: `59f52f7b9686e9d8bfc53e820f5fad3bf95058fb`
- Main commit before integration: `dbafbd62480aa044f025e764e04967682d3e25c9`
- Final main commit will be recorded after its hosted rerun.

## Change

The Rust CNOT top-level controller coordinates failures while constructing the
identity, resolving the native stage runner, and building compatibility stages.
All three operations use the existing checked CNOT path. Capability negotiation
uses the stable Gloo control group through `World.all_ranks_support`. Search
semantics, scores, precision, candidate ordering, and NCCL payload transport
remain unchanged.

## Hosted checks

- RED: [run 44750014338](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44750014338)
  failed the new rank-wide setup assertions on the pre-fix implementation
  (4 failed, 296 passed, 2 skipped).
- Intermediate [run 44750990888](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44750990888)
  stopped at rustfmt before compiling or running tests.
- Intermediate [run 44751261363](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44751261363)
  passed Rust checks and PyO3 build; three old test expectations still asserted
  the previous data-plane capability vote.
- GREEN: [run 44751923980](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44751923980)
  passed Rust CNOT checks, PyO3 build, 300 Python compatibility tests
  (2 skipped), and NCCL probe syntax.
- Artifact: [cnot-rust-domain-fallback-cpu-validation.zip](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44751923980/artifacts/14636575322),
  SHA-256 `d69aa55acf31b468732a5ce30dfb0318ddab15d9bb07eda06e21922e70fc3f19`.

## Acceptance boundary

This is hosted CPU/Gloo and build evidence. Physical CUDA/NCCL behavior,
checkpoint inference, and physical 1-/2-GPU search were not run. No Vast
instance was started or rented for this change.
