# CNOT selection completion error coordination

Source commit: `ccd0657789e26ade55b5b5060dc9a409c44b8da1`

## Regression

A two-rank Gloo test injects an exception on rank one in the local ordering
performed after the global selection collectives. Before the fix, rank zero
advanced into candidate redistribution and observed a peer-closed Gloo
connection. After the fix, both ranks receive the same
`select CNOT frontier failed: rank 1: RuntimeError: ...` diagnostic, then
successfully complete a follow-up Gloo all-reduce.

The fix checks the completed result of the CNOT selection stage through the
existing Gloo control group before the next stage begins. It does not change
selection scores, global width, radix cutoff, lexical tie-breaking, or selected
candidates. Failures that happen inside selection before one of its internal
collectives remain outside this guard.

## Hosted evidence

- Expected RED: [run 44795813454](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44795813454)
  failed with rank zero reporting `Gloo ... Connection closed by peer` instead
  of the injected rank-one error. This demonstrated the missing coordination.
- GREEN CNOT CPU suite: [run 44796514366](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514366)
  passed 302 tests, 2 skipped. Artifact
  [14659601192](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514366/artifacts/14659601192),
  SHA-256 `87ae53a9f6ab336e160eff33ec3d7a1c6b282e5043614ffe03398d5b302e686f`.
- Installed wheel: [run 44796514465](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514465)
  built and installed the PEP 517 wheel outside the checkout, passed
  `pip check`, public import/export checks, and worker protocol
  `resident-search-v1`. Artifact
  [14658528465](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514465/artifacts/14658528465),
  SHA-256 `8b7d94b169c9f3bf82b95b94a91a7ec94bcbe037fc7ed83b99cee165a55b924e`.
- Read-only existing-Vast check: [run 44796514430](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514430)
  found instance 61080396 exited. GPU execution was skipped; no start, stop,
  or rent request was made.

These results cover hosted CPU/Gloo and installed-wheel behavior. CUDA,
NCCL, checkpoint inference on GPU, and physical 1-/2-GPU search remain
unverified.
