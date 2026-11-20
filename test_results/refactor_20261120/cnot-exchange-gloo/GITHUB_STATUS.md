# CNOT record exchange preparation

Implementation commit: `7da474ac2e38f62ee942985b8cde3a64207704d7`  
Regression-test commit: `61a374dfe4572adb0ea77b82822cba3e51cb7f35`

## Covered failure boundaries

Two two-rank Gloo tests exercise the fallback record exchange:

- **Packet preparation:** rank one fails in local `torch.bincount` while building
  its send packet. The earlier RED run showed rank zero reaching all-to-all and
  receiving a peer-closed Gloo error. The fixed path agrees on the local
  preparation result before entering the data-plane collective.
- **Payload preparation:** after the ranks exchange send counts, rank one fails
  to allocate its receive tensor. The fixed path agrees on size conversion and
  allocation before starting the payload all-to-all.

Both green cases check that each rank receives the same stage diagnostic and
that the Gloo group can complete another all-reduce afterward.

## Hosted evidence

- Expected RED for packet preparation:
  [run 44797521897](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44797521897)
  failed because rank zero received `Gloo ... Connection closed by peer`.
- CPU/Gloo GREEN:
  [run 44798848987](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798848987)
  passed 304 tests, 2 skipped. Artifact
  [14661050464](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798848987/artifacts/14661050464),
  SHA-256 `eee8773cdfa27fb8e5da41ec48a5afceaf6c8b3a4299520f7328f2c87c504a40`.
- Installed wheel:
  [run 44798339035](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798339035)
  built and installed the PEP 517 wheel outside the checkout, passed
  `pip check`, public import/export checks, and worker protocol
  `resident-search-v1`. Artifact
  [14660825488](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798339035/artifacts/14660825488),
  SHA-256 `525eaeeeb26246dc0f4b70675748e9c5fd960818c261e4d46aafb0d157354469`.
- Read-only existing-Vast check:
  [run 44798339107](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798339107)
  found instance 61080396 exited; physical GPU checks were skipped without any
  instance state change.

The Gloo votes cover rank-local preparation around the collectives. They do not
prove behavior when a low-level CUDA/NCCL data-plane collective itself fails.
Physical 1-/2-GPU exchange remains unverified.
