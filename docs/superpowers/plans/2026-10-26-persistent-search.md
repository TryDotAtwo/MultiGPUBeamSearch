# Persistent search implementation plan

Goal: reuse native workers, model weights, static GPU buffers and CUDA Graphs across sequential public searches. No changes to selection/dedup architecture.

Architecture: explicit NativeSession context binds calls via a context-local owner. Extend the existing resident task loop using private atomic JSON request files, per-rank completion receipts and per-request logs. Compatible graph/model/runner/devices/beam/depth/options reuse workers; incompatibility closes all ranks before new admission. Each task resets native state and history. Session failure invalidates the entire worker group. close is idempotent and bounded.

- [ ] Native request protocol and reset lifecycle.
- [ ] Python single-owner transport, context API, backend reuse with exact compatibility.
- [ ] Unit/integration tests for sequential requests, incompatibility, corruption, timeout, closure and replay.
- [ ] Native compilation and bounded GPU acceptance where available; do not claim GPU acceptance from CPU tests.
- [ ] Document API and evidence, publish only verified scope.
