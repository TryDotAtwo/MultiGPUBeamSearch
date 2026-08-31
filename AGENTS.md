# Project Agent Rules

## Operating Rule
- Read `AGENTS.md` before project edits.
- Update `memory/CHANGELOG.md` after meaningful code, config, or architecture changes.
- Update `memory/PROMPTS.md` with user prompts that define requirements.
- Store test outputs, logs, and verification notes under `test_results/`.
- Prefer agent-centered code: explicit contracts, deterministic tests, small files, low hidden state.

## Architecture Source
- Primary architecture contract: `ARCHITECTURE_NEED.md`.
- Current Stream 4 implementation rule: use threshold + compact + CUB/fixed-temp sort/reduce + compact; do not use Stream 4 hash-table dedup.

## Implementation Guardrails
- The default Megaminx build has `STATE_LEN=120` and `STATE_STORAGE_LEN=128`:
  logical bytes are `v[0..119]` and padding is `v[120..127]`.
- Shape-specialized builds use `STATE_LEN=N` and an aligned
  `STATE_STORAGE_LEN >= N + 4`; `State128` is the retained compatibility alias
  for this build-specific `StatePacked` layout.
- Persistent frontier padding `v[STATE_LEN..STATE_STORAGE_LEN-1]` must be zero.
- `FinalResponse` may temporarily store `target_local_idx` in
  `v[STATE_LEN..STATE_LEN+3]`.
- `Hash128` is one logical 128-bit key stored as `{lo, hi}`.
- `CandidateMeta` must remain 32 bytes and 32-byte aligned.
- Stream 3 payload id must be original candidate id, not compact index.
- Stream 4 must not apply shard top-k or semantic shard cap.
