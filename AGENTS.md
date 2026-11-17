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
- `State128` logical bytes are `v[0..119]`.
- `State128` padding bytes are `v[120..127]`.
- Persistent frontier padding must be zero.
- `FinalResponse` may temporarily store `target_local_idx` in `v[120..123]`.
- `Hash128` is one logical 128-bit key stored as `{lo, hi}`.
- `CandidateMeta` must remain 32 bytes and 32-byte aligned.
- Stream 3 payload id must be original candidate id, not compact index.
- Stream 4 must not apply shard top-k or semantic shard cap.

## Remote Compute and Local Storage

- The Windows checkout may contain repository source, code, configuration and documentation. Do not download datasets, model weights, checkpoints, logs, run results, build outputs or archives to the PC.
- Run builds and remote commands through GitHub-hosted Linux Actions and the existing project Vast instance. Remote SSH/SCP may run only inside those Actions, never from the Windows checkout; do not open interactive windows for remote work.
- Check the fixed instance's current status before remote access. Workflows must not start, stop or rent instances. Keep raw private logs on Vast and publish only sanitized summaries to Actions; never publish credentials.
